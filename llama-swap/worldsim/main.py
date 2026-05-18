#!/usr/bin/env python3
"""
worldsim — Interactive web world model simulation using llama-swap.

Agent + World Model loop with sliding window context management.
The agent has conversation memory (knows what it already did).
The world model uses sliding window (last N turns only) to avoid context overflow.

Usage:
    llama-swap-cli worldsim
    llama-swap-cli worldsim --manual
    llama-swap-cli worldsim --agent qwen3.5-4b --world webworld-8b --steps 15
"""

import sys
import os
import json
import time
import re
import argparse
from typing import Optional

import requests
import questionary
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

# --- Config ---
BASE_URL = os.environ.get(
    "WORLDSIM_URL",
    f"http://{os.environ.get('LLAMA_SWAP_HOST', '127.0.0.1')}"
    f":{os.environ.get('LLAMA_SWAP_PORT', '12434')}/v1",
)

# WebWorld system prompt (from official model card)
WORLD_SYSTEM = (
    "You are a web world model. I will provide you with an initial page state "
    "and a sequence of actions. For each action, predict the resulting page state.\n"
    "Strictly maintain the original format. Output only the full page state "
    "without explanations, code, or truncation."
)

CONTINUE_PROMPT = (
    "Continue the trajectory. Given the previous state, "
    "predict the next page state after this action.\n\n"
    "Action: '{action}'\n\nNext Page State:"
)

# Agent system prompt — explains A11y format and task structure
AGENT_SYSTEM = """You are a web navigation agent. Your goal is to complete tasks on websites.

RULES:
1. You see page states in A11y Tree format. Elements have IDs in square brackets like [5], [13], etc.
2. Use these IDs in your actions — always with brackets, e.g. click([13]), fill([7], "text")
3. Available actions:
   - click([id]) — Click an element
   - fill([id], "text") — Type text into a field
   - keyboard_press("Enter") — Press a key
   - goto("url") — Navigate to URL
   - scroll(dx, dy) — Scroll the page
   - go_back() — Go back in browser history
4. If the task is complete or impossible, respond with: DONE <reason>
5. Output ONLY one action per turn. No explanations.
6. Look at the page state carefully — check cart counts, form values, page titles for progress."""

# Default initial states
DEFAULT_STATES = {
    "search_portal": """RootWebArea 'Global Start - Your Daily Portal', focused
\t[1] banner 'Top Header', visible
\t\t[2] link 'Set as Homepage', clickable, visible
\t\t[3] link 'Feedback', clickable, visible
\t\t[5] region 'Weather Widget', visible
\t\t\tStaticText 'New York, USA'
\t\t\t[6] image 'Sunny', visible
\t\t\tStaticText '24°C'
\t\t[8] link 'Sign In', clickable, visible
\t[10] region 'Search Area', visible
\t\t[11] image 'Global Start Logo', visible
\t\tStaticText 'Search the entire web'
\t\t[12] tablist 'Search Engine Selector', orientation='horizontal'
\t\t\t[13] tab 'Google', selected=True, clickable
\t\t\t[14] tab 'Bing', selected=False, clickable
\t\t\t[15] tab 'DuckDuckGo', selected=False, clickable
\t\t[18] combobox 'Web Search', clickable, visible, autocomplete='both', expanded=False
\t\t\t[19] textbox 'Type keywords or URL...', clickable, visible, editable, value=''
\t\t[20] button 'Search', clickable, visible
\t[30] navigation 'Category Bar', visible
\t\t[31] link 'Home', clickable, selected=True
\t\t[32] link 'News', clickable
\t\t[33] link 'Video', clickable
\t\t[34] link 'Shopping', clickable
\t\t[35] link 'Social', clickable
\t[50] main 'Site Directory', visible
\t\t[51] region 'Top Recommended', visible
\t\t\t[52] heading 'Most Popular', visible
\t\t\t[53] list 'Top Sites Grid', visible
\t\t\t\t[54] link 'Facebook', clickable
\t\t\t\t[56] link 'YouTube', clickable
\t\t\t\t[58] link 'Amazon', clickable
\t\t\t\t[60] link 'Twitter / X', clickable
\t\t\t\t[62] link 'Instagram', clickable
\t\t\t\t[64] link 'Wikipedia', clickable
\t\t\t\t[66] link 'Netflix', clickable
\t\t\t\t[68] link 'LinkedIn', clickable""",

    "github_homepage": """RootWebArea 'GitHub', focused
\t[1] banner 'Top Header', visible
\t\t[2] link 'Pull requests', clickable, visible
\t\t[3] link 'Issues', clickable, visible
\t\t[4] link 'Actions', clickable, visible
\t\t[5] textbox 'Search or jump to...', clickable, visible, editable, value=''
\t\t[6] button 'Sign in', clickable, visible
\t[10] main 'Content', visible
\t\t[11] heading 'Welcome to GitHub', visible
\t\t[12] link 'Explore repositories', clickable
\t\t[13] link 'Trending', clickable
\t\t[14] link 'Marketplace', clickable""",

    "shopping_site": """RootWebArea 'TechStore - Electronics', focused
\t[1] banner 'Navigation', visible
\t\t[2] link 'Home', clickable, selected=True, visible
\t\t[3] link 'Laptops', clickable, visible
\t\t[4] link 'Phones', clickable, visible
\t\t[5] link 'Accessories', clickable, visible
\t\t[6] link 'Cart (0)', clickable, visible
\t\t[7] textbox 'Search products...', clickable, visible, editable, value=''
\t\t[8] button 'Search', clickable, visible
\t\t[9] link 'Sign In', clickable, visible
\t[10] main 'Products', visible
\t\t[11] heading 'Featured Products', visible
\t\t[12] list 'Product Grid', visible
\t\t\t[13] link 'MacBook Pro 14" - $1,999', clickable, visible
\t\t\t[14] link 'iPhone 16 - $999', clickable, visible
\t\t\t[15] link 'AirPods Pro - $249', clickable, visible
\t\t\t[16] link 'iPad Air - $599', clickable, visible
\t[20] contentinfo 'Footer', visible""",
}

# --- Tasks per template ---
DEFAULT_TASKS = {
    "search_portal": "Find news about AI technology using the search",
    "github_homepage": "Search for Python repositories and find the most starred one",
    "shopping_site": "Find and add a MacBook Pro to the cart",
}


def chat_completion(model: str, messages: list, max_tokens: int = 2048,
                    temperature: float = 0.0, timeout: int = 180) -> tuple:
    """Call llama-swap chat completion. Returns (content, elapsed_seconds, total_tokens)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=timeout, stream=False)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, elapsed, tokens
    except requests.exceptions.Timeout:
        return "[TIMEOUT]", time.time() - t0, 0
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0, 0


def strip_reason(text: str) -> str:
    """Remove <reason>...</reason> CoT tags from world model output."""
    return re.sub(r"<reason>.*?</reason>\s*", "", text, flags=re.DOTALL).strip()


def extract_action(text: str) -> str:
    """Extract action from agent response. Returns 'DONE <reason>' or the action."""
    text = text.strip()
    # Check for task completion
    if re.search(r"\bDONE\b", text, re.IGNORECASE):
        return text.strip()
    # Try to find a Python-style function call
    match = re.search(
        r"(click|fill|goto|scroll|keyboard_press|select_option|hover|go_back|"
        r"send_msg_to_user|noop|infeasible)\([^)]*\)",
        text,
    )
    if match:
        return match.group(0)
    # Fallback: first non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else "noop(1000)"


def truncate_state(state: str, max_lines: int = 50) -> str:
    """Truncate A11y Tree state to max_lines for agent context."""
    lines = state.strip().split("\n")
    if len(lines) <= max_lines:
        return state
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def sliding_window(messages: list, keep_turns: int = 2) -> list:
    """
    Keep system message + last N turn pairs from world model context.
    Each turn = user + assistant message pair.
    """
    if len(messages) <= 1 + keep_turns * 2:
        return messages

    system = messages[0]  # Always keep system
    # Keep last keep_turns * 2 messages (user+assistant pairs)
    tail = messages[-(keep_turns * 2):]
    return [system] + tail


def run_agent(agent_model: str, state: str, task: str,
              action_history: list, step: int) -> tuple:
    """
    Run agent to decide next action.
    Returns (action_str, elapsed_seconds, tokens).
    """
    # Build context: task + current state (truncated) + history
    history_text = ""
    if action_history:
        history_text = "\n\nActions taken so far:\n"
        for i, a in enumerate(action_history):
            history_text += f"  {i + 1}. {a}\n"

    state_truncated = truncate_state(state, max_lines=40)

    user_msg = f"Task: {task}\n\nCurrent Page State:\n{state_truncated}{history_text}\n\nWhat action should be taken next? (output ONE action, or DONE if task is complete)"

    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    content, elapsed, tokens = chat_completion(agent_model, messages, max_tokens=128, temperature=0.1)
    action = extract_action(content)
    return action, elapsed, tokens


def run_world_model(world_model: str, state: str, action: str,
                    wm_messages: list, step: int, window_turns: int = 2) -> tuple:
    """
    Run world model to predict next page state.
    Uses sliding window on wm_messages to avoid context overflow.
    Returns (new_state, updated_wm_messages, elapsed_seconds, tokens).
    """
    # Build prompt for this turn
    if step == 0:
        user_msg = f"Initial Page State:\n{state}\n\nFirst Action: '{action}'\n\nNext Page State:"
        new_messages = [
            {"role": "system", "content": WORLD_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
    else:
        user_msg = CONTINUE_PROMPT.format(action=action)
        new_messages = wm_messages + [{"role": "user", "content": user_msg}]

    # Apply sliding window BEFORE sending (keep system + last N turns)
    new_messages_windowed = sliding_window(new_messages, keep_turns=window_turns)

    content, elapsed, tokens = chat_completion(world_model, new_messages_windowed, max_tokens=4096, temperature=0.0)
    new_state = strip_reason(content)

    # Update multi-turn context with FULL (un-windowed) messages for accurate history tracking
    if step == 0:
        updated = [
            {"role": "system", "content": WORLD_SYSTEM},
            {"role": "user", "content": f"Initial Page State:\n{state}\n\nFirst Action: '{action}'\n\nNext Page State:"},
            {"role": "assistant", "content": content},
        ]
    else:
        updated = wm_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": content},
        ]

    return new_state, updated, elapsed, tokens


def fetch_models():
    """Fetch available models from llama-swap."""
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        console.print(f"[red]Error fetching models: {e}[/red]")
        sys.exit(1)


def select_model(prompt_text: str, models: list, default: Optional[str] = None) -> str:
    """Interactive model selection."""
    if default and default in models:
        choices = [default] + [m for m in models if m != default]
    else:
        choices = models

    result = questionary.select(
        prompt_text,
        choices=choices,
        style=questionary.styles.Default,
    ).ask()

    if result is None:
        console.print("[red]Cancelled.[/red]")
        sys.exit(0)
    return result


def main():
    parser = argparse.ArgumentParser(description="Web World Model Simulation")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: you choose actions instead of an agent")
    parser.add_argument("--agent", type=str, help="Agent model (default: interactive selection)")
    parser.add_argument("--world", type=str, help="World model (default: interactive selection)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task description for the agent")
    parser.add_argument("--state", type=str, choices=list(DEFAULT_STATES.keys()),
                        help="Initial page state template")
    parser.add_argument("--steps", type=int, default=15,
                        help="Maximum simulation steps (default: 15)")
    parser.add_argument("--window", type=int, default=2,
                        help="Sliding window turns for world model (default: 2)")
    parser.add_argument("--no-truncate", action="store_true",
                        help="Don't truncate states sent to agent")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🌐 WorldSim[/bold cyan] — Agent + World Model → Web Trajectory Simulation\n"
        "[dim]Sliding window context · Agent with memory · Step-by-step display[/dim]",
        border_style="bright_blue",
        box=box.HEAVY,
    ))

    # Fetch models
    all_models = fetch_models()

    # World model selection
    if args.world:
        world_model = args.world
    else:
        world_candidates = [m for m in all_models if "webworld" in m.lower() or "world" in m.lower()]
        if not world_candidates:
            world_candidates = all_models
        world_model = select_model("🌍 Select World Model:", world_candidates)

    # Agent selection
    agent_model: str = ""
    if not args.manual:
        if args.agent:
            agent_model = args.agent
        else:
            agent_candidates = [m for m in all_models if "webworld" not in m.lower() and "world" not in m.lower()]
            if not agent_candidates:
                agent_candidates = all_models
            agent_model = select_model("🤖 Select Agent Model:", agent_candidates)

    # Initial state
    custom_state_lines: list[str] = []
    if args.state:
        state_key = args.state
    else:
        state_choices = list(DEFAULT_STATES.keys()) + ["[custom]"]
        selected = questionary.select("📄 Select initial page state:", choices=state_choices).ask()
        if selected == "[custom]":
            console.print("[dim]Paste your initial page state (Ctrl+D to finish):[/dim]")
            try:
                while True:
                    custom_state_lines.append(input())
            except EOFError:
                pass
            state_key = "custom"
        else:
            state_key = selected

    initial_state = DEFAULT_STATES.get(state_key, "")
    if state_key == "custom" and custom_state_lines:
        initial_state = "\n".join(custom_state_lines)

    # Task
    default_task = DEFAULT_TASKS.get(state_key, "Explore the website")
    if args.task:
        task = args.task
    elif not args.manual:
        task = questionary.text("🎯 Enter task for the agent:", default=default_task).ask()
    else:
        task = ""

    if not task and not args.manual:
        task = default_task

    # --- Simulation Loop ---
    console.print()
    console.rule("[bold]🚀 Simulation Starting[/bold]")
    console.print(f"  World Model: [cyan]{world_model}[/cyan]")
    console.print(f"  Mode: {'👤 Manual' if args.manual else f'🤖 Agent ({agent_model})'}")
    console.print(f"  Task: [italic]{task if task else 'N/A (manual)'}[/italic]")
    console.print(f"  Max steps: {args.steps} | Window: last {args.window} turns")
    console.rule()

    # Print initial state
    console.print(Panel(
        Text(initial_state[:1500] + ("..." if len(initial_state) > 1500 else ""),
             style="cyan"),
        title="[bold]📄 Initial Page State[/bold]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    current_state = initial_state
    wm_messages = []   # World model multi-turn context (full history)
    action_history = []  # List of actions taken
    trajectory = []     # List of (action, state_after, agent_time, wm_time) tuples

    step = 0
    while step < args.steps:
        # --- Decide action ---
        if args.manual:
            console.print("\n[yellow]Available actions:[/yellow]")
            console.print("  [dim]click([id])  fill([id], \"text\")  goto(\"url\")  scroll(dx, dy)[/dim]")
            console.print("  [dim]keyboard_press(\"key\")  select_option([id], \"opt\")  hover([id])[/dim]")
            console.print("  [dim]go_back()  noop(ms)  infeasible(\"reason\")[/dim]")
            console.print("  [dim]DONE — end simulation (task complete)[/dim]")

            action = questionary.text(f"  Step {step + 1} → Action:").ask()
            if action is None or action.strip().lower() in ("quit", "exit", "q"):
                console.print("[yellow]Simulation ended by user.[/yellow]")
                break
            action = action.strip()
            if not action:
                continue
            source = "manual"
            agent_time = 0
            agent_tokens = 0
        else:
            console.print(f"\n[bold]Step {step + 1}[/bold]")
            with console.status("[bold green]🤖 Agent deciding...[/bold green]"):
                action, agent_time, agent_tokens = run_agent(
                    agent_model, current_state, task, action_history, step
                )
            source = "agent"

        # Check for completion
        if action.upper().startswith("DONE") or action.lower().startswith("done"):
            console.print(f"\n[bold green]✅ Task complete![/bold green] {action}")
            break

        action_history.append(action)
        console.print(f"  {'👤' if source == 'manual' else '🤖'} Action: [bold yellow]{action}[/bold yellow] [dim]({agent_time:.1f}s, {agent_tokens} tok)[/dim]")

        # --- World model predicts next state ---
        with console.status("[bold cyan]🌍 World model predicting...[/bold cyan]"):
            new_state, wm_messages, wm_time, wm_tokens = run_world_model(
                world_model, current_state, action,
                wm_messages, step, window_turns=args.window,
            )

        trajectory.append((action, new_state, agent_time, wm_time))
        current_state = new_state

        # Display new state
        display = current_state[:1500] + ("..." if len(current_state) > 1500 else "")
        console.print(Panel(
            Text(display, style="cyan"),
            title=f"[bold]📄 State after Step {step + 1}[/bold] [dim]({wm_time:.1f}s, {wm_tokens} tok)[/dim]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        ))

        step += 1

        # Progress check
        console.print(f"[dim]  Steps: {step}/{args.steps} | Actions: {' → '.join(action_history[-5:])}[/dim]")

    # --- Summary ---
    console.print()
    console.rule("[bold]📊 Simulation Summary[/bold]")
    summary_table = Table(show_header=True, box=box.SIMPLE)
    summary_table.add_column("Step", style="bold", width=5)
    summary_table.add_column("Action", style="yellow")
    summary_table.add_column("Agent", justify="right", width=6)
    summary_table.add_column("World", justify="right", width=6)

    for i, (act, state, at, wt) in enumerate(trajectory):
        summary_table.add_row(str(i + 1), act[:50], f"{at:.1f}s", f"{wt:.1f}s")

    console.print(summary_table)
    console.print(f"\n[bold green]✅ Simulation complete — {len(trajectory)} steps[/bold green]")
    if action_history:
        console.print(f"[dim]Full trajectory: {' → '.join(action_history)}[/dim]")


if __name__ == "__main__":
    main()