#!/usr/bin/env python3
"""
worldsim — Interactive web world model simulation using llama-swap.

Runs an Agent + World Model loop:
  1. Agent model decides an action based on the current page state
  2. World model predicts the next page state given (state + action)
  3. Repeat until task is done or max steps reached

Supports manual mode where the user chooses actions instead of an agent.

Usage:
    llama-swap-cli worldsim
    llama-swap-cli worldsim --manual
    llama-swap-cli worldsim --agent lfm2.5-1.2b --world webworld-8b
"""

import sys
import os
import json
import time
import argparse
from typing import Optional

import requests
import questionary
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich import box

console = Console()

# --- Config ---
BASE_URL = os.environ.get(
    "WORLDSIM_URL",
    f"http://{os.environ.get('LLAMA_SWAP_HOST', '127.0.0.1')}"
    f":{os.environ.get('LLAMA_SWAP_PORT', '12434')}/v1",
)

# WebWorld system prompt (from official model card)
WORLD_MODEL_SYSTEM = (
    "You are a web world model. I will provide you with an initial page state "
    "and a sequence of actions. For each action, predict the resulting page state.\n"
    "Strictly maintain the original format. Output only the full page state "
    "without explanations, code, or truncation."
)

AGENT_SYSTEM = (
    "You are a web navigation agent. Given a page state and a task, "
    "decide the next action to take. "
    "Output ONLY a single action in Python function call format, nothing else.\n"
    "Available actions: click(bid), fill(bid, text), goto(url), "
    "scroll(dx, dy), keyboard_press(key), select_option(bid, options), "
    "hover(bid), go_back(), send_msg_to_user(text), noop(wait_ms), infeasible(reason)\n"
    "Example output: click([5])"
)

CONTINUE_PROMPT = (
    "Continue the trajectory. Given the previous state, "
    "predict the next page state after this action.\n\n"
    "Action: '{action}'\n\nNext Page State:"
)

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
\t\t\t[53] list 'Top Sites Grid', visible""",

    "github_homepage": """RootWebArea 'GitHub', focused
\t[1] banner 'Top Header', visible
\t\t[2] link 'Pull requests', clickable, visible
\t\t[3] link 'Issues', clickable, visible
\t\t[4] link 'Actions', clickable, visible
\t\t[5] textbox 'Search or jump to...', clickable, visible, editable, value=''
\t\t[6] button 'Sign in', clickable, visible
\t[10] main 'Content', visible
\t\t[11] heading 'Welcome to GitHub', visible
\t\t[12] link 'Explore repositories', clickable, visible
\t\t[13] link 'Trending', clickable, visible
\t\t[14] link 'Marketplace', clickable, visible""",

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


def get_world_models(models):
    """Filter models likely to be world models (tag-based or name-based)."""
    world_models = [m for m in models if "webworld" in m.lower() or "world" in m.lower()]
    return world_models if world_models else models


def get_agent_models(models):
    """Filter models suitable as agents (have tools capability or are small/chat models)."""
    # Prefer models with tools, exclude world models
    agents = [m for m in models if "webworld" not in m.lower() and "world" not in m.lower()]
    return agents if agents else models


def chat_completion(model: str, messages: list, max_tokens: int = 2048,
                    temperature: float = 0.0, timeout: int = 120) -> str:
    """Call llama-swap chat completion API."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            timeout=timeout,
            stream=False,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Strip <reason>...</reason> tags from world model output
        # (WebWorld uses CoT but we want just the state prediction)
        import re
        content = re.sub(r"<reason>.*?</reason>\s*", "", content, flags=re.DOTALL)

        return content.strip()
    except requests.exceptions.Timeout:
        return "[TIMEOUT - model may still be loading]"
    except Exception as e:
        return f"[ERROR: {e}]"


def extract_action(text: str) -> str:
    """Extract action from agent response (may contain extra text)."""
    import re
    # Try to find a Python-style function call: word(args)
    match = re.search(r"(click|fill|goto|scroll|keyboard_press|select_option|hover|go_back|send_msg_to_user|noop|infeasible)\([^)]*\)", text)
    if match:
        return match.group(0)
    # Fallback: return first non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else "noop(1000)"


def run_step(world_model: str, agent_model: str, state: str, action: str,
             messages: list, step: int) -> tuple:
    """Run one simulation step. Returns (new_state, updated_messages)."""

    # Build the world model prompt
    if step == 0:
        user_msg = f"Initial Page State:\n{state}\n\nFirst Action: '{action}'\n\nNext Page State:"
        wm_messages = [
            {"role": "system", "content": WORLD_MODEL_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
    else:
        user_msg = CONTINUE_PROMPT.format(action=action)
        wm_messages = messages + [{"role": "user", "content": user_msg}]

    new_state = chat_completion(world_model, wm_messages, max_tokens=4096,
                                temperature=0.0, timeout=120)

    # Update messages for multi-turn context
    if step == 0:
        updated = [
            {"role": "system", "content": WORLD_MODEL_SYSTEM},
            {"role": "user", "content": f"Initial Page State:\n{state}\n\nFirst Action: '{action}'\n\nNext Page State:"},
            {"role": "assistant", "content": new_state},
        ]
    else:
        updated = messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": new_state},
        ]

    return new_state, updated


def decide_action(agent_model: str, state: str, task: str) -> str:
    """Ask the agent model to decide the next action."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nCurrent Page State:\n{state}\n\nWhat action should be taken next?"},
    ]
    response = chat_completion(agent_model, messages, max_tokens=256,
                               temperature=0.3, timeout=60)
    return extract_action(response)


def print_state(state: str, title: str = "Page State"):
    """Pretty-print a page state."""
    # Truncate very long states for display
    display = state if len(state) <= 3000 else state[:3000] + "\n... (truncated)"
    console.print(Panel(
        Text(display, style="cyan"),
        title=f"[bold]{title}[/bold]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def print_action(action: str, step: int, source: str = "agent"):
    """Pretty-print an action."""
    color = "green" if source == "agent" else "yellow"
    label = "🤖 Agent" if source == "agent" else "👤 Manual"
    console.print(f"\n[{label} | Step {step}] Action: [bold {color}]'{action}'[/bold {color}]")


def print_summary(steps: list, final_state: str):
    """Print trajectory summary."""
    console.print("\n" + "─" * 60)
    console.print("[bold]📊 Simulation Summary[/bold]")
    console.print(f"  Steps: {len(steps)}")
    console.print(f"  Actions: {' → '.join(steps)}")
    console.print("─" * 60)


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
    parser.add_argument("--task", type=str, default="Find information about Python programming",
                        help="Task description for the agent")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum simulation steps (default: 10)")
    parser.add_argument("--state", type=str, choices=list(DEFAULT_STATES.keys()),
                        help="Initial page state template")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🌐 WebWorld Simulator[/bold cyan]\n"
        "Agent + World Model → Simulated Web Trajectory",
        border_style="bright_blue",
        box=box.HEAVY,
    ))

    # Fetch models
    all_models = fetch_models()

    # World model selection
    if args.world:
        world_model = args.world
    else:
        world_candidates = get_world_models(all_models)
        world_model = select_model("🌍 Select World Model:", world_candidates)

    # Agent selection (skip in manual mode)
    agent_model: str = ""
    if not args.manual:
        if args.agent:
            agent_model = args.agent
        else:
            agent_candidates = get_agent_models(all_models)
            agent_model = select_model("🤖 Select Agent Model:", agent_candidates)

    # Initial state selection
    if args.state:
        initial_state = DEFAULT_STATES[args.state]
    else:
        state_choices = list(DEFAULT_STATES.keys()) + ["[custom]"]
        selected = questionary.select(
            "📄 Select initial page state:",
            choices=state_choices,
        ).ask()

        if selected == "[custom]":
            console.print("[dim]Paste your initial page state (Ctrl+D to finish):[/dim]")
            lines = []
            try:
                while True:
                    lines.append(input())
            except EOFError:
                pass
            initial_state = "\n".join(lines)
            if not initial_state.strip():
                console.print("[red]Empty state. Exiting.[/red]")
                sys.exit(1)
        else:
            initial_state = DEFAULT_STATES[selected]

    # Task (for agent mode)
    if not args.manual:
        task = questionary.text(
            "🎯 Enter task for the agent:",
            default=args.task,
        ).ask()
    else:
        task = ""

    # --- Simulation Loop ---
    console.print("\n" + "═" * 60)
    console.print(f"[bold]🚀 Starting Simulation[/bold]")
    console.print(f"  World Model: [cyan]{world_model}[/cyan]")
    console.print(f"  Mode: {'👤 Manual' if args.manual else f'🤖 Agent ({agent_model})'}")
    console.print(f"  Task: [italic]{task if task else 'N/A (manual mode)'}[/italic]")
    console.print(f"  Max steps: {args.max_steps}")
    console.print("═" * 60 + "\n")

    # Print initial state
    print_state(initial_state, title="Initial Page State")

    current_state = initial_state
    wm_messages = []  # World model multi-turn context
    trajectory = []   # List of (action, resulting_state) pairs

    step = 0
    while step < args.max_steps:
        # Decide action
        if args.manual:
            console.print("\n[yellow]Available actions:[/yellow]")
            console.print("  click(bid)  fill(bid, text)  goto(url)  scroll(dx, dy)")
            console.print("  keyboard_press(key)  select_option(bid, opts)  hover(bid)")
            console.print("  go_back()  send_msg_to_user(text)  noop(ms)  infeasible(reason)")
            console.print("  [dim]quit — end simulation[/dim]")

            action = questionary.text(
                f"  Step {step + 1} → Action:",
                default="",
            ).ask()

            if action is None or action.strip().lower() in ("quit", "exit", "q"):
                console.print("\n[yellow]Simulation ended by user.[/yellow]")
                break

            action = action.strip()
            if not action:
                continue

            source = "manual"
        else:
            action = decide_action(agent_model, current_state, task)
            source = "agent"

        print_action(action, step + 1, source=source)

        # Run world model
        with console.status("[bold green]World model predicting next state...[/bold green]"):
            new_state, wm_messages = run_step(
                world_model, agent_model, current_state, action,
                wm_messages, step,
            )

        trajectory.append((action, new_state))
        current_state = new_state

        # Print resulting state
        print_state(current_state, title=f"Page State After Step {step + 1}")

        step += 1

        # Check if agent declared task done/infeasible
        if "infeasible" in action.lower() or "send_msg_to_user" in action.lower():
            console.print("\n[yellow]Agent declared task done or infeasible.[/yellow]")
            break

        if not args.manual:
            # Ask if user wants to continue
            cont = questionary.confirm("Continue simulation?", default=True).ask()
            if not cont:
                break

    # Summary
    print_summary([a for a, _ in trajectory], current_state)
    console.print("[bold green]✅ Simulation complete.[/bold green]")


if __name__ == "__main__":
    main()