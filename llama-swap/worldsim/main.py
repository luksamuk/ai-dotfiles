#!/usr/bin/env python3
"""
worldsim — Interactive world model simulation using llama-swap.

Supports 7 AgentWorld domains (MCP, Search, Terminal, SWE, Android, Web, OS)
plus legacy WebWorld-8B for web navigation.

Agent + World Model loop with single-turn world model calls.
The agent has conversation memory (knows what it already did).
The world model receives only current state + action (no trajectory history).

Display: Rich console.print() incremental — no Live(), no Textual.
Each turn is printed once and stays in terminal scrollback.
Streaming uses console.print(text, end="") for real-time token display.

Usage:
    llama-swap-cli worldsim
    llama-swap-cli worldsim --domain terminal
    llama-swap-cli worldsim --domain web --manual
    llama-swap-cli worldsim --agent qwen3.5-4b --world agentworld-35b --steps 15
"""

import sys
import os
import json
import time
import re
import argparse
from typing import Optional, Callable, Union

import requests
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich import box

from domains import DOMAINS, get_domain, list_domains

console = Console()

# --- Config ---
BASE_URL = os.environ.get(
    "WORLDSIM_URL",
    f"http://127.0.0.1:12434/v1",
)

_raw_host = os.environ.get("LLAMA_SWAP_HOST", "")
_raw_port = os.environ.get("LLAMA_SWAP_PORT", "")
if _raw_host and not os.environ.get("WORLDSIM_URL"):
    if _raw_host.startswith("http://") or _raw_host.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(_raw_host)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or int(_raw_port or "12434")
        BASE_URL = f"http://{host}:{port}/v1"
    else:
        port = _raw_port or "12434"
        BASE_URL = f"http://{_raw_host}:{port}/v1"


# ─── API ─────────────────────────────────────────────────────────────────────

def chat_completion(model: str, messages: list, max_tokens: int = 2048,
                    temperature: float = 0.0, timeout: int = 600,
                    on_chunk: Union[Callable[[str], None], None] = None) -> tuple:
    """Call llama-swap chat completion. Returns (content, elapsed_seconds, tokens).

    Non-streaming only. Reasoning content is captured by the server (--reasoning on)
    but NEVER used as content — it stays separate in reasoning_content.
    The on_chunk parameter is accepted for API compatibility but ignored.
    """
    t0 = time.time()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", json=payload,
                             timeout=timeout, stream=False)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # Reasoning is captured but NEVER used as content — it stays separate.
        # If content is empty, return empty (caller handles it).
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, elapsed, tokens
    except requests.exceptions.Timeout:
        return "[TIMEOUT]", time.time() - t0, 0
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0, 0


# ─── Parsing ──────────────────────────────────────────────────────────────────

def strip_thinking_and_extract(text: str, response_tag: Optional[str],
                                thinking_tag: str) -> tuple:
    """Split world model output into (thinking, observation).

    Handles two formats:
    1. WebWorld: <reason>...</reason> then raw state
    2. AgentWorld: <think>...</think> then <predicted_observation>...</predicted_observation>
    3. AgentWorld with --reasoning on: thinking in reasoning_content, clean content
    """
    thinking = ""
    observation = text.strip()

    if response_tag is None:
        if thinking_tag == "reason":
            match = re.search(r"<reason>(.*?)</reason>\s*", text, flags=re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                observation = text[match.end():].strip()
        return thinking, observation
    else:
        think_pattern = rf"<{re.escape(thinking_tag)}>(.*?)(?:</{re.escape(thinking_tag)}>|$)"
        think_match = re.search(think_pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()
            text = text[:think_match.start()] + text[think_match.end():]

        start_pattern = rf"<{re.escape(response_tag)}>"
        start_matches = list(re.finditer(start_pattern, text, re.IGNORECASE))
        if start_matches:
            last_start = start_matches[-1].end()
            close_pattern = rf"</{re.escape(response_tag)}>"
            close_match = re.search(close_pattern, text[last_start:], re.IGNORECASE)
            if close_match:
                observation = text[last_start:last_start + close_match.start()].strip()
            else:
                observation = text[last_start:].strip()
        else:
            marker = "**Environment Observation:**"
            if marker in text:
                idx = text.index(marker) + len(marker)
                observation = text[idx:].strip()
            else:
                observation = text.strip()

        return thinking, observation


def extract_action(text: str, domain_name: str) -> str:
    """Extract action from agent response."""
    text = text.strip()
    if re.search(r"\bDONE\b", text, re.IGNORECASE):
        return text.strip()

    domain = DOMAINS[domain_name]
    fmt = domain["action_format"]

    if fmt == "web":
        match = re.search(
            r"(click|fill|goto|scroll|keyboard_press|select_option|hover|go_back|"
            r"send_msg_to_user|noop|infeasible)\([^)]*\)",
            text,
        )
        if match:
            return match.group(0)

    elif fmt == "terminal":
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return match.group(0)

    elif fmt in ("mcp", "swe"):
        match = re.search(r'\{[^{}]*"name"[^{}]*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return match.group(0)

    elif fmt == "android":
        match = re.search(
            r"(tap|swipe|type|press|back|home)\([^)]*\)",
            text,
        )
        if match:
            return match.group(0)

    elif fmt == "os":
        match = re.search(
            r"(pyautogui\.\w+\([^)]*\)|BrowserTools\.\w+\([^)]*\))",
            text,
        )
        if match:
            return match.group(0)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else "noop(1000)"


def truncate_state(state: str, max_lines: int = 50) -> str:
    """Truncate state to max_lines for agent context."""
    lines = state.strip().split("\n")
    if len(lines) <= max_lines:
        return state
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def build_world_user_message(state: str, action: str, domain_name: str) -> str:
    """Build the user message for the world model, formatted per domain."""
    domain = DOMAINS[domain_name]
    fmt = domain["action_format"]

    if fmt == "web":
        return f"Initial Page State:\n{state}\n\nFirst Action: '{action}'\n\nNext Page State:"
    elif fmt == "terminal":
        return f"### Turn 1\n**Current Terminal State:**\n{state}\n\n**Action:**\n```json\n{action}\n```\n\n**Next Terminal State:**"
    elif fmt in ("mcp", "swe"):
        return f"### Turn 1\n**Action:**\n```json\n{action}\n```\n\n**Environment Observation:**"
    elif fmt == "android":
        return f"### Turn 1\n**Current Screen State:**\n{state}\n\n**Action:**\n{action}\n\n**Next Screen State:**"
    elif fmt == "os":
        return f"### Turn 1\n**Current Desktop State:**\n{state}\n\n**Action:**\n```python\n{action}\n```\n\n**Next Desktop State:**"
    else:
        return f"Current State:\n{state}\n\nAction: {action}\n\nNext State:"


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_header(domain_name: str, world_model: str, agent_model: str,
                 task: str, steps: int, manual: bool):
    """Print simulation header."""
    domain = DOMAINS[domain_name]
    console.print()
    console.print(Panel(
        "[bold cyan]WorldSim[/bold cyan] — Agent + World Model -> Environment Simulation\n"
        "[dim]7 domains | Single-turn world model | Agent with memory | Incremental display[/dim]",
        border_style="bright_blue",
        box=box.HEAVY,
    ))
    console.print()
    console.print(Rule("[bold]Simulation Starting[/bold]"))
    console.print(f"  Domain: [bold]{domain_name}[/bold] - {domain['label']}")
    console.print(f"  World Model: [cyan]{world_model}[/cyan]")
    console.print(f"  Mode: {'Manual' if manual else f'Agent ({agent_model})'}")
    console.print(f"  Task: [italic]{task if task else 'N/A (manual)'}[/italic]")
    console.print(f"  Max steps: {steps}")
    console.print(Rule())


def print_initial_state(state: str):
    """Print the initial state panel."""
    display = state[:2000] + ("..." if len(state) > 2000 else "")
    console.print(Panel(
        Text(display, style="cyan"),
        title="[bold]Initial State[/bold]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def print_step_separator(step: int):
    """Print a visual separator between steps."""
    console.print()
    console.print(Rule(f"[bold dim]-- Step {step} --[/bold dim]"))


def print_agent_result(action: str, elapsed: float, tokens: int, manual: bool):
    """Print the agent's chosen action."""
    icon = "Manual" if manual else "Agent"
    console.print(f"\n  [{icon}] [bold yellow]Action:[/bold yellow] {action[:120]}")
    if not manual:
        console.print(f"     [dim]({elapsed:.1f}s, {tokens} tok)[/dim]")


def stream_world_model(world_model: str, state: str, action: str,
                       step: int, domain_name: str) -> tuple:
    """Run world model and display result.

    Non-streaming — reasoning is suppressed, only content (prediction) is shown.
    Returns (observation, thinking, elapsed, tokens).
    """
    domain = DOMAINS[domain_name]
    world_system = domain["world_system"]
    response_tag = domain["response_tag"]
    thinking_tag = domain["thinking_tag"]
    max_tokens = domain["max_tokens"]

    user_msg = build_world_user_message(state, action, domain_name)
    messages = [
        {"role": "system", "content": world_system},
        {"role": "user", "content": user_msg},
    ]

    console.print(f"\n  [bold cyan]World Model generating...[/bold cyan]")

    content, elapsed, tokens = chat_completion(
        world_model, messages, max_tokens=max_tokens, temperature=0.6,
    )

    thinking, observation = strip_thinking_and_extract(content, response_tag, thinking_tag)
    return observation, thinking, elapsed, tokens


def print_state_panel(state: str, step: int, elapsed: float, tokens: int,
                      thinking: str):
    """Print the final state panel after streaming completes."""
    parts = []
    if thinking:
        think_short = thinking[:300] + ("..." if len(thinking) > 300 else "")
        parts.append(Text("Thinking: ", style="dim bold"))
        parts.append(Text(think_short, style="dim italic"))
        parts.append(Text("\n\n", style=""))
    display = state[:2000] + ("..." if len(state) > 2000 else "")
    parts.append(Text(display, style="cyan"))

    console.print(Panel(
        Text.assemble(*parts),
        title=f"[bold]State after Step {step}[/bold] [dim]({elapsed:.1f}s, {tokens} tok)[/dim]",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def print_summary(trajectory: list):
    """Print simulation summary table."""
    console.print()
    console.print(Rule("[bold]Simulation Summary[/bold]"))
    summary_table = Table(show_header=True, box=box.SIMPLE)
    summary_table.add_column("Step", style="bold", width=5)
    summary_table.add_column("Action", style="yellow")
    summary_table.add_column("Agent", justify="right", width=6)
    summary_table.add_column("World", justify="right", width=6)

    for i, (act, state, at, wt) in enumerate(trajectory):
        summary_table.add_row(str(i + 1), act[:50], f"{at:.1f}s", f"{wt:.1f}s")

    console.print(summary_table)
    console.print(f"\n[bold green]Simulation complete - {len(trajectory)} steps[/bold green]")


# ─── Model/agent runners ─────────────────────────────────────────────────────

def run_agent(agent_model: str, state: str, task: str,
              action_history: list, step: int, domain_name: str) -> tuple:
    """Run agent to decide next action. Returns (action_str, elapsed_seconds, tokens)."""
    domain = DOMAINS[domain_name]
    agent_system = domain["agent_system"]

    history_text = ""
    if action_history:
        history_text = "\n\nActions taken so far:\n"
        for i, a in enumerate(action_history):
            history_text += f"  {i + 1}. {a}\n"

    state_truncated = truncate_state(state, max_lines=40)

    user_msg = (
        f"Task: {task}\n\n"
        f"Current State:\n{state_truncated}"
        f"{history_text}\n\n"
        f"What action should be taken next? (output ONE action, or DONE if task is complete)"
    )

    messages = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": user_msg},
    ]

    content, elapsed, tokens = chat_completion(
        agent_model, messages, max_tokens=32768, temperature=0.1,
    )
    action = extract_action(content, domain_name)
    return action, elapsed, tokens


def run_world_model_simple(world_model: str, state: str, action: str,
                           domain_name: str) -> tuple:
    """Run world model without streaming. Returns (observation, thinking, elapsed, tokens)."""
    domain = DOMAINS[domain_name]
    world_system = domain["world_system"]
    response_tag = domain["response_tag"]
    thinking_tag = domain["thinking_tag"]
    max_tokens = domain["max_tokens"]

    user_msg = build_world_user_message(state, action, domain_name)
    messages = [
        {"role": "system", "content": world_system},
        {"role": "user", "content": user_msg},
    ]

    content, elapsed, tokens = chat_completion(
        world_model, messages, max_tokens=max_tokens, temperature=0.6,
    )
    thinking, observation = strip_thinking_and_extract(content, response_tag, thinking_tag)
    return observation, thinking, elapsed, tokens


# ─── Model selection ─────────────────────────────────────────────────────────

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


def select_model(prompt_text: str, models: list, default: Optional[str] = None,
                 filter_fn: Optional[Callable] = None) -> str:
    """Interactive model selection with optional filtering."""
    candidates = models
    if filter_fn:
        candidates = [m for m in models if filter_fn(m)]
    if not candidates:
        candidates = models

    if default and default in candidates:
        choices = [default] + [m for m in candidates if m != default]
    else:
        choices = candidates

    result = questionary.select(
        prompt_text,
        choices=choices,
        style=questionary.styles.DEFAULT_STYLE,
    ).ask()

    if result is None:
        console.print("[red]Cancelled.[/red]")
        sys.exit(0)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="World Model Simulation (7 domains)")
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()),
                        help="Simulation domain (default: interactive selection)")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: you choose actions instead of an agent")
    parser.add_argument("--agent", type=str, help="Agent model (default: interactive selection)")
    parser.add_argument("--world", type=str, help="World model (default: interactive selection)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task description for the agent")
    parser.add_argument("--state", type=str, default=None,
                        help="Initial state template key")
    parser.add_argument("--steps", type=int, default=10,
                        help="Max simulation steps (default: 10)")
    parser.add_argument("--no-truncate", action="store_true",
                        help="Don't truncate states sent to agent")
    args = parser.parse_args()

    # --- Domain selection ---
    if args.domain:
        domain_name = args.domain
    else:
        domain_labels = [label for _, label in list_domains()]
        selected = questionary.select(
            "Select simulation domain:",
            choices=domain_labels,
            style=questionary.styles.DEFAULT_STYLE,
        ).ask()
        if selected is None:
            sys.exit(0)
        domain_name = [name for name, label in list_domains() if label == selected][0]

    domain = DOMAINS[domain_name]

    # --- Fetch models ---
    all_models = fetch_models()

    # --- World model selection ---
    if args.world:
        world_model = args.world
    else:
        hint = domain.get("model_hint", "")
        world_candidates = [m for m in all_models if hint and hint in m.lower()]
        if not world_candidates:
            world_candidates = [m for m in all_models if "world" in m.lower()]
        if not world_candidates:
            world_candidates = all_models
        # Prefer :think variant for world models — world models need thinking
        # to simulate environments properly (AgentWorld puts reasoning in thinking,
        # prediction in content). Without :think, reasoning leaks into content.
        think_first = [m for m in world_candidates if m.endswith(":think")]
        non_think = [m for m in world_candidates if not m.endswith(":think")]
        world_candidates = think_first + non_think
        world_model = select_model("Select World Model:", world_candidates)

    # --- Agent selection ---
    agent_model = ""
    if not args.manual:
        if args.agent:
            agent_model = args.agent
        else:
            agent_candidates = [m for m in all_models if hint not in m.lower() and "world" not in m.lower()]
            if not agent_candidates:
                agent_candidates = all_models
            agent_model = select_model("Select Agent Model:", agent_candidates)

    # --- Initial state ---
    states = domain["initial_states"]
    tasks = domain["default_tasks"]

    custom_state_lines = []
    if args.state and args.state in states:
        state_key = args.state
    else:
        state_choices = list(states.keys()) + ["[custom]"]
        selected = questionary.select(
            "Select initial state:",
            choices=state_choices,
            style=questionary.styles.DEFAULT_STYLE,
        ).ask()
        if selected == "[custom]":
            console.print("[dim]Paste your initial state (Ctrl+D to finish):[/dim]")
            try:
                while True:
                    custom_state_lines.append(input())
            except EOFError:
                pass
            state_key = "custom"
        else:
            state_key = selected

    initial_state = states.get(state_key, "")
    if state_key == "custom" and custom_state_lines:
        initial_state = "\n".join(custom_state_lines)

    # --- Task ---
    default_task = tasks.get(state_key, "Explore the environment")
    if args.task:
        task = args.task
    elif not args.manual:
        task = questionary.text("Enter task for the agent:", default=default_task).ask()
    else:
        task = ""

    if not task and not args.manual:
        task = default_task

    # --- Print header ---
    print_header(domain_name, world_model, agent_model, task, args.steps, args.manual)
    print_initial_state(initial_state)

    # --- Simulation Loop ---
    current_state = initial_state
    action_history = []
    trajectory = []

    step = 0
    while step < args.steps:
        print_step_separator(step + 1)

        # --- Decide action ---
        if args.manual:
            console.print("\n[yellow]Available actions:[/yellow]")
            console.print("  [dim]Type your action (domain-specific format)[/dim]")
            console.print("  [dim]DONE - end simulation[/dim]")

            action = questionary.text(f"  Step {step + 1} -> Action:").ask()
            if action is None or action.strip().lower() in ("quit", "exit", "q"):
                console.print("[yellow]Simulation ended by user.[/yellow]")
                break
            action = action.strip()
            if not action:
                continue
            agent_time = 0
            agent_tokens = 0
        else:
            console.print(f"\n  [dim]Agent deciding...[/dim]")
            action, agent_time, agent_tokens = run_agent(
                agent_model, current_state, task, action_history, step, domain_name,
            )

        if action.upper().startswith("DONE") or action.lower().startswith("done"):
            console.print(f"\n[bold green]Task complete![/bold green] {action}")
            break

        action_history.append(action)
        print_agent_result(action, agent_time, agent_tokens, args.manual)

        # --- World model predicts next state ---
        console.print(f"\n  [dim]World model predicting...[/dim]")
        new_state, thinking, wm_time, wm_tokens = run_world_model_simple(
            world_model, current_state, action, domain_name,
        )
        print_state_panel(new_state, step + 1, wm_time, wm_tokens, thinking)

        trajectory.append((action, new_state, agent_time, wm_time))
        current_state = new_state

        step += 1
        console.print(
            f"[dim]  Steps: {step}/{args.steps} | "
            f"Actions: {' -> '.join(a[:30] for a in action_history[-5:])}[/dim]"
        )

    # --- Summary ---
    print_summary(trajectory)
    if action_history:
        console.print(f"[dim]Full trajectory: {' -> '.join(a[:30] for a in action_history)}[/dim]")


if __name__ == "__main__":
    main()