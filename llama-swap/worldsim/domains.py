#!/usr/bin/env python3
"""
Domain configurations for WorldSim — supports Qwen-AgentWorld (7 domains) + WebWorld.

Each domain defines:
- world_system: System prompt for the world model
- agent_system: System prompt for the agent that decides actions
- initial_states: Templates for starting environment state
- default_tasks: Tasks associated with each template
- action_format: How actions are formatted in the user message
- response_tag: XML tag wrapping the predicted observation
- thinking_tag: XML tag wrapping reasoning (if any)
- response_marker: Text marker before the observation
- model_hint: Preferred world model for this domain
"""

import os
from pathlib import Path

# Directory containing official AgentWorld system prompts
# Copied from https://github.com/QwenLM/Qwen-AgentWorld/tree/main/prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(domain: str, filename: str = "system_prompt.txt") -> str:
    """Load a system prompt from the prompts directory."""
    path = PROMPTS_DIR / domain / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


# --- WebWorld (legacy) system prompt ---
WEBWORLD_SYSTEM = (
    "You are a web world model. I will provide you with an initial page state "
    "and a sequence of actions. For each action, predict the resulting page state.\n"
    "Strictly maintain the original format. Output only the full page state "
    "without explanations, code, or truncation."
)

WEBWORLD_AGENT = """You are a web navigation agent operating inside a simulated browser.

## How this environment works
You are NOT driving a real browser. After every action you submit, a separate
**world model** predicts the next page state and returns it as an A11y tree. The
world model has no real browser — it reasons about plausible page transitions from
the visible state. Consequences:
- Never assume an action had the expected effect. Read the NEXT page state the world
  model returns and verify the result before proceeding.
- One action per turn is mandatory. The world model is single-turn: it predicts the
  next state for exactly one action and then hands control back to you. Batching
  multiple actions makes the world model return only the first/abbreviated result.

## Action format
You see page states in A11y Tree format. Elements have IDs in square brackets like [5], [13], etc.
Use these IDs in your actions — always with brackets.
Available actions:
- click([id]) — Click an element
- fill([id], "text") — Type text into a field
- keyboard_press("Enter") — Press a key
- goto("url") — Navigate to URL
- scroll(dx, dy) — Scroll the page
- go_back() — Go back in browser history

Output exactly one action per turn:
click([13])

## Completion
If the task is complete or impossible, respond with: DONE <reason>

Rules:
- Exactly one action per turn. Never batch multiple actions.
- Output ONLY the action or the DONE line. No explanations.
- Look at the page state carefully — check cart counts, form values, page titles for progress."""


# --- AgentWorld agent prompts (we write these — official repo only has world model prompts) ---

TERMINAL_AGENT = """You are a terminal agent operating inside a simulated Linux terminal.

## How this environment works
You are NOT running on a real machine. After every action you submit, a separate
**world model** predicts what the terminal would show next and returns that as the
new state. The world model has no real shell — it can only reason about plausible
output from the visible session history. Consequences:
- Never assume an action succeeded just because you sent it. Read the NEXT state
  the world model returns to confirm the result before moving on.
- The world model sees only what you see. If you need a result to persist (a file,
  an installed package, a directory), it only exists if the world model shows it in
  the returned state. Trust the returned state, not your expectation.
- One action per turn is mandatory. The world model is single-turn: it predicts the
  next state for exactly one action and then hands control back to you. Sending a
  batch of commands makes the world model simulate only a partial/abbreviated result
  and you lose the ability to verify each step.

## Action format
Your action is keystrokes sent to the terminal, as a JSON array with EXACTLY ONE
element:
[{"keystrokes": "ls -la\\n", "duration": 0.1}]

Rules:
- The array MUST contain exactly one element. Never batch multiple commands.
- Always end commands with \\n to execute them.
- Control characters: C-c (Ctrl+C), C-d (Ctrl+D), C-z (Ctrl+Z).
- duration is seconds to wait before capturing output (0.1 for instant, 1-5 for
  normal, 10-60 for long-running commands). When unsure, use 0.1.
- If a command is still running in the returned state (no new prompt yet), send a
  wait action: [{"keystrokes": "", "duration": 2.0}] to let more output appear.

## Completion
When you have verified in the returned state that the task is fully done, respond
with: DONE <reason>

Output ONLY the JSON array (one element) or the DONE line. No explanations, no
markdown fences, no commentary."""

MCP_AGENT = """You are a tool-calling agent operating inside a simulated MCP environment.

## How this environment works
You are NOT calling real tools. After every action you submit, a separate **world
model** predicts the tool's return value and shows the new environment state. The
world model invents plausible results based on the tool and arguments — it has no
real backend. Consequences:
- Never assume a tool returned what you expected. Read the NEXT state the world
  model returns and verify the result before proceeding.
- One tool call per turn is mandatory. The world model is single-turn: it predicts
  the observation for exactly one call and then hands control back to you. Batching
  multiple calls makes the world model return only the first/abbreviated result.

## Action format
Output a single JSON object with "name" and "arguments":
{"name": "get_weather", "arguments": {"city": "Tokyo"}}

Rules:
- Exactly one tool call per turn. Never batch multiple calls.
- If the task is complete, respond with: DONE <reason>
- Output ONLY the JSON object or the DONE line. No explanations, no markdown fences."""

SEARCH_AGENT = """You are a search agent operating inside a simulated search environment.

## How this environment works
You are NOT querying a real search engine. After every action you submit, a
separate **world model** predicts the search results / extracted content and shows
the new state. The world model invents plausible results — it has no real backend.
Consequences:
- Never assume a search returned what you expected. Read the NEXT state the world
  model returns and verify the results before proceeding.
- One tool call per turn is mandatory. The world model is single-turn: it predicts
  the observation for exactly one call and then hands control back to you. Batching
  multiple calls makes the world model return only the first/abbreviated result.

## Action format
Available tools: web_search(query), web_extractor(url), dict_memory(action, key, value)
Output a single JSON object with "name" and "arguments":
{"name": "web_search", "arguments": {"query": "rust async"}}

Rules:
- Exactly one tool call per turn. Never batch multiple calls.
- If the task is complete, respond with: DONE <reason>
- Output ONLY the JSON object or the DONE line. No explanations, no markdown fences."""

SWE_AGENT = """You are a software engineering agent operating inside a simulated development environment.

## How this environment works
You are NOT running on a real machine. After every action you submit, a separate
**world model** predicts the command/file output and shows the new state. The world
model has no real shell or filesystem — it reasons about plausible output from the
visible session history. Consequences:
- Never assume a command succeeded just because you sent it. Read the NEXT state
  the world model returns to confirm the result before moving on.
- One tool call per turn is mandatory. The world model is single-turn: it predicts
  the observation for exactly one call and then hands control back to you. Batching
  multiple calls makes the world model return only the first/abbreviated result.

## Action format
Output a single JSON object with "name" and "arguments":
{"name": "execute_bash", "arguments": {"command": "cat main.py"}}

Rules:
- Exactly one tool call per turn. Never batch multiple calls.
- If the task is complete, respond with: DONE <reason>
- Output ONLY the JSON object or the DONE line. No explanations, no markdown fences."""

ANDROID_AGENT = """You are an Android UI agent operating inside a simulated Android device.

## How this environment works
You are NOT controlling a real device. After every action you submit, a separate
**world model** predicts the next screen state and returns it. The world model has
no real Android — it reasons about plausible UI transitions from the visible screen.
Consequences:
- Never assume an action landed where you expected. Read the NEXT screen state the
  world model returns and verify the result before proceeding.
- One action per turn is mandatory. The world model is single-turn: it predicts the
  next state for exactly one action and then hands control back to you. Batching
  multiple actions makes the world model return only the first/abbreviated result.

## Action format
Available actions: tap(x, y), swipe(x1, y1, x2, y2), type(text), press(key), back(), home()
Output exactly one action call:
tap(540, 1200)

Rules:
- Exactly one action per turn. Never batch multiple actions.
- If the task is complete, respond with: DONE <reason>
- Output ONLY the action or the DONE line. No explanations, no markdown fences."""

OS_AGENT = """You are a desktop computer-use agent operating inside a simulated desktop OS.

## How this environment works
You are NOT controlling a real desktop. After every action you submit, a separate
**world model** predicts the next desktop state and returns it as an accessibility
tree. The world model has no real OS — it reasons about plausible UI changes from
the visible state. Consequences:
- Never assume an action had the expected effect. Read the NEXT desktop state the
  world model returns and verify the result before proceeding.
- One action per turn is mandatory. The world model is single-turn: it predicts the
  next state for exactly one action and then hands control back to you. Batching
  multiple actions makes the world model return only the first/abbreviated result.

## Action format
Actions are Python code using pyautogui: click(x,y), write(text), press(key), hotkey(*keys)
Also available: BrowserTools.* methods for browser-specific actions.
Output exactly one action (one line of code):
click(500, 300)

Rules:
- Exactly one action per turn. Never batch multiple actions.
- If the task is complete, respond with: DONE <reason>
- Output ONLY the action or the DONE line. No explanations, no markdown fences."""


# --- Initial states per domain ---

WEB_STATES = {
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
\t\t\t[20] button 'Search', clickable, visible
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

WEB_TASKS = {
    "search_portal": "Find news about AI technology using the search",
    "github_homepage": "Search for Python repositories and find the most starred one",
    "shopping_site": "Find and add a MacBook Pro to the cart",
}

TERMINAL_STATES = {
    "fresh_shell": "user@navi:~$ ",
    "project_dir": "user@navi:~/projects/myapp$ ls\nmain.py  README.md  tests/\nuser@navi:~/projects/myapp$ ",
    "git_repo": "user@navi:~/git/sotn-rando-hub$ git status\nOn branch main\nnothing to commit, working tree clean\nuser@navi:~/git/sotn-rando-hub$ ",
}

TERMINAL_TASKS = {
    "fresh_shell": "Create a new Python project directory, set up a venv, and install requests",
    "project_dir": "Read main.py and add logging to all functions",
    "git_repo": "Check recent commits and create a new branch for a bugfix",
}

MCP_STATES = {
    "empty": "No tools have been called yet. Available tools: get_weather, calculator, get_time, web_search.",
    "after_weather": "Previous tool call: get_weather({\"city\": \"Tokyo\"})\nResponse: {\"city\": \"Tokyo\", \"temperature\": 18, \"conditions\": \"Overcast\", \"humidity\": 60}",
}

MCP_TASKS = {
    "empty": "Check the weather in Diamantina and London, then calculate the temperature difference",
    "after_weather": "Calculate the temperature difference between Tokyo and London",
}

SEARCH_STATES = {
    "empty": "No searches performed yet.",
    "after_search": "Previous search: web_search({\"query\": \"rust async runtime comparison\"})\nResults: 5 results found.",
}

SEARCH_TASKS = {
    "empty": "Find information about the latest Qwen3.6 model release",
    "after_search": "Extract the full content of the first search result about rust async",
}


# --- Domain configuration registry ---

DOMAINS = {
    "web": {
        "label": "🌐 Web — Browser navigation (A11y Tree)",
        "world_system": WEBWORLD_SYSTEM,  # Uses WebWorld format (compatible)
        "agent_system": WEBWORLD_AGENT,
        "initial_states": WEB_STATES,
        "default_tasks": WEB_TASKS,
        "action_format": "web",  # click([id]), fill([id], "text"), etc.
        "response_tag": None,  # WebWorld outputs raw state (no XML tags)
        "thinking_tag": "reason",  # WebWorld uses <reason>...</reason>
        "response_marker": None,
        "model_hint": "webworld",  # Prefer WebWorld-8B for web
        "max_tokens": 4096,
    },
    "terminal": {
        "label": "💻 Terminal — Linux shell simulation",
        "world_system": _load_prompt("terminal"),
        "agent_system": TERMINAL_AGENT,
        "initial_states": TERMINAL_STATES,
        "default_tasks": TERMINAL_TASKS,
        "action_format": "terminal",  # JSON array of keystrokes
        "response_tag": "predicted_observation",
        "thinking_tag": "think",  # AgentWorld uses <think>...</think>
        "response_marker": "**Environment Observation:**",
        "model_hint": "agentworld",  # Needs AgentWorld
        "max_tokens": 32768,  # AgentWorld needs room to think (~8K reasoning + prediction)
    },
    "mcp": {
        "label": "🔧 MCP — Tool calling simulation",
        "world_system": _load_prompt("mcp"),
        "agent_system": MCP_AGENT,
        "initial_states": MCP_STATES,
        "default_tasks": MCP_TASKS,
        "action_format": "mcp",  # JSON tool call
        "response_tag": "predicted_observation",
        "thinking_tag": "think",
        "response_marker": "**Environment Observation:**",
        "model_hint": "agentworld",
        "max_tokens": 32768,
    },
    "search": {
        "label": "🔍 Search — Web search & retrieval simulation",
        "world_system": _load_prompt("search"),
        "agent_system": SEARCH_AGENT,
        "initial_states": SEARCH_STATES,
        "default_tasks": SEARCH_TASKS,
        "action_format": "mcp",  # Same JSON tool call format
        "response_tag": "predicted_observation",
        "thinking_tag": "think",
        "response_marker": "**Environment Observation:**",
        "model_hint": "agentworld",
        "max_tokens": 32768,
    },
    "swe": {
        "label": "📦 SWE — Software engineering environment",
        "world_system": _load_prompt("swe"),
        "agent_system": SWE_AGENT,
        "initial_states": TERMINAL_STATES,  # Reuse terminal states
        "default_tasks": TERMINAL_TASKS,
        "action_format": "mcp",  # JSON tool call
        "response_tag": "predicted_observation",
        "thinking_tag": "think",
        "response_marker": "**Environment Observation:**",
        "model_hint": "agentworld",
        "max_tokens": 32768,
    },
    "android": {
        "label": "📱 Android — Mobile UI simulation",
        "world_system": _load_prompt("android"),
        "agent_system": ANDROID_AGENT,
        "initial_states": {"home_screen": "Android Home Screen\n\t[1] icon 'Phone'\n\t[2] icon 'Messages'\n\t[3] icon 'Settings'\n\t[4] icon 'Chrome'\n\t[5] icon 'Camera'"},
        "default_tasks": {"home_screen": "Open Settings and enable Airplane Mode"},
        "action_format": "android",
        "response_tag": "predicted_observation",
        "thinking_tag": "think",
        "response_marker": "**Environment Observation:**",
        "model_hint": "agentworld",
        "max_tokens": 32768,
    },
    "os": {
        "label": "🖥️ OS — Desktop computer-use simulation",
        "world_system": _load_prompt("os"),
        "agent_system": OS_AGENT,
        "initial_states": {"desktop": "Desktop accessibility tree:\n\t[1] window 'Firefox — Qwen AI'\n\t\t[2] button 'New Tab'\n\t\t[3] textbox 'Search or enter address'\n\t[4] window 'Terminal'\n\t\t[5] text 'user@navi:~$'\n\t[6] icon 'Files'\n\t[7] icon 'Settings'"},
        "default_tasks": {"desktop": "Open the browser and search for 'Qwen AgentWorld'"},
        "action_format": "os",
        "response_tag": "predicted_observation",
        "thinking_tag": "think",
        "response_marker": "**Environment Observation:**",
        "model_hint": "agentworld",
        "max_tokens": 32768,
    },
}


def get_domain(name: str) -> dict:
    """Get domain configuration by name."""
    if name not in DOMAINS:
        raise ValueError(f"Unknown domain: {name}. Available: {list(DOMAINS.keys())}")
    return DOMAINS[name]


def list_domains() -> list[tuple[str, str]]:
    """Return list of (name, label) for all domains."""
    return [(name, cfg["label"]) for name, cfg in DOMAINS.items()]