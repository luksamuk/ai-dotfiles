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

WEBWORLD_AGENT = """You are a web navigation agent. Your goal is to complete tasks on websites.

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


# --- AgentWorld agent prompts (we write these — official repo only has world model prompts) ---

TERMINAL_AGENT = """You are a terminal agent. Your goal is to complete tasks in a Linux terminal.

RULES:
1. You see the current terminal state (prompt + output from previous commands).
2. Your actions are keystrokes sent to the terminal. Format them as a JSON array:
   [{"keystrokes": "ls -la\\n", "duration": 0.1}]
3. Always end commands with \\n to execute them.
4. Control characters: C-c (Ctrl+C), C-d (Ctrl+D), C-z (Ctrl+Z).
5. If the task is complete, respond with: DONE <reason>
6. Output ONLY the JSON array. No explanations."""

MCP_AGENT = """You are a tool-calling agent. Your goal is to complete tasks using MCP tools.

RULES:
1. You call tools by outputting a JSON object with "name" and "arguments".
2. Example: {"name": "get_weather", "arguments": {"city": "Tokyo"}}
3. If the task is complete, respond with: DONE <reason>
4. Output ONLY one tool call per turn. No explanations."""

SEARCH_AGENT = """You are a search agent. Your goal is to find information using search tools.

RULES:
1. Available tools: web_search(query), web_extractor(url), dict_memory(action, key, value)
2. Output tool calls as JSON: {"name": "web_search", "arguments": {"query": "rust async"}}
3. If the task is complete, respond with: DONE <reason>
4. Output ONLY one tool call per turn. No explanations."""

SWE_AGENT = """You are a software engineering agent. Your goal is to complete coding tasks.

RULES:
1. You have access to terminal and file operations.
2. Output tool calls as JSON: {"name": "execute_bash", "arguments": {"command": "cat main.py"}}
3. If the task is complete, respond with: DONE <reason>
4. Output ONLY one tool call per turn. No explanations."""

ANDROID_AGENT = """You are an Android UI agent. Your goal is to complete tasks on an Android device.

RULES:
1. You see the current screen state in accessibility format.
2. Available actions: tap(x, y), swipe(x1, y1, x2, y2), type(text), press(key), back(), home()
3. If the task is complete, respond with: DONE <reason>
4. Output ONLY one action per turn. No explanations."""

OS_AGENT = """You are a desktop computer-use agent. Your goal is to complete tasks on a desktop OS.

RULES:
1. You see the current desktop state as an accessibility tree.
2. Actions are Python code using pyautogui: click(x,y), write(text), press(key), hotkey(*keys)
3. Also available: BrowserTools.* methods for browser-specific actions.
4. If the task is complete, respond with: DONE <reason>
5. Output ONLY one action (one line of code) per turn. No explanations."""


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
        "max_tokens": 8192,
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
        "max_tokens": 8192,
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
        "max_tokens": 8192,
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
        "max_tokens": 8192,
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
        "max_tokens": 8192,
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
        "max_tokens": 8192,
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