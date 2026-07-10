"""System prompts for LLM enhancement — loaded from YAML files."""
from __future__ import annotations

from pathlib import Path

import yaml

from diffuse.paths import PROMPTS_DIR

# ── Prompt name → YAML file mapping ────────────────────────────────────────
_PROMPT_FILES: dict[str, str] = {
    "ideogram_enhance": "ideogram_enhance.yaml",
    "vision_enhance": "vision_enhance.yaml",
    "edit_enhance": "edit_enhance.yaml",
    "edit_vision": "edit_vision.yaml",
    "vision_analysis": "vision_analysis.yaml",
    "video_enhance": "video_enhance.yaml",
}

# ── Cache ───────────────────────────────────────────────────────────────────
_cache: dict[str, str] = {}


def _load_prompt(name: str) -> str:
    """Load a system prompt from YAML, with caching."""
    if name in _cache:
        return _cache[name]

    filename = _PROMPT_FILES.get(name)
    if filename is None:
        raise ValueError(f"Unknown prompt: {name!r}. Available: {list(_PROMPT_FILES)}")

    path = PROMPTS_DIR / filename
    if not path.exists():
        # Fallback: try to find it relative to the package
        path = Path(__file__).resolve().parent.parent / "prompts" / filename

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    content = data.get("content", "").strip()
    if not content:
        raise ValueError(f"Prompt file {path} has empty 'content' field")

    _cache[name] = content
    return content


# ── Convenience accessors ───────────────────────────────────────────────────
def get_ideogram_enhance_prompt() -> str:
    return _load_prompt("ideogram_enhance")


def get_vision_enhance_prompt() -> str:
    return _load_prompt("vision_enhance")


def get_edit_enhance_prompt() -> str:
    return _load_prompt("edit_enhance")


def get_edit_vision_prompt() -> str:
    return _load_prompt("edit_vision")


def get_vision_analysis_prompt() -> str:
    return _load_prompt("vision_analysis")


def get_video_enhance_prompt() -> str:
    return _load_prompt("video_enhance")