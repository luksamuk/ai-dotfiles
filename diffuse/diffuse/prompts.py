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


def _toggle_safety_rules(content: str, nsfw: bool) -> str:
    """When nsfw=False, uncomment #-prefixed safety avoidance lines.
    When nsfw=True, keep them commented (disabled)."""
    if nsfw:
        return content

    lines = content.split('\n')
    result = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('#'):
            uncommented = stripped[2:] if stripped.startswith('# ') else stripped[1:]
            uncommented = uncommented.replace(' — disabled (LoRA bypasses the content filter)', '')
            result.append(uncommented)
        else:
            result.append(line)
    return '\n'.join(result)


# ── Convenience accessors ───────────────────────────────────────────────────
def get_ideogram_enhance_prompt(nsfw: bool = False) -> str:
    return _toggle_safety_rules(_load_prompt("ideogram_enhance"), nsfw)


def get_vision_enhance_prompt(nsfw: bool = False) -> str:
    return _toggle_safety_rules(_load_prompt("vision_enhance"), nsfw)


def get_edit_enhance_prompt(nsfw: bool = False) -> str:
    return _toggle_safety_rules(_load_prompt("edit_enhance"), nsfw)


def get_edit_vision_prompt(nsfw: bool = False) -> str:
    return _toggle_safety_rules(_load_prompt("edit_vision"), nsfw)


def get_vision_analysis_prompt(nsfw: bool = False) -> str:
    return _toggle_safety_rules(_load_prompt("vision_analysis"), nsfw)


def get_video_enhance_prompt(nsfw: bool = False) -> str:
    return _toggle_safety_rules(_load_prompt("video_enhance"), nsfw)