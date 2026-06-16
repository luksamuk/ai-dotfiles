"""Path constants and environment setup for diffuse."""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent  # diffuse/ repo root
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
TRITON_CACHE_DIR = OUTPUTS_DIR / ".triton_cache"
GEMLITE_PERSIST_PATH = OUTPUTS_DIR / ".gemlite_cache" / "autotune.json"
SD_CLI_PATH = SCRIPT_DIR / "bin" / "sd-cli"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

DEFAULT_VISION_MODEL = "minicpm-v-4.6"

# ── LLM swap URL ──────────────────────────────────────────────────────────
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")


def setup_environment() -> None:
    """Set cache directories before any torch/triton imports."""
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GEMLITE_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))