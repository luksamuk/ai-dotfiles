"""Backend registry — load_pipeline and unload_pipeline dispatch to the right backend."""
from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

from diffuse.paths import MODELS_DIR
from diffuse.models import MODELS

log = logging.getLogger("diffuse")


# ── Shared helpers ──────────────────────────────────────────────────────────
def _find_subdir(root: Path, *hints: str) -> Path:
    """Find a child directory whose name contains any of the hints."""
    matches = [p for p in root.iterdir() if p.is_dir() and any(h in p.name for h in hints)]
    if not matches:
        present = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "(empty)"
        raise FileNotFoundError(f"No subdir matching {hints!r} under {root}. Present: {present}")
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def require_model_dir(model_name: str) -> Path:
    """Ensure model weights are present and return path."""
    model_info = MODELS[model_name]
    model_root = MODELS_DIR / model_info["dir"]
    if not model_root.exists():
        print(f"\n  ✗ Model not found: {model_root}")
        print(f"    Run: diffuse download {model_name.split('-')[0]}")
        sys.exit(1)
    return model_root


# ── Backend dispatch ─────────────────────────────────────────────────────────
def load_pipeline(model_name: str, editing: bool = False) -> tuple:
    """Load a pipeline based on the model's backend_type.

    Returns (pipeline_or_config, load_time_seconds).
    """
    model_info = MODELS[model_name]
    backend_type = model_info.get("backend_type", "gemlite")

    if backend_type == "gemlite":
        from diffuse.backends.gemlite import load_pipeline_gemlite
        return load_pipeline_gemlite(model_name)
    elif backend_type == "sd_cpp":
        from diffuse.backends.sd_cpp import load_pipeline_sd_cpp
        return load_pipeline_sd_cpp(model_name)
    elif backend_type == "hidream":
        from diffuse.backends.hidream import load_pipeline_hidream
        return load_pipeline_hidream(model_name, editing=editing)
    elif backend_type == "framepack":
        from diffuse.backends.framepack import load_pipeline as load_pipeline_framepack
        return load_pipeline_framepack(editing=editing)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def unload_pipeline() -> None:
    """Force-unload pipeline from VRAM and system memory."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    log.info("Pipeline unloaded from VRAM")