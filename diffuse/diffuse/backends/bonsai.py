"""Bonsai image backend — subprocess into bonsai-venv.

Bonsai requires diffusers 0.38+ (Flux2Pipeline), which conflicts with the
diffuse main venv (diffusers 0.33 for HiDream). We run the Bonsai pipeline
in a dedicated venv via subprocess, similar to how LingBot ran.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from diffuse.models import MODELS

log = logging.getLogger("diffuse")

BONSAI_VENV_PY = Path(__file__).resolve().parent.parent.parent / "bonsai-venv" / "bin" / "python"
BONSAI_RUNNER = Path(__file__).resolve().parent.parent.parent / "scripts" / "bonsai_runner.py"


def load_pipeline_bonsai(model_name: str) -> tuple:
    """Return runner config. Actual loading happens in subprocess."""
    model_info = MODELS[model_name]
    model_root = Path(__file__).resolve().parent.parent.parent / "models" / model_info["dir"]

    if not BONSAI_VENV_PY.exists():
        raise FileNotFoundError(
            f"Bonsai venv not found: {BONSAI_VENV_PY}\n"
            f"  Run: cd ~/git/ai-dotfiles/diffuse && uv venv bonsai-venv && uv pip install --python bonsai-venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128 && uv pip install --python bonsai-venv/bin/python 'diffusers>=0.38' 'transformers>=4.46' accelerate hqq gemlite einops pillow safetensors sentencepiece tokenizers"
        )
    if not BONSAI_RUNNER.exists():
        raise FileNotFoundError(f"Bonsai runner script not found: {BONSAI_RUNNER}")

    config = {
        "venv_py": str(BONSAI_VENV_PY),
        "runner": str(BONSAI_RUNNER),
        "model_dir": str(model_root),
    }
    return config, 0.0


def generate_image_bonsai(
    config: dict,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    output_path: Path,
) -> tuple:
    """Generate image by invoking bonsai_runner.py in the Bonsai venv.

    Returns (output_path, wall_time, peak_vram_mb).
    """
    cmd = [
        config["venv_py"],
        config["runner"],
        "--model-dir", config["model_dir"],
        "--prompt", prompt,
        "--seed", str(seed),
        "--steps", str(steps),
        "--width", str(width),
        "--height", str(height),
        "--guidance", str(guidance),
        "--output", str(output_path),
    ]

    log.info("Bonsai: seed=%d %dx%d steps=%d", seed, width, height, steps)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False, text=True)
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Bonsai runner failed (rc={result.returncode})")

    if not output_path.exists():
        raise FileNotFoundError(f"Bonsai runner did not produce output: {output_path}")

    return output_path, wall_time, 0.0