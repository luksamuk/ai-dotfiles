"""LingBot-Video backend — text-to-video via subprocess (separate venv).

LingBot requires diffusers 0.39 + transformers 5.x, which conflict with the
diffuse venv (diffusers 0.33 + transformers 4.57 for HiDream). We run the
LingBot pipeline in a dedicated venv at ~/git/lingbot-video/.venv via
subprocess, similar to how sd_cpp uses sd-cli.

The runner script (scripts/lingbot_runner.py) handles model loading, GPU
dispatch, and video export. This module just builds the command and captures
output.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from diffuse.models import MODELS

log = logging.getLogger("diffuse")

LINGBOT_VENV_PY = Path.home() / "git" / "lingbot-video" / ".venv" / "bin" / "python"
LINGBOT_RUNNER = Path(__file__).resolve().parent.parent.parent / "scripts" / "lingbot_runner.py"


def load_pipeline_lingbot(model_name: str) -> tuple:
    """Return runner config. Actual loading happens in subprocess."""
    model_info = MODELS[model_name]
    model_root = Path(os.path.expanduser(f"~/.llama-models/{model_info['dir']}"))

    if not LINGBOT_VENV_PY.exists():
        raise FileNotFoundError(
            f"LingBot venv not found: {LINGBOT_VENV_PY}\n"
            f"  Run: cd ~/git/lingbot-video && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e ."
        )
    if not LINGBOT_RUNNER.exists():
        raise FileNotFoundError(f"LingBot runner script not found: {LINGBOT_RUNNER}")

    config = {
        "venv_py": str(LINGBOT_VENV_PY),
        "runner": str(LINGBOT_RUNNER),
        "model_dir": str(model_root),
    }
    return config, 0.0


def generate_video_lingbot(
    config: dict,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    video_frames: int,
    fps: int,
    steps: int,
    cfg_scale: float,
    shift: float,
    output_path: Path,
) -> tuple:
    """Generate video by invoking lingbot_runner.py in the LingBot venv.

    Returns (output_path, wall_time, 0.0).
    """
    cmd = [
        config["venv_py"],
        config["runner"],
        "--model-dir", config["model_dir"],
        "--prompt", prompt,
        "--negative-prompt", negative_prompt,
        "--width", str(width),
        "--height", str(height),
        "--video-frames", str(video_frames),
        "--fps", str(fps),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--shift", str(shift),
        "--seed", str(seed),
        "--output", str(output_path),
    ]

    log.info("LingBot T2V: seed=%d %dx%d frames=%d steps=%d", seed, width, height, video_frames, steps)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False, text=True)
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(f"LingBot runner failed (rc={result.returncode})")

    if not output_path.exists():
        raise FileNotFoundError(f"LingBot runner did not produce output: {output_path}")

    return output_path, wall_time, 0.0