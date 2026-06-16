"""sd-cli / stable-diffusion.cpp backend — Ideogram 4 Q4."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from diffuse.paths import SD_CLI_PATH
from diffuse.models import MODELS
from diffuse.backends import require_model_dir

log = logging.getLogger("diffuse")


def load_pipeline_sd_cpp(model_name: str) -> tuple:
    """Prepare sd-cli configuration. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)

    sd_cli = str(SD_CLI_PATH)
    if not Path(sd_cli).exists():
        # Try stable-diffusion.cpp build location
        alt = Path.home() / "git" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
        if alt.exists():
            sd_cli = str(alt)
        else:
            raise FileNotFoundError(f"sd-cli not found at {SD_CLI_PATH}. Run: diffuse build-sd-cpp")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(model_root / "ideogram4-Q4_0.gguf"),
        "uncond_diffusion_model": str(model_root / "ideogram4_uncond-Q4_0.gguf"),
        "llm": str(model_root / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "vae": str(model_root / "vae" / "flux2-vae.safetensors"),
    }

    # Verify all files exist
    for key, path in config.items():
        if key == "sd_cli" and not Path(path).exists():
            raise FileNotFoundError(f"{key} not found: {path}")

    return config, 0.0


def generate_image_sd_cpp(config: dict, prompt: str, seed: int, width: int, height: int, output_path: Path, cpu_fallback: bool = False) -> tuple:
    """Generate image using sd-cli. Returns (output_path, wall_time_seconds, 0.0)."""
    log.info("Generating via sd-cli: seed=%d size=%dx%d cpu_fallback=%s", seed, width, height, cpu_fallback)

    cmd = [
        config["sd_cli"],
        "--diffusion-model", config["diffusion_model"],
        "--uncond-diffusion-model", config["uncond_diffusion_model"],
        "--llm", config["llm"],
        "--vae", config["vae"],
        "-p", prompt,
        "--diffusion-fa",
        "--offload-to-cpu",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "--max-vram", "5.1",
        "--stream-layers",
        "-H", str(height),
        "-W", str(width),
        "--seed", str(seed),
        "-o", str(output_path),
    ]

    # CPU fallback: remove VRAM limits and force everything on CPU
    if cpu_fallback:
        log.warning("Retrying with CPU-only backend — this will be very slow (~30+ minutes)")
        cmd = [
            config["sd_cli"],
            "--diffusion-model", config["diffusion_model"],
            "--uncond-diffusion-model", config["uncond_diffusion_model"],
            "--llm", config["llm"],
            "--vae", config["vae"],
            "-p", prompt,
            "--backend", "cpu",
            "-H", str(height),
            "-W", str(width),
            "--seed", str(seed),
            "-o", str(output_path),
        ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)  # no timeout — let sd-cli finish naturally
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        # Print last 20 lines of stderr for debugging
        stderr_lines = result.stderr.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(f"sd-cli failed (rc={result.returncode}). Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}")

    if not output_path.exists():
        raise FileNotFoundError(f"sd-cli did not produce output: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("sd-cli completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0