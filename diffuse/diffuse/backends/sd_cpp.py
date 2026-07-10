"""sd-cli / stable-diffusion.cpp backend — Ideogram 4 Q4 + Wan2.2 I2V video."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from diffuse.paths import SD_CLI_PATH
from diffuse.models import MODELS
from diffuse.backends import require_model_dir

log = logging.getLogger("diffuse")


def _resolve_sd_cli() -> str:
    """Find sd-cli binary."""
    sd_cli = str(SD_CLI_PATH)
    if not Path(sd_cli).exists():
        alt = Path.home() / "git" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
        if alt.exists():
            sd_cli = str(alt)
        else:
            raise FileNotFoundError(f"sd-cli not found at {SD_CLI_PATH}. Run: diffuse build-sd-cpp")
    return sd_cli


def load_pipeline_sd_cpp(model_name: str) -> tuple:
    """Prepare sd-cli configuration. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    sd_cli = _resolve_sd_cli()

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


def load_pipeline_sd_cpp_video(model_name: str) -> tuple:
    """Prepare sd-cli config for Wan2.2 I2V video generation. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    sd_cli = _resolve_sd_cli()

    # The AllInOne GGUF contains both low-noise and high-noise in one file
    gguf_name = model_info.get("gguf_file", "wan2.2-i2v-rapid-aio-v10-nsfw-Q2_K.gguf")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(model_root / gguf_name),
        "vae": str(model_root / "vae" / "wan_2.1_vae.safetensors"),
        "t5xxl": str(model_root / "text_encoder" / "umt5-xxl-encoder-Q8_0.gguf"),
        # clip_vision must be GGUF format — safetensors has a 5D tensor that
        # sd-cli's internal GGUF converter can't handle (patch_embedding.weight).
        # Converted manually: clip_vision_h.safetensors → clip_vision_h.gguf
        "clip_vision": str(model_root / "clip_vision" / "clip_vision_h.gguf"),
    }

    # Verify all files exist
    for key, path in config.items():
        if not Path(path).exists():
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


def generate_video_sd_cpp(
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
    flow_shift: float,
    input_image: str,
    output_path: Path,
    max_vram: float = 5.1,
) -> tuple:
    """Generate video using sd-cli (Wan2.2 I2V).

    The AllInOne GGUF merges low-noise + high-noise into one file, so we
    only need --diffusion-model (no --high-noise-diffusion-model).

    sd-cli outputs a PNG sequence; we assemble into MP4 with ffmpeg.
    Returns (mp4_path, wall_time_seconds, 0.0).
    """
    log.info(
        "Generating Wan2.2 I2V: seed=%d %dx%d frames=%d fps=%d steps=%d cfg=%.1f",
        seed, width, height, video_frames, fps, steps, cfg_scale,
    )

    # Output: sequence of PNGs in a temp dir, then ffmpeg → mp4
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = str(frame_dir / "frame_%04d.png")

    cmd = [
        config["sd_cli"],
        "-M", "vid_gen",
        "--diffusion-model", config["diffusion_model"],
        "--vae", config["vae"],
        "--t5xxl", config["t5xxl"],
        "--clip_vision", config["clip_vision"],
        "-i", input_image,
        "-p", prompt,
        "-n", negative_prompt,
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", "euler",
        "--steps", str(steps),
        "-W", str(width),
        "-H", str(height),
        "--seed", str(seed),
        "--video-frames", str(video_frames),
        "--fps", str(fps),
        "--flow-shift", str(flow_shift),
        "--diffusion-fa",
        "--offload-to-cpu",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "--stream-layers",
        "--max-vram", str(max_vram),
        "--vae-tiling",
        "-o", frame_pattern,
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(
            f"sd-cli video failed (rc={result.returncode}). "
            f"Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}"
        )

    # Assemble PNG sequence → MP4 via ffmpeg
    frame_files = sorted(frame_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(
            f"sd-cli did not produce any frames in {frame_dir}"
        )

    log.info("Assembling %d frames → %s", len(frame_files), output_path)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "16",
        str(output_path),
    ]
    ff_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if ff_result.returncode != 0:
        log.error("ffmpeg: %s", ff_result.stderr[-500:])
        raise RuntimeError(f"ffmpeg failed (rc={ff_result.returncode})")

    # Clean up frame PNGs (keep the mp4)
    for f in frame_files:
        f.unlink()
    frame_dir.rmdir()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(
        "sd-cli video completed in %.1fs, output %.2f MiB (%d frames @ %d fps)",
        wall_time, file_size_mb, len(frame_files), fps,
    )

    return output_path, wall_time, 0.0