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


def _resolve_sd_cli_zimage() -> str:
    """Find sd-cli-zimage binary (compiled from z-image-omini-base branch)."""
    base = Path(SD_CLI_PATH).parent
    zimage_cli = base / "sd-cli-zimage"
    if zimage_cli.exists():
        return str(zimage_cli)
    raise FileNotFoundError(
        f"sd-cli-zimage not found at {zimage_cli}\n"
        f"  Build it: cd ~/git/stable-diffusion.cpp && git checkout z-image-omini-base && "
        f"cd build && make -j$(nproc) sd-cli && cp bin/sd-cli ~/git/ai-dotfiles/diffuse/bin/sd-cli-zimage"
    )


def load_pipeline_sd_cpp(model_name: str) -> tuple:
    """Prepare sd-cli configuration. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    sd_cli = _resolve_sd_cli()

    # Z-Image-Turbo uses different model files than Ideogram 4
    backend_type = model_info.get("backend_type", "sd_cpp")
    if backend_type == "zimage_sd_cpp":
        return load_pipeline_sd_cpp_zimage(model_name, model_root, sd_cli)
    if backend_type == "mageflow_sd_cpp":
        return load_pipeline_sd_cpp_mageflow(model_name, model_root, sd_cli)

    lora_dir = model_root / "lora"
    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(model_root / "ideogram4-Q4_0.gguf"),
        "uncond_diffusion_model": str(model_root / "ideogram4_uncond-Q4_0.gguf"),
        "llm": str(model_root / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "vae": str(model_root / "vae" / "flux2-vae.safetensors"),
        "lora_dir": str(lora_dir) if lora_dir.exists() else None,
    }

    # Verify all files exist
    for key, path in config.items():
        if key == "sd_cli" and not Path(path).exists():
            raise FileNotFoundError(f"{key} not found: {path}")

    return config, 0.0


def load_pipeline_sd_cpp_zimage(model_name: str, model_root: Path, sd_cli: str) -> tuple:
    """Prepare sd-cli config for Z-Image-Turbo. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]

    # Use the dedicated z-image sd-cli binary (compiled from z-image-omini-base branch)
    sd_cli = _resolve_sd_cli_zimage()

    # Find the best GGUF for our VRAM
    import torch
    vram_gb = torch.cuda.mem_get_info()[1] / 1e9 if torch.cuda.is_available() else 999
    if vram_gb <= 6:
        preferred = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M"]
    else:
        preferred = ["Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S"]

    gguf_files = list(model_root.glob("z_image_turbo-*.gguf"))
    dit_gguf = None
    for pref in preferred:
        matches = [f for f in gguf_files if pref.lower() in f.name.lower()]
        if matches:
            dit_gguf = str(matches[0])
            break
    if not dit_gguf and gguf_files:
        dit_gguf = str(gguf_files[0])
    if not dit_gguf:
        raise FileNotFoundError(f"No Z-Image GGUF found in {model_root}")

    # Find Qwen3-4B text encoder GGUF
    llm_files = list(model_root.glob("Qwen3-4B-*.gguf"))
    llm_gguf = str(llm_files[0]) if llm_files else None
    if not llm_gguf:
        raise FileNotFoundError(f"No Qwen3-4B GGUF found in {model_root}")

    # VAE: use the Z-Image pipeline VAE (same as Flux)
    vae_path = str(model_root / "pipeline" / "vae" / "diffusion_pytorch_model.safetensors")
    if not Path(vae_path).exists():
        # Fallback to Ideogram 4's Flux VAE
        vae_path = str(model_root.parent / "ideogram-4-Q4_0" / "vae" / "flux2-vae.safetensors")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": dit_gguf,
        "llm": llm_gguf,
        "vae": vae_path,
        "is_zimage": True,
    }

    return config, 0.0


def load_pipeline_sd_cpp_mageflow(model_name: str, model_root: Path, sd_cli: str) -> tuple:
    """Prepare sd-cli config for Mage-Flow-Edit-Turbo. Returns (config_dict, 0.0).

    Mage-Flow-Edit uses:
    - DiT GGUF (NVFP4) as diffusion model
    - Qwen3-VL-4B GGUF as text encoder (--llm)
    - Qwen3-VL-4B mmproj F16 as vision encoder (--llm_vision)
    - Mage-VAE GGUF as VAE
    - Reference image via -r flag (instruction-based editing, no masks)
    - Turbo: 4 steps, cfg=1.0
    """
    dit_gguf = str(model_root / "mageflow-edit-turbo-nvfp4.gguf")
    vae_gguf = str(model_root / "pig_mageflow_vae_fp32-f16.gguf")
    llm_gguf = str(model_root / "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
    mmproj_gguf = str(model_root / "mmproj-Qwen3VL-4B-Instruct-F16.gguf")

    for label, path in [("DiT", dit_gguf), ("VAE", vae_gguf), ("LLM", llm_gguf), ("mmproj", mmproj_gguf)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": dit_gguf,
        "llm": llm_gguf,
        "llm_vision": mmproj_gguf,
        "vae": vae_gguf,
        "is_mageflow": True,
    }

    return config, 0.0


def generate_image_mageflow_sd_cpp(
    config: dict, prompt: str, seed: int, width: int, height: int,
    output_path: Path, ref_image: str, cpu_fallback: bool = False,
) -> tuple:
    """Generate edited image using sd-cli with Mage-Flow-Edit-Turbo.

    Mage-Flow-Edit is instruction-based: it takes a reference image and a text
    instruction (e.g. "change the background to a beach") and produces an
    edited image. No masks needed. Turbo = 4 steps, cfg=1.0.
    """
    log.info("Generating via sd-cli Mage-Flow-Edit: seed=%d ref=%s", seed, ref_image)

    cmd = [
        config["sd_cli"],
        "--diffusion-model", config["diffusion_model"],
        "--llm", config["llm"],
        "--llm_vision", config["llm_vision"],
        "--vae", config["vae"],
        "-r", ref_image,
        "-p", prompt,
        "--cfg-scale", "1.0",
        "--steps", "4",
        "--sampling-method", "euler",
        "--diffusion-fa",
        "--offload-to-cpu",
        "-v",
        "--seed", str(seed),
        "-o", str(output_path),
    ]

    if cpu_fallback:
        cmd += ["--backend", "cpu"]
        log.warning("Retrying with CPU-only backend — this will be very slow")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(f"sd-cli failed (rc={result.returncode}). Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}")

    if not output_path.exists():
        raise FileNotFoundError(f"sd-cli did not produce output: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("sd-cli Mage-Flow completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0


def load_pipeline_sd_cpp_video(model_name: str) -> tuple:
    """Prepare sd-cli config for Wan2.2 video generation. Returns (config_dict, 0.0).

    Supports both the 5B TI2V (dense, unified T2V+I2V) and the 14B A14B (MoE).
    The 5B uses wan2.2_vae (high-compression 4x16x16); the 14B uses wan_2.1_vae.
    clip_vision is only needed for I2V — it's optional (loaded if present).
    """
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    sd_cli = _resolve_sd_cli()

    gguf_name = model_info.get("gguf_file", "Wan2.2-TI2V-5B-Q4_K_M.gguf")

    # VAE: 5B TI2V uses wan2.2_vae; 14B uses wan_2.1_vae
    vae_name = model_info.get("vae_file", "wan2.2_vae.safetensors")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(model_root / gguf_name),
        "vae": str(model_root / "vae" / vae_name),
        "t5xxl": str(model_root / "text_encoder" / "umt5-xxl-encoder-Q8_0.gguf"),
    }

    # clip_vision is only needed for I2V — load if present, but don't fail if missing
    clip_vision_gguf = model_root / "clip_vision" / "clip_vision_h.gguf"
    if clip_vision_gguf.exists():
        config["clip_vision"] = str(clip_vision_gguf)

    # Verify required files exist
    for key, path in config.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{key} not found: {path}")

    return config, 0.0


def generate_image_sd_cpp(config: dict, prompt: str, seed: int, width: int, height: int, output_path: Path, cpu_fallback: bool = False, nsfw: bool = False) -> tuple:
    """Generate image using sd-cli. Returns (output_path, wall_time_seconds, 0.0)."""
    log.info("Generating via sd-cli: seed=%d size=%dx%d cpu_fallback=%s nsfw=%s", seed, width, height, cpu_fallback, nsfw)

    is_zimage = config.get("is_zimage", False)

    # Z-Image-Turbo: no uncond model, CFG=0, 9 steps, flux_flow prediction
    if is_zimage:
        cmd = [
            config["sd_cli"],
            "--diffusion-model", config["diffusion_model"],
            "--llm", config["llm"],
            "--vae", config["vae"],
            "-p", prompt,
            "--diffusion-fa",
            "--offload-to-cpu",
            "--clip-on-cpu",
            "--vae-on-cpu",
            "--vae-tiling",
            "-H", str(height),
            "-W", str(width),
            "--seed", str(seed),
            "-o", str(output_path),
        ]
        # Add steps and cfg from config if provided
        if "steps" in config:
            cmd += ["--steps", str(config["steps"])]
        if "cfg" in config:
            cmd += ["--cfg-scale", str(config["cfg"])]
    else:
        cmd = [
            config["sd_cli"],
            "--diffusion-model", config["diffusion_model"],
        ]
        if not nsfw and "uncond_diffusion_model" in config:
            cmd += ["--uncond-diffusion-model", config["uncond_diffusion_model"]]
        cmd += [
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
        if nsfw:
            lora_dir = config.get("lora_dir")
            if lora_dir:
                import os as _os
                loras = [f for f in _os.listdir(lora_dir) if f.endswith((".safetensors", ".gguf", ".pt"))]
                if loras:
                    cmd += ["--lora-model-dir", lora_dir]
                    if "<lora:" not in prompt:
                        prompt_lora = " ".join(f"<lora:{f.rsplit('.', 1)[0]}:0.6>" for f in loras)
                        cmd[cmd.index("-p") + 1] = prompt + " " + prompt_lora

    # CPU fallback: remove VRAM limits and force everything on CPU
    if cpu_fallback:
        log.warning("Retrying with CPU-only backend — this will be very slow (~30+ minutes)")
        if is_zimage:
            cmd = [
                config["sd_cli"],
                "--diffusion-model", config["diffusion_model"],
                "--llm", config["llm"],
                "--vae", config["vae"],
                "-p", prompt,
                "--backend", "cpu",
                "-H", str(height),
                "-W", str(width),
                "--seed", str(seed),
                "-o", str(output_path),
            ]
            if "steps" in config:
                cmd += ["--steps", str(config["steps"])]
            if "cfg" in config:
                cmd += ["--cfg-scale", str(config["cfg"])]
        else:
            cmd = [
                config["sd_cli"],
                "--diffusion-model", config["diffusion_model"],
            ]
            if not nsfw and "uncond_diffusion_model" in config:
                cmd += ["--uncond-diffusion-model", config["uncond_diffusion_model"]]
            cmd += [
                "--llm", config["llm"],
                "--vae", config["vae"],
                "-p", prompt,
                "--backend", "cpu",
                "-H", str(height),
                "-W", str(width),
                "--seed", str(seed),
                "-o", str(output_path),
            ]
            if nsfw:
                lora_dir = config.get("lora_dir")
                if lora_dir:
                    import os as _os
                    loras = [f for f in _os.listdir(lora_dir) if f.endswith((".safetensors", ".gguf", ".pt"))]
                    if loras:
                        cmd += ["--lora-model-dir", lora_dir]
                        if "<lora:" not in prompt:
                            prompt_lora = " ".join(f"<lora:{f.rsplit('.', 1)[0]}:0.6>" for f in loras)
                            cmd[cmd.index("-p") + 1] = prompt + " " + prompt_lora

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
    input_image: str | None,
    output_path: Path,
    max_vram: float = 5.1,
    vae_on_cpu: bool = False,
) -> tuple:
    """Generate video using sd-cli (Wan2.2 T2V or I2V).

    For T2V: input_image=None, clip_vision not needed.
    For I2V: input_image required, clip_vision used if present in config.

    sd-cli outputs a PNG sequence; we assemble into MP4 with ffmpeg.
    Returns (mp4_path, wall_time_seconds, 0.0).
    """
    is_i2v = input_image is not None
    mode_label = "I2V" if is_i2v else "T2V"
    log.info(
        "Generating Wan2.2 %s: seed=%d %dx%d frames=%d fps=%d steps=%d cfg=%.1f",
        mode_label, seed, width, height, video_frames, fps, steps, cfg_scale,
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
        "--stream-layers",
        "--max-vram", str(max_vram),
        "--vae-tiling",
        "-o", frame_pattern,
    ]

    # VAE on CPU only for large models (14B) — 5B fits VAE on GPU
    if vae_on_cpu:
        cmd.insert(cmd.index("--stream-layers"), "--vae-on-cpu")

    # I2V: add clip_vision and input image
    if is_i2v:
        if "clip_vision" in config:
            cmd += ["--clip_vision", config["clip_vision"]]
        cmd += ["-i", str(input_image)]

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