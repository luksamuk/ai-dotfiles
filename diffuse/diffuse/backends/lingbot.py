"""LingBot-Video backend — text-to-video and text+image-to-video with custom diffusers pipeline.

LingBot-Video Dense 1.3B uses a custom pipeline (LingBotVideoPipeline) with:
  - LingBotVideoTransformer3DModel (DiT, 2.79 GB bf16)
  - Qwen3VLForConditionalGeneration (text encoder, 8.88 GB bf16 — CPU offloaded)
  - AutoencoderKLWan (VAE, shared with Wan2.2)
  - FlowUniPCMultistepScheduler (vendored scheduler)

Requires the lingbot-video repo cloned at ~/git/lingbot-video for the custom
pipeline, transformer, and scheduler code.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

from diffuse.models import MODELS

log = logging.getLogger("diffuse")

LINGBOT_REPO = Path(os.path.expanduser("~/git/lingbot-video"))


def load_pipeline_lingbot(model_name: str) -> tuple:
    """Load LingBotVideoPipeline with CPU offload for 6GB VRAM.

    Returns (pipeline, load_time_seconds).
    """
    import torch
    from accelerate import infer_auto_device_map, dispatch_model

    model_info = MODELS[model_name]
    model_root = Path(os.path.expanduser(f"~/.llama-models/{model_info['dir']}"))
    lingbot_repo = Path(os.path.expanduser(model_info.get("lingbot_repo", "~/git/lingbot-video")))

    if not lingbot_repo.exists():
        print(f"\n  ✗ LingBot repo not found: {lingbot_repo}")
        print(f"    Clone from: https://github.com/Robbyant/lingbot-video")
        sys.exit(1)

    if not model_root.exists():
        print(f"\n  ✗ Model not found: {model_root}")
        print(f"    Run: diffuse download lingbot")
        sys.exit(1)

    # Add repo to sys.path for custom pipeline/transformer/scheduler imports
    if str(lingbot_repo) not in sys.path:
        sys.path.insert(0, str(lingbot_repo))

    os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_flash")

    torch.cuda.empty_cache()
    gc.collect()

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    t0 = time.perf_counter()
    print("  Loading LingBot pipeline on CPU...")

    from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

    dtype_map = {
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "vae": torch.float32,
    }

    pipe = LingBotVideoPipeline.from_pretrained(
        str(model_root),
        trust_remote_code=True,
        torch_dtype=dtype_map,
    )

    load_time = time.perf_counter() - t0
    print(f"  Pipeline loaded on CPU in {load_time:.1f}s")

    # Dispatch transformer to GPU with memory limit, keep text encoder and VAE on CPU.
    # DiT is 2.79 GB bf16 — fits in 6 GB with headroom for activations.
    # Text encoder (8.88 GB) stays on CPU (only needed for prompt encoding).
    # VAE (0.25 GB) goes to GPU for decode.
    headroom_gb = 1.5
    max_vram_mb = int((total / 1e9 - headroom_gb) * 1024)
    print(f"  Max GPU for DiT layers: {max_vram_mb} MiB")

    device_map = infer_auto_device_map(
        pipe.transformer,
        max_memory={0: f"{max_vram_mb}MiB", "cpu": "24GiB"},
    )
    gpu_layers = len([v for v in device_map.values() if v == 0 or v == torch.device(0)])
    cpu_layers = len([v for v in device_map.values() if v == "cpu" or v == torch.device("cpu")])
    print(f"  DiT device map: {gpu_layers} layers on GPU, {cpu_layers} on CPU")

    pipe.transformer = dispatch_model(pipe.transformer, device_map=device_map)

    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM after dispatch: {free/1e9:.1f} GB free")

    return pipe, load_time


def generate_video_lingbot(
    pipeline,
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
    """Generate video using LingBot pipeline. Returns (mp4_path, wall_time, 0.0)."""
    import torch
    import numpy as np
    from diffusers.utils import export_to_video

    generator = torch.Generator("cpu").manual_seed(seed)

    log.info(
        "LingBot T2V: seed=%d %dx%d frames=%d fps=%d steps=%d cfg=%.1f shift=%.1f",
        seed, width, height, video_frames, fps, steps, cfg_scale, shift,
    )

    # num_frames must be 1 or 4n+1 for Wan VAE temporal compression
    if video_frames != 1 and (video_frames - 1) % 4 != 0:
        nearest = ((video_frames - 1) // 4) * 4 + 1
        log.warning("video_frames %d not 4n+1, adjusting to %d", video_frames, nearest)
        video_frames = nearest

    print(f"  🎬 LingBot T2V: {width}×{height}, {video_frames} frames @ {fps} fps ({video_frames/fps:.1f}s)")
    print(f"     Steps: {steps}, CFG: {cfg_scale}, Shift: {shift}, Seed: {seed}")

    t0 = time.perf_counter()

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=video_frames,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            shift=shift,
            generator=generator,
            output_type="np",
            batch_cfg=True,
        )

    wall_time = time.perf_counter() - t0

    frames = result.frames
    if isinstance(frames, list):
        frames = np.stack(frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=fps)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("LingBot video completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0