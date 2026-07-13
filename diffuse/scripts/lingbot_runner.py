#!/usr/bin/env python3
"""LingBot video generation runner — called as subprocess by diffuse.

Uses the lingbot-video venv which has diffusers 0.39 + transformers 5.x.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add lingbot-video repo to sys.path for custom pipeline imports
LINGBOT_REPO = Path(os.path.expanduser("~/git/lingbot-video"))
if str(LINGBOT_REPO) not in sys.path:
    sys.path.insert(0, str(LINGBOT_REPO))

os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_flash")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-frames", type=int, default=33)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    import numpy as np
    from diffusers.utils import export_to_video
    from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total", flush=True)

    t0 = time.perf_counter()
    print("  Loading LingBot pipeline on CPU...", flush=True)

    dtype_map = {
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "vae": torch.float32,
    }

    pipe = LingBotVideoPipeline.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype_map,
    )

    load_time = time.perf_counter() - t0
    print(f"  Pipeline loaded in {load_time:.1f}s", flush=True)

    # Reload VAE with materialized weights (pipeline may leave it in meta state).
    # Use fp16 instead of fp32 to save VRAM for decode on 6 GB cards.
    from diffusers import AutoencoderKLWan
    vae_path = os.path.join(args.model_dir, "vae")
    pipe.vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32)

    # Move transformer to GPU. VAE stays on CPU initially — offloaded during
    # denoise and only moved to GPU for decode (saves ~0.3 GB during diffusion).
    pipe.transformer.to("cuda")
    # Force _execution_device to cuda via property override (read-only property
    # normally returns first module's device, which is CPU since text_encoder
    # is on CPU). We pre-compute prompt embeddings on CPU and pass them
    # directly, so encode_prompt is never called inside __call__.
    type(pipe)._execution_device = property(lambda self: torch.device("cuda", 0))

    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM after move: {free/1e9:.1f} GB free", flush=True)

    # num_frames must be 1 or 4n+1
    vf = args.video_frames
    if vf != 1 and (vf - 1) % 4 != 0:
        vf = ((vf - 1) // 4) * 4 + 1
        print(f"  Adjusted video_frames to {vf} (must be 4n+1)", flush=True)

    print(f"  Generating: {args.width}x{args.height}, {vf} frames, {args.steps} steps, seed={args.seed}", flush=True)

    # Pre-compute prompt embeddings on CPU (text_encoder is on CPU).
    # This avoids device mismatch inside __call__ when _execution_device=cuda
    # but text_encoder is on CPU.
    print("  Encoding prompt on CPU...", flush=True)
    prompt_embeds, prompt_mask = pipe.encode_prompt(args.prompt, device="cpu")
    neg_embeds, neg_mask = pipe.encode_prompt(args.negative_prompt, device="cpu")

    generator = torch.Generator("cpu").manual_seed(args.seed)

    t1 = time.perf_counter()
    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=vf,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            shift=args.shift,
            generator=generator,
            output_type="np",
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_mask=neg_mask,
            offload_vae_during_denoise=True,
        )
    gen_time = time.perf_counter() - t1

    frames = result.frames
    if isinstance(frames, list):
        frames = np.stack(frames)

    print(f"  Frames shape: {frames.shape}, dtype: {frames.dtype}", flush=True)

    # Ensure frames are uint8 with 3 channels in THWC format for imageio
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    # Squeeze batch dimension if present: (1, T, H, W, C) -> (T, H, W, C)
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    # Handle NCHW -> NHWC conversion
    if frames.ndim == 4:
        if frames.shape[1] == 3 or frames.shape[1] == 1:
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] > 3:
            frames = frames[..., :3]

    print(f"  Output frames: {frames.shape}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(frames), str(output), fps=args.fps)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Done in {gen_time:.1f}s, output {size_mb:.2f} MiB", flush=True)
    print(f"  Output: {output}", flush=True)


if __name__ == "__main__":
    main()