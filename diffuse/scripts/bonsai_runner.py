#!/usr/bin/env python3
"""Bonsai image generation runner — called as subprocess by diffuse.

Uses the bonsai-venv which has diffusers 0.39 + Flux2Pipeline support.
The main diffuse venv has diffusers 0.33 (for HiDream) which lacks Flux2Pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add vendored image-studio to sys.path for backend_gpu imports
VENDOR_STUDIO = Path(__file__).resolve().parent.parent / "vendor" / "image-studio"
if str(VENDOR_STUDIO) not in sys.path:
    sys.path.insert(0, str(VENDOR_STUDIO))

GEMLITE_CACHE = Path(__file__).resolve().parent.parent / "outputs" / ".gemlite_cache" / "autotune.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tiled-vae", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from backend_gpu.pipeline_gpu import GpuPipeline
    from backend_gpu.diffusion_klein import _encode_klein_qwen3_prompt
    from gemlite.core import GemLiteLinearTriton

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total", flush=True)

    t0 = time.perf_counter()
    print("  Loading Bonsai pipeline...", flush=True)

    model_root = Path(args.model_dir)

    def _find_subdir(root, *hints):
        matches = [p for p in root.iterdir() if p.is_dir() and any(h in p.name for h in hints)]
        if not matches:
            raise FileNotFoundError(f"No subdir matching {hints} under {root}")
        matches.sort(key=lambda p: len(p.name), reverse=True)
        return matches[0]

    text_encoder_dir = _find_subdir(model_root, "text_encoder")
    transformer_dir = _find_subdir(model_root, "transformer")
    vae_dir = _find_subdir(model_root, "vae")

    pipeline = GpuPipeline(
        backend="bonsai-ternary-gemlite",
        binary_transformer_path=str(transformer_dir),
        ternary_transformer_path=str(transformer_dir),
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(vae_dir),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )

    if GEMLITE_CACHE.exists():
        GemLiteLinearTriton.load_config(str(GEMLITE_CACHE), print_error=False)

    pipeline.prewarm()
    load_time = time.perf_counter() - t0
    GemLiteLinearTriton.cache_config(str(GEMLITE_CACHE))
    print(f"  Pipeline loaded in {load_time:.1f}s", flush=True)

    # Enable VAE tiling for large resolutions
    if args.tiled_vae:
        pipeline._vae.enable_tiling()
        print("  VAE tiling enabled", flush=True)

    free, _ = torch.cuda.mem_get_info()
    print(f"  VRAM after load: {free/1e9:.1f} GB free", flush=True)

    # Encode prompt on GPU, then offload text encoder to CPU
    text_encoder = pipeline._text_encoder
    tokenizer = pipeline._tokenizer

    print("  Encoding prompt...", flush=True)
    t_enc = time.perf_counter()
    prompt_embeds = _encode_klein_qwen3_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_sequence_length=512,
    )
    print(f"  Prompt encoded in {time.perf_counter() - t_enc:.2f}s", flush=True)

    # Offload text encoder to CPU — frees ~2.8 GB VRAM
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    free, _ = torch.cuda.mem_get_info()
    print(f"  VRAM after text encoder offload: {free/1e9:.1f} GB free", flush=True)

    print(f"  Generating: {args.width}x{args.height}, {args.steps} steps, seed={args.seed}", flush=True)

    t1 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt="",
        seed=args.seed,
        steps=args.steps,
        height=args.height,
        width=args.width,
        guidance=args.guidance,
        precomputed_prompt_embeds=prompt_embeds,
    )
    gen_time = time.perf_counter() - t1

    peak_hbm = pipeline.last_peak_memory_mb or 0.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png_bytes)
    size_mb = output.stat().st_size / (1024 * 1024)

    print(f"  Done in {gen_time:.1f}s, output {size_mb:.2f} MiB, peak VRAM {peak_hbm:.0f} MiB", flush=True)
    print(f"  Output: {output}", flush=True)


if __name__ == "__main__":
    main()