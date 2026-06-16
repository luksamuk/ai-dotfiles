"""Gemlite backend — Bonsai Image 4B with HQQ kernels on CUDA."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from diffuse.paths import GEMLITE_PERSIST_PATH
from diffuse.models import MODELS
from diffuse.backends import require_model_dir, _find_subdir

log = logging.getLogger("diffuse")


def load_pipeline_gemlite(model_name: str) -> tuple:
    """Load a gemlite-backed pipeline. Returns (pipeline, load_time_seconds)."""
    from backend_gpu.pipeline_gpu import GpuPipeline
    from gemlite.core import GemLiteLinearTriton

    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    text_encoder_dir = _find_subdir(model_root, "text_encoder")
    transformer_dir = _find_subdir(model_root, "transformer")

    t0 = time.perf_counter()

    pipeline = GpuPipeline(
        backend=model_info["backend_id"],
        binary_transformer_path=str(transformer_dir),
        ternary_transformer_path=str(transformer_dir),
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(_find_subdir(model_root, "vae")),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )

    if GEMLITE_PERSIST_PATH.exists():
        GemLiteLinearTriton.load_config(str(GEMLITE_PERSIST_PATH), print_error=False)

    pipeline.prewarm()
    load_time = time.perf_counter() - t0

    GemLiteLinearTriton.cache_config(str(GEMLITE_PERSIST_PATH))

    return pipeline, load_time


def generate_image_gemlite(pipeline, prompt: str, seed: int, steps: int, width: int, height: int) -> tuple:
    """Generate a PNG image using gemlite pipeline with text-encoder offload.

    Text encoder (2.84 GB) is moved to CPU after encoding the prompt, freeing
    VRAM for the diffusion loop and VAE decode. This allows larger resolutions
    without OOMing on 6 GB cards.

    Returns (png_bytes, diffusion_time, peak_hbm_mb).
    """
    import torch
    from backend_gpu.diffusion_klein import _encode_klein_qwen3_prompt

    log.info("Generating: prompt=%r seed=%d steps=%d size=%dx%d", prompt, seed, steps, width, height)

    text_encoder = pipeline._text_encoder
    tokenizer = pipeline._tokenizer

    # 1. Encode prompt (text encoder on GPU)
    log.info("Encoding prompt (text encoder on GPU)...")
    t_enc = time.perf_counter()
    prompt_embeds = _encode_klein_qwen3_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
        max_sequence_length=512,
    )
    log.info("Prompt encoded in %.2fs", time.perf_counter() - t_enc)

    # 2. Offload text encoder to CPU — frees ~2.8 GB VRAM
    if torch.cuda.is_available():
        log.info("Offloading text encoder to CPU...")
        text_encoder.to("cpu")
        torch.cuda.empty_cache()
        log.info("Text encoder offloaded to CPU (~2.8 GB VRAM freed)")

    # 3. Diffusion + VAE decode (only transformer + VAE on GPU)
    t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt="",  # Ignored — we pass precomputed embeds
        seed=seed,
        steps=steps,
        height=height,
        width=width,
        precomputed_prompt_embeds=prompt_embeds,
    )
    diffusion_time = time.perf_counter() - t0

    # 4. Move text encoder back to GPU for next call
    if torch.cuda.is_available():
        text_encoder.to(pipeline.device)
        log.info("Text encoder restored to GPU")

    peak_hbm = pipeline.last_peak_memory_mb or 0.0
    log.info("Diffusion done in %.2fs (peak HBM %.1f MiB)", diffusion_time, peak_hbm)

    return png_bytes, diffusion_time, peak_hbm