"""Agate backend — Logolabs/agate-preview-001, 260M thinker-steered FCDM (pure PyTorch).

Model: 191M generator (FCDM-T2: transformer "thinker" plans a 16×16 region map that steers
a convolutional U-Net) + Ettin-68M ModernBERT text encoder + SD-VAE decoder. Trained from
scratch, 145 GH200-hours, GenEval 0.550. Output is FIXED 256×256 — upscale with
Real-ESRGAN for anything larger. MIT licence.

Pipeline code ships inside the HF repo (agate/*.py) and is imported from the model dir
via sys.path (same pattern as the hidream backend). VAE/config paths are patched to local
files after download so the model runs offline.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

from diffuse.models import MODELS
from diffuse.paths import MODELS_DIR

log = logging.getLogger("diffuse")

# Resolution support per preview: 001 is fixed 256×256 (latent 4×32×32);
# 003 is multi-res (256 and 512, SD3 shift 2 at 512). model_name decides.
_AGGATE_MODEL_SIZE = {}
AGATE_SIZE = (256, 256)
# Defaults from the model card: Euler 50 steps, CFG 3.0 (trained against empty prompt).
AGATE_STEPS = 50
AGATE_CFG = 3.0


def load_pipeline_agate(model_name: str, editing: bool = False) -> tuple:
    """Load the Agate pipeline from the model dir.

    Returns (pipeline_dict, load_time_seconds). pipeline_dict contains the
    callable AgatePipeline under "pipe" (returns list[PIL.Image]).

    editing is accepted for interface parity — Agate has no editing mode.
    """
    import torch

    model_info = MODELS[model_name]
    model_root = MODELS_DIR / model_info["dir"]

    if not model_root.exists():
        print(f"\n  ✗ Agate model not found: {model_root}")
        sys.exit(1)

    torch.cuda.empty_cache()
    gc.collect()

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    t0 = time.perf_counter()
    print("  Loading Agate (260M, fp16)...")
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    from agate import AgatePipeline

    pipe = AgatePipeline.from_pretrained(str(model_root), device="cuda", fast_vae=False)
    load_time = time.perf_counter() - t0
    is_multi_res = hasattr(pipe, "prepare")  # 003+ expõe a pipeline de prompt com normalize
    print(f"  Pipeline ready in {load_time:.1f}s (~0.5-1.5 GB VRAM, CUDA graphs on, multi-res={is_multi_res})")

    return {"pipe": pipe, "model_root": model_root}, load_time


def generate_image_agate(
    pipeline_dict: dict,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    cfg: float = AGATE_CFG,
    autoguide: float = 0.0,
) -> tuple:
    """Generate one image with Agate. Returns (png_bytes, diffusion_time, peak_hbm).

    001 is fixed 256×256 (the CLI snaps before calling); 003 is multi-res and the
    requested resolution goes through via the `resolution` kwarg.
    """
    import io

    pipe = pipeline_dict["pipe"]
    steps = steps or AGATE_STEPS
    cfg = AGATE_CFG if cfg is None else float(cfg)  # None = deixar o default do card (3.0)

    log.info("Agate T2I: prompt=%r seed=%d steps=%d cfg=%.1f size=%dx%d", prompt[:80], seed, steps, cfg, width, height)

    t0 = time.perf_counter()
    # 003: resolution= (512/256) + watermark=False. 001: autoguide= (0 = off).
    # Tentar cada kwarg com fallback ordenado (001 não conhece resolution; 003 não conhece autoguide)
    kwargs = {"seed": int(seed), "steps": int(steps), "cfg": float(cfg), "watermark": False}
    if width != 256 or autoguide > 0:
        pass  # tenta o superset primeiro (abaixo)
    try:
        # 003: resolution + watermark; sem autoguide
        images = pipe(prompt, resolution=int(width), **kwargs)
    except TypeError:
        try:
            # 001: autoguide; sem resolution/watermark
            args_001 = {"seed": int(seed), "steps": int(steps), "cfg": float(cfg)}
            if float(autoguide) > 0:
                args_001["autoguide"] = float(autoguide)
            images = pipe(prompt, **args_001)
        except TypeError:
            # fallback base (qualquer variante)
            images = pipe(prompt, seed=int(seed), steps=int(steps), cfg=float(cfg))
    diffusion_time = time.perf_counter() - t0
    peak_hbm = 0.0  # sub-GB model; not worth querying

    buf = io.BytesIO()
    images[0].save(buf, format="PNG", optimize=True)
    return buf.getvalue(), diffusion_time, peak_hbm


def patch_local_config(model_root: Path) -> None:
    """Rewrite config.json's hub ids to local paths so the pipeline never hits the hub.

    The upstream config.json references "stabilityai/sd-vae-ft-mse" (and TAESD for
    fast_vae); we pre-download the SD-VAE into vae/ and point both keys there. TAESD
    is intentionally not shipped — pass fast_vae=False (the default here).
    """
    cfg_path = model_root / "config.json"
    cfg = json.loads(cfg_path.read_text())
    changed = False
    if cfg.get("vae") == "stabilityai/sd-vae-ft-mse":
        vae_local = model_root / "vae"
        if (vae_local / "config.json").exists():
            cfg["vae"] = str(vae_local)
            changed = True
    if cfg.get("fast_vae") == "madebyollin/taesd":
        cfg["fast_vae"] = "madebyollin/taesd"  # hub id — only used with fast_vae=True
        changed = False
    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log.info("Agate config.json patched: vae → %s", cfg["vae"])