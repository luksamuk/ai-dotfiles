"""HiDream backend — HiDream-O1-Image-Dev SDNQ with transformers + accelerate CPU offload."""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

from diffuse.models import MODELS

log = logging.getLogger("diffuse")

# Default repo path (overridden by model_info["hidream_repo"] if present)
HIDREAM_REPO = Path(os.path.expanduser("~/git/HiDream-O1-Image"))


def load_pipeline_hidream(model_name: str, editing: bool = False) -> tuple:
    """Load HiDream-O1-Image-Dev SDNQ model with accelerate CPU offload.

    Returns (pipeline_dict, load_time_seconds).
    pipeline_dict contains model, processor, tokenizer — not a callable pipeline.

    If editing=True, reserves more VRAM headroom for reference image encoding.
    """
    import torch
    from accelerate import infer_auto_device_map, dispatch_model

    # Disable cuDNN to avoid CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH crash
    # with Conv3d + accelerate CPU offload on RTX 3050
    torch.backends.cudnn.enabled = False

    model_info = MODELS[model_name]
    model_root = Path(os.path.expanduser(f"~/.llama-models/{model_info['dir']}"))
    hidream_repo = Path(os.path.expanduser(model_info["hidream_repo"]))

    # Model path already validated in main() — skip duplicate check here
    if not hidream_repo.exists():
        print(f"\n  ✗ HiDream repo not found: {hidream_repo}")
        print(f"    Clone from: https://github.com/HiDream-ai/HiDream-O1-Image")
        sys.exit(1)

    # Add repo to sys.path so we can import sdnq, models, inference
    if str(hidream_repo) not in sys.path:
        sys.path.insert(0, str(hidream_repo))

    import sdnq
    from transformers import AutoProcessor
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    from inference import add_special_tokens, get_tokenizer

    torch.cuda.empty_cache()
    gc.collect()

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    t0 = time.perf_counter()
    print("  Loading SDNQ model on CPU...")

    processor = AutoProcessor.from_pretrained(str(model_root))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_root),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()

    load_time = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded on CPU in {load_time:.1f}s ({param_count/1e9:.1f}B params)")

    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    # Dispatch model across GPU/CPU with memory limit.
    # Editing mode (ref_image) needs ~1 GB more VRAM for reference image encoding.
    # T2I works fine with 1.8 GB headroom; editing needs 3.0+ GB.
    headroom_gb = 3.0 if editing else 1.8
    max_vram_mb = int((total / 1e9 - headroom_gb) * 1024)
    print(f"  Mode: {'editing' if editing else 'T2I'} — reserving {headroom_gb:.1f} GB for activations")
    print(f"  Max GPU for model layers: {max_vram_mb} MiB")

    device_map = infer_auto_device_map(
        model,
        max_memory={0: f"{max_vram_mb}MiB", "cpu": "24GiB"},
        no_split_module_classes=["Qwen3VLDecoderLayer"],
    )

    gpu_layers = len([v for v in device_map.values() if v == 0 or v == torch.device(0)])
    cpu_layers = len([v for v in device_map.values() if v == "cpu" or v == torch.device("cpu")])
    print(f"  Device map: {gpu_layers} layers on GPU, {cpu_layers} on CPU")

    model = dispatch_model(model, device_map=device_map)

    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM after dispatch: {free/1e9:.1f} GB free")

    return {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "hidream_repo": hidream_repo,
    }, load_time


def generate_image_hidream(
    pipeline_dict: dict,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None = None,
) -> tuple:
    """Generate an image using HiDream pipeline. Returns (output_path, diffusion_time, peak_hbm).

    If ref_image_paths is provided, performs image editing (instruction-based).
    """
    import torch
    from models.pipeline import generate_image

    model = pipeline_dict["model"]
    processor = pipeline_dict["processor"]
    is_editing = ref_image_paths is not None and len(ref_image_paths) > 0

    # Dev model defaults: 28 steps, guidance_scale=0, shift=1.0
    # Editing uses flash scheduler (flow_match causes cuDNN Conv3d crash with CPU offload on RTX 3050)
    num_inference_steps = steps or 28
    guidance_scale = 0.0
    shift = 1.0
    scheduler_name = "flash"
    noise_scale_start = 7.5
    noise_scale_end = 7.5
    noise_clip_std = 2.5
    extra_kwargs = {
        "noise_scale_start": noise_scale_start,
        "noise_scale_end": noise_scale_end,
        "noise_clip_std": noise_clip_std,
    }

    mode_str = "editing" if is_editing else "T2I"
    log.info("HiDream %s: prompt=%r seed=%d steps=%d size=%dx%d",
             mode_str, prompt[:80], seed, num_inference_steps, width, height)

    t0 = time.perf_counter()

    image = generate_image(
        model=model,
        processor=processor,
        prompt=prompt,
        ref_image_paths=ref_image_paths,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        scheduler_name=scheduler_name,
        seed=seed,
        **extra_kwargs,
    )

    diffusion_time = time.perf_counter() - t0
    peak_hbm = 0.0  # HiDream doesn't expose peak HBM easily

    # Save to temp PNG bytes
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    return png_bytes, diffusion_time, peak_hbm