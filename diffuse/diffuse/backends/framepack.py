"""FramePack I2V backend — Image-to-Video generation with HunyuanVideo.

Uses FramePack's next-frame prediction with DynamicSwapInstaller for 6GB VRAM.
Requires:
  - lllyasviel/FramePackI2V_HY (transformer, ~24GB)
  - hunyuanvideo-community/HunyuanVideo (text encoders, VAE, tokenizers, ~15GB)
  - lllyasviel/flux_redux_bfl (image encoder, ~0.8GB)
  - diffusers_helper vendored from FramePack repo
"""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import torch

from diffuse.paths import MODELS_DIR, OUTPUTS_DIR

log = logging.getLogger("diffuse.backends.framepack")

# ── Model paths ─────────────────────────────────────────────────────────────
FRAMEPACK_DIR = MODELS_DIR / "framepack-i2v"
HUNYUAN_DIR = MODELS_DIR / "hunyuanvideo-community" / "HunyuanVideo"
FLUX_REDUX_DIR = MODELS_DIR / "flux_redux_bfl"

# ── DynamicSwap helper path ─────────────────────────────────────────────────
# Parent dir that *contains* diffusers_helper as a package
VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "framepack"


def _add_helper_to_path() -> None:
    """Add vendor dir to sys.path so diffusers_helper imports work."""
    import sys
    vendor_dir = str(VENDOR_DIR)
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)


def check_models() -> dict[str, bool]:
    """Check if all required model directories exist."""
    return {
        "transformer": FRAMEPACK_DIR.exists() and any(FRAMEPACK_DIR.glob("*.safetensors")),
        "hunyuan": HUNYUAN_DIR.exists() and (HUNYUAN_DIR / "text_encoder").exists(),
        "image_encoder": FLUX_REDUX_DIR.exists() and (FLUX_REDUX_DIR / "image_encoder").exists(),
    }


def require_models() -> None:
    """Raise if any required models are missing."""
    status = check_models()
    missing = [k for k, v in status.items() if not v]
    if missing:
        missing_dirs = {
            "transformer": str(FRAMEPACK_DIR),
            "hunyuan": str(HUNYUAN_DIR),
            "image_encoder": str(FLUX_REDUX_DIR),
        }
        msg = f"Missing FramePack models: {', '.join(missing)}\n"
        for m in missing:
            msg += f"  {m}: download to {missing_dirs[m]}\n"
        msg += "\nDownload commands:\n"
        msg += f"  hf download lllyasviel/FramePackI2V_HY --local-dir {FRAMEPACK_DIR}\n"
        msg += f"  hf download hunyuanvideo-community/HunyuanVideo --include 'text_encoder/*' 'text_encoder_2/*' 'tokenizer/*' 'tokenizer_2/*' 'vae/*' --local-dir {HUNYUAN_DIR}\n"
        msg += f"  hf download lllyasviel/flux_redux_bfl --include 'image_encoder/*' 'feature_extractor/*' --local-dir {FLUX_REDUX_DIR}\n"
        raise FileNotFoundError(msg)


def load_pipeline(editing: bool = False) -> tuple:
    """Load all FramePack models with DynamicSwap for 6GB VRAM.

    Returns: (pipeline_dict, load_time_seconds)
    pipeline_dict keys: transformer, text_encoder, text_encoder_2, tokenizer,
                         tokenizer_2, vae, feature_extractor, image_encoder
    """
    _add_helper_to_path()

    # Ensure PyTorch uses expandable segments for fragmented VRAM on 6GB cards.
    # Must be 'True'/'False' (not '1'/'0') — PyTorch raises ValueError otherwise.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        LlamaModel,
        LlamaTokenizerFast,
        SiglipImageProcessor,
        SiglipVisionModel,
    )
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.memory import DynamicSwapInstaller, gpu

    require_models()

    t0 = time.perf_counter()

    free_mem_gb = torch.cuda.mem_get_info()[1] / (1024**3) if torch.cuda.is_available() else 0
    high_vram = free_mem_gb > 60
    log.info("Free VRAM: %.1f GB, high_vram=%s", free_mem_gb, high_vram)

    # Load models to CPU first
    text_encoder = LlamaModel.from_pretrained(
        str(HUNYUAN_DIR), subfolder="text_encoder", torch_dtype=torch.float16
    ).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        str(HUNYUAN_DIR), subfolder="text_encoder_2", torch_dtype=torch.float16
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained(
        str(HUNYUAN_DIR), subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        str(HUNYUAN_DIR), subfolder="tokenizer_2"
    )
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        str(HUNYUAN_DIR), subfolder="vae", torch_dtype=torch.float16
    ).cpu()
    feature_extractor = SiglipImageProcessor.from_pretrained(
        str(FLUX_REDUX_DIR), subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        str(FLUX_REDUX_DIR), subfolder="image_encoder", torch_dtype=torch.float16
    ).cpu()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        str(FRAMEPACK_DIR), torch_dtype=torch.bfloat16
    ).cpu()

    # Eval mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # Low VRAM optimizations
    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()
        transformer.high_quality_fp32_output_for_inference = True
        log.info("transformer.high_quality_fp32_output_for_inference = True")

    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    # Freeze all params
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    # DynamicSwap for low VRAM (3x faster than sequential offload)
    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

    load_time = time.perf_counter() - t0
    log.info("FramePack models loaded in %.1fs (high_vram=%s)", load_time, high_vram)

    pipeline_dict = {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "vae": vae,
        "feature_extractor": feature_extractor,
        "image_encoder": image_encoder,
        "high_vram": high_vram,
    }
    return pipeline_dict, load_time


def unload_pipeline() -> None:
    """Force-unload all FramePack models from VRAM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    log.info("FramePack pipeline unloaded from VRAM")


def generate_video_framepack(
    pipeline: dict,
    input_image_path: str,
    prompt: str,
    seed: int = 31337,
    total_second_length: float = 5.0,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 4.5,
    rs: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = True,
    mp4_crf: int = 16,
    output_path: str | None = None,
) -> tuple[str, float]:
    """Generate a video from an input image using FramePack I2V.

    Args:
        pipeline: dict from load_pipeline()
        input_image_path: path to input image
        prompt: text prompt describing motion
        seed: random seed
        total_second_length: video length in seconds (1-120)
        steps: denoising steps (default 25, changing not recommended)
        cfg: CFG scale (default 1.0)
        gs: distilled guidance scale (default 4.5)
        rs: guidance rescale (default 0.0)
        gpu_memory_preservation: GB to preserve on GPU (default 6.0)
        use_teacache: faster but may reduce quality (default True)
        mp4_crf: MP4 quality (default 16)
        output_path: output MP4 path (auto-generated if None)

    Returns:
        (output_path, generation_time_seconds)
    """
    _add_helper_to_path()

    import numpy as np
    import einops
    from PIL import Image

    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
    from diffusers_helper.utils import (
        save_bcthw_as_mp4, resize_and_center_crop, generate_timestamp,
        soft_append_bcthw, crop_or_pad_yield_mask,
    )
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import (
        gpu, get_cuda_free_memory_gb,
        move_model_to_device_with_memory_preservation,
        offload_model_from_device_for_memory_preservation,
        fake_diffusers_current_device,
        DynamicSwapInstaller,
        unload_complete_models,
        load_model_as_complete,
    )
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket

    transformer = pipeline["transformer"]
    text_encoder = pipeline["text_encoder"]
    text_encoder_2 = pipeline["text_encoder_2"]
    tokenizer = pipeline["tokenizer"]
    tokenizer_2 = pipeline["tokenizer_2"]
    vae = pipeline["vae"]
    feature_extractor = pipeline["feature_extractor"]
    image_encoder = pipeline["image_encoder"]
    high_vram = pipeline["high_vram"]
    latent_window_size = 9  # fixed for FramePack I2V

    n_prompt = ""  # negative prompt (not used with cfg=1)

    # Load input image
    input_image_pil = Image.open(input_image_path).convert("RGB")
    input_image_np = np.array(input_image_pil)

    H, W, C = input_image_np.shape
    height, width = find_nearest_bucket(H, W, resolution=640)
    input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

    input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
    input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

    total_latent_sections = int(max(round(total_second_length * 30 / (latent_window_size * 4)), 1))
    log.info("FramePack I2V: %dx%d, %.1fs, %d sections, %d steps, seed=%d",
             width, height, total_second_length, total_latent_sections, steps, seed)

    t_start = time.perf_counter()

    # ── Text encoding ──────────────────────────────────────────────────────
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        fake_diffusers_current_device(text_encoder, gpu)

    load_model_as_complete(text_encoder_2, target_device=gpu)
    llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    if cfg == 1:
        llama_vec_n = torch.zeros_like(llama_vec)
        clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    # ── VAE encode input image ──────────────────────────────────────────────
    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)
    start_latent = vae_encode(input_image_pt, vae)

    # ── CLIP Vision encode ──────────────────────────────────────────────────
    if not high_vram:
        load_model_as_complete(image_encoder, target_device=gpu)
    image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
    image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

    # ── Dtype alignment ─────────────────────────────────────────────────────
    llama_vec = llama_vec.to(transformer.dtype)
    llama_vec_n = llama_vec_n.to(transformer.dtype)
    clip_l_pooler = clip_l_pooler.to(transformer.dtype)
    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

    # ── Sampling loop ────────────────────────────────────────────────────────
    rnd = torch.Generator("cpu").manual_seed(seed)
    num_frames = latent_window_size * 4 - 3

    history_latents = torch.zeros(
        size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
        dtype=torch.float32,
    ).cpu()
    history_pixels = None
    total_generated_latent_frames = 0

    # Latent padding sequence
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
    else:
        latent_paddings = list(reversed(range(total_latent_sections)))

    last_output_path = None

    for section_idx, latent_padding in enumerate(latent_paddings):
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * latent_window_size

        log.info("Section %d/%d: latent_padding=%d, is_last=%s",
                 section_idx + 1, len(latent_paddings), latent_padding, is_last_section)

        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        (
            clean_latent_indices_pre,
            blank_indices,
            latent_indices,
            clean_latent_indices_post,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
        ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split(
            [1, 2, 16], dim=2
        )
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        if not high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(
                transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        if use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        else:
            transformer.initialize_teacache(enable_teacache=False)

        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler="unipc",
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )

        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

        # ── VAE decode ──────────────────────────────────────────────────────
        if not high_vram:
            offload_model_from_device_for_memory_preservation(
                transformer, target_device=gpu, preserved_memory_gb=8
            )
            load_model_as_complete(vae, target_device=gpu)

        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

        if history_pixels is None:
            history_pixels = vae_decode(real_history_latents, vae).cpu()
        else:
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
            overlapped_frames = latent_window_size * 4 - 3
            current_pixels = vae_decode(
                real_history_latents[:, :, :section_latent_frames], vae
            ).cpu()
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

        if not high_vram:
            unload_complete_models()

        # Save intermediate result
        ts = generate_timestamp()
        if output_path:
            out_file = output_path
        else:
            out_dir = OUTPUTS_DIR / "framepack-i2v"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = str(out_dir / f"framepack_{ts}_{total_generated_latent_frames}.mp4")

        save_bcthw_as_mp4(history_pixels, out_file, fps=30, crf=mp4_crf)
        log.info("Saved section %d: %s (%d frames, %.1fs)",
                 section_idx + 1, out_file, total_generated_latent_frames * 4 - 3,
                 max(0, (total_generated_latent_frames * 4 - 3) / 30))
        last_output_path = out_file

        if is_last_section:
            break

    gen_time = time.perf_counter() - t_start

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    return last_output_path or "", gen_time