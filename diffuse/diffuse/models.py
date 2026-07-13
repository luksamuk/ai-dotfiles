"""Model registry — all supported models and their configurations."""
from __future__ import annotations

# ── Shared component registry ──────────────────────────────────────────────
# Components that can be reused across models. Keyed by a shared_id;
# each entry specifies the canonical path under models/shared/.
SHARED_COMPONENTS: dict[str, dict] = {
    "wan-vae": {
        "path": "shared/vae/wan",
        "description": "AutoencoderKLWan (Wan 2.1 VAE)",
        "size_gb": 0.25,
    },
}

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry defines:
#   backend_type: which backend module handles generation
#   bits: quantization description
#   dir: model directory under MODELS_DIR (or ~/.llama-models/ for hidream)
#   description: human-readable description
#   enhance_model: default LLM for prompt enhancement
#   enhance_type: "ideogram" (JSON) or "vision" (natural language)
#   default_size: (width, height) tuple, optional
#   hf_files: list of {repo, files, subdir?, rename?, hf_path?} for auto-download
#   shared: list of shared component IDs this model uses
#   components: list of {name, path, size_gb} for diffuse list display
#   category: "image" or "video"

MODELS: dict[str, dict] = {
    # ── Image generation ────────────────────────────────────────────────────
    # HiDream-O1-Image-Dev SDNQ (transformers + accelerate CPU offload)
    "hidream-sdnq": {
        "dir": "HiDream-O1-Image-Dev-SDNQ-last8",
        "backend_type": "hidream",
        "category": "image",
        "bits": "4-bit SDNQ (uint4+svd-r32 last8-odown-bf16)",
        "description": "HiDream-O1-Image-Dev SDNQ — 8B unified (T2I + editing + IP), ~3min/2048² on 6GB VRAM",
        "default_size": (1024, 1024),
        "hidream_repo": "~/git/HiDream-O1-Image",
        "enhance_model": "qwen3.6-35b-a3b",
        "enhance_type": "vision",
        "components": [
            {"name": "Qwen3-VL 8B SDNQ (unified DiT + text encoder)", "path": "~/.llama-models/HiDream-O1-Image-Dev-SDNQ-last8", "size_gb": 7.3},
        ],
    },
    # Ideogram 4 (sd-cli / stable-diffusion.cpp)
    # Models auto-download from HuggingFace on first use (no pre-download needed).
    "ideogram4-q4": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "category": "image",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
        "enhance_model": "qwen3.6-35b-a3b",
        "enhance_type": "ideogram",
        "default_size": (480, 480),
        "hf_files": [
            {"repo": "leejet/ideogram-4-GGUF", "files": ["ideogram4-Q4_0.gguf", "ideogram4_uncond-Q4_0.gguf"]},
            {"repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF", "files": ["Qwen3-VL-8B-Instruct-Q4_K_M.gguf"],
             "rename": {"Qwen3-VL-8B-Instruct-Q4_K_M.gguf": "Qwen3VL-8B-Instruct-Q4_K_M.gguf"}},
            {"repo": "black-forest-labs/FLUX.2-dev", "files": ["ae.safetensors"], "subdir": "vae",
             "rename": {"ae.safetensors": "flux2-vae.safetensors"}},
        ],
        "components": [
            {"name": "ideogram4-Q4_0.gguf (DiT)", "path": "ideogram-4-Q4_0/ideogram4-Q4_0.gguf", "size_gb": 5.0},
            {"name": "ideogram4_uncond-Q4_0.gguf (uncond DiT)", "path": "ideogram-4-Q4_0/ideogram4_uncond-Q4_0.gguf", "size_gb": 5.0},
            {"name": "Qwen3VL-8B-Instruct-Q4_K_M.gguf (text encoder)", "path": "ideogram-4-Q4_0/Qwen3VL-8B-Instruct-Q4_K_M.gguf", "size_gb": 4.7},
            {"name": "flux2-vae.safetensors (VAE)", "path": "ideogram-4-Q4_0/vae/flux2-vae.safetensors", "size_gb": 0.2},
        ],
    },
    # ── Video generation ────────────────────────────────────────────────────
    # Wan2.2 I2V (sd-cli / stable-diffusion.cpp)
    # AllInOne GGUF merges low-noise + high-noise into one file.
    # Fine-tune "Rapid" uses 3 accelerators (lightx2v + WAN2.2 Lightning + rCM) → 4 steps, 1 CFG.
    # NSFW fine-tune removes safety filters from base Wan2.2 I2V.
    "wan22-i2v": {
        "dir": "wan22-i2v",
        "backend_type": "sd_cpp_video",
        "category": "video",
        "bits": "4-bit (Q4_K_S)",
        "gguf_file": "wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf",
        "description": "Wan2.2 I2V A14B Rapid AllInOne — image-to-video, 4-step accelerator, no content filters",
        "default_size": (832, 480),
        "default_video_frames": 33,
        "default_fps": 24,
        "default_steps": 4,
        "default_cfg": 1.0,
        "default_flow_shift": 3.0,
        "enhance_model": "qwen3.6-35b-a3b",
        "enhance_type": "vision",
        "shared": ["wan-vae"],
        "hf_files": [
            {"repo": "desirel/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10",
             "files": ["wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf"]},
            {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
             "files": ["wan_2.1_vae.safetensors"], "subdir": "vae",
             "hf_path": "split_files/vae/wan_2.1_vae.safetensors",
             "shared_id": "wan-vae"},
            {"repo": "city96/umt5-xxl-encoder-gguf",
             "files": ["umt5-xxl-encoder-Q8_0.gguf"], "subdir": "text_encoder"},
            {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
             "files": ["clip_vision_h.safetensors"], "subdir": "clip_vision",
             "hf_path": "split_files/clip_vision/clip_vision_h.safetensors"},
        ],
        "components": [
            {"name": "wan2.2-i2v-rapid-aio Q4_K_S (DiT)", "path": "wan22-i2v/wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf", "size_gb": 9.3},
            {"name": "umt5-xxl-encoder Q8_0 (text encoder)", "path": "wan22-i2v/text_encoder/umt5-xxl-encoder-Q8_0.gguf", "size_gb": 5.7},
            {"name": "clip_vision_h (vision encoder)", "path": "wan22-i2v/clip_vision/clip_vision_h.gguf", "size_gb": 1.2},
            {"name": "wan_2.1_vae [shared]", "path": "shared/vae/wan", "size_gb": 0.25},
        ],
    },
    # LingBot-Video Dense 1.3B (transformers/diffusers — custom pipeline)
    # T2V + TI2V (text-to-video and text+image-to-video).
    # DiT 2.79GB bf16 fits in 6GB VRAM; text encoder (Qwen3-VL 4B bf16, 8.88GB) runs on CPU.
    # Shares Wan VAE with wan22-i2v.
    "lingbot-t2v": {
        "dir": "lingbot-t2v",
        "backend_type": "lingbot",
        "category": "video",
        "bits": "bf16",
        "description": "LingBot-Video Dense 1.3B — text-to-video + text+image-to-video, custom pipeline, 4-step FlowUniPC",
        "default_size": (832, 480),
        "default_video_frames": 33,
        "default_fps": 24,
        "default_steps": 4,
        "default_cfg": 1.0,
        "enhance_model": "qwen3.6-35b-a3b",
        "enhance_type": "vision",
        "shared": ["wan-vae"],
        "lingbot_repo": "~/git/lingbot-video",
        "hf_files": [
            {"repo": "robbyant/lingbot-video-dense-1.3b",
             "files": ["model_index.json",
                       "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
                       "text_encoder/config.json", "text_encoder/configuration.json",
                       "text_encoder/generation_config.json", "text_encoder/model-00001-of-00002.safetensors",
                       "text_encoder/model-00002-of-00002.safetensors", "text_encoder/model.safetensors.index.json",
                       "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json",
                       "text_encoder/vocab.json", "text_encoder/merges.txt",
                       "text_encoder/chat_template.json", "text_encoder/preprocessor_config.json",
                       "text_encoder/video_preprocessor_config.json",
                       "processor/config.json", "processor/configuration.json",
                       "processor/generation_config.json", "processor/tokenizer.json",
                       "processor/tokenizer_config.json", "processor/vocab.json",
                       "processor/merges.txt", "processor/chat_template.json",
                       "processor/preprocessor_config.json", "processor/video_preprocessor_config.json",
                       "scheduler/scheduler_config.json", "scheduling_flow_unipc.py"]},
            # VAE is shared with wan22-i2v — symlink, not re-download
        ],
        "components": [
            {"name": "LingBot 1.3B DiT bf16 (transformer)", "path": "lingbot-t2v/transformer/diffusion_pytorch_model.safetensors", "size_gb": 2.79},
            {"name": "Qwen3-VL 4B bf16 (text encoder, 2 shards)", "path": "lingbot-t2v/text_encoder", "size_gb": 8.88},
            {"name": "wan_2.1_vae [shared]", "path": "shared/vae/wan", "size_gb": 0.25},
        ],
    },
}