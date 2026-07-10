"""Model registry — all supported models and their configurations."""
from __future__ import annotations

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry defines:
#   backend_type: which backend module handles generation
#   bits: quantization description
#   dir: model directory under MODELS_DIR (or ~/.llama-models/ for hidream)
#   description: human-readable description
#   enhance_model: default LLM for prompt enhancement
#   enhance_type: "ideogram" (JSON) or "vision" (natural language)
#   default_size: (width, height) tuple, optional
#   backend_id: gemlite pipeline backend ID (gemlite only)
#   hf_repo: huggingface repo for download (optional)
#   transformer_kwarg: gemlite kwarg name (gemlite only)
#   hidream_repo: path to HiDream repo (hidream only)

MODELS: dict[str, dict] = {
    # Bonsai Image 4B (gemlite CUDA)
    "binary-gemlite": {
        "backend_id": "bonsai-binary-gemlite",
        "hf_repo": "prism-ml/bonsai-image-binary-4B-gemlite-1bit",
        "dir": "bonsai-image-4B-binary-gemlite",
        "backend_type": "gemlite",
        "bits": "1-bit",
        "transformer_kwarg": "binary_transformer_path",
        "description": "1-bit {−1, +1} — 0.93 GB transformer, 88% of FP16 quality",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
    },
    "ternary-gemlite": {
        "backend_id": "bonsai-ternary-gemlite",
        "hf_repo": "prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
        "dir": "bonsai-image-4B-ternary-gemlite",
        "backend_type": "gemlite",
        "bits": "1.58-bit",
        "transformer_kwarg": "ternary_transformer_path",
        "description": "1.58-bit {−1, 0, +1} — 1.21 GB transformer, 95% of FP16 quality",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
    },
    # Ideogram 4 (sd-cli / stable-diffusion.cpp)
    # Models auto-download from HuggingFace on first use (no pre-download needed).
    "ideogram4-q4": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "ideogram",
        "default_size": (480, 480),
        # Auto-download: files fetched from HuggingFace on first use via require_model_dir()
        # VAE source: black-forest-labs/FLUX.2-dev is a gated repo — user must run
        #   hf auth login && hf download black-forest-labs/FLUX.2-dev
        # once to accept the license. After that, auto-download works.
        "hf_files": [
            {"repo": "leejet/ideogram-4-GGUF", "files": ["ideogram4-Q4_0.gguf", "ideogram4_uncond-Q4_0.gguf"]},
            {"repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF", "files": ["Qwen3-VL-8B-Instruct-Q4_K_M.gguf"],
             "rename": {"Qwen3-VL-8B-Instruct-Q4_K_M.gguf": "Qwen3VL-8B-Instruct-Q4_K_M.gguf"}},
            {"repo": "black-forest-labs/FLUX.2-dev", "files": ["ae.safetensors"], "subdir": "vae",
             "rename": {"ae.safetensors": "flux2-vae.safetensors"}},
        ],
    },
    # HiDream-O1-Image-Dev SDNQ (transformers + accelerate CPU offload)
    "hidream-sdnq": {
        "dir": "HiDream-O1-Image-Dev-SDNQ-last8",
        "backend_type": "hidream",
        "bits": "4-bit SDNQ (uint4+svd-r32 last8-odown-bf16)",
        "description": "HiDream-O1-Image-Dev SDNQ — 8B unified (T2I + editing + IP), ~3min/2048² on 6GB VRAM",
        "default_size": (1024, 1024),
        "hidream_repo": "~/git/HiDream-O1-Image",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
    },
    # FramePack I2V — disabled: models removed (~41GB, 12min/1s video on RTX 3050)
    # To re-enable: download models + uncomment this entry
    # "framepack-i2v": {
    #     "dir": "framepack-i2v",
    #     "backend_type": "framepack",
    #     "bits": "bf16 transformer (DynamicSwap on 6GB VRAM)",
    #     "description": "FramePack I2V — image-to-video with HunyuanVideo, progressive next-frame prediction",
    #     "default_size": (640, 640),
    #     "enhance_model": "qwen3.5-4b",
    #     "enhance_type": "vision",
    #     "default_seconds": 5.0,
    #     "default_steps": 25,
    # },
    # Wan2.2 I2V (sd-cli / stable-diffusion.cpp)
    # AllInOne GGUF merges low-noise + high-noise into one file.
    # Fine-tune "Rapid" uses 3 accelerators (lightx2v + WAN2.2 Lightning + rCM) → 4 steps, 1 CFG.
    # NSFW fine-tune removes safety filters from base Wan2.2 I2V.
    # Extras: VAE (wan_2.1_vae), UMT5-XXL text encoder (Q8_0), clip_vision_h (for I2V conditioning).
    "wan22-i2v": {
        "dir": "wan22-i2v",
        "backend_type": "sd_cpp_video",
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
        "hf_files": [
            {"repo": "desirel/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10",
             "files": ["wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf"]},
            {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
             "files": ["wan_2.1_vae.safetensors"], "subdir": "vae",
             "hf_path": "split_files/vae/wan_2.1_vae.safetensors"},
            {"repo": "city96/umt5-xxl-encoder-gguf",
             "files": ["umt5-xxl-encoder-Q8_0.gguf"], "subdir": "text_encoder"},
            {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
             "files": ["clip_vision_h.safetensors"], "subdir": "clip_vision",
             "hf_path": "split_files/clip_vision/clip_vision_h.safetensors"},
        ],
    },
}