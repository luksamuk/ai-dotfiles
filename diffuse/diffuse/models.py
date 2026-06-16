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
    "ideogram4-q4": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "ideogram",
        "default_size": (480, 480),
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
    # FramePack I2V (HunyuanVideo-based image-to-video)
    "framepack-i2v": {
        "dir": "framepack-i2v",
        "backend_type": "framepack",
        "bits": "bf16 transformer (DynamicSwap on 6GB VRAM)",
        "description": "FramePack I2V — image-to-video with HunyuanVideo, progressive next-frame prediction, ~5s video in 3-8min on 6GB",
        "default_size": (640, 640),
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
        "default_seconds": 5.0,
        "default_steps": 25,
    },
}