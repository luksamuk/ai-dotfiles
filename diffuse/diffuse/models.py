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
#   hf_files: list of {repo, files, subdir?, rename?, hf_path?} for auto-download
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
}