# diffuse — Local Diffusion Image Generation CLI

Diffusion generation CLI for NVIDIA GPUs. Supports image generation (HiDream, Ideogram 4) with automatic VRAM management.

> **Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

## Fleet

| Model | Type | Backend | Components | Size |
|-------|------|---------|-----------|------|
| **hidream-sdnq** | T2I + editing | transformers + accelerate | Qwen3-VL 8B SDNQ (unified) | 7.3 GB |
| **ideogram4-q4** | T2I (text rendering) | sd-cli (GGUF) | DiT + Qwen3-VL 8B GGUF + FLUX2 VAE | 14.9 GB |

## Quick Start

```bash
# List all models, dependencies, and sizes
diffuse list

# Download a model
diffuse download hidream
diffuse download ideogram4

# Generate an image
diffuse generate -m hidream-sdnq -p "a cat on the moon"
diffuse generate -m ideogram4-q4 -p "text: HELLO WORLD" --enhance

# Enhance prompt via LLM before generation
diffuse generate -m hidream-sdnq -p "cyberpunk city" --enhance

# Free VRAM by evicting LLM models from llama-swap
diffuse evict
```

## Architecture

```
Prompt (text)
    │
    ▼
┌──────────────────────┐
│  Text Encoder        │  Encodes prompt → embeddings (offloaded after use)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Diffusion Transformer│  N denoising steps guided by text embeddings
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  VAE                  │  Decodes latent → pixel image
└──────────┬───────────┘
           │
           ▼
      Image (PNG)
```

**Memory lifecycle on 6GB VRAM:**
1. Text encoder loads, encodes prompt, unloads from VRAM
2. Diffusion transformer runs denoising steps
3. VAE decodes the result
4. LLM models (llama-swap) are evicted before loading to free VRAM

## File Structure

```
diffuse/
├── README.md
├── run.sh                  # Entry point (symlinked as `diffuse`)
├── download-model.sh       # Model download script
├── pyproject.toml
├── models/                 # Model weights (not in git)
│   └── ideogram-4-Q4_0/    # Ideogram 4 (when downloaded)
├── diffuse/                # Python package
│   ├── cli.py              # CLI — argument parsing, orchestration
│   ├── models.py           # Model registry
│   ├── paths.py            # Path constants
│   ├── backends/           # Backend modules (one per pipeline type)
│   │   ├── __init__.py     # Backend dispatch + auto-download
│   │   ├── hidream.py      # HiDream (transformers + accelerate)
│   │   ├── sd_cpp.py       # Ideogram 4 (stable-diffusion.cpp)
│   │   ├── gemlite.py     # Bonsai (legacy, code preserved)
│   │   └── framepack.py    # FramePack (disabled, code preserved)
│   ├── enhance.py          # LLM-based prompt enhancement
│   ├── llm.py              # llama-swap integration
│   └── output.py          # Output path resolution + debrief
├── vendor/                 # Vendored dependencies (pinned SHAs)
└── outputs/                 # Generated images
```

## Model Details

### HiDream-O1 SDNQ (`hidream-sdnq`)
- **Backend:** transformers + accelerate (CPU offload)
- **Components:** Qwen3-VL 8B SDNQ (unified DiT + text encoder, 7.3 GB)
- **Capabilities:** T2I, instruction-based image editing (`--edit`)
- **Default size:** 1024×1024 (snaps to 2048² or 2560×1440)
- **Speed:** ~3min T2I, ~8min editing on 6GB VRAM

### Ideogram 4 Q4 (`ideogram4-q4`)
- **Backend:** sd-cli (stable-diffusion.cpp, GGUF)
- **Components:** DiT Q4_0 (5.0 GB) + uncond DiT (5.0 GB) + Qwen3-VL 8B GGUF (4.7 GB) + FLUX2 VAE (0.2 GB)
- **Capabilities:** T2I, best-in-class text rendering
- **Note:** VAE source (FLUX.2-dev) is a gated repo — run `hf auth login` and accept license first
- **Default size:** 480×480

## License

Models: respective licenses (Apache 2.0 for HiDream; see HuggingFace pages)
This CLI tool: MIT