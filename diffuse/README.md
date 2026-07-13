# diffuse — Local Diffusion Image & Video Generation CLI

Diffusion generation CLI for NVIDIA GPUs. Supports image generation (HiDream, Ideogram 4) and video generation (Wan2.2 I2V, LingBot T2V) with shared components and automatic VRAM management.

> **Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

## Fleet

| Model | Type | Backend | Components | Size |
|-------|------|---------|-----------|------|
| **hidream-sdnq** | T2I + editing | transformers + accelerate | Qwen3-VL 8B SDNQ (unified) | 7.3 GB |
| **ideogram4-q4** | T2I (text rendering) | sd-cli (GGUF) | DiT + Qwen3-VL 8B GGUF + FLUX2 VAE | 14.9 GB |
| **wan22-i2v** | I2V (image-to-video) | sd-cli (GGUF) | Wan2.2 A14B DiT + UMT5-XXL + CLIP Vision + Wan VAE | 16.4 GB |
| **lingbot-t2v** | T2V (text-to-video) | diffusers (custom pipeline) | LingBot 1.3B DiT + Qwen3-VL 4B + Wan VAE | 11.9 GB |

**Shared component:** Wan VAE (AutoencoderKLWan, 256 MB) — used by both wan22-i2v and lingbot-t2v.

## Quick Start

```bash
# List all models, dependencies, and sizes
diffuse list

# Download a model
diffuse download hidream
diffuse download lingbot

# Generate an image
diffuse generate -m hidream-sdnq -p "a cat on the moon"
diffuse generate -m ideogram4-q4 -p "text: HELLO WORLD" --enhance

# Generate a video
diffuse generate -m wan22-i2v --input-image photo.png -p "camera pan right"
diffuse generate -m lingbot-t2v -p "a cat playing with yarn"

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
│  VAE                  │  Decodes latent → pixel image/video frames
└──────────┬───────────┘
           │
           ▼
      Image (PNG) / Video (MP4)
```

**Memory lifecycle on 6GB VRAM:**
1. Text encoder loads, encodes prompt, unloads from VRAM
2. Diffusion transformer runs denoising steps
3. VAE decodes the result
4. LLM models (llama-swap) are evicted before loading to free VRAM

## Shared Components

Models can share components to avoid duplicate downloads:

```
models/shared/
└── vae/
    └── wan/                    # AutoencoderKLWan (Wan 2.1 VAE)
        ├── config.json
        └── diffusion_pytorch_model.safetensors
```

Models that use a shared component link to it via symlink instead of downloading a copy:
```
models/wan22-i2v/vae/wan_2.1_vae.safetensors → ../../shared/vae/wan/diffusion_pytorch_model.safetensors
~/.llama-models/lingbot-t2v/vae → ../../git/ai-dotfiles/diffuse/models/shared/vae/wan
```

Run `diffuse list` to see which components are shared and how much disk space is saved.

## File Structure

```
diffuse/
├── README.md
├── run.sh                  # Entry point (symlinked as `diffuse`)
├── download-model.sh       # Model download script
├── pyproject.toml
├── models/                 # Model weights (not in git)
│   ├── shared/             # Shared components (deduplicated)
│   │   └── vae/wan/        # AutoencoderKLWan
│   ├── wan22-i2v/          # Wan2.2 I2V
│   └── ideogram-4-Q4_0/    # Ideogram 4 (when downloaded)
├── diffuse/                # Python package
│   ├── cli.py              # CLI — argument parsing, orchestration
│   ├── models.py           # Model registry + shared component registry
│   ├── paths.py            # Path constants
│   ├── backends/           # Backend modules (one per pipeline type)
│   │   ├── __init__.py     # Backend dispatch + auto-download + shared component logic
│   │   ├── hidream.py      # HiDream (transformers + accelerate)
│   │   ├── sd_cpp.py       # Ideogram 4 + Wan2.2 (stable-diffusion.cpp)
│   │   ├── lingbot.py      # LingBot (diffusers custom pipeline)
│   │   ├── gemlite.py     # Bonsai (legacy, code preserved)
│   │   └── framepack.py    # FramePack (disabled, code preserved)
│   ├── enhance.py          # LLM-based prompt enhancement
│   ├── llm.py              # llama-swap integration
│   └── output.py          # Output path resolution + debrief
├── vendor/                 # Vendored dependencies (pinned SHAs)
└── outputs/                 # Generated images/videos
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

### Wan2.2 I2V (`wan22-i2v`)
- **Backend:** sd-cli (stable-diffusion.cpp, GGUF)
- **Components:** Wan2.2 A14B Rapid AIO Q4_K_S (9.3 GB) + UMT5-XXL Q8_0 (5.7 GB) + CLIP Vision (1.2 GB) + shared Wan VAE
- **Capabilities:** Image-to-video, 4-step accelerator
- **Default:** 832×480, 33 frames @ 24fps (~1.3s)

### LingBot T2V (`lingbot-t2v`)
- **Backend:** diffusers (custom pipeline, `LingBotVideoPipeline`)
- **Components:** LingBot 1.3B DiT bf16 (2.79 GB) + Qwen3-VL 4B bf16 text encoder (8.88 GB, CPU) + shared Wan VAE
- **Capabilities:** Text-to-video, text+image-to-video
- **Requires:** `lingbot-video` repo cloned at `~/git/lingbot-video` (pipeline code)
- **Default:** 832×480, 33 frames @ 24fps, 4 steps

## License

Models: respective licenses (Apache 2.0 for LingBot, HiDream; see HuggingFace pages)
This CLI tool: MIT