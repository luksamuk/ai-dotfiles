# diffuse — Local Diffusion Image Generation CLI

Diffusion image generation CLI for NVIDIA GPUs. Currently ships with [Bonsai Image 4B](https://prismml.com/news/bonsai-image-4b) (1-bit / ternary), designed to support additional diffusion backends in the future.

> **Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

## Nomenclature

| Term | What it is | Example |
|------|-----------|---------|
| **LLM** | Large Language Model — generates text | Qwen3, Gemma, Llama |
| **Diffusion Model** (DiT) | Diffusion Transformer — generates images by iteratively denoising | FLUX, Stable Diffusion, Bonsai Image |
| **Text Encoder** | Reads your prompt and produces vector embeddings. Pipeline-specific, NOT replaceable across models | Qwen3-4B (4-bit HQQ, bundled with Bonsai) |
| **Diffusion Transformer** | The core image-generation engine — runs N denoising steps | Bonsai 1-bit (0.93 GB) or Ternary (1.21 GB) |
| **VAE** | Variational Autoencoder — encodes/decodes between pixel space and latent space | Flux2 32-channel (bundled with Bonsai) |
| **Pipeline** | The full assembly: Text Encoder → Diffusion Transformer → VAE | Bonsai Image 4B pipeline |

## Architecture

```
Prompt (text)
    │
    ▼
┌──────────────────────────┐
│  Text Encoder (Qwen3-4B)  │  2.84 GB (HQQ 4-bit) — OFFLOADED after encode
│  → produces text embeddings│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Diffusion Transformer   │  1.08 GB (gemlite INT1) or 1.35 GB (gemlite 2-bit)
│  4 steps of denoising     │  Guided by text embeddings, iteratively refines noise → image latent
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  VAE (Flux2 32ch)        │  0.17 GB (FP16) — decodes latent → pixel image
└──────────┬───────────────┘
           │
           ▼
      Image (PNG)
```

**Memory lifecycle:**
1. Text encoder loads (~2.84 GB), encodes prompt, then **unloads from VRAM**
2. Diffusion transformer runs 4 denoising steps (~1-1.35 GB + activations)
3. VAE decodes the result
4. **Peak HBM:** ~6.4 GiB at 1024×1024 (fits in 6 GB VRAM)

## Available Models

| Model | Weights | Transformer Size | Quality | Speed on RTX 3050 |
|-------|---------|-----------------|---------|-------------------|
| `binary-gemlite` | 1-bit {−1, +1} | 0.93 GB | 88% of FP16 | 512² ~4s, 1024² ~25s |
| `ternary-gemlite` | 1.58-bit {−1, 0, +1} | 1.21 GB | 95% of FP16 | 512² ~6s, 1024² ~30s |

**Recommended for RTX 3050:** `ternary-gemlite` — better quality, fits comfortably at 512×512 default resolution.

## Quick Start

```bash
# Download model weights (first time only)
diffuse download ternary

# Generate an image
diffuse generate -p "A bonsai tree under moonlight"

# Interactive mode (prompts for input)
diffuse generate

# Custom size and seed
diffuse generate -p "Cyberpunk cityscape" --size 1024x1024 --seed 42
```

## File Structure

```
diffuse/
├── README.md              # This file
├── generate.py             # CLI: load → prompt → generate → stats → unload
├── pyproject.toml          # Dependencies
├── download-model.sh       # Download model weights from HuggingFace
├── run.sh                  # Runner script (entry point for diffuse symlink)
├── models/                 # Model weights (downloaded, not in git)
└── outputs/                # Generated images
```

## CLI Usage

```
usage: diffuse generate [-h] [-m MODEL] [-p PROMPT] [--seed SEED] [--steps STEPS]
                        [--size SIZE] [--output OUTPUT]

Local diffusion image generation CLI

options:
  -h, --help            show this help message and exit
  -m, --model          Model variant (default: binary-gemlite)
  -p, --prompt         Text prompt (interactive if omitted)
  --seed               Random seed (random if not set)
  --steps              Denoising steps (default: 4)
  --size               Image size WxH (default: 512x512)
  --output             Output PNG path (auto-generated if not set)
```

After generation, the CLI prints a debrief:

```
═══ diffuse — Generation Report ═══
  Model:       binary-gemlite (1-bit)
  Prompt:      "A bonsai tree under moonlight"
  Seed:        9909
  Resolution:  512 × 512
  Steps:       4

  Timings:
    Setup:       25.3 s   (imports + model load + kernel JIT)
    Diffusion:    4.2 s   (4 denoising steps + VAE decode)
    ─────────────────────
    Wall:        29.5 s

  Memory:
    Peak HBM:   4,120 MiB

  Output: outputs/binary-gemlite/image_20260527_143020_seed9909.png
══════════════════════════════════════
```

## Why Not ComfyUI / vLLM / llama.cpp?

| Tool | Works? | Reason |
|------|--------|--------|
| **ComfyUI** | ⚠️ No native low-bit support | ComfyUI can't read gemlite/MLX packed weights. Would need GGUF conversion or custom node |
| **vLLM** | ❌ | LLM serving engine — not designed for diffusion models |
| **llama.cpp** | ❌ | Same — text token prediction, not image denoising |
| **diffuse** | ✅ | Direct pipeline, minimal overhead, model-agnostic design, load-and-unload |

## Text Encoder — Can I Swap It?

No. The text encoder is **integral** to each pipeline:
- Its embeddings must match the dimensionalities the diffusion transformer expects
- It's quantized specifically for its pipeline
- It unloads after prompt encoding anyway — doesn't compete for VRAM during denoising

Think of it as the "power supply" of the pipeline — it has to match the voltage and connector exactly.

## Adding New Models

To add a new diffusion model backend:
1. Add model metadata to `MODELS` dict in `generate.py`
2. Add download config to `download-model.sh`
3. If the model uses a different pipeline class, add a new loader function

## License

Bonsai Image 4B models: **Apache 2.0** (PrismML)
This CLI tool: MIT