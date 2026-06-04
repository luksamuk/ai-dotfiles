# magenta-rt — Local Music Generation with Magenta RealTime 2

Magenta RealTime 2 (MRT2) running locally on NVIDIA RTX 3050 (6GB) via JAX backend (offline inference).

> **Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)
> **Note:** Real-time streaming requires Apple Silicon. On NVIDIA, only offline (non-real-time) inference is supported via the Python JAX backend.

## Architecture

```
Prompt (text) + Audio reference + MIDI
    │
    ▼
┌──────────────────────────┐
│  MusicCoCa (style model) │  Embeds text + audio into 768-dim vectors
│  → 12 quantized tokens    │  Quantized to 10-bit codes, 12 RVQ depth
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Decoder Transformer LLM │  mrt2_small (230M, 12 layers)
│  Frame-wise autoregression│  or mrt2_base (2.4B, 20 layers)
│  25Hz, windowed attention │  Generates 12 RVQ tokens per frame
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  SpectroStream (codec)   │  Decodes tokens → 48kHz stereo audio
│  25Hz, 64 RVQ, 10-bit    │  16kbps bitrate
└──────────┬───────────────┘
           │
           ▼
      Audio (WAV, 48kHz stereo)
```

**Memory lifecycle (offline inference):**
1. MusicCoCa loads (~small footprint), encodes prompt, stays in memory
2. Transformer LLM generates audio tokens frame-by-frame
3. SpectroStream decodes tokens to waveform
4. **Peak VRAM estimate (small):** ~1.5-2GB for 230M model
5. **Peak VRAM estimate (base):** ~5-6GB for 2.4B model (tight on 6GB)

## Available Models

| Model | Parameters | Layers | Attention Window | Real-time (Apple Silicon) | Quality | Disk Size |
|-------|-----------|--------|-----------------|--------------------------|---------|-----------|
| `mrt2_small` | 230M | 12 | 41 frames (~1.6s) | Any M-series Mac ✅ | Good | ~460MB |
| `mrt2_base` | 2.4B | 20 | 25 frames (1s) | Pro Max only ✅ | Better | ~4.8GB |

## Quick Start

```bash
# Create venv and install
cd ~/git/ai-dotfiles/magenta-rt
uv venv --python 3.12
source .venv/bin/activate
uv pip install "magenta-rt[jax]"

# Download resources (MusicCoCa + SpectroStream)
mrt models init

# Download small model (recommended for RTX 3050)
mrt models download --model=mrt2_small

# Generate 4 seconds of music
mrt jax generate --prompt "disco funk with bass" --duration 4.0 --model=mrt2_small

# Generate with base model (heavier, may not fit)
mrt jax generate --prompt "ambient pads with sub bass" --duration 4.0 --model=mrt2_base
```

## MIDI Control

MRT2 accepts 128-dim multihot MIDI vectors per frame:
- 0 = Off
- 1 = Sustain
- 2 = Onset
- 3 = Sustain or Onset (model decides)

This enables note-by-note control of the generated music — live instruments, DAW integration, or programmatic composition.

## File Structure

```
magenta-rt/
├── README.md              # This file
├── pyproject.toml          # Dependencies and metadata
├── generate.py             # CLI: prompt → generate → save WAV
├── run.sh                  # Runner script (entry point for symlink)
├── models/                 # Model weights (downloaded, not in git)
└── outputs/                # Generated audio files
```

## Performance Estimates (RTX 3050 6GB)

| Model | VRAM (est.) | Generation Speed | Quality |
|-------|-------------|-----------------|---------|
| mrt2_small | ~1.5-2GB | Likely faster than real-time offline | Good |
| mrt2_base | ~5-6GB | Tight on VRAM, may need batch size 1 | Better |

**Strategy:**
- Start with `mrt2_small` — should run comfortably
- Try `mrt2_base` only if small works well — watch for OOM
- Both use JAX with CUDA backend on NVIDIA

## Model Components

| Component | Purpose | Size |
|-----------|---------|------|
| **SpectroStream** | Audio codec (48kHz stereo ↔ tokens, 25Hz, 64 RVQ) | Part of resources |
| **MusicCoCa** | Joint text+audio embeddings (768-dim) | Part of resources |
| **mrt2_small** | Decoder-only Transformer (230M) | ~460MB |
| **mrt2_base** | Decoder-only Transformer (2.4B) | ~4.8GB |

## Why JAX (not MLX)?

MLX is Apple Silicon only. For NVIDIA GPUs, the JAX backend provides CUDA acceleration. The `magenta-rt[jax]` extra installs JAX with CUDA support.

## Comparison with Other Music Models

| Model | Type | Params | License | Local? | Real-time? | Control |
|-------|------|--------|---------|--------|------------|---------|
| **MagentaRT 2** | Music gen | 230M / 2.4B | Apache 2.0 + CC-BY 4.0 | ✅ | Apple Silicon only | Text + Audio + MIDI |
| MusicGen (Meta) | Music gen | 300M-3.3B | MIT | ✅ | ❌ | Text + Audio |
| Stable Audio Open | Music gen | 1.2B | Stability AI License | ✅ | ❌ | Text |

MagentaRT 2 is unique in offering real-time streaming + MIDI control + genuinely open weights (CC-BY 4.0).

## License

- Code: Apache 2.0
- Model weights: CC-BY 4.0 (Google)
- This CLI wrapper: MIT