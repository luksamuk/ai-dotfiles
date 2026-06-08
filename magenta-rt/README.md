# magenta-rt — Local Music Generation with Magenta RealTime 2

Magenta RealTime 2 (MRT2) running locally on NVIDIA RTX 3050 (6GB) via JAX CUDA backend (offline inference).

> **Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)
> **Note:** Real-time streaming requires Apple Silicon. On NVIDIA, only offline (non-real-time) inference is supported via the JAX backend. However, generation is still faster than real-time (~37 steps/s vs 25 target).

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
│  25Hz, 64 RVQ, 10-bit     │  16kbps bitrate
└──────────┬───────────────┘
           │
           ▼
      Audio (WAV, 48kHz stereo)
```

**Memory lifecycle (offline inference on RTX 3050):**
1. MusicCoCa loads (TFLite, CPU), encodes prompt
2. Transformer LLM compiles on GPU (JIT, ~23s first time, ~5s warm)
3. Generation: 100 frames in ~2.6s (37.9 steps/s — faster than real-time!)
4. SpectroStream decodes
5. **Peak VRAM: ~4.1GB** (fits in 6GB with compilation overhead)
6. JIT compilation may trigger OOM warnings but completes successfully

## Available Models

| Model | Parameters | Layers | Attention Window | Quality | Disk Size |
|-------|-----------|--------|-----------------|---------|-----------|
| `mrt2_small` | 230M | 12 | 41 frames (~1.6s) | Good | ~1.1GB checkpoint |
| `mrt2_base` | 2.4B | 20 | 25 frames (1s) | Better | ~4.8GB (estimated) |

**Recommended for RTX 3050:** `mrt2_small` — fits comfortably, generates faster than real-time.

## Setup

```bash
# 1. First-time setup (creates venv, installs deps including jax[cuda12])
magenta-rt setup

# 2. Download resources (MusicCoCa + SpectroStream)
magenta-rt download resources

# 3. Download models
#    JAX backend needs checkpoints (safetensors):
magenta-rt download checkpoints-small

#    MLX backend needs models (mlxfn) — Apple Silicon only:
magenta-rt download small

# 4. Generate!
magenta-rt generate -p "disco funk with bass"
```

## Commands

```
magenta-rt generate -p "prompt"           Generate audio (interactive if no prompt)
magenta-rt generate -p "jazz" --duration 8.0  Longer generation
magenta-rt generate -p "ambient" --evict-llm  Free VRAM first
magenta-rt generate -m mrt2_base -p "epic"   Use base model (2.4B)

magenta-rt setup                           Install dependencies
magenta-rt download resources              Download MusicCoCa + SpectroStream
magenta-rt download small                  Download mrt2_small MLX format
magenta-rt download checkpoints-small     Download JAX checkpoint (REQUIRED for CUDA)
magenta-rt download base                  Download mrt2_base MLX format
magenta-rt download all                   Download everything

magenta-rt status                         Check prerequisites and models
magenta-rt evict                          Evict LLM models from llama-swap
magenta-rt clean                          Remove generated audio and caches
```

## Important: JAX Checkpoints vs MLX Models

The `mrt models download` command downloads **MLX format** files (`*.mlxfn`) which are for Apple Silicon only. For NVIDIA GPU (JAX backend), you need **safetensors checkpoints**:

```bash
# MLX format (Apple Silicon) — downloaded by 'mrt models download'
~/Documentos/Magenta/magenta-rt-v2/models/mrt2_small/

# JAX checkpoint (NVIDIA GPU) — downloaded by 'mrt checkpoints download' OR manual download
~/Documentos/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors
```

If `mrt checkpoints download mrt2_small` fails (404 on HuggingFace), download manually:

```bash
cd ~/git/ai-dotfiles/magenta-rt
.venv/bin/python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='google/magenta-realtime-2',
    filename='checkpoints/mrt2_small.safetensors',
    local_dir='$HOME/Documentos/Magenta/magenta-rt-v2',
)
"
```

## Performance (RTX 3050 6GB, mrt2_small)

| Metric | Value |
|--------|-------|
| Generation speed | 37.9 steps/s (CPU: 3.3 steps/s) |
| 4s audio generation | ~2.6s (GPU) vs ~30.5s (CPU) |
| JIT compilation | ~23s first run, ~5s warm |
| Peak VRAM | ~4.1GB (fits in 6GB) |
| Target real-time | 25 steps/s — **exceeded** ✅ |

## MIDI Control

MRT2 accepts 128-dim multihot MIDI vectors per frame:
- 0 = Off
- 1 = Sustain
- 2 = Onset
- 3 = Sustain or Onset (model decides)

## LLM Eviction

MagentaRT and llama-swap share the same GPU. Use `--evict-llm` or `magenta-rt evict` to free VRAM:

```bash
# Evict LLM models before generating
magenta-rt generate -p "ambient" --evict-llm

# Or evict separately
magenta-rt evict
```

## File Structure

```
magenta-rt/
├── README.md              # This file
├── generate.py             # CLI: prompt → generate → stats → save
├── run.sh                  # Runner script (entry point for symlink)
├── pyproject.toml          # Dependencies: magenta-rt[jax] + jax[cuda12]
├── .gitignore
├── models/                 # Model weights (downloaded, gitignored)
└── outputs/                # Generated audio (gitignored)
```

## Comparison with Other Music Models

| Model | Type | Params | License | Local? | Real-time? | Control |
|-------|------|--------|---------|--------|------------|---------|
| **MagentaRT 2** | Music gen | 230M / 2.4B | Apache 2.0 + CC-BY 4.0 | ✅ | Apple Silicon only | Text + Audio + MIDI |
| MusicGen (Meta) | Music gen | 300M-3.3B | MIT | ✅ | ❌ | Text + Audio |
| Stable Audio Open | Music gen | 1.2B | Stability AI License | ✅ | ❌ | Text |

## License

- Code: Apache 2.0
- Model weights: CC-BY 4.0 (Google)
- This CLI wrapper: MIT