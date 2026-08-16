# music3 — MiniMax Music 3 CLI wrapper

Local text-to-music generation using MiniMax Music 3 via diffusers.
Designed for low-VRAM GPUs (RTX 3050 6GB) with CPU offload.

## Structure

```
music3/
├── music3              # Bash shortcut (like h3 for Wan2GP)
├── generate.py          # Python generation script (main)
├── prompt_enhancer.py   # Local LLM-based caption rewriter (optional)
├── templates/           # Example lyrics + captions
│   ├── blues.txt        # Official example caption
│   └── blues_lyrics.txt # Official example lyrics
└── README.md            # This file
```

## Setup

1. Model: `~/git/minimax-music3/models/` (downloaded from HF)
2. Venv: `~/git/minimax-music3/.venv` (diffusers PR commit)
3. Install shortcut: `ln -s ~/git/ai-dotfiles/music3/music3 ~/.local/bin/music3`

## Usage

```bash
# Simple
music3 "warm acoustic pop, female vocals, fingerpicked guitar" --lyrics-file lyrics.txt

# With structured caption file
music3 --caption-file caption.txt --lyrics-file lyrics.txt --duration 120 --seed 42

# Enhance a short prompt into structured caption via local LLM
music3 "dark synthwave 128 BPM male vocals" --lyrics-file lyrics.txt --enhance

# Dry run
music3 "jazz" --lyrics "[Verse]\nTest" --dry-run
```

## VRAM Strategy

RTX 3050 6GB. Model card says 8GB minimum with group offload.
We use:
1. `enable_auto_cpu_offload` — automatic component offloading
2. `apply_group_offloading(leaf_level, stream=True)` — layer-by-layer streaming of the 8B LLM
3. bf16 dtype

If OOM: reduce duration, or add group offload to flow_matching_model.

## Prompt Enhancement

`--enhance` uses llama-swap (localhost:12434) to expand a short natural-language
prompt into the full 3-section Structured Caption format (Global Metadata,
Vocal Details, Arrangement). Similar to h3's --enhance for video prompts.

The enhancer system prompt is in `prompt_enhancer.py`. It follows the same
pattern as the H3 prompt enhancer patch.