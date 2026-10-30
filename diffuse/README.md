# diffuse — Local Diffusion Image Generation CLI

Diffusion generation CLI for NVIDIA GPUs. Wan2GP is the canonical image backend (sd.cpp retired 2026-10-20). Automatic VRAM management on 6GB via MMGP profile 5 offload.

> **Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

## Fleet (16 entries)

| Model | Type | Backend | Highlights |
|-------|------|---------|-----------|
| **qwen-image-2.1** | T2I + editing + RGBA | Wan2GP | DEFAULT — uncensored (abenzerps UC convrot + abliterated TE); `--turbo` = distilled 8-step LoRA |
| **qwen-image-2.1-turbo** | T2I + editing | Wan2GP | official Alibaba turbo checkpoint, teacache |
| **moss-image** / **-base** / **-kromoss** | T2I (JSON layout-native) | Wan2GP | MOSS family; kromoss = full fine-tune of base |
| **flux-3** | T2I + editing (SFW) | Wan2GP | FLUX 3 Klein, best text render |
| **agate-preview-001/-003** | T2I | native python | 260M FCDM, seconds/image, fixed 256² / multi-res 512² |
| **ternary-gemlite** | T2I | gemlite (Bonsai) | 1.58-bit fast drafts, 512×512 |
| **anima-turbo-v1.1** | T2I (anime) | sd.cpp | **30/10 — 2B CircleStone×Comfy; distilled 10 steps CFG 1, er_sde; uncensored (Unbound) — recomendado pra começar; primeira família de imagem TAMBÉM na frota llama-swap (sd-server) |
| **anima-aesthetic-v1.1** | T2I (anime) | sd.cpp | fine-tune de consistência, 40 steps CFG 3 — o melhor default art style |
| **anima-base-v1.0** | T2I (anime) | sd.cpp | Q8; base pra treinar LoRA da Anima; 40 steps CFG 5 |
| **anima-preview3** | T2I (anime) | sd.cpp | Q8; próxima geração (preview public na HF; Civitai ainda não lista) |

## Quick Start

```bash
diffuse list                          # all models, deps, sizes
diffuse loras                         # LoRA inventory per family (catalog x disk)
diffuse evict                         # free VRAM (evict llama-swap LLMs)

# Default generation
diffuse generate -m qwen-image-2.1 --evict-llm -p "a cat on the moon"

# Fast (8 steps, distilled LoRA over the base)
diffuse generate -m qwen-image-2.1 --turbo --evict-llm -p "a cat on the moon"

# Editing
diffuse generate -m qwen-image-2.1 --edit photo.png -p "add a red hat"

# Manual LoRA tag (name WITHOUT .safetensors)
diffuse generate -m qwen-image-2.1 -p "<lora:FlatAnime_Qwen_2.1:1.0> an anime girl, flat colors"

# Prompt enhance (LLM) + LoRA selection (llm | decision model)
diffuse generate -m qwen-image-2.1 --enhance --nsfw -p "..."
diffuse generate -m qwen-image-2.1 --enhance --nsfw --lora-selector decision -p "..."
```

## LoRA selection

Unified catalogs live in `~/.local/share/diffuse/loras_<family>.json` (`nsfw: bool` +
`compat` per variant; IDs are stable — never renumber). Two selectors, configured in
`~/.local/share/fleet/enhance-models.yaml` (flags: `--lora-selector`, `--decision-model`,
`--decision-threshold`, `--decision-max`):

- **llm** (default): ColBERT rerank → enhance LLM picks 0-2 via a `LORAS:` line
- **decision**: fleet decision model (`d1-3b` via llama-swap `/v1/systemone`) scores
  every eligible LoRA independently (one `noul` question each); cutoff 0.5, max 4

Manual `<lora:NAME:mult>` tags override the selector and skip it entirely. Distilled
LoRAs (turbo/DMD) are never selectable — they enter via `--turbo` or tags only.

## Architecture

```
Prompt (text) → Text Encoder (CPU offload) → DiT (MMGP profile 5, block streaming)
→ VAE decode → PNG in the CALLER's cwd
```

Memory lifecycle on 6GB: llama-swap LLMs evicted first → TE encodes on CPU → DiT
denoises with block-by-block VRAM streaming (spills to RAM) → VAE decodes.

## File Structure

```
diffuse/
├── run.sh                  # Entry point (symlinked as `diffuse`); `generate` subcommand
├── pyproject.toml
├── diffuse/                # Python package
│   ├── cli.py              # CLI — parsing, dispatch, runners
│   ├── models.py           # Model registry (16 entries)
│   ├── lora_selector.py    # Unified LoRA selection (llm | decision)
│   ├── paths.py            # Path constants
│   ├── backends/           # wan2gp.py (canonical), gemlite, agate, bonsai, framepack
│   ├── enhance.py          # LLM prompt enhancement
│   ├── llm.py              # llama-swap integration
│   └── output.py           # Output resolution + debrief
└── outputs/                # Generated images registry
```

## License

Models: respective licenses (see HuggingFace pages; UC/uncensored builds per their repos).
This CLI tool: MIT