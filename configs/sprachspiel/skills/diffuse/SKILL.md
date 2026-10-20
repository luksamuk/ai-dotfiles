---
name: diffuse
description: Generate images locally with the diffuse CLI. Wan2GP-only fleet (qwen-image-2.1, moss-image, flux-3, agate, bonsai), prompt guidelines per family, LoRA tags, and the --nsfw catalog selector. Emit enhanced prompts yourself instead of using --enhance.
---

# diffuse — Local Image Generation

The `diffuse` CLI generates images from text prompts using local models on the GPU (Wan2GP is the canonical image backend — sd.cpp was retired 2026-10-20). It coordinates with llama-swap via `--evict-llm` to free VRAM before loading.

**Important:** Some models may not be installed. If a model fails to load or is not listed by `diffuse --list`, inform the user and suggest an alternative from the list below.

## CRITICAL: Do NOT use --enhance. Emit the enhanced prompt yourself.

The `--enhance` flag calls a local LLM (qwen3.6-heretic via llama-swap) to expand prompts and select LoRAs. **Do not use it** — you are a more capable prompt engineer than the local LLMs. Craft the enhanced prompt yourself and pass it directly with `-p`. (Exception: `--enhance` is acceptable as a fallback when you cannot see an input image the user references.)

All image models today take **natural language** — a rich English paragraph with subject, art style, lighting, composition, and mood. The structured-JSON prompt format belonged to Ideogram 4, which was retired.

## Model Selection

**Default model: `qwen-image-2.1`** (Qwen-Image 2.1 Uncensored via Wan2GP) — best overall quality, no refusals, text rendering, and native editing. Use it for most tasks.

| Need | Model | Why |
|------|-------|-----|
| Default image generation | `qwen-image-2.1` | Best quality; uncensored (UC weights, abliterated TE) |
| Fast generation (8 steps) | `qwen-image-2.1 --turbo` | Same base + distilled LoRA; or `qwen-image-2.1-turbo` (official checkpoint) |
| Image editing / multi-reference | `qwen-image-2.1 --edit` | Instruction-based editing; up to 4 reference images |
| Flat anime / cel-shaded illustration | `qwen-image-2.1` + FlatAnime LoRA (see below) | Style LoRA from the catalog |
| Structured JSON-native illustration (MOSS family) | `moss-image` / `moss-image-base` / `moss-image-kromoss` | JSON layout-native; kromoss is a full fine-tune of the base |
| Text-rendering poster / typography | `flux-3` | FLUX 3 Klein, best-in-class text render, SFW |
| Logos / icons / small drafts | `agate-preview-001` / `agate-preview-003` | 260M FCDM, seconds per image |
| Fast 1-bit draft | `ternary-gemlite` | Bonsai ternary, 512×512 |

LoRA selection is done by catalog + reranker when `--enhance`/`--nsfw` is used; for manual control, use **LoRA tags** (below).

## LoRA Tags (manual, no --enhance needed)

Manual tags in the prompt override everything and disable the selector:

```bash
# tag syntax: <lora:NAME:multiplier> — NAME WITHOUT the .safetensors extension
diffuse generate -m qwen-image-2.1 -p "<lora:FlatAnime_Qwen_2.1:1.0> An anime illustration of a girl, flat colors, cel shading"
```

Rules:

- The tag name is the filename **without** `.safetensors` (e.g. `FlatAnime_Qwen_2.1`, `nsfw_image_edit_aio`).
- Multiplier 0.4-1.0 typical; 1.0 for styles, 0.6-0.8 for anatomy/fixes.
- Style LoRAs need the style vocabulary **in the prompt** — a FlatAnime tag with a photographic prompt still yields a photo (measured: the LoRA applies, but the medium follows the prompt).
- Browse what exists: `diffuse loras` (per-family inventory: cataloged entries, orphans on disk, compat per variant).

## The `--nsfw` Flag (catalog selector)

Unlike the retired Ideogram-bypass semantics, `--nsfw` today means: **the LoRA selector may pick from the NSFW entries of the family catalog** (unified `loras_<family>.json`, field `nsfw: bool`). The qwen/moss models themselves are uncensored by training (UC weights / abliterated TE) — the flag does not "unlock" them; it only widens the LoRA candidate pool and relaxes the negative prompt.

Selection modes (config in `~/.local/share/fleet/enhance-models.yaml`, override via `--lora-selector llm|decision`):

- `llm` (default): ColBERT reranks the eligible catalog, then the enhance LLM picks 0-2 via a `LORAS:` line.
- `decision`: a fleet decision model (d1-3b via `/v1/systemone`) scores each eligible LoRA independently; cutoff 0.5, max 4. Validated at 93% recall@3.

## Prompt Guidelines (all models: natural language)

Write a single rich English paragraph. Include:

- **Subject**: described physically (the model does not know named characters)
- **Art style**: medium and aesthetic ("cinematic photography", "anime illustration with flat colors and clean lineart")
- **Lighting**: direction, quality, color temperature
- **Composition**: framing, camera angle, depth of field
- **Mood**: emotional tone

Example:

> A lone samurai standing under a torrential downpour at dusk, rain streaking across the frame in silver threads. The samurai wears weathered indigo armor, water beading on the lacquered plates, a tachi sword at his side. Cinematic low-angle shot, shallow depth of field with rain drops in sharp foreground. Moody blue-teal palette with warm amber reflections from distant lanterns. Concept art illustration style, dramatic rim lighting, volumetric mist.

For **editing** (`--edit photo.png`): write a clear instruction of what to change, plus preservation constraints (age, proportions) and the parts that must stay untouched.

## Usage Patterns

```bash
# Default generation (always --evict-llm to free VRAM)
diffuse generate -m qwen-image-2.1 --evict-llm -p "<your enhanced prompt>"

# Fast: same base + distilled turbo LoRA (8 steps)
diffuse generate -m qwen-image-2.1 --turbo --evict-llm -p "<your enhanced prompt>"

# Editing (qwen21 or flux-3)
diffuse generate -m qwen-image-2.1 --evict-llm --edit photo.png -p "<edit instruction>"

# Manual LoRA tag (no --enhance needed)
diffuse generate -m qwen-image-2.1 --evict-llm -p "<lora:FlatAnime_Qwen_2.1:1.0> <anime prompt>"

# Moss family (JSON layout native)
diffuse generate -m moss-image-base --evict-llm -p "<your enhanced prompt>"

# Custom resolution / seed
diffuse generate -m qwen-image-2.1 --evict-llm -p "<prompt>" --size 1360x768 --seed 42
```

## Important Notes

- **Always use `--evict-llm`** before generation to free VRAM from llama-swap models
- **Default model is `qwen-image-2.1`**
- Images save to the **caller's current working directory**, not the diffuse project dir
- **`diffuse --list`** shows all models; **`diffuse loras`** shows the LoRA inventory per family
- Qwen-Image 2.1 dimensions are multiples of 32; ~1MP is the sweet spot (1024×1024, 1424×800)
- Moss family: 50 steps, cfg 3.0 by default (~3min on 6GB); qwen21 base: 40 steps cfg 4.0; `--turbo`: 8 steps
- **Two concurrent diffuse processes freeze VRAM** — never run in parallel; wait for one to finish

## Pitfalls

- Do not stack quality tags ("masterpiece, best quality, 4k") — models ignore or degrade from them
- One mood per prompt — conflicting vibes produce incoherent images
- Seed matters more than prompt refinement — iterate seeds before rewriting prompts
- **LoRA of style ≠ filter**: the tag applies the weights, but the prompt still decides the medium — ask for the style in words too
- **Distilled LoRAs (turbo/DMD) are never selectable** — they enter via `--turbo` or tags only, by house decree
- The `qwen-image-2.1-fix-1.0-comfy` LoRA (DoRA format) was **removed entirely** (2026-10-20): the WanGP handler rejected it and killed the render; do not re-add it or tag it
- Music generation is a separate CLI (`magenta-rt`), not part of diffuse; video generation is currently decommissioned in this fleet