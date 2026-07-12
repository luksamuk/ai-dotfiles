---
name: diffuse
description: Generate images and video locally with the diffuse CLI. Covers all available models, prompt enhancement strategy (emit enhanced prompts yourself instead of using --enhance), and per-model formatting rules.
---

# diffuse — Local Image & Video Generation

The `diffuse` CLI generates images and video from text prompts using local AI models on the GPU. It coordinates with llama-swap via `--evict-llm` to free VRAM before loading.

**Important:** Some models may not be installed. If a model fails to load or is not listed by `diffuse --list`, inform the user and suggest an alternative from the list below.

## CRITICAL: Do NOT use --enhance. Emit the enhanced prompt yourself.

The `diffuse` CLI has an `--enhance` flag that calls a local LLM to expand prompts. **Do not use it.** You are a more capable prompt engineer than the local LLMs. Instead, craft the enhanced prompt yourself and pass it directly with `-p`.

This means:

- For **vision-type models** (Bonsai, HiDream, Wan2.2): write a rich, detailed English paragraph describing the scene, characters, art style, lighting, and composition. Pass it directly: `diffuse generate -m <model> -p "<your enhanced prompt>"`.
- For **ideogram-type models** (Ideogram 4): write a complete JSON object with the required structure (see below) and pass it directly: `diffuse generate -m ideogram4-q4 -p '<your JSON>'`.
- For **video models** (Wan2.2): if you have a reference image, use your own vision capabilities to analyze it, then write a detailed motion-focused prompt. Do not rely on `--enhance` to analyze the image for you.

If you cannot see the input image (no vision capability in the current session), ask the user to describe it, or use `--enhance` as a fallback only in that specific case.

## Available Models

### Image Models

| Model | Flag | Backend | Best for | Default size | Enhance type |
|-------|------|---------|----------|-------------|--------------|
| `binary-gemlite` | `-m binary-gemlite` | gemlite (Bonsai) | Fast generation, 1-bit weights | 512x512 | vision (natural language) |
| `ternary-gemlite` | `-m ternary-gemlite` | gemlite (Bonsai) | Default, 95% of FP16 quality | 512x512 | vision (natural language) |
| `ideogram4-q4` | `-m ideogram4-q4` | sd.cpp | Best text rendering in images | 480x480 | ideogram (JSON) |
| `hidream-sdnq` | `-m hidream-sdnq` | transformers (HiDream) | T2I + image editing, high-res native | 2048x2048 (snapped) | vision (natural language) |

### Video Models

| Model | Flag | Backend | Best for | Notes |
|-------|------|---------|----------|-------|
| `wan22-i2v` | `-m wan22-i2v` | sd.cpp (GGUF) | Image-to-video, Wan2.2 A14B | Requires `--input-image`. 4-step accelerator. ~48min on 6GB for 33 frames. Use Q4_K_S quant (never Q2_K). |

### Music Models (separate CLI: `magenta-rt`)

| Model | Flag | Best for | Notes |
|-------|------|----------|-------|
| `mrt2_small` | `-m mrt2_small` | Recommended for 6GB VRAM | 230M params, faster than real-time on GPU |
| `mrt2_base` | `-m mrt2_base` | Higher quality but OOMs on 6GB | 2.4B params, CPU fallback ~10x slower |

## Prompt Enhancement Guidelines Per Model

### Bonsai (binary-gemlite, ternary-gemlite) — Natural Language

Write a single rich English paragraph. Include:

- **Subject**: who/what is in the scene, described physically (do not assume the model knows named characters — describe them as if it doesn't)
- **Art style**: medium, aesthetic, visual references (e.g. "oil painting style", "anime illustration", "cinematic photography")
- **Lighting**: direction, quality, color temperature (e.g. "golden hour backlight", "dramatic chiaroscuro")
- **Composition**: framing, camera angle, depth of field
- **Mood/atmosphere**: emotional tone of the image

Example enhanced prompt:

> A lone samurai standing under a torrential downpour at dusk, rain streaking across the frame in silver threads. The samurai wears weathered indigo armor, water beading on the lacquered plates, a tachi sword at his side. Cinematic low-angle shot, shallow depth of field with rain drops in sharp foreground. Moody blue-teal color palette with warm amber reflections from distant lanterns. Concept art illustration style, dramatic rim lighting, volumetric mist.

### Ideogram 4 (ideogram4-q4) — Structured JSON

Ideogram 4 **requires** structured JSON. Plain text produces garbage output. Write a complete JSON object with these fields:

```json
{
  "high_level_description": "One-sentence scene summary",
  "style_description": {
    "aesthetics": "art style, visual genre",
    "lighting": "lighting description",
    "color_palette": ["#RRGGBB", "#RRGGBB", "#RRGGBB"]
  },
  "compositional_deconstruction": {
    "canvas": "canvas description (size, orientation)",
    "background": "what's behind the subject",
    "layout": "spatial arrangement of elements",
    "elements": [
      {"name": "element name", "description": "visual details"}
    ]
  }
}
```

Rules for Ideogram 4 JSON:

- Always use medium: "illustration" or "painting" to avoid photo-mode safety filters
- Avoid trigger words: "vampire", "blood", "death", "gothic", "horror", "oil on canvas" — use safe alternatives ("pale immortal noble", "crimson", "distinguished", "ornate medieval", "richly painted artwork")
- Include 3-5 colors in `color_palette` as hex codes — this anchors the model and reduces structural collapse
- The `elements` array should describe each visible component of the image

### HiDream (hidream-sdnq) — Natural Language + Editing

For T2I: same guidelines as Bonsai (rich English paragraph). HiDream snaps all resolutions to a minimum of 2048x2048 — there is no way to generate below that. Use Bonsai or Ideogram 4 for smaller images.

For image editing (`--edit photo.png`): write a clear instruction describing what to change. Include:

- **Age preservation**: if editing a photo of a person, explicitly state their approximate age and skin quality (e.g. "a man in his early 30s with smooth, youthful skin, no wrinkles") — HiDream tends to age subjects without this
- **Body proportions**: include "full body shot, natural body proportions, head proportional to body" to avoid the "big head" distortion
- **Safety avoidance**: same trigger word rules as Ideogram 4

Example edit prompt:

> Add a red scarf around the puppy's neck. The puppy is a young golden retriever with soft fluffy fur, keep its exact age and appearance. Full body shot, natural body proportions. Warm lighting, cozy atmosphere, illustration style.

### Wan2.2 I2V (wan22-i2v) — Motion-Focused Video Prompt

Video prompts must emphasize **motion and temporal dynamics**, not just static image qualities. Include:

- **Character description**: physical appearance (the video model needs visual anchoring)
- **Motion description**: what moves, how, direction, speed (e.g. "slowly turns his head to the right", "hair flows in the wind", "cape billows behind him")
- **Camera movement**: pan, zoom, tilt, dolly (e.g. "camera slowly dollies in", "static camera with subject movement")
- **Physics/environment**: ambient effects (candles flicker, leaves scatter, dust particles float)
- **Atmosphere**: mood that carries through the temporal sequence

The AllInOne 4-step accelerator tends to produce near-zero motion with subtle prompts. Be aggressive: use words like "dramatic", "strongly", "swiftly", "billowing", "flying".

Example video prompt:

> Silver the Hedgehog, a young anthropomorphic hedgehog with silver fur, golden eyes, white gloves, and cyan wristbands, stands in a dark gothic cathedral. He dramatically raises his hand, psychokinetic energy surging around him as golden particles explode outward. His golden pocket watch swings wildly. Camera slowly dollies in from a low angle. Candles flicker violently, stone dust rises from the floor, his silver fur ripples from the energy pulse. Dramatic lighting with intense golden highlights against deep shadows. 5 seconds of dynamic motion.

## Usage Patterns

```bash
# Image generation (always use --evict-llm to free VRAM)
diffuse generate -m ternary-gemlite --evict-llm -p "<your enhanced prompt>"
diffuse generate -m ideogram4-q4 --evict-llm -p '<your JSON>'
diffuse generate -m hidream-sdnq --evict-llm -p "<your enhanced prompt>"

# Image editing (HiDream only)
diffuse generate -m hidream-sdnq --evict-llm --edit photo.png -p "<your edit instruction>"

# Video generation (Wan2.2)
diffuse generate -m wan22-i2v --evict-llm --input-image photo.png -p "<your motion prompt>"

# Custom resolution
diffuse generate -m ternary-gemlite --evict-llm -p "..." --size 768x768

# With seed for reproducibility
diffuse generate -m ternary-gemlite --evict-llm -p "..." --seed 42

# Music generation (separate CLI)
magenta-rt generate -p "high-energy 16-bit era platformer music, FM synthesis, driving bassline, 140bpm" --duration 8.0 --evict-llm
```

## Important Notes

- **Always use `--evict-llm`** before generation to free VRAM from llama-swap models
- **Images save to the caller's current working directory**, not the diffuse project dir
- **HiDream resolution snapping**: all outputs are minimum 2048x2048 — use Bonsai or Ideogram 4 for smaller images
- **Ideogram 4 max safe size on 6GB**: 480x480 (512 may OOM)
- **Wan2.2 video takes ~48 minutes** on 6GB VRAM for 33 frames — plan accordingly
- **`diffuse --list`** shows all available models, backends, and sizes
- **Some models may not be installed** — if `diffuse generate -m <model>` fails, check `diffuse --list` and suggest an alternative
- **FramePack I2V was decommissioned** — use `wan22-i2v` for video generation instead
- **Music prompts**: describe the sound, not the context. "Sega Genesis style, FM synthesis brass, driving bassline, 140bpm" works. "Sonic the Hedgehog music" does not — the model doesn't know game franchises

## Pitfalls

- Do not stack quality tags like "masterpiece, best quality, 4k" — the models ignore or degrade from them
- One mood per prompt — conflicting vibes produce incoherent images
- Seed matters more than prompt refinement — iterate seeds before rewriting prompts
- Ideogram 4 content filter may trigger on gothic/horror vocabulary even with safe alternatives — the refusal is baked into the weights
- HiDream ages subjects in editing — always include explicit age preservation instructions
- Wan2.2 Q2_K quant produces severe hallucinations — Q4_K_S is the minimum viable quant
- Wan2.2 4-step accelerator produces near-zero motion with subtle prompts — be aggressive with motion language