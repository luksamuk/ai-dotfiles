# Changelog

### 2026-05-31 — AFM-4.5B, Backend Rebuild, Template Versioning

**New model:**
- `afm-4.5b`: Arcee Fusion Model 4.5B dense (Q5_K_M, ~3.11 GB). Upstream backend only (ik/Bee lack arcee arch). Tool calling via Hermes 2 Pro template override (`--chat-template-file`). No native thinking. 64K context (rope scaling x20 from 4096).

**Templates:**
- `afm-4.5b-tool_use.jinja`: Versioned copy of Hermes 2 Pro template from llama.cpp b766. Required for AFM tool calling — the GGUF ships with an empty `tool_use` chat_template. Verified: `finish_reason=tool_calls`, correct JSON arguments, ~44 tok/s.
- `afm-4.5b-tool_use.md`: Documentation covering format detection, template behavior, and pitfalls.

**Backend builds (2026-05-30):**
- `llama.cpp`: b766 (was b604)
- `ik_llama.cpp`: v4550 (was v4524)
- `BeeLlama.cpp`: b9868 (rebuilt)
- `llama-swap`: v219 (was v216)

**Known issues:**
- Custom Jinja templates that deviate from known format signatures cause `COMMON_CHAT_FORMAT_CONTENT_ONLY` fallback in llama.cpp's PEG auto-parser, breaking tool calling. Always use the original Hermes 2 Pro template for AFM, not a modified version.
- Laguna XS.2: upstream PR exists but not merged. No ik/Bee support yet.

### 2026-05-28 — Engine Updates, Bee/TurboQuant, ik Hadamard, Config Overhaul

**Binaries updated:**
- `llama.cpp`: b604 (was b585)
- `ik_llama.cpp`: v4524 (was v4517) — MTP fixes for Qwen3.6-MoE, `-khad`/`-vhad`, `--defer-experts`, `--flash-attn auto`, `--jinja`
- `BeeLlama.cpp`: rebuilt v9351+ with FA flags — TurboQuant/TurboQuant-TCQ, DFlash
- `llama-swap`: v216 (was v214) — performance monitoring, Prometheus/Grafana, Anthropic endpoints

**Config system overhaul:**
- Fragment system: Model IDs are now canonical (`qwen3.5-4b`, not `qwen3.5-4b-bee`)
- `build-config.py`: Fixed `unlisted: true` injection for quoted YAML keys and indent
- `build-config.py`: Fixed duplicate model key collision (suffixed IDs in `_disabled/`)
- All `_disabled/` fragments: `unlisted: true` auto-injected with correct indent

**Backend assignments:**
- `qwen3.5-4b`: upstream → **Bee/TurboQuant** (41.2 vs 27.7 tok/s, ~49% faster)
- `qwen3.5-9b`: upstream → **ik + hadamard** (Bee segfaults with UD-Q3_K_XL)
- `gemma4-e4b`: upstream → **ik + hadamard** (Bee was 36 vs 40 tok/s)
- `qwopus-coder-9b`: stays **ik + hadamard** (Bee variant in _disabled for testing)
- `webworld-8b`: upstream → **ik + hadamard** (ik v4524 fixed qwen3 segfault)
- `qwen3.6-35b-moe`: `--fit-margin 1536` → **768** (80-83% VRAM, was 67%)
- `gemma4-26b-moe`: `--fit-target 256` → **96** (more GPU layers)

**New flags applied across models:**
- `-khad`/`-vhad`: Hadamard KV cache transform (all ik models)
- `--defer-experts`: MoE deferred mmap (qwen3.6-35b-moe, qwopus-35b)
- `--flash-attn auto`: ik only, disables FA on short contexts
- `--jinja`: Required for Gemma 4 and Qwen with tools
- `--parallel-tool-calls`: Required for multi-tool calling (qwen3.5-9b, qwopus-coder-9b)

**TurboQuant benchmarks (BeeLlama.cpp `turbo3_tcq`):**
- qwen3.5-4b: 41.2 tok/s (Bee) vs 27.7 tok/s (ik) → **+49%**
- gemma4-e4b: 36 tok/s (Bee) vs 40 tok/s (ik) → ik wins here
- qwen3.5-9b: **CRASH** (Bee segfaults with UD-Q3_K_XL quantization)

**Gemma 4 MTP investigation:**
- MTP uses separate assistant/drafter model (not built-in like Qwen3.6)
- `Gemma4AssistantForCausalLM` GGUF conversion not yet in upstream llama.cpp
- RTX 3050 6GB: E4B + drafter = ~8.2GB VRAM → **doesn't fit**
- Conclusion: Gemma 4 MTP **not viable** on this hardware

### 2026-05-16 — MTP Benchmark Results, Gemma 4 MTP Assistants

**Binaries (unchanged from 2026-05-15):**
- `llama.cpp`: **b9174+13** (v528) — MTP PR #22673 merged
- `ik_llama.cpp`: **v4504** — MTP re-quant output tensor, Gemma 4 MTP
- `llama-swap`: **v214** (unchanged)

**MTP Benchmark Results on RTX 3050 (6GB VRAM):**

| Model | Baseline | With MTP | Notes |
|-------|----------|----------|-------|
| Gemma 4 E2B | **38.3 tok/s** | 10.6 tok/s | MTP overhead > speedup |
| Gemma 4 E4B | **27.5 tok/s** | OOM | Exceeds 6GB VRAM |
| Qwen 3.5 0.8B | 136 tok/s | OOM | GDN compute buffer blocks MTP |
| Qwen 3.5 4B | 25 tok/s | OOM | GDN compute buffer blocks MTP |
| Qwen 3.5 9B | 12 tok/s | OOM | GDN compute buffer blocks MTP |

**Conclusion: MTP is NOT beneficial for small dense models on RTX 3050 6GB.**