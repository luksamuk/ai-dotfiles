# Changelog

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