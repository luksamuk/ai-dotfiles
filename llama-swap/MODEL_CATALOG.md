# Model Catalog — llama-swap on RTX 3050 6GB

> **Incremental wiki** — last updated 2026-05-25  
> All models served by llama-swap with parameters, inference engine, and benchmark data.

---

## Hardware & Environment

| Item | Value |
|------|-------|
| **GPU** | NVIDIA RTX 3050 Laptop (6 GB VRAM) |
| **VRAM budget** | ~5.2 GB usable (after CUDA/driver overhead) |
| **Model storage** | `~/.llama-models/` |
| **Config repo** | `~/git/ai-dotfiles/llama-swap/` |
| **llama-swap** | v212+ required |

---

## Inference Engines

| Engine | Build | Binary | Strengths | Models |
|--------|-------|--------|-----------|--------|
| **llama.cpp** (upstream) | v674 `35c9b1f39` | `~/git/llama.cpp/build/bin/llama-server` | `--fit`/`--fit-target`, VLM/mmproj via libmtmd, MTP speculative decoding, best dense model support | Dense models, VLMs, LFM2.5 |
| **ik_llama.cpp** | v4542 `b4e1d916` | `~/git/ik_llama.cpp/build/bin/llama-server` | `--k-cache-hadamard`/`--v-cache-hadamard` (hadamard KV), `--defer-experts` (faster MoE load), `--flash-attn auto`, pinned memory for MoE expert offload | MoE models, dense models with hadamard KV |
| **BeeLlama.cpp** | v9459/v0.2.0 `07ac3cec6` | `~/git/beellama.cpp/build/bin/llama-server` | `turbo3_tcq`/`turbo4_tcq` KV cache (~5× compression), DFlash speculative decoding | TurboQuant KV cache models (no `--fit`, must use `--n-gpu-layers 99`) |

### Engine Selection Notes

- **ik_llama.cpp** is preferred for MoE models due to pinned memory and `--defer-experts` reducing load time.
- **BeeLlama.cpp** lacks `--fit`/`--fit-margin`; models must fit entirely in VRAM (`--n-gpu-layers 99`).
- **BeeLlama.cpp** `turbo3_tcq` crashes with MoE (256 experts) — bug reported in v0.1.2, may be fixed in v0.2.0.
- **ik_llama.cpp** segfaults with LittleLamb 0.3B (qwen3 arch) — use upstream only.
- **ik_llama.cpp** MTP does NOT work on 6 GB VRAM for dense models (SSM buffer or overhead causes OOM).

---

## Active Models

### 1. gemma4-26b-a4b — Gemma 4 26B A4B MoE

| Parameter | Value |
|-----------|-------|
| **Source** | Google Gemma 4 26B (Activations 4B, MoE) |
| **Quant** | APEX I-Compact (~15.5 GB) — mixed-precision MoE: edges Q4_K, middle Q3_K, shared Q6_K, attn Q4_K |
| **Backend** | `llama_server` (upstream v674) |
| **KV cache** | `q4_0` K + `q4_0` V + `attn_rot` (iSWA fix, b8815+) |
| **Flash attention** | `--flash-attn on` |
| **Context** | 16K–128K dynamic, `--fit on --fit-target 96 --fit-ctx 16384 --ctx-size 131072` |
| **Thinking** | ✅ Yes — dual mode (`:think` variant available) |
| **Tool calling** | ✅ Yes |
| **Vision** | ❌ DISABLED — mmproj crashes on CUDA (issue [#21402](https://github.com/ggml-org/llama.cpp/issues/21402)) |
| **Sampling** | `temp=1.0 / top-p=0.95` (think mode), `temp=0.7 / top-p=0.95` (chat mode) |
| **TTL** | 300s |
| **Known issues** | mmproj crash on CUDA; upstream backend used because mmproj crashes on CUDA even with ik |

---

### 2. gemma4-e2b — Gemma 4 E2B Dense

| Parameter | Value |
|-----------|-------|
| **Source** | Google Gemma 4 E2B (2B dense) |
| **Quant** | Q4_K_M (~2.89 GB) |
| **Backend** | `llama_server` (upstream v674) |
| **KV cache** | `${gemma_cache_k}`/`${gemma_cache_v}` = `q4_0`/`q4_0` + `attn_rot` (iSWA fix) |
| **Flash attention** | `--flash-attn on` |
| **Context** | 32K–128K dynamic, `--fit on --fit-target 1024` |
| **Thinking** | ✅ Yes — dual mode |
| **Tool calling** | ✅ Yes |
| **Vision** | ❌ DISABLED (`--no-mmproj` flag) |
| **Sampling** | `temp=0.7 / top-p=0.9` (chat mode), `temp=0.6 / top-p=0.9` (think mode) |
| **TTL** | 60s |
| **Known issues** | MTP dense OOM — Gemma4 E2B 3.6× slower with MTP on 6 GB; text-only |

---

### 3. gemma4-e4b — Gemma 4 E4B Dense

| Parameter | Value |
|-----------|-------|
| **Source** | Google Gemma 4 E4B (4B dense) |
| **Quant** | Q4_K_M (~4.63 GB) |
| **Backend** | `ik_llama_server` (ik v4542) |
| **KV cache** | `q8_0` K + `q4_0` V + hadamard (`-khad`/`-vhad`) |
| **Flash attention** | `--flash-attn auto` |
| **Context** | 64K–128K dynamic, `--fit --fit-margin 512` |
| **Thinking** | ✅ Yes — dual mode |
| **Tool calling** | ✅ Yes |
| **Vision** | ❌ DISABLED — mmproj crash on CUDA (#21402) |
| **Sampling** | `temp=0.7 / top-p=0.9` (chat mode), `temp=0.6 / top-p=0.9` (think mode) |
| **TTL** | 60s |
| **Known issues** | MTP dense OOM on 6 GB VRAM; mmproj crash; hadamard KV cache has ~0% tok/s impact on dense models with high quality benefit at q4_0 |

---

### 4. gpt-oss-20b — GPT-OSS 20B MoE

| Parameter | Value |
|-----------|-------|
| **Source** | GPT-OSS 20B (MoE architecture) |
| **Quant** | Q4_K_M (~10.8 GB) |
| **Backend** | `ik_llama_server` (ik v4542) |
| **KV cache** | `${large_cache_k}`/`${large_cache_v}` = `q4_0`/`q4_0` + hadamard + `attn_rot` |
| **Flash attention** | `--flash-attn auto` |
| **Context** | 16K–128K, `--fit --fit-margin 768 --defer-experts` |
| **Thinking** | ✅ ALWAYS ON — Harmony format (not optional) |
| **Tool calling** | ✅ Yes |
| **Vision** | ❌ No (not a VLM) |
| **Sampling** | `temp=0.6 / top-p=0.9` |
| **Batching** | `--ubatch-size 2048 --batch-size 2048` |
| **TTL** | 120s |
| **Known issues** | — |

---

### 5. hy-mt2-1.8b — Hunyuan MT2 1.8B Translation

| Parameter | Value |
|-----------|-------|
| **Source** | Tencent Hunyuan MT2 1.8B (translation model) |
| **Quant** | Q4_K_M (~1.1 GB) |
| **Backend** | `llama_server` (upstream v674) — ik doesn't support `hunyuan_v1_dense` |
| **KV cache** | `${small_cache_k}`/`${small_cache_v}` = `q8_0`/`q8_0` |
| **Flash attention** | `--flash-attn on` |
| **Context** | 32K–128K dynamic, `--fit on --fit-target 768` |
| **Thinking** | ❌ No |
| **Tool calling** | ❌ No |
| **Vision** | ❌ No |
| **Sampling** | `temp=0.7 / top-p=0.6 / top-k=20 / repeat-penalty=1.05` |
| **TTL** | 60s |
| **Known issues** | ik_llama.cpp does not support `hunyuan_v1_dense` architecture — must use upstream |

---

### 6. lfm2.5-1.2b — LFM2.5 1.2B Instruct Dense

| Parameter | Value |
|-----------|-------|
| **Source** | Liquid AI LFM2.5 1.2B Instruct (dense hybrid) |
| **Quant** | Q8_0 (~1.25 GB) |
| **Backend** | `llama_server` (upstream v674) |
| **KV cache** | `${small_cache_k}`/`${small_cache_v}` = `q8_0`/`q8_0` + `attn_rot` |
| **Flash attention** | `--flash-attn on` |
| **Context** | 32K–128K dynamic, `--fit on --fit-target 768` |
| **Thinking** | ❌ No |
| **Tool calling** | ✅ Yes — best LFM2 model for agentic tasks |
| **Vision** | ❌ No |
| **attn_rot** | ✅ `head_dim=64` → `n_embd_head % 64 == 0` |
| **Sampling** | `temp=0.3 / top-p=0.9 / top-k=40 / min-p=0.15 / repeat-penalty=1.05` |
| **TTL** | 60s |
| **Known issues** | — |

---

### 7. lfm2.5-vl-450m — LFM2.5-VL 450M Vision

| Parameter | Value |
|-----------|-------|
| **Source** | Liquid AI LFM2.5-VL 450M (vision-language model) |
| **Quant** | Q8_0 (~0.22 GB model + ~0.18 GB mmproj) |
| **Backend** | `llama_server` (upstream v674) |
| **KV cache** | `f16` — vision model, precision priority |
| **Flash attention** | `--flash-attn on` |
| **Context** | 32K–128K dynamic, `--fit on --fit-target ${vram_margin}` |
| **Thinking** | ❌ No (`--reasoning off`) |
| **Tool calling** | ❌ No |
| **Vision** | ✅ Yes — mmproj included |
| **Sampling** | `temp=0.1 / top-p=0.9 / top-k=40 / min-p=0.15` |
| **TTL** | 60s |
| **Known issues** | `attn_rot` not applicable (vision model uses f16 cache) |

---

### 8. littlelamb-0.3b-tc — LittleLamb 0.3B Tool Calling

| Parameter | Value |
|-----------|-------|
| **Source** | LittleLamb 0.3B (Qwen3-style JSON tool calling) |
| **Quant** | Q8_0 (~303 MB) |
| **Backend** | `llama_server` (upstream v674) — ik segfaults with this model |
| **KV cache** | `${small_cache_k}`/`${small_cache_v}` = `q8_0`/`q8_0` |
| **Flash attention** | `--flash-attn on` |
| **Context** | 8K–40K dynamic, `--fit on --fit-target 768 --parallel 4 --cont-batching` |
| **Thinking** | ✅ Yes — dual mode |
| **Tool calling** | ✅ Yes — Qwen3-style JSON, BFCL v4 51.5% |
| **Vision** | ❌ No |
| **Sampling** | Uses `${code_temp}`/`${code_top_p}`/`${code_top_k}`/`${code_min_p}`/`${code_repeat_penalty}` macros |
| **TTL** | 60s |
| **Known issues** | ik_llama segfaults with this model (qwen3 arch) — upstream only |

---

### 9. minicpm-v-4.6 — MiniCPM-V 4.6 VLM

| Parameter | Value |
|-----------|-------|
| **Source** | MiniCPM-V 4.6 (vision-language model) |
| **Quant** | Q5_K_M (~552 MB model + ~1.1 GB mmproj) |
| **Backend** | `llama_server` (upstream v674) — mmproj uses libmtmd |
| **KV cache** | `f16` — vision model, precision priority |
| **Flash attention** | `--flash-attn on` |
| **Context** | 8K–256K dynamic, `--fit on --fit-ctx 8192 --parallel 4 --cont-batching` |
| **Thinking** | ✅ Yes — dual mode |
| **Tool calling** | ❌ No |
| **Vision** | ✅ Yes — mmproj included |
| **Sampling** | `temp=${default_temp} / top-p=0.8 / top-k=100 / min-p=${default_min_p} / repeat-penalty=1.05` |
| **TTL** | 60s |
| **Known issues** | MiniCPM5-1B tool calling broken (llama.cpp autoparser TAG_WITH_TAGGED boundary bug) — this is MiniCPM-V, not affected |

---

### 10. qwen3.5-4b — Qwen3.5 4B Dense

| Parameter | Value |
|-----------|-------|
| **Source** | Qwen3.5 4B (dense) |
| **Quant** | i1-Q4_K_M (~2.52 GB) |
| **Backend** | `bee_server` (BeeLlama v9459/v0.2.0) — TurboQuant KV cache |
| **KV cache** | `turbo3_tcq` K+V (~5× compression) |
| **Flash attention** | `--flash-attn on` |
| **Context** | 131072 (fixed ctx, `--n-gpu-layers 99`) |
| **Thinking** | ✅ Yes — dual mode |
| **Tool calling** | ✅ Yes |
| **Vision** | ❌ No (`--no-mmproj`) |
| **Sampling** | `temp=${default_temp} / top-p=${default_top_p} / top-k=${default_top_k} / min-p=${default_min_p}` |
| **TTL** | 60s |
| **Benchmarks** | BeeLlama turbo4 on Qwen3.5-4B: **+29–35% speedup** at 8–16K context, **~2×** at 32K+ |
| **Known issues** | BeeLlama lacks `--fit`; `turbo3_tcq` crashes with MoE models |

---

### 11. qwen3.5-9b — Qwen3.5 9B Dense

| Parameter | Value |
|-----------|-------|
| **Source** | Qwen3.5 9B (dense) |
| **Quant** | UD-Q4_K_XL (~5.55 GB) |
| **Backend** | `ik_llama_server` (ik v4542) — BeeLlama segfaults with this model |
| **KV cache** | `q8_0` K + `q4_0` V + hadamard (`-khad`/`-vhad`) |
| **Flash attention** | `--flash-attn auto` |
| **Context** | Dynamic, `--fit --fit-margin 512` |
| **Thinking** | ✅ Yes — dual mode |
| **Tool calling** | ✅ Yes (`--parallel-tool-calls`) |
| **Vision** | ❌ No — mmproj exists but blocks offload |
| **Sampling** | `temp=${default_temp} / top-p=${default_top_p} / top-k=${default_top_k} / min-p=${default_min_p}` |
| **TTL** | 60s |
| **Known issues** | BeeLlama segfaults with this model; mmproj blocks partial offload so `--no-mmproj` required |

---

### 12. qwen3.6-35b-a3b — Qwen3.6 35B A3B MoE

| Parameter | Value |
|-----------|-------|
| **Source** | Qwen3.6 35B (Activations 3B, MoE) |
| **Quant** | APEX I-Compact (~17.3 GB) — mixed-precision, MoE offload to RAM |
| **Backend** | `ik_llama_server` (ik v4542) — better MoE performance via pinned memory |
| **KV cache** | `${large_cache_k}`/`${large_cache_v}` = `q4_0`/`q4_0` + hadamard |
| **Flash attention** | `--flash-attn auto` |
| **Context** | Dynamic, `--fit --fit-margin 768 --defer-experts` |
| **Thinking** | ✅ Yes — dual mode (`:think` variant) |
| **Tool calling** | ✅ Yes (`--parallel-tool-calls`) |
| **Vision** | ❌ No (`--no-mmproj` — mmproj blocks partial expert offload) |
| **Sampling** | `temp=${code_temp} / top-p=${code_top_p}` (coding-optimized: 0.6/0.85) |
| **TTL** | 300s |
| **Known issues** | Very large model; relies heavily on `--defer-experts` for load time; mmproj would block partial expert offload |

---

### 13. qwopus-coder-9b — Qwopus 3.5-9B Coder

| Parameter | Value |
|-----------|-------|
| **Source** | Qwopus 3.5-9B Coder (specialized agentic coding fine-tune) |
| **Quant** | Q4_K_M (~5.63 GB) |
| **Backend** | `ik_llama_server` (ik v4542) |
| **KV cache** | `fit/fit-margin 512` + `--jinja` + `--parallel-tool-calls` |
| **Flash attention** | (ik default, auto) |
| **Context** | 131072, `--fit --fit-margin 512` |
| **Thinking** | ✅ Yes |
| **Tool calling** | ✅ Yes — specialized for agentic coding + tool calling |
| **Vision** | ❌ No — mmproj blocks partial expert offload |
| **Sampling** | `temp=${code_temp} / top-p=${code_top_p}` (coding-optimized: 0.6/0.85) |
| **TTL** | Not specified |

---

### 14. translategemma-4b — TranslateGemma 4B

| Parameter | Value |
|-----------|-------|
| **Source** | Google TranslateGemma 4B (translation model) |
| **Quant** | Q4_K_M (~2.7 GB) |
| **Backend** | `llama_server` (upstream v674) |
| **KV cache** | Default cache (not overridden) |
| **Flash attention** | (not specified, likely off) |
| **Context** | 32K–128K dynamic, `--fit on --fit-target 768` |
| **Template** | `--no-jinja --chat-template gemma` |
| **Thinking** | ❌ No |
| **Tool calling** | ❌ No |
| **Vision** | ❌ No |
| **Sampling** | `temp=0.2` |
| **TTL** | 60s |
| **Known issues** | Uses Gemma chat template directly (no jinja) |

---

### 15. webworld-8b — WebWorld 8B

| Parameter | Value |
|-----------|-------|
| **Source** | WebWorld 8B (world model, NOT a chatbot) |
| **Quant** | i1-Q5_K_M |
| **Backend** | `ik_llama_server` (ik v4542) |
| **KV cache** | Default (not overridden) |
| **Context** | 40960, `--fit --fit-margin 768` |
| **Batching** | `--parallel 2 --cont-batching` |
| **Thinking** | ❌ No |
| **Tool calling** | ❌ No — NOT a chatbot, world model |
| **Vision** | ❌ No |
| **Sampling** | `temp=0.6 / top-p=0.9` |
| **TTL** | Not specified |
| **Known issues** | — |

---

## Disabled Models

Configs preserved in `models/_disabled/` (GGUF deleted, can be re-downloaded):

| Config | Description | Reason Disabled |
|--------|-------------|-----------------|
| `gemma4-e4b-bee.yaml` | Gemma 4 E4B on BeeLlama backend | Superseded by ik_llama backend |
| `granite-3.3-8b-vllm.yaml` | Granite 3.3 8B on vLLM | vLLM OOM on 6 GB VRAM |
| `granite-4.0-h-1b*.yaml` | Granite 4.0 1B variants | — |
| `hunyuan-7b.yaml` | Hunyuan 7B | — |
| `lfm2.5-1.2b-think.yaml` | LFM2.5 1.2B with thinking | Merged into main config? |
| `lfm2.5-1.2b-vllm.yaml` | LFM2.5 1.2B on vLLM | vLLM OOM on 6 GB VRAM |
| `lfm2.5-sgl.yaml` | LFM2.5 on SGLang | SGLang OOM on 6 GB VRAM |
| `ministral-3-3b.yaml` | Ministral 3B | — |
| `qwen3.5-0.8b*.yaml` | Qwen3.5 0.8B variants | — |
| `qwen3.5-9b-bee.yaml` | Qwen3.5 9B on BeeLlama | BeeLlama segfaults with this model |
| `qwen3-8b.yaml` | Qwen3 8B | — |
| `qwopus-35b.yaml` | Qwopus 35B | — |
| `qwopus-coder-9b-bee.yaml` | Qwopus Coder 9B on BeeLlama | Alternative backend config |
| `qwopus-coder-9b-ik.yaml` | Qwopus Coder 9B on ik (alt config) | Alternative config, merged |
| `smolllm3-3b.yaml` | SmolLM3 3B | — |

### Notable Disabled Backend Configs

- **`qwen3.5-4b-upstream.yaml`**: Original upstream llama.cpp config with `q8_0` cache + `--fit` → replaced by BeeLlama turbo3_tcq backend
- **`qwen3.5-9b-upstream.yaml`**: Original upstream llama.cpp config with `q4_0` cache + `--fit` → replaced by ik_llama hadamard backend

---

## Removed Models

Configs in `models/_removed/` (dead code):

| Config | Description | Reason Removed |
|--------|-------------|----------------|
| `ds-r1-distill-14b-32b.yaml` | DeepSeek R1 distill 14B/32B | Too large for 6 GB VRAM |

---

## Known Issues

| Issue | Affected Models | Details |
|-------|-----------------|---------|
| **Gemma 4 mmproj CUDA crash** | gemma4-26b-a4b, gemma4-e2b, gemma4-e4b | llama.cpp issue [#21402](https://github.com/ggml-org/llama.cpp/issues/21402) — all Gemma 4 models are text-only |
| **ik_llama segfault with LittleLamb 0.3B** | littlelamb-0.3b-tc | qwen3 arch incompatibility — use upstream only |
| **ik_llama MTP OOM on 6 GB for dense** | Dense models with MTP | SSM buffer or overhead causes OOM — MTP disabled for dense models |
| **BeeLlama turbo3_tcq MoE crash** | MoE models with 256 experts | Bug in Bee v0.1.2, may be fixed in v0.2.0 — avoid turbo3_tcq with MoE |
| **MiniCPM5-1B tool calling broken** | (not in active roster) | llama.cpp autoparser `TAG_WITH_TAGGED` boundary bug |
| **SGLang/vLLM OOM** | Any SGLang or vLLM backend | Both backends OOM on 6 GB VRAM — not viable |
| **Gemma4 iSWA attn_rot** (FIXED) | gemma4-26b-a4b, gemma4-e2b, gemma4-e4b | Fixed in commit `4eb19514d` (build b8815+) — `q4_0` cache now works correctly |
| **BeeLlama lacks --fit** | qwen3.5-4b | Must use `--n-gpu-layers 99` instead; cannot use dynamic VRAM fitting |

---

## Macro Reference

Key macros from `config-base.yaml`:

| Macro | Value | Purpose |
|-------|-------|---------|
| `llama_server` | `~/git/llama.cpp/build/bin/llama-server` | Upstream llama.cpp v674 |
| `ik_llama_server` | `~/git/ik_llama.cpp/build/bin/llama-server` | ik_llama.cpp v4542 |
| `bee_server` | `~/git/beellama.cpp/build/bin/llama-server` | BeeLlama v9459/v0.2.0 |
| `models_dir` | `~/.llama-models` | GGUF storage |
| `media_path` | `~/testfiles/vision` | Vision test media |
| `small_cache_k` / `v` | `q8_0` / `q8_0` | Small model KV cache (~47% vs f16) |
| `large_cache_k` / `v` | `q4_0` / `q4_0` | Large model KV cache (~72% vs f16) |
| `gemma_cache_k` / `v` | `q4_0` / `q4_0` | Gemma-specific KV cache (iSWA fixed) |
| `code_temp` | `0.6` | Coding temperature |
| `code_top_p` | `0.85` | Coding top-p |
| `code_top_k` | `40` | Coding top-k |
| `code_min_p` | `0.02` | Coding min-p |
| `code_repeat_penalty` | `1.05` | Coding repeat penalty |
| `default_temp` | `0.7` | General temperature |
| `default_top_p` | `0.9` | General top-p |
| `default_top_k` | `20` | General top-k |
| `default_min_p` | `0.01` | General min-p |
| `default_repeat_penalty` | `1.05` | General repeat penalty |
| `think_temp` | `0.6` | Thinking temperature |
| `think_top_p` | `0.9` | Thinking top-p |
| `vram_margin` | `512` | VRAM safety margin (MiB) |

---

## Benchmark Data

| Model | Metric | Value | Notes |
|-------|--------|-------|-------|
| MiniCPM5-1B | tok/s (no-think) | ~106 | (Disabled, reference only) |
| MiniCPM5-1B | tok/s (think) | ~127 | (Disabled, reference only) |
| MTP dense (6 GB) | Qwen3.5 E2B | OOM | MTP not viable for dense on 6 GB |
| MTP dense (6 GB) | Gemma4 E2B | 3.6× slower | Worse than no MTP |
| MTP dense (6 GB) | Gemma4 E4B | OOM | MTP not viable |
| BeeLlama turbo4 | Qwen3.5-4B @ 8–16K | +29–35% speedup | vs standard q8_0 cache |
| BeeLlama turbo4 | Qwen3.5-4B @ 32K+ | ~2× speedup | vs standard q8_0 cache |
| BeeLlama turbo3_tcq | MoE (256 experts) | CRASH | Bug in v0.1.2, may be fixed v0.2.0 |
| Hadamard KV | Dense models | ~0% tok/s impact | High quality benefit at q4_0 |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-25 | Initial catalog created — 15 active models, 3 inference engines, benchmark data |