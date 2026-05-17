# Changelog

### 2026-05-16 — MTP Benchmark Results, Gemma 4 MTP Assistants

**Binaries (unchanged from 2026-05-15):**
- `llama.cpp`: **b9174+13** (v528) — MTP PR #22673 merged
- `ik_llama.cpp`: **v4504** — MTP re-quant output tensor, Gemma 4 MTP
- `llama-swap`: **v214** (unchanged)

**Gemma 4 MTP assistant GGUFs created:**
- `gemma-4-E2B-it-assistant-Q4_K_M.gguf` (75 MB) — converted from `google/gemma-4-E2B-it-assistant`
- `gemma-4-E4B-it-assistant-Q4_K_M.gguf` (75 MB) — converted from `google/gemma-4-E4B-it-assistant`
- Both converted using ik_llama.cpp's `convert_hf_to_gguf.py` (arch: `Gemma4AssistantForCausalLM` → `gemma4_mtp`)
- F16 originals also available (165-166 MB)

**MTP Benchmark Results on RTX 3050 (6GB VRAM):**

| Model | Baseline | With MTP | Notes |
|-------|----------|----------|-------|
| Gemma 4 E2B | **38.3 tok/s** | 10.6 tok/s | MTP overhead > speedup |
| Gemma 4 E4B | **27.5 tok/s** | OOM | Exceeds 6GB VRAM |
| Qwen 3.5 0.8B | 136 tok/s | OOM | GDN compute buffer blocks MTP |
| Qwen 3.5 4B | 25 tok/s | OOM | GDN compute buffer blocks MTP |
| Qwen 3.5 9B | 12 tok/s | OOM | GDN compute buffer blocks MTP |

**Conclusion: MTP is NOT beneficial for small dense models on RTX 3050 6GB.** The
speculative decoding verification overhead exceeds any draft acceptance gains on models
that already run fast. Only the Qwen 3.6 MoE (with ik_llama.cpp pinned memory offload)
benefits from MTP on this hardware.

See [MTP-NOTES.md](MTP-NOTES.md) for detailed MTP configuration and architecture notes.

### 2026-05-14 — llama-swap v212, Binary Rebuilds, BeeLlama Monitoring

**Binaries upgraded:**
- `llama.cpp`: b9124 → **b9158+** (upstream, version 500)
- `ik_llama.cpp`: v4486 → **v4496** (Iwan Kawrakow's fork)
- `llama-swap`: v211 → **v212**
- `vLLM`: 0.20.2 (unchanged)

**What's new in llama.cpp b9124 → b9158+:**
- **Qwen3.5 tokenizer fix** — Prevents stack overflow on long Qwen3.5 prompts.
- **WebGPU gpt-oss support** — Can now run gpt-oss-20b via WebGPU backend.
- **Server: /v1/models modalities** — Models endpoint now exposes vision/audio/text capabilities.
- **NCCL-free Tensor Parallelism** (build b9095) — Dual GPU without NCCL.
- **attn-rot PR #21038** — Gerganov opened PR for Hadamard rotation before KV quantization (~80% of TurboQuant benefit). **NOT YET MERGED**.
- **MTP PR #22673** — Still open/draft. Not in mainline yet.
- **TurboQuant/TCQ** — CPU-only PR #21089 pending. CUDA versions only in community forks.
- **BeeLlama.cpp** (github.com/Anbeeld/beellama.cpp, v0.1.2) — Performance fork combining DFlash speculative decoding, TurboQuant/TCQ KV cache compression, adaptive draft, and reasoning-loop protection. **Monitoring only** — not adopted yet. TurboQuant KV cache is the killer feature for 6GB VRAM (4-5× more context). `turbo4` ≈ lossless, `turbo3_tcq` achieves PPL lower than FP16. Waiting for upstream merge or stabilization.

**What's new in ik_llama.cpp v4486 → v4496:**
- **Gemma 4 MTP KV fix** (PR #1786) — MTP avoids casting KV cache to f32, saving VRAM on Gemma 4 models.
- **Gemma 4 full tensor mapping + imatrix** (PR #1796) — Complete model support with imatrix for better quantization.
- **`--threads-mtmd`** (PR #1797) — Independent thread count for multimodal processing. ik-only flag (upstream doesn't have this).
- **MTP faster recurrent state restore** (PR #1791) — Speed improvement for MTP on recurrent models.
- **mmproj: inflate n_batch only for GPU-offloaded** (PR #1788) — CPU mmproj no longer gets inflated batch size.
- **Cache tokens reset fix** (PR #1787) — Server resets cache tokens after prompt processing stops.
- **Hadamard KV/V-cache transforms** (PRs #1033/#1034/#1527) — ik's answer to TurboQuant rotation, already available.
- **Low perplexity Q4_0 KV cache** (PRs #1547/#1556) — Alternative to Hadamard approach.

**What's new in llama-swap v211 → v212:**
- **Prometheus metrics** — Performance monitoring endpoint.
- **Fix: data race in `/running` endpoint** — Race condition during model status queries.
- **Fix: ignore LACT devices with zero VRAM** — Prevents errors on systems with virtual displays.
- **v208: `reasoning_content` in UI** — Shows thinking/reasoning in the web interface.
- **v205: SIGHUP config reload** — No restart needed for config changes.
- **v203: zstd compression for captures + race condition fix during swap**.

**Config changes:**
- Updated build version comments in `config.yaml` header (b9158+, v4496+, v212+).
- `--threads-mtmd` available in ik_llama.cpp but **not used** in current config — vision models use upstream binary which doesn't have this flag.

**What's new in 2025-05-16 — MTP Upstream, Build Upgrades, Qwen 3.5 MTP Investigation**

**Binaries upgraded:**
- `llama.cpp`: b9158 → **b9174+13** (v528, upstream)
- `ik_llama.cpp`: v4503 → **v4504** (Iwan Kawrakow's fork)
- `llama-swap`: v214 (unchanged)

**What's new in llama.cpp b9158 → b9174+:**
- **MTP Support** (PR #22673) — `--spec-type draft-mtp` for Qwen3.5/3.6 dense models.
  Supports Qwen3.5/3.6 dense and MoE, plus Qwen3.5-MoE in ik_llama.cpp.
  See [MTP-NOTES.md](MTP-NOTES.md) for details and RTX 3050 limitations.
- **New spec-types**: `draft-mtp`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`
- **Draft model support**: `--spec-draft-model`, `--spec-draft-type-k/v` for separate draft model cache types
- **GDN partial rollback** (for speculative decoding with Gated Delta Net models)
- **Various**: Qwen3 ASR conversion fix, Codex CLI support, AIME 2026 dataset, UI timeout for MCP tools

**What's new in ik_llama.cpp v4503 → v4504:**
- **MTP re-quantized output tensor** (PR #1809) — Better TG performance with MTP enabled
- imatrix fix: use data for ffn_up when data for ffn_gate is missing (PR #1806)
- **Dual speculative decoding** (PR #1789) — Combine draft model + ngram/MTP

**Qwen 3.5 Dense MTP Investigation:**
- Downloaded and tested Unsloth MTP GGUF variants for Qwen 3.5 (0.8B, 4B, 9B)
- **Result: MTP does NOT work on RTX 3050 6GB** — Gated Delta Net requires ~1.3GB
  compute buffer per context, and MTP doubles this for the draft head
- Even the 0.8B model fails with OOM when `--spec-type draft-mtp` is enabled
- Qwen 3.5 dense models remain on non-MTP UD-Q3_K_XL variants
- Qwen 3.6 MoE (ik_llama.cpp) continues to work with MTP as before
- See [MTP-NOTES.md](MTP-NOTES.md) for full details

**Benchmark (Qwen 3.5 dense, non-MTP, hot cache):**

| Model | Type | Backend | Prompt t/s | Decode t/s |
|-------|------|---------|------------|------------|
| qwen3.5-0.8b | Dense Q3 | upstream | 1356 | **136** |
| qwen3.5-4b | Dense Q3 | upstream | 62 | **25** |
| qwen3.5-9b | Dense Q3 | upstream | 23 | **12** |

> Second request (model pre-loaded). MTP variants failed with OOM on 6GB VRAM.

**Previous: 2025-05-15 — vLLM 0.21, LFM2 Tool Parser, Qwopus**

**Binaries upgraded:**
- `llama.cpp`: b9066 → **b9124** (upstream)
- `ik_llama.cpp`: v4481 → **v4486** (Iwan Kawrakow's fork)
- `vLLM`: 0.20.1 → **0.20.2**
- `llama-swap`: v211 (unchanged, current)

**What's new in llama.cpp b9066 → b9124:**
- **Speculative Checkpointing** (PR #19493) — Saves/restores KV state during speculative decoding, reducing VRAM usage by up to 40% and boosting throughput by up to 20%. Useful with draft-model setups.
- **Gemma 4 KV Cache Fix** (PR #21534) — Fixes over-allocation of VRAM for Gemma 4 models that use shared KV layers. All Gemma 4 variants now use less memory.
- **Flash Attention enabled by default** — `-fa` is now on by default in recent builds. RTX 3050 (Ampere, compute 8.6) supports it.
- **Qwen3Next graph optimization** (PR #19375) — Reduces redundant copy operations.
- **NCCL-free Tensor Parallelism** (build b9095) — Dual GPU without NCCL library.
- **TurboQuant** (PR #21089) — **NOT YET MERGED**. Game-changer for 6GB VRAM: compresses KV cache from 16-bit to 2–4 bits (~4.5× compression). Will enable 4–5× longer contexts once merged.

**What's new in ik_llama.cpp v4481 → v4486:**
- **MTP for Gemma 4** (PR #1744) — Multi-Token Prediction native support for Gemma 4 models.
- **MTP for Qwen3.5-MoE** (PR #1745) — MTP tail layer support for Qwen3.5/Qwen3.6 MoE models.
- **MTP improvements** — Multiple PRs optimizing MTP: async copies for recurrent state, AVX2 greedy speculative sampling, fix crashes in speculative decoding, avoid per-step SSM copy.
- **Gemma4 partial offload fix** (PR #1657) — Fixed Gemma 4 not fitting correctly with partial offload.
- **Gemma4 MoE better routing** (PRs #1610, #1615) — Fused ops and optimized routing for Gemma4-MoE.
- **Expiring Logit Bias** (PR #1731) — New feature for bias that expires automatically.

**What's new in vLLM 0.20.1 → 0.20.2:**
- Bug fixes for DeepSeek V4, Qwen3-VL, and gpt-oss.
- **Breaking:** `reasoning_content` message field removed (deprecated in 0.20.0).
- **Breaking:** BitBlas and Marlin 24-bit quantization removed.
- AWQ, GPTQ-Marlin, GGUF, and FP8 quantization remain available.

**Config changes:**
- Added `--multi-token-prediction` flag to `qwen3.6-35b-moe` and `qwen3.6-35b-qwopus` (ik_llama.cpp MoE models).
- ⚠️ **Pitfall:** Use `--multi-token-prediction` (long form only). The `--mtp` shorthand causes **exit code 1** in `llama-server`. The CLI `llama-cli` accepts it, but the server binary does not.
- Updated build version comments in `config.yaml` header.

**Test results (post-upgrade, models pre-loaded / hot):**

| Model | Backend | Prompt t/s | Gen t/s | Cold Start | Notes |
|-------|---------|-----------|---------|------------|-------|
| lfm2.5-vl-450m | upstream | 773 | **244** | ~4s | Fastest generate in fleet |
| qwen3.5-0.8b | upstream | 1113 | **126** | ~5s | Smallest Qwen |
| lfm2.5-1.2b | upstream | 1646 | **107** | ~2s | Fast text model |
| lfm2.5-1.2b-think | upstream | 1635 | **107** | ~4s | Thinking variant |
| gemma4-e2b | upstream | 677 | **63** | ~8s | |
| lfm2-24b | ik | 118 | **43** | ~10s | MoE, offloaded experts |
| gemma4-e4b | upstream | 407 | **39** | ~5s | Best small dense |
| qwen3.6-35b-moe | ik + MTP | 95 | **27** | ~12s | MoE, MTP enabled |
| qwen3.5-4b | upstream | 58 | **24** | ~10s | |
| gemma4-26b-moe | upstream | 86 | **24** | ~18s | MoE, 128 experts |
| qwen3.5-9b | upstream | 23 | **12** | ~15s | Largest dense fit |

> Method: prompt "Comer miojo de galinha caipira na sexta-feira santa é pecado?", 200 max tokens, temp 0.7.
> Second request (model pre-loaded in VRAM). Prompt t/s from llama-server timings.
> gpt-oss-20b: template parsing error with Portuguese diacritics — known quirk, not a regression.
| qwen3.6-35b-qwopus | ik + MTP | 91.2 | 40.4 | ~12s (warm) |
| qwen3.5-4b | upstream | — | — | ~6s |
| qwen3.5-9b | upstream | 26.0 | 15.3 | ~8s |
| gemma4-e4b | upstream | 138.5 | 49.2 | ~6s |
| gemma4-26b-moe | upstream | 86.5 | 31.0 | ~16s |