# MTP (Multi-Token Prediction) Notes

## What is MTP?

Multi-Token Prediction (MTP) is a speculative decoding technique where the model
predicts multiple tokens per forward pass using a dedicated "draft head" trained
alongside the main model. This can improve generation throughput by accepting
draft tokens in parallel.

As of llama.cpp b9174+ (PR #22673), MTP is supported via `--spec-type draft-mtp`.

## Supported Models

| Model | MTP Support | Notes |
|-------|-------------|-------|
| Qwen3.5/3.6 dense | ✅ (upstream b9174+) | Qwen3.5-0.8B, 2B, 4B, 9B, 27B |
| Qwen3.5/3.6 MoE | ✅ (ik v4496+) | Qwen3.6-35B-A3B, Qwopus |
| Gemma 4 dense/MoE | ✅ (ik v4496+) | Gemma 4 E2B, E4B, 26B MoE, 31B. Uses external assistant GGUF via `--model-draft` |
| GLM-4.7 Flash | ❌ (upstream only, no graph) | Tensors load but MTP graph not implemented |

## RTX 3050 6GB Limitation: Qwen 3.5 Dense + MTP

**MTP does NOT work with Qwen 3.5 dense models on the RTX 3050 (6GB VRAM).**

Qwen 3.5 uses a hybrid SSM+attention architecture (Gated Delta Net) that requires
a large fixed compute buffer (~1.3 GB) for the recurrent state, independent of
context length. When MTP is enabled, the draft head needs its own copy of this
state, roughly doubling the memory requirement.

This makes it impossible to fit even the smallest Qwen 3.5 (0.8B) with MTP
on 6GB VRAM — the GDN buffer alone exceeds available memory after model loading.

**For Qwen 3.5 dense models, continue using them without `--spec-type draft-mtp`.**

The Qwen 3.6 **MoE** models (35B-A3B) work fine with MTP because they use ik_llama.cpp
with pinned memory for expert offloading, which handles the memory pressure differently.

## RTX 3050 6GB Limitation: Gemma 4 Dense + MTP

**MTP is NOT beneficial for Gemma 4 dense models on the RTX 3050 (6GB VRAM).**

Gemma 4 MTP uses a SEPARATE external assistant model (not built into the main GGUF like
Qwen 3.5/3.6). This requires loading the assistant (~75MB) alongside the main model via
`--model-draft` on ik_llama.cpp.

Test results (2025-05-16):

| Model | Without MTP | With MTP | Notes |
|-------|-------------|----------|-------|
| Gemma 4 E2B | 38.3 tok/s | 10.6 tok/s | MTP overhead > speedup |
| Gemma 4 E4B | 27.5 tok/s | OOM | Exceeds 6GB VRAM |

For small dense models that already run fast, the MTP verification overhead exceeds any
draft acceptance gains. The assistant head adds compute that the 3050 can't offset.

**For Gemma 4 dense models, continue using them without MTP speculative decoding.**

The Gemma 4 26B MoE might benefit from MTP on higher-VRAM GPUs, but not on the 3050.

## MTP Flags Reference (llama.cpp upstream b9174+)

### Basic Usage
```bash
--spec-type draft-mtp                    # Enable MTP speculative decoding
```

### Advanced Flags
```bash
--spec-draft-n-max N                    # Max tokens to draft (default: 16)
--spec-draft-n-min N                    # Min tokens to draft (default: 0)
--spec-draft-p-min P                    # Min probability for greedy draft (default: 0.75)
--spec-draft-p-split P                  # Split probability (default: 0.10)
--spec-draft-type-k TYPE                # KV cache type for draft head (default: f16)
--spec-draft-type-v TYPE                # V cache type for draft head (default: f16)
```

## ik_llama.cpp MTP Flags

ik_llama.cpp supports two MTP modes:

### Built-in MTP (Qwen 3.5/3.6 style)
```bash
--multi-token-prediction               # Enable MTP for Qwen-style built-in heads
--mtp                                   # ❌ CAUSES EXIT CODE 1 in llama-server
```

### External MTP Assistant (Gemma 4 style)
```bash
--model-draft <path>                   # Load external MTP assistant GGUF
--spec-type mtp                        # Enable MTP speculative decoding
--draft-max N                          # Max draft tokens (default: 16)
--mtp-requantize-output-tensor TYPE    # Requantize MTP output tensor (e.g., q4_0)
```

Gemma 4 assistants are tiny (~75MB Q4_K_M) and converted from Google's
`google/gemma-4-{E2B,E4B}-it-assistant` safetensors using ik_llama.cpp's
`convert_hf_to_gguf.py` (arch: `Gemma4AssistantForCausalLM` → `gemma4_mtp`).

## Unsloth MTP GGUF Variants

Unsloth provides MTP-trained GGUF files at:
- `unsloth/Qwen3.5-{0.8B,2B,4B,9B,27B,35B-A3B}-MTP-GGUF`
- Available in all standard quantizations (Q3_K_M through Q8_0 and UD variants)

These files include the MTP draft head weights. They can be used without `--spec-type draft-mtp`
as regular models (MTP tensors are loaded but ignored).

## Build Versions

| Binary | Version | MTP Support |
|--------|---------|-------------|
| llama.cpp | b9174+ (v528) | `--spec-type draft-mtp` |
| ik_llama.cpp | v4496+ | `--multi-token-prediction` |
| llama.cpp (pre-b9174) | none | ❌ No MTP |