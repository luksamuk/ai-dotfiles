# Testing K2-Horizon-MoVA-36B-A4B on ik_llama.cpp — findings from an RTX 3050 6GB (36B MoE, 4B active)

Reproducing @kassane's setup on a smaller card: RTX 3050 6GB VRAM, 31GB RAM. Same model file (abenzerps Q3_K_M, SHA256 verified), same chat template (abenzerps `chat_template.jinja`), same arch branch (kassane's `k2-horizon`), fork point `fe215a8c` of ikawrakow/ik_llama.cpp.

All findings below are reproducible; deterministic tests ran at temperature 0. The tokenizer patch is attached to the issue reply (`k2-horizon-vocab-fix.patch`).

---

## 1. Bug: merged-out tokenizer regex (solar-open body lost inside k2-horizon branch)

In `src/llama-vocab.cpp` (branch `k2-horizon`, as of `00adc77`):

```cpp
} else if (tokenizer_pre == "solar-open") {
    // empty body — solar-open falls through to "unknown pre-tokenizer" error
} else if (tokenizer_pre == "k2-horizon") {
    pre_type = LLAMA_VOCAB_PRE_TYPE_K2HORIZON;
    clean_spaces = false;
    pre_type = LLAMA_PRE_TYPE_SOLAR_OPEN;   // ← overwrites K2HORIZON
    clean_spaces = false;
}
```

**Effect:** the custom K2 regex (digit grouping 1-3 digits, ZWJ/combining-mark handling) is dead code — every k2-horizon model tokenizes with the solar-open regex, which splits digits one-by-one. Quality divergence vs the official IFM tokenizer on numbers/unicode.

**Fix (tested, deterministic count 1..50 exact after fix):** attached `k2-horizon-vocab-fix.patch`. One-line explanation: restore solar-open's body and let the k2-horizon branch set only its own pre_type.

---

## 2. K-cache Hadamard corrupts k2-horizon output (deterministic)

With `--cache-type-k q4_0 --cache-type-v q4_0` + `--k-cache-hadamard` (+/− `--v-cache-hadamard`), the model emits glitched tokens **deterministically at temp 0**. Prompt: "Count from 1 to 50, one number per line" → `33 33 ... 36 36 ... 44 44` style duplication glitches.

Matrix (ctx 8192, temp 0, same prompt every row):

| Config | Result |
|---|---|
| K+V hadamard | glitched ("33 33", "36 36", "44 44") |
| no hadamard | exact 1..50 |
| K only | glitched ("44 44") |
| V only | exact 1..50 (deterministic) |

**Attribution:** corruption is 100% K-side. The V path — which is exactly where MoVA lives (routed value-experts → V-cache → read-back de-transform) — handles hadamard + de-transform correctly end-to-end. The K side loses its de-transform in the `llm_build_kv` path that k2-horizon is forced to use (the comment in `build_k2horizon.cpp` itself says quantized K @ head_dim 128 is unsupported in IQK FA). So this is a **path-coverage bug**, not an architectural conflict with MoVA. Fix options: wire the K de-transform into the `llm_build_kv` path, or gate `--k-cache-hadamard` off for k2-horizon.

(FYI: V-only measured 21.7 t/s vs 20.4 baseline — possible small gain, not adopted.)

---

## 3. `--fit` doesn't work for this architecture (manual mapping workaround, measured)

`--fit` only treats `ffn_*_exps` as MoE tensors — the MoVA value-expert tensors (`attn_v_exps`, ~3.2GB at Q3_K_M) are **not** offloaded, leaving 5.835 MiB of residue on GPU vs 5.042 MiB available → "Unable to auto-fit". And `--override-tensor` cannot be combined with `--fit` (explicit error).

**Workaround** (works, measured at ctx 131072):

```bash
-ngl 99
--override-tensor "ffn_gate_exps=CPU,ffn_down_exps=CPU,ffn_up_exps=CPU,tok_embd=CPU"
--no-kv-offload    # KV @131K = 6.4GB in RAM (KV is only ~49KB/token: 48 layers, 8 kv-heads × 128, q4_0)
--ubatch-size 256
env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1   # OOM insurance only — see note
```

Note on `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`: we measured it as a **primary serving path** on a different MoE model (Ornith-1.5-35B): 8-10x slower than the tuned offload pipeline (page-by-page migration vs overlapped pinned copies), so we use it here only as OOM insurance — free when everything fits, converts OOM into a run.

Measured (decode / prefill / VRAM / RAM):

| Stage | decode | prefill | VRAM | RAM |
|---|---|---|---|---|
| manual, R1 params @8K | 16.3 t/s | 48.8 t/s | 3.3GB (54%) | ~17GB |
| + output tensor on GPU @8K | 20.9 t/s | 49.2 | 3.7GB | ~24GB |
| + MoVA value-experts on GPU, ubatch 256, ctx 131072 | 20.4 t/s | 67.2 | 5.3GB (86%) | 24GB (77%) |

The output tensor placement matters a lot: keeping `output` on GPU = **+30% decode** (15.2 → 20.9 t/s @8K) for +0.9GB VRAM.

Context: 131072 runs fine (RAM ~24GB total incl. system). The model's native 512K would be impossible on this card (KV alone would be ~25GB).

---

## 4. The "base" alias of the example config THINKS

Without `enable_thinking: false` in the base-alias filter, the template falls back to its default `reasoning_effort=high` — the "base" alias burns its token budget on thinking. Fix: `chat_template_kwargs: {enable_thinking: false}` on the base alias (or `--reasoning off`). Also recommended: `--reasoning-format deepseek` (thinking lands in `reasoning_content`) and `--reasoning-budget 16384` as a runaway ceiling (8192 truncates "high" mode frequently).

---

## 5. Temperature: the model card's "temp 1.0" breaks agentic reading (use 0.7)

Following the model card's "reasoning effort: always high, temperature 1.0" for tool-calling turns, the model read "London" as "Tokyo" **3 times in a row** on the same prompt (same seed-free sampling). Same task at temperature **0.7** (thinking still ON, everything else equal): 4 parallel calls, correct cities. The 1.0/0.95 rule is for benchmark scoring; for agents that must read the prompt, 0.7 is the right operating point.

Related: both configs (kassane's 0.7/0.6 @ top_p 0.9 and ours 0.7/0.7 @ 0.95) run BELOW 1.0 in practice — the `--temp 1.0` in the cmd is overridden by `stripParams` + `setParamsByID` in both.

---

## 6. Tool calls: well-formed calls parse fine; malformed ones leak raw tags

With the abenzerps template + `--parallel-tool-calls`, well-formed calls get parsed into structured tool_calls (tested: 2 and 4 parallel calls in one turn). **But** when the model emits a malformed call (swapped arguments — happens at higher temps), the raw `<ifm|tool_calls>` tags leak into `content`. Improvement candidate: tolerant parser or a strip fallback; also worth testing `tool_call_format: json` via `--chat-template-kwargs`.

Harness-side lesson (not a fork/model bug): in a tool-loop client, the round-trip turn must keep the tools in the payload with `tool_choice: "auto"`. With `"none"` the server **skips** the tool parser (`server-common.cpp:814`) and any re-emission leaks raw tags; with no tools at all, the chat template drops tool definitions from the system block and the model re-plans instead of answering. Our agent simulator (ai-dotfiles `testchat`) now implements the natural ReAct loop — same context as round 1 (system prompt included), tools always present, loop ends when the model answers without calls.

---

## 7. Chat template provenance

The abenzerps GGUF ships `chat_template.jinja` (llama.cpp-Jinja-compatible adaptation — no `dict()`/`sameas`, think-tags parsed from content, `enable_thinking=false` escape) and `chat_template.upstream.jinja` (the original IFM template). The IFM-embedded one presumes the `k2_horizon` parser of the MBZUAI-IFM fork, which doesn't exist in ik_llama.cpp — so the abenzerps template is the right one to use.

---

## 8. Quant landscape (as of Sep 8)

- **NANI-Nithin GGUF**: full i-quant ladder IQ1_M (8.1GB) → IQ4_XS (18.7GB) + MXFP4_MOE (20.2GB) + BF16
- **vincespeed APEX-GGUF**: i-quality 23.9GB / i-balanced 26.3GB / i-compact 17.6GB
- **abenzerps** (used in our tests): Q3_K_M 16.4GB … Q8_0 37.1GB
- **ONYX: nothing yet. APEX configs from mudler: nothing yet. Upstream llama.cpp arch PR: nothing yet** (the kassane fork remains the only runtime with k2-horizon support).
- hermitdave `oQ4e`: safetensors for the IFM Python runtime — not GGUF, out of llama.cpp scope
- MXFP4 types exist in this fork (`MXFP4_R8`) — the 20.2GB MXFP4_MOE is a future quality-upgrade candidate
- IFM also shipped smaller siblings: 7B, 3.7B, 0.9B (+ a dense 32B) — candidates for smaller devices
- Full family: K2-Horizon-375B-A23B exists (flagship; Baekpica mixed-quant GGUF) — 26 GGUF repos total in search

---

## Bottom line

The kassane fork runs this 36B/4B MoE **great on a 6GB card**: 20.4 t/s decode, 67.2 t/s prefill at full 131K context, 86% VRAM / 77% RAM. The vocab merge bug is a 2-line fix (attached). K-hadamard needs a de-transform wire-in on the `llm_build_kv` path (V-side works perfectly, MoVA included). Everything else is measured, documented, and reproducible.