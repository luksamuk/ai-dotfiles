# llama-swap: ik_llama.cpp vs upstream (Dual Binary Setup)

## Why Two Binaries?

**ik_llama.cpp** ([github.com/ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp))
is a performance-focused fork of llama.cpp by Iwan Kawrakow. Its key advantages for
MoE (Mixture of Experts) models on CPU+GPU hybrid setups:

- **Pinned CUDA_Host memory** (`cudaHostAlloc`) for expert offload — reduces PCIe transfer
  latency and improves prompt processing throughput significantly
- **Fused MoE FFN kernel** — combines gate+up projection into a single kernel, reducing memory
  bandwidth
- **Smart Expert Reduction (SER)** — dynamically reduces active experts when GPU memory is limited
- **`--fit` support** (PR #1501/#1504, merged Mar 2026) — automatic layer distribution like upstream

**Benchmark results on RTX 3050 6GB (hybrid CPU+GPU):**

| Model | Type | Backend | Prompt tok/s | Decode tok/s | Notes |
|-------|------|---------|-------------|-------------|-------|
| Qwen3.6 35B MoE | MoE (APEX I-Compact) | ik_llama.cpp + MTP | **91.2** | **40.4** | +41% prompt vs upstream, MTP enabled |
| Qwen3.6 35B Qwopus | MoE (APEX I-Compact) | ik_llama.cpp + MTP | **80.5** | **38.8** | Same arch, Qwopus SFT |
| Qwen3.6 35B MoE | MoE (APEX I-Compact) | upstream | 60.8–67.1 | 30.5 | Baseline |
| GPT-OSS 20B | Dense (Q4_K_M) | upstream | **66.8–104.6** | **31.8** | +42–148% vs ik |
| GPT-OSS 20B | Dense (Q4_K_M) | ik_llama.cpp | 42.1–53.9 | 22.4 | Slower for dense |
| Gemma 4 26B MoE | MoE (APEX I-Compact) | upstream | 86.5 | 31.0 | ik `--fit` fails; upstream only |
| Qwen3.5 9B | Dense (UD-Q3_K_XL) | upstream | 26.0 | 15.3 | Fits VRAM with partial offload |
| Gemma 4 E4B | Dense (UD-Q3_K_XL) | upstream | 138.5 | 49.2 | Fits VRAM, fast |

**Decision rule:** Use ik_llama.cpp for MoE models, upstream for dense models.

### Key Differences Between Binaries

| Feature | llama.cpp (upstream) | ik_llama.cpp |
|---------|---------------------|--------------|
| `--fit` flag | `--fit on` | `--fit` (no arg) |
| VRAM margin | `--fit-target N` (MiB free target) | `--fit-margin N` (MiB safety margin) |
| Best for | Dense models (Q4_K_M, small) | MoE models (APEX I-Compact, expert offload) |
| Pinned memory | No (uses mmap) | Yes (CUDA_Host, automatic for experts) |
| APEX I-Compact GGUF | ✅ Works | ✅ Works |
| Unsloth `_XL` GGUF | ✅ Works | ❌ Known incompatibility |

### MoE Models Using ik_llama.cpp

The following models in `config.yaml` are configured to use ik_llama.cpp:

- `gemma4-26b-moe` — Gemma 4 26B MoE (128 experts, 4B active)
- `qwen3.6-35b-moe` — Qwen3.6 35B MoE (256 experts, 3B active)
- `qwen3.6-35b-qwopus` — Qwopus 3.6 35B (same arch, Qwopus SFT)

All other models use the upstream llama.cpp binary.

> **Note:** Gemma 4 26B MoE was tested with ik_llama.cpp but its `--fit` algorithm
> cannot fit the model on 6GB VRAM (requires 9.9GB even after offloading all MoE tensors),
> so it uses the upstream binary. This may change in future ik releases.

### ik_llama.cpp Flags for MoE Models

MoE models using ik_llama.cpp require these flag differences from upstream:

| Flag | Upstream (dense models) | ik_llama.cpp (MoE models) |
|------|------------------------|---------------------------|
| `--fit` | `--fit on` | `--fit` (no arg) |
| VRAM margin | `--fit-target N` | `--fit-margin N` |
| Context floor | `--fit-ctx N` | ❌ Not supported — use `--ctx-size` only |
| Vision disable | `--no-mmproj` | ❌ Not supported — omit for text-only models |
| Tool calling | Automatic | `--jinja` required |
| Parallel tool calls | Automatic | `--parallel-tool-calls` required |
| Reasoning | `--reasoning on` | `--reasoning on` (no `--reasoning-format` needed) |
| Multi-Token Pred | ❌ Not available | `--multi-token-prediction` (**long form only**, `--mtp` causes exit 1) |

### Known Limitations of ik_llama.cpp

1. **No `--reasoning-format` in streaming**: The `deepseek` format puts reasoning in
   `reasoning_content` for non-streaming, but in streaming mode it behaves as `none`
   (tags stay in `content`). The `deepseek-legacy` format keeps tags in `content` in
   both modes. Use `--jinja` for proper chat template and tool support instead.

2. **Sequential tool calls** (FIXED): ik_llama.cpp requires `--parallel-tool-calls` to
   generate multiple tool calls in a single response. Without it, the model only generates
   1 tool call per turn. Upstream enables this by default.

3. **No `--no-mmproj` or `--fit-ctx`**: These upstream flags don't exist in ik.
   For text-only models, simply omit `--no-mmproj`. Use `--ctx-size` as the ceiling
   instead of `--fit-ctx`.

### Build ik_llama.cpp

```bash
# Clone and build (same process as llama.cpp)
cd ~/git
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Binary at: ~/git/ik_llama.cpp/build/bin/llama-server
```

Build both from source with CUDA. The AUR package doesn't include `--fit`.