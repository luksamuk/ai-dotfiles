# Model Selection Guide

### Active Listed Models (2026-05-28)

| Model | Engine | VRAM | Context | Features |
|-------|--------|------|---------|----------|
| **qwen3.5-4b** | Bee/TurboQuant | ~3GB | 64K-128K | `thinking`, `tools` |
| **qwen3.5-9b** | ik + hadamard | ~5GB | 64K-128K | `thinking`, `tools`, `parallel-tool-calls` |
| **gemma4-e4b** | ik + hadamard | ~5GB | 64K-128K | `thinking`, `tools` |
| **gemma4-e2b** | upstream | ~3GB | 32K-128K | `thinking`, `vision` |
| **gemma4-26b-moe** | upstream | ~15.5GB offload | 64K-256K | `thinking`, `vision` |
| **qwen3.6-35b-moe** | ik + hadamard | ~17.3GB offload | 64K-128K | `thinking`, `tools`, `parallel-tool-calls`, `mtp` |
| **qwopus-coder-9b** | ik + hadamard | ~5.6GB | 64K-128K | `thinking`, `tools`, `parallel-tool-calls` |
| **webworld-8b** | ik + hadamard | ~5.9GB | 4K-40K | world model, no tools |
| **lfm2.5-1.2b** | upstream | ~1.4GB | 32K-128K | `tools` |
| **lfm2.5-vl-450m** | upstream | ~0.5GB | 32K-128K | `vision` |
| **ministral-3-3b** | upstream | ~2.4GB | 8K-128K | `tools` |
| **minicpm-v-4.6** | upstream | ~3GB | 8K-128K | `vision`, `thinking` |
| **littlelamb-0.3b-tc** | upstream | ~0.4GB | 32K-128K | `tools` |
| **granite-4.0-h-1b** | upstream | ~1.5GB | 8K-128K | hybrid Mamba-2 |

### Disabled Models (unlisted, available via direct ID)

| Model | Why Disabled |
|-------|-------------|
| qwen3.5-4b-upstream | Replaced by Bee/TurboQuant variant |
| qwen3.5-9b-upstream | Replaced by ik + hadamard variant |
| qwen3.5-9b-bee | Bee/TurboQuant segfaults with UD-Q3_K_XL quant |
| gemma4-e4b-bee | Bee was slower (36 vs 40 tok/s with ik) |
| qwopus-coder-9b-bee | Bee variant for benchmarking |
| qwopus-coder-9b-ik | ik backup (identical to active) |
| qwopus-35b | Unlisted MoE variant |
| smolllm3-3b | Unlisted testing |

### Backend Summary

| Backend | Use For | Key Flags |
|---------|---------|-----------|
| **ik_llama.cpp** | MoE models, most dense models | `--fit --fit-margin`, `-khad/-vhad`, `--defer-experts`, `--flash-attn auto`, `--jinja`, `--parallel-tool-calls` |
| **BeeLlama.cpp** | Small dense models (Q4_K_M only) | `--n-gpu-layers 99`, `--cache-type-k/v turbo3_tcq`, `--flash-attn on` |
| **llama.cpp upstream** | Models incompatible with ik/Bee | `--fit on --fit-target`, `--no-mmproj` |

### Known Incompatibilities

- **Bee + UD-Q3_K_XL**: Segfaults with TurboQuant. Use ik instead for UD-quantized models.
- **Bee + `--parallel-tool-calls`**: Not supported. Use ik for tool-calling.
- **ik + `--no-mmproj`**: Flag doesn't exist. ik ignores mmproj files automatically.
- **ik + Gemma 4**: Must use `--jinja` for custom chat templates.