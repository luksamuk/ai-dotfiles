# Model Selection Guide

### Active Listed Models (2026-05-31)

|| Model | Engine | VRAM | Context | Features ||
-------|--------|------|---------|----------|--
|| **afm-4.5b** | upstream (Hermes2Pro template) | ~3.1GB | 64K | `tools`, `parallel-tool-calls` ||
|| **qwen3.5-4b** | Bee/TurboQuant | ~3GB | 64K-128K | `thinking`, `tools` ||
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
| **cafe-llama.cpp** | Contexto longo (262K) e MoE que não cabe em RAM/VRAM | `--jinja` (template do GGUF), `-hmoe` (pinned RAM) ou `-ssd`, `-ctk/-ctv turbo4`, `--fit-target` |
| **llama.cpp upstream** | Models incompatible with ik/cafe | `--fit on --fit-target`, `--no-mmproj` |

### Backend notes — cafe-llama.cpp (fork quimmedes/Ark, adotado Set/2026)

Substituiu o BeeLlama.cpp (removido). Usar quando o ik não dá conta:

- **Contexto longo em MoE**: único backend que carrega `qwen35moe` @262K na 3050 6GB
  (turbo4 KV; ik OOMa com q4_0 nesse ctx)
- **MoE offload**: `-hmoe` = experts pinned em RAM (~12-14GB fixos, sem warmup) |
  `-ssd` = experts streamados do NVMe por mmap (RSS baixo, mas warmup de 1-2 prompts
  por domínio novo de conteúdo)
- **Tradeoffs medidos**: prefill em prompt FRIO ~3.4x mais caro que o ik (82 vs 281 t/s
  @15K tok) por causa do KV turbo reconstruído no kernel FA + ausência de
  `--prefetch-experts`. Cache quente é equivalente.
- **Não tem** (flags ik-only): `--parallel-tool-calls`, `--defer-experts`,
  `--prefetch-experts`, `--k/v-cache-hadamard`, `--no-graph-reuse`, `--fit-margin`
  (usar `--fit-target`)
- **Nunca** `--pipeline-parallel` com ctx ≥131K (buffer DMA ~ctx → OOM)
- **turbo KV em arch híbrida SWA** (laguna) CORROMPE — usar q4_0 lá
- Detalhes completos: skill `llama-swap-fleet` → `references/cafe-llama-cpp-eval.md`

### Known Incompatibilities

- **cafe + turbo em laguna (SWA híbrida)**: turbo2/3/4 corrompem a saída mesmo em ctx curto — usar q4_0.
- **cafe + `--pipeline-parallel`**: buffer DMA cresce com o ctx (48GB @131K) — proibido em ctx longo.
- **ik + `--no-mmproj`**: Flag doesn't exist. ik ignores mmproj files automatically.
- **ik + Gemma 4**: Must use `--jinja` for custom chat templates.
- **AFM-4.5B**: GGUF has empty `tool_use` chat_template. Must use `--chat-template-file` with Hermes 2 Pro template for tool calling. Without it, AFM runs in plain chat mode only.