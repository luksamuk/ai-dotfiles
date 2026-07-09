# llama-swap Configuration

This directory contains configuration files for running local LLMs with 
[llama-swap](https://github.com/mostlygeek/llama-swap) - a model swapping proxy 
for llama.cpp.

> **📋 Config Architecture:** The config is now split into modular fragments.
> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full structure, workflow, and build process.
> **Do not edit `config.yaml` directly** — edit model fragments in `models/` and run `python3 build-config.py`.

## Overview

**Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

This configuration uses a **dual-binary setup** for optimal performance:
- **llama.cpp** (upstream) for dense models — better mmap performance
- **ik_llama.cpp** (Iwan Kawrakow's fork) for MoE models — faster prompt processing via pinned memory

For details on why two binaries, benchmark results, and flag differences, see [docs/BINARIES.md](docs/BINARIES.md).

## Documentation

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Config fragment system — how `models/`, `config-base.yaml`, and `build-config.py` work |
| [docs/BINARIES.md](docs/BINARIES.md) | ik_llama.cpp vs upstream — benchmarks, flags, known limitations |
| [docs/SETUP.md](docs/SETUP.md) | Prerequisites, installation, model downloads |
| [docs/USAGE.md](docs/USAGE.md) | Running llama-swap, CLI commands, testchat |
| [docs/MODEL-GUIDE.md](docs/MODEL-GUIDE.md) | Model selection — standard, thinking, tool-calling, vLLM models |
| [docs/CONFIG-DETAILS.md](docs/CONFIG-DETAILS.md) | Key parameters, `--fit` behavior, VRAM considerations |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | OOM errors, model not loading, slow inference |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Version history and changes |

## Quick Start

```bash
# Start the service
systemctl --user restart llama-swap

# List available (active) models
llama-swap-cli list

# Chat with a model
llama-swap-cli testchat qwen3.5-4b

# Rebuild config after editing fragments
cd ~/git/ai-dotfiles/llama-swap
python3 build-config.py
```

See [docs/USAGE.md](docs/USAGE.md) for full usage details and [docs/SETUP.md](docs/SETUP.md) for installation.

## Related Files

| File | Purpose |
|------|---------|
| `config.yaml` | ⚡ Generated config — do not edit directly |
| `config-base.yaml` | Header + macros (source of truth) |
| `config-footer.yaml` | Hooks + matrix (source of truth) |
| `build-config.py` | Assembles fragments into `config.yaml` |
| `models/*.yaml` | Active model definitions |
| `models/_disabled/*.yaml` | Unlisted models (each has `unlisted: true`) |
| `models/_removed/*.yaml` | Dead code (reference only, not built) |
| `download-models.sh` | Download GGUF models from HuggingFace |
| `run.sh` | Run llama-swap directly (no systemd) |
| `llama-swap-cli` | CLI helper for model management |
| `testchat/` | Interactive chat test script |
| `llama-swap.service.template` | systemd user service template |
| `MTP-NOTES.md` | Multi-Token Prediction benchmarks and notes |
| `templates/` | Custom Jinja chat templates (see below) |

## Custom Chat Templates

Some models use custom Jinja chat templates loaded via `--chat-template-file`
instead of the template embedded in their GGUF metadata.

| Template | Models | Description |
|----------|--------|-------------|
| `qwen-fixed-chat-template.jinja` | qwen3.5-4b, qwen3.5-4b-abliterated, qwen3.5-9b, qwen3.6-35b-a3b | Fixed Jinja chat template for Qwen 3.5 & 3.6 — fixes agentic loop stalling, KV cache invalidation, minijinja compatibility, and tool calling errors |
| `lfm2.5-chat-template.jinja` | lfm2.5-8b-a1b | LFM2.5 chat template for empty-template GGUFs |
| `laguna-chat-template.jinja` | (disabled) | Laguna XS chat template |

### Qwen Fixed Chat Template

The `qwen-fixed-chat-template.jinja` file is sourced from
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v21.3, Apache-2.0 license).

**Attribution:**
- Original models: Alibaba Cloud (Qwen team)
- Template fixes: [froggeric](https://huggingface.co/froggeric)
- C++ AST optimizations: barubary / spiritbuun

**Key fixes over the stock Qwen template:**
- Cured "Empty Think" poisoning that causes agentic loop stalls
- Two-tier error escalation for consecutive tool call failures
- Enforced chronological history for improved KV cache hit rate
- Flattened Jinja AST for better llama.cpp throughput
- 100% minijinja-safe (compatible with BeeLlama.cpp)
- Dynamic payload truncation for long tool responses
- Anthropic `message.thinking` payload support
- Auto-injected closing tags before tool boundaries

**Applied to:** Qwen 3.5 and 3.6 models only (4B, 4B-abliterated, 9B, 35B-A3B).
Ornith models are excluded — they have a modified template from post-training.

**Original templates backup:** `templates/_backup-qwen-original/` contains the
stock `tokenizer.chat_template` extracted from each GGUF before the override
was applied. To revert, remove the `--chat-template-file` line from the model
fragment and rebuild the config.