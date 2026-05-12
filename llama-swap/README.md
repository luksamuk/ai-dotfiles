# llama-swap Configuration

This directory contains configuration files for running local LLMs with 
[llama-swap](https://github.com/mostlygeek/llama-swap) - a model swapping proxy 
for llama.cpp.

## Overview

**Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

This configuration uses a **dual-binary setup** for optimal performance:
- **llama.cpp** (upstream) for dense models — better mmap performance
- **ik_llama.cpp** (Iwan Kawrakow's fork) for MoE models — faster prompt processing via pinned memory

Both binaries share the same GGUF format (standard quants), same server API, and same
CLI flags — just swap the binary. The config uses the macro `${llama_server}` for upstream
and `${ik_llama_server}` for ik, routing models to the best backend automatically.

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
| Qwen3.6 35B MoE | MoE (APEX I-Compact) | ik_llama.cpp | **87.1** | 31.1 | +41% prompt vs upstream |
| Qwen3.6 35B MoE | MoE (APEX I-Compact) | upstream | 60.8–67.1 | 30.5 | Baseline |
| GPT-OSS 20B | Dense (Q4_K_M) | upstream | **66.8–104.6** | **31.8** | +42–148% vs ik |
| GPT-OSS 20B | Dense (Q4_K_M) | ik_llama.cpp | 42.1–53.9 | 22.4 | Slower for dense |

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

## Prerequisites

### 1. Install llama.cpp with CUDA support

**Option A: Build from source (recommended - includes `--fit` feature)**

The `--fit` feature for automatic VRAM-aware parameter fitting requires llama.cpp
built from source (PR #16653, Dec 2025). The AUR package `llama.cpp-cuda` may not
have this feature yet.

```bash
# Clone and build with CUDA
cd ~/git
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Binaries will be in build/bin/
# llama-server is at: ~/git/llama.cpp/build/bin/llama-server
```

**Option B: AUR package (without `--fit`)**

```bash
# Arch Linux (AUR) - may not have --fit feature
yay -S llama.cpp-cuda
```

If using Option B, you'll need to manually configure `--gpu-layers` in the config.

### 2. Install llama-swap

```bash
# Arch Linux (AUR)
yay -S llama-swap-bin

# Or download from GitHub
# https://github.com/mostlygeek/llama-swap/releases
```

### 3. Install Hugging Face CLI (for downloading models)

```bash
# Using pip
pip install hf-transfer huggingface_hub

# Or the standalone installer
curl -LsSf https://hf.co/cli/install.sh | bash
```

## Model Downloads

The models are stored in `~/.llama-models/`. Create the directory and download:

```bash
# Create models directory
mkdir -p ~/.llama-models

# Download Qwen3.5-4B (2.63 GB - fits entirely in VRAM)
hf download unsloth/Qwen3.5-4B-GGUF \
  Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir ~/.llama-models

# Download Qwen3.5-9B (5.68 GB - requires partial offload)
hf download unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-Q4_K_M.gguf \
  --local-dir ~/.llama-models

# Download Gemma-4-E4B (4.98 GB - requires partial offload)
hf download unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf \
  --local-dir ~/.llama-models

# Download Gemma-4-E2B (3.11 GB - fits in VRAM)
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf \
  --local-dir ~/.llama-models

# Download Nemotron-3-Nano-4B (2.90 GB - fits in VRAM, great for tool-calling)
hf download unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF \
  NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf \
  --local-dir ~/.llama-models
```

### Alternative Quantizations

If you need different quality/size trade-offs:

| Model | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|-------|--------|--------|------|------|
| Qwen3.5-4B | 2.63 GB | 3.09 GB | 3.59 GB | 4.65 GB |
| Qwen3.5-9B | 5.68 GB | 6.58 GB | 7.46 GB | 9.53 GB |
| Gemma-4-E4B | 4.98 GB | 5.48 GB | - | - |
| Gemma-4-E2B | 3.11 GB | 3.36 GB | 4.50 GB | 5.05 GB |
| Nemotron-3-Nano-4B | 2.90 GB | 3.16 GB | 4.06 GB | 4.23 GB |

```bash
# Example: Download Q5_K_M variant
hf download unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-Q5_K_M.gguf \
  --local-dir ~/.llama-models \
  --local-dir-use-symlinks False
```

## Installation

### System-wide (requires root)

```bash
# Copy config to system location
sudo cp config.yaml /etc/llama-swap/config.yaml

# Start systemd service
sudo systemctl enable --now llama-swap
```

### User-level (recommended)

```bash
# Create user config directory
mkdir -p ~/.config/llama-swap

# Copy config
cp config.yaml ~/.config/llama-swap/

# Run manually (for testing)
llama-swap -config ~/.config/llama-swap/config.yaml -listen 127.0.0.1:12434
```

## Usage

### Quick Start (No Systemd)

Run directly in foreground - no installation needed:

```bash
# Check prerequisites first
./run.sh status

# Run in foreground (press Ctrl+C to stop)
./run.sh run

# Or use LLAMA_SWAP_PORT to change port
LLAMA_SWAP_PORT=8080 ./run.sh run
```

### Systemd User Service

```bash
# List available models
curl http://127.0.0.1:12434/v1/models

# Chat completion
curl http://127.0.0.1:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Systemd User Service

For running as a background service:

```bash
# Install service (copies config if needed)
./run.sh install

# Start service
systemctl --user start llama-swap

# Check status
systemctl --user status llama-swap

# View logs
./run.sh logs
# or
journalctl --user -u llama-swap -f

# Stop service
systemctl --user stop llama-swap

# Enable/disable autostart
systemctl --user enable llama-swap
systemctl --user disable llama-swap

# Uninstall service (keeps config and models)
./run.sh uninstall
```

The service file is installed at `~/.config/systemd/user/llama-swap.service`
and uses the config from `~/.config/llama-swap/config.yaml`.

### Testing the API

Once running, llama-swap provides an OpenAI-compatible API:

```bash
# List available models
curl http://127.0.0.1:12434/v1/models

# Chat completion
curl http://127.0.0.1:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_SWAP_CONFIG` | Config file path | `~/.config/llama-swap/config.yaml` |
| `LLAMA_SWAP_PORT` | Port to listen on | `12434` |

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:12434/v1",
    api_key="not-needed"  # llama-swap doesn't require auth by default
)

response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Using with Ollama-compatible tools

The API is OpenAI-compatible, so most tools work out of the box:
- [Continue](https://continue.dev/) - VS Code extension
- [Open WebUI](https://github.com/open-webui/open-webui)
- [LibreChat](https://github.com/danny-avila/LibreChat)

## Model Selection Guide

### Standard Models (General Use)

| Model | VRAM | Context | Features |
|-------|------|---------|----------|
| **qwen3.5-4b** | ~3GB | 32K-128K | - |
| **qwen3.5-9b** | ~5GB + RAM | 16K-128K | - |
| **gemma4-e4b** | ~4.5GB + RAM | 16K-128K | - |
| **gemma4-e2b** | ~3GB | 32K-128K | - |
| **nemotron-3-nano-4b** | ~3GB | 32K-128K | `tools` |

### Thinking Models (With Reasoning)

| Model | VRAM | Context | Features |
|-------|------|---------|----------|
| **qwen3.5-4b-think** | ~3GB | 32K-128K | `thinking` |
| **qwen3.5-9b-think** | ~5GB + RAM | 16K-128K | `thinking` |
| **gemma4-e4b-think** | ~4.5GB + RAM | 16K-128K | `thinking` |
| **gemma4-e2b-think** | ~3GB | 32K-128K | `thinking` |
| **nemotron-3-nano-4b-think** | ~3GB | 32K-128K | `thinking`, `tools` |

### Tool-Calling Models (Reasoning + Tools)

These models are fine-tuned for function calling and always use reasoning:

| Model | VRAM | Context | Features |
|-------|------|---------|----------|
| **qwopus-4b** | ~3GB | 32K-128K | `thinking`, `tools` |
| **qwopus-9b** | ~5GB + RAM | 16K-128K | `thinking`, `tools` |

### vLLM Backend Models

Experimental vLLM backends for API compatibility testing. Not for daily use — slower startup (~30-90s), higher overhead.

| Model | VRAM | Context | Features | Notes |
|-------|------|---------|----------|-------|
| **qwen3.5-0.8b-vllm** | ~2.5-3GB | 8K | `tools` | vLLM safetensors, auto-download from HF |
| **qwen3.5-2b-vllm** | ~5GB | 2K | `tools` | vLLM safetensors, no vision (`--skip-mm-profiling`) |

**vLLM-specific flags:**
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` — Required for tool calling
- `--skip-mm-profiling` + `--limit-mm-per-prompt '{"image": 0}'` — Skip ViT profiling (2B only)
- `--default-chat-template-kwargs '{"enable_thinking": false}'` — Disable reasoning in output

### Context Size Behavior

Context is **dynamic** - automatically adjusts based on available VRAM:

| Model Type | Minimum | Maximum | Behavior |
|------------|---------|---------|----------|
| Small (4B, E2B) | 32K | 128K | Fits entirely in VRAM |
| Large (9B, E4B) | 16K | 128K | May use RAM offload |

The `--fit` feature ensures:
- **Never crashes** - reduces context if VRAM is tight
- **Maximum utilization** - expands context when VRAM is free
- **Dynamic adjustment** - adapts to current system state

### Feature Flags

| Flag | Description |
|------|-------------|
| `thinking` | Model has reasoning/thinking capability enabled |
| `tools` | Model excels at function calling/tool use |
| `vision` | Model supports image input (not yet available) |

### Inference Parameters

Based on Unsloth recommendations:

| Parameter | Standard | Thinking | Code/Tools |
|-----------|----------|----------|------------|
| `temp` | 0.7 | 0.6 | 0.6 |
| `top_p` | 0.9 | 0.9 | 0.85 |
| `top_k` | 20 | 20 | 40 |
| `min_p` | 0.01 | 0.0 | 0.02 |
| `repeat_penalty` | 1.05 | 1.0 | 1.05 |
| `reasoning` | - | `auto` | - |

### Model Names (Ollama-style)

Models use the format `model:size` for consistency with Ollama:

| Primary Name | Aliases |
|--------------|---------|
| `qwen3.5-4b` | `qwen3.5-4b`, `qwen3.5-4b-q4` |
| `qwen3.5-4b-think` | `qwen3.5-4b-think`, `qwen3.5-4b-reasoning` |
| `qwen3.5-9b` | `qwen3.5-9b`, `qwen3.5-9b-q4` |
| `qwen3.5-9b-think` | `qwen3.5-9b-think`, `qwen3.5-9b-reasoning` |
| `gemma4-e4b` | `gemma4-e4b`, `gemma-4-e4b` |
| `gemma4-e4b-think` | `gemma4-e4b-think`, `gemma-4-e4b-think` |
| `gemma4-e2b` | `gemma4-e2b`, `gemma-4-e2b` |
| `gemma4-e2b-think` | `gemma4-e2b-think`, `gemma-4-e2b-think` |
| `nemotron-3-nano-4b` | `nemotron-4b`, `nemotron-3-nano-4b`, `nemotron` |
| `nemotron-3-nano-4b-think` | `nemotron-4b-think`, `nemotron-think` |
| `qwopus-4b` | `qwopus4b`, `qwopus-4b` |
| `qwopus-9b` | `qwopus`, `qwopus9b` |

## CLI: llama-swap-cli

A companion CLI for managing llama-swap models from the terminal.

```bash
# List configured models
llama-swap-cli list [--pretty]

# Show running models (VRAM, RSS, tok/s, GPU layers)
llama-swap-cli ps [--pretty]

# Detailed metrics (tokens, speed, queue)
llama-swap-cli stats [--pretty]

# Unload all models (or a specific one)
llama-swap-cli unload [MODEL]

# Recent logs
llama-swap-cli logs [N]

# Interactive chat with model selection
llama-swap-cli testchat
```

Supports both llama.cpp and vLLM backends. The `ps` and `stats` commands detect vLLM metrics automatically.

## Interactive Chat (testchat)

An interactive terminal chat with streaming, reasoning display, and tool calling.

```bash
# Via CLI
llama-swap-cli testchat

# Or directly
cd testchat && uv run main.py
```

### Features

- **Model selection** with feature icons (🤔 thinking, 🛠️ tools, 👁️ vision)
- **Tool calling** with mock tools (get_weather, calculator, get_time) — auto-enabled for models with `tools: true`
- **Reasoning panel** — split-screen display for thinking models
- **Streaming** — real-time token display with timing stats

### Tool Calling Flow

When a model with `tools: true` is selected, mock tools are sent automatically. The model decides whether to call a tool, and the testchat provides simulated responses:

1. Model decides to call a tool → displays `🔧 tool_name(args)`
2. Testchat generates mock response → displays inline
3. Model formats final answer using mock data

### Waybar Integration

The `inference-status.py` Waybar module shows loaded models and allows unloading via click:

```json
// ~/.config/waybar/config.jsonc
"custom/inference": {
  "exec": "~/.config/waybar/scripts/inference-status.py llamaswap status",
  "return-type": "json",
  "on-click": "~/.config/waybar/scripts/inference-status.py llamaswap eject_all"
}
```

Uses the llama-swap `/running` API — works with both llama.cpp and vLLM backends.

## Configuration Details

### Key Parameters

- **`--fit on`**: Automatically adjusts GPU layers and context size to fit in VRAM
- **`--fit-target 512`**: Safety margin in MiB (prevents OOM)
- **`--fit-ctx 4096`**: Minimum context size when downscaling
- **`--ctx-size`**: Maximum context length (8192 or 16384 depending on model)
- **`--flash-attn`**: Flash Attention for better performance
- **`--temp 0.7`**: Temperature for sampling diversity
- **`--top-p 0.85`**: Nucleus sampling threshold
- **`--top-k 40`**: Top-K sampling

### How `--fit` Works

The `--fit` feature automatically:

1. Detects available VRAM on each GPU
2. Calculates optimal number of GPU layers (`-ngl`)
3. Reduces context size if necessary
4. Prioritizes dense weights for MoE models
5. Leaves a safety margin (configurable)

**Note:** The flag syntax differs between binaries:
- **upstream llama.cpp:** `--fit on --fit-target 512` (target free VRAM in MiB)
- **ik_llama.cpp:** `--fit --fit-margin 512` (safety margin in MiB)

In `config.yaml`, MoE models use `${ik_llama_server}` with `--fit --fit-margin`,
while dense models use `${llama_server}` with `--fit on --fit-target`.

This is especially useful for:
- **Mixed GPU setups** - automatically balances layers
- **Memory pressure** - prevents OOM crashes
- **Different models** - no manual tuning per model

### VRAM Considerations for RTX 3050 (6GB)

With ~5GB free VRAM (after desktop environment):

| Model | Strategy |
|-------|----------|
| Qwen3.5-4B | Fits entirely in VRAM - fastest inference |
| Qwen3.5-9B | Requires partial offload - `--fit` handles automatically |
| Gemma-4-E4B | Requires partial offload - `--fit` handles automatically |

The `--fit` flag will:
1. Detect available VRAM
2. Calculate optimal number of GPU layers
3. Reduce context size if necessary
4. Move remaining layers to system RAM

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Increase the fit margin
--fit-margin 1024  # instead of 512
```

### Model Not Loading

```bash
# Check llama-swap logs
journalctl -u llama-swap -f

# Or if running manually
llama-swap -config config.yaml -listen 127.0.0.1:12434
```

### Slow Inference

```bash
# Check if GPU is being used
nvidia-smi  # should show llama-server process

# Verify Flash Attention is enabled (-fa flag in config)
```

## Resources

- [llama-swap GitHub](https://github.com/mostlygeek/llama-swap)
- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)
- [Unsloth Models](https://unsloth.ai/)
- [Qwen3.5 Documentation](https://unsloth.ai/docs/models/qwen3.5)
- [Gemma 4 Documentation](https://unsloth.ai/docs/models/gemma-4)

## Related Files in This Repo

- `../modelfiles/` - Ollama modelfiles for similar models
- `../configs/ask-ai/` - Configuration for the ask-ai CLI tool