# llama-swap Configuration

This directory contains configuration files for running local LLMs with 
[llama-swap](https://github.com/mostlygeek/llama-swap) - a model swapping proxy 
for llama.cpp.

## Overview

**Hardware Target:** NVIDIA RTX 3050 Laptop (6GB VRAM)

This configuration uses llama.cpp's **`--fit`** feature for automatic VRAM-aware
parameter fitting, which automatically adjusts GPU layers (`-ngl`) and context
size based on available VRAM.

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

The `--fit` feature (PR #16653) automatically:

1. Detects available VRAM on each GPU
2. Calculates optimal number of GPU layers (`-ngl`)
3. Reduces context size if necessary
4. Prioritizes dense weights for MoE models
5. Leaves a safety margin (configurable via `--fit-margin`)

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