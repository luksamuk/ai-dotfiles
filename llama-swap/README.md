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

```bash
# Arch Linux (AUR)
yay -S llama.cpp-cuda

# Or build from source
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
sudo cmake --install build
```

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
  --local-dir ~/.llama-models \
  --local-dir-use-symlinks False

# Download Qwen3.5-9B (5.68 GB - requires partial offload)
hf download unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-Q4_K_M.gguf \
  --local-dir ~/.llama-models \
  --local-dir-use-symlinks False

# Download Gemma-4-E4B (4.98 GB - requires partial offload)
hf download unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf \
  --local-dir ~/.llama-models \
  --local-dir-use-symlinks False
```

### Alternative Quantizations

If you need different quality/size trade-offs:

| Model | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|-------|--------|--------|------|------|
| Qwen3.5-4B | 2.63 GB | 3.09 GB | 3.59 GB | 4.65 GB |
| Qwen3.5-9B | 5.68 GB | 6.58 GB | 7.46 GB | 9.53 GB |
| Gemma-4-E4B | 4.98 GB | 5.48 GB | - | - |

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

| Model | VRAM Usage | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| **qwen3.5-4b** | ~3GB | Fastest | Good | Quick tasks, code completion |
| **qwen3.5-9b** | ~5GB + RAM | Medium | Better | Complex reasoning, longer responses |
| **gemma4-e4b** | ~5GB + RAM | Medium | Good | General purpose, multilingual |

## Configuration Details

### Key Parameters

- **`--fit on`**: Automatically adjusts `-ngl` (GPU layers) based on available VRAM
- **`--fit-margin 512`**: Safety margin in MiB (prevents OOM)
- **`--ctx-size`**: Context length (reduced if needed by `--fit`)
- **`-fa`**: Flash Attention for better performance
- **`--temp 0.7`**: Temperature for sampling diversity
- **`--top-p 0.85`**: Nucleus sampling threshold
- **`--top-k 40`**: Top-K sampling

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