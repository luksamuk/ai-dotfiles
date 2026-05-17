# llama-swap: Setup & Installation

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