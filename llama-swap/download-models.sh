#!/bin/bash
# Download GGUF models for llama-swap
# Usage: ./download-models.sh [model-name]
#
# Available models:
#   qwen3.5-4b    - Qwen3.5-4B Q4_K_M (2.63 GB)
#   qwen3.5-9b    - Qwen3.5-9B Q4_K_M (5.68 GB)
#   gemma4-e4b    - Gemma-4 E4B Q4_K_M (4.98 GB)
#   nemotron-4b   - Nemotron-3-Nano-4B Q4_K_M (2.90 GB) - tool-calling
#   all           - Download all models
#
# If no argument, downloads qwen3.5-4b (fits entirely in 6GB VRAM)

set -e

MODELS_DIR="${HOME}/.llama-models"

# Create directory
mkdir -p "$MODELS_DIR"

# Model definitions
declare -A MODELS=(
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf"
  ["nemotron-4b"]="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
)

# Alternative quantizations (for reference)
# declare -A QWEN_9B_VARIANTS=(
#   ["q4_k_m"]="Qwen3.5-9B-Q4_K_M.gguf 5.68 GB"
#   ["q5_k_m"]="Qwen3.5-9B-Q5_K_M.gguf 6.58 GB"
#   ["q6_k"]="Qwen3.5-9B-Q6_K.gguf 7.46 GB"
#   ["q8_0"]="Qwen3.5-9B-Q8_0.gguf 9.53 GB"
# )

download_model() {
  local key="$1"
  local repo_file="${MODELS[$key]}"
  local repo="${repo_file%% *}"
  local file="${repo_file#* }"
  
  echo "Downloading $file from $repo..."
  
  if [[ -f "$MODELS_DIR/$file" ]]; then
    echo "  ✓ Already exists: $MODELS_DIR/$file"
    return 0
  fi
  
  hf download "$repo" "$file" --local-dir "$MODELS_DIR"
  
  echo "  ✓ Downloaded: $MODELS_DIR/$file"
}

show_sizes() {
  echo ""
  echo "Model Sizes (Q4_K_M):"
  echo "  Qwen3.5-4B:    2.63 GB  - Fits entirely in 6GB VRAM"
  echo "  Qwen3.5-9B:    5.68 GB  - Requires partial offload"
  echo "  Gemma-4-E4B:   4.98 GB  - Requires partial offload"
  echo "  Nemotron-4B:   2.90 GB  - Fits in VRAM, great for tool-calling"
  echo ""
}

# Main
case "${1:-qwen3.5-4b}" in
  "qwen3.5-4b"|"qwen3.5-9b"|"gemma4-e4b"|"nemotron-4b")
    show_sizes
    download_model "$1"
    ;;
  "all")
    show_sizes
    echo "Downloading all models..."
    for key in "${!MODELS[@]}"; do
      download_model "$key"
    done
    ;;
  "sizes")
    show_sizes
    ;;
  *)
    echo "Unknown model: $1"
    echo "Available: qwen3.5-4b, qwen3.5-9b, gemma4-e4b, nemotron-4b, all, sizes"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"