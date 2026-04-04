#!/bin/bash
# Download GGUF models for llama-swap
# Usage: ./download-models.sh [model-name]
#
# Available models:
#   qwen3.5:4b      - Qwen3.5-4B Q4_K_M (2.63 GB) - fits in VRAM
#   qwen3.5:9b      - Qwen3.5-9B Q4_K_M (5.68 GB) - partial offload
#   gemma4:e4b      - Gemma-4 E4B Q4_K_M (4.98 GB) - partial offload
#   gemma4:e2b      - Gemma-4 E2B Q4_K_M (3.11 GB) - fits in VRAM
#   nemotron-3-nano:4b - Nemotron-3-Nano-4B Q4_K_M (2.90 GB) - tool-calling
#   all             - Download all models
#
# If no argument, downloads qwen3.5:4b (fits entirely in 6GB VRAM)

set -e

MODELS_DIR="${HOME}/.llama-models"

# Create directory
mkdir -p "$MODELS_DIR"

# Model definitions (name: repo filename)
declare -A MODELS=(
  ["qwen3.5:4b"]="unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"
  ["qwen3.5:9b"]="unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf"
  ["gemma4:e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf"
  ["gemma4:e2b"]="unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf"
  ["nemotron-3-nano:4b"]="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
)

# Legacy aliases (for backwards compatibility)
declare -A ALIASES=(
  ["qwen3.5-4b"]="qwen3.5:4b"
  ["qwen3.5-9b"]="qwen3.5:9b"
  ["gemma4-e4b"]="gemma4:e4b"
  ["gemma4-e2b"]="gemma4:e2b"
  ["nemotron-4b"]="nemotron-3-nano:4b"
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
  
  # Resolve legacy aliases
  if [[ -n "${ALIASES[$key]}" ]]; then
    echo "Note: '$key' is deprecated, use '${ALIASES[$key]}' instead"
    key="${ALIASES[$key]}"
  fi
  
  local repo_file="${MODELS[$key]}"
  if [[ -z "$repo_file" ]]; then
    echo "Error: Unknown model '$key'"
    return 1
  fi
  
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
  echo "Model Sizes (Q4_K_M) - Ollama-style names:"
  echo "  qwen3.5:4b          2.63 GB  - Fits in VRAM"
  echo "  qwen3.5:9b          5.68 GB  - Partial offload"
  echo "  gemma4:e4b          4.98 GB  - Partial offload"
  echo "  gemma4:e2b          3.11 GB  - Fits in VRAM"
  echo "  nemotron-3-nano:4b  2.90 GB  - Fits in VRAM, tool-calling"
  echo ""
  echo "Legacy names (still work):"
  echo "  qwen3.5-4b → qwen3.5:4b"
  echo "  qwen3.5-9b → qwen3.5:9b"
  echo "  gemma4-e4b → gemma4:e4b"
  echo "  gemma4-e2b → gemma4:e2b"
  echo "  nemotron-4b → nemotron-3-nano:4b"
  echo ""
}

# Main
case "${1:-qwen3.5:4b}" in
  "qwen3.5:4b"|"qwen3.5:9b"|"gemma4:e4b"|"gemma4:e2b"|"nemotron-3-nano:4b")
    show_sizes
    download_model "$1"
    ;;
  "qwen3.5-4b"|"qwen3.5-9b"|"gemma4-e4b"|"gemma4-e2b"|"nemotron-4b")
    # Legacy aliases
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
    echo "Available: qwen3.5:4b, qwen3.5:9b, gemma4:e4b, gemma4:e2b, nemotron-3-nano:4b, all, sizes"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"