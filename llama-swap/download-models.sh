#!/bin/bash
# Download GGUF models for llama-swap
# Usage: ./download-models.sh [model-name]
#
# Available models:
#   qwen3.5-4b          - Qwen3.5-4B Q4_K_M (2.63 GB) - fits in VRAM
#   qwen3.5-9b          - Qwen3.5-9B Q4_K_M (5.68 GB) - partial offload
#   qwen3.5-27b         - Qwen3.5-27B Q4_K_M (~17 GB) - heavy offload, slow
#   gemma4-e4b          - Gemma-4 E4B Q4_K_M (4.98 GB) - partial offload
#   gemma4-e2b          - Gemma-4 E2B Q4_K_M (3.11 GB) - fits in VRAM
#   nemotron-3-nano-4b  - Nemotron-3-Nano-4B Q4_K_M (2.90 GB) - tool-calling
#   all                 - Download all models
#
# If no argument, downloads qwen3.5-4b (fits entirely in 6GB VRAM)

set -e

MODELS_DIR="${HOME}/.llama-models"

# Create directory
mkdir -p "$MODELS_DIR"

# Model definitions
# Format: "repo filename [local_filename]"
# If local_filename is provided, the file is renamed after download
declare -A MODELS=(
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf"
  ["qwen3.5-27b"]="unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf"
  ["gemma4-e2b"]="unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf"
  ["gemma4-31b"]="unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-Q4_K_M.gguf"
  ["nemotron-3-nano-4b"]="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
)

# Legacy aliases with colons (for backwards compatibility)
declare -A ALIASES=(
  ["qwen3.5:4b"]="qwen3.5-4b"
  ["qwen3.5:9b"]="qwen3.5-9b"
  ["gemma4:e4b"]="gemma4-e4b"
  ["gemma4:e2b"]="gemma4-e2b"
  ["nemotron-3-nano:4b"]="nemotron-3-nano-4b"
)

download_model() {
  local key="$1"
  
  # Resolve legacy aliases (colon -> hyphen)
  if [[ -n "${ALIASES[$key]}" ]]; then
    echo "Note: '$key' is deprecated, use '${ALIASES[$key]}' instead"
    key="${ALIASES[$key]}"
  fi
  
  local repo_file="${MODELS[$key]}"
  if [[ -z "$repo_file" ]]; then
    echo "Error: Unknown model '$key'"
    return 1
  fi
  
  # Parse arguments: repo, remote_filename, [local_filename]
  local repo remote_file local_file
  read -r repo remote_file local_file <<< "$repo_file"
  
  # If no local filename specified, use remote filename
  local_file="${local_file:-$remote_file}"
  
  echo "Downloading $remote_file from $repo..."
  
  if [[ -f "$MODELS_DIR/$local_file" ]]; then
    echo "  ✓ Already exists: $MODELS_DIR/$local_file"
    return 0
  fi
  
  # Download to temp directory first
  local temp_dir=$(mktemp -d)
  trap "rm -rf $temp_dir" EXIT
  
  hf download "$repo" "$remote_file" --local-dir "$temp_dir"
  
  # Rename if needed
  if [[ "$remote_file" != "$local_file" ]]; then
    echo "  Renaming: $remote_file → $local_file"
    mv "$temp_dir/$remote_file" "$MODELS_DIR/$local_file"
  else
    mv "$temp_dir/$remote_file" "$MODELS_DIR/$local_file"
  fi
  
  echo "  ✓ Downloaded: $MODELS_DIR/$local_file"
}

show_sizes() {
  echo ""
  echo "Model Sizes (Q4_K_M):"
  echo "  qwen3.5-4b          2.63 GB  - Fits in VRAM"
  echo "  qwen3.5-9b          5.68 GB  - Partial offload"
  echo "  qwen3.5-27b        ~17.00 GB  - Heavy offload, slow (TOMBSTONE)"
  echo "  gemma4-e4b          4.98 GB  - Partial offload"
  echo "  gemma4-e2b          3.11 GB  - Fits in VRAM"
  echo "  gemma4-31b         ~18.00 GB  - Heavy offload, slow"
  echo "  nemotron-3-nano-4b  2.90 GB  - Fits in VRAM, tool-calling"
  echo ""
  echo "Legacy names with colons (still work):"
  echo "  qwen3.5:4b   → qwen3.5-4b"
  echo "  qwen3.5:9b   → qwen3.5-9b"
  echo "  gemma4:e4b   → gemma4-e4b"
  echo "  gemma4:e2b   → gemma4-e2b"
  echo "  nemotron-3-nano:4b → nemotron-3-nano-4b"
  echo ""
}

# Main
case "${1:-qwen3.5-4b}" in
  "qwen3.5-4b"|"qwen3.5-9b"|"qwen3.5-27b"|"gemma4-e4b"|"gemma4-e2b"|"nemotron-3-nano-4b")
    show_sizes
    download_model "$1"
    ;;
  "qwen3.5:4b"|"qwen3.5:9b"|"gemma4:e4b"|"gemma4:e2b"|"nemotron-3-nano:4b")
    # Legacy aliases with colons
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
    echo "Available: qwen3.5-4b, qwen3.5-9b, qwen3.5-27b, gemma4-e4b, gemma4-e2b, nemotron-3-nano-4b, all, sizes"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"