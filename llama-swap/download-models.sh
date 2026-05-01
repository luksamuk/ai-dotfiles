#!/bin/bash
# Download GGUF models for llama-swap
# Usage: ./download-models.sh [model-name]
#
# Available models:
#   qwen3.5-4b           - Qwen3.5-4B Q4_K_M (2.63 GB) - fits in VRAM
#   qwen3.5-9b           - Qwen3.5-9B UD-Q3_K_XL (~5.05 GB) - fits in VRAM + mmproj
#   qwen3-14b            - Qwen3-14B Q4_K_M (~8.5 GB) - REMOVED (GGUF deleted)
#   qwen3-coder-30b     - REMOVED (replaced by qwen3.6-35b-moe)
#   gemma4-e4b           - Gemma-4 E4B UD-Q3_K_XL (~4.5 GB) - fits in VRAM + mmproj
#   gemma4-e2b           - Gemma-4 E2B Q4_K_M (3.11 GB) - fits in VRAM
#   nemotron-3-nano-4b   - Nemotron-3-Nano-4B Q4_K_M (2.90 GB) - tool-calling
#   lfm2.5-vl-450m       - LFM2.5-VL-450M Q4_0 (0.22 GB) + mmproj F16 - vision/OCR
#   qwen3.6-27b          - Qwen3.6-27B UD-Q3_K_XL (~14.5 GB) - Dense vision + coding (Dynamic 2.0)
#   qwen3.6-35b-moe      - Qwen3.6-35B-A3B UD-Q3_K_XL (~13.8 GB) - MoE coding + tools (Dynamic 2.0)
#   nemotron3-omni-30b   - Nemotron-3-Nano-Omni-30B-A3B UD-Q3_K_XL (~17 GB) + mmproj - Omni
#   qwopus-glm-18b       - REMOVED (GGUF deleted, no advantage over qwen3.6-27b)
#   all                  - Download all models
#
# If no argument, downloads qwen3.5-4b (fits entirely in 6GB VRAM)

set -e

MODELS_DIR="${HOME}/.llama-models"

# Create directory
mkdir -p "$MODELS_DIR"

# HF cache on disk (not tmpfs) — /tmp can be too small for models >16GB
export HF_HUB_CACHE="${HOME}/.cache/huggingface"

# Model definitions
# Format: "repo filename [local_filename]"
# If local_filename is provided, the file is renamed after download
declare -A MODELS=(
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q3_K_XL.gguf"
  ["qwen3.5-27b"]="TOMBSTONE - use qwen3-14b instead"
  ["qwen3-14b"]="TOMBSTONE - GGUF deleted, use qwen3.6-27b"
  ["qwopus3.5-4b"]="TOMBSTONE - GGUF removed, model unlisted"
  ["qwopus3.5-9b"]="TOMBSTONE - GGUF removed, model unlisted"
  ["carnice-9b"]="TOMBSTONE - GGUF removed, model unlisted"
  ["qwen3-coder-30b"]="TOMBSTONE - replaced by qwen3.6-35b-moe (better coding model)"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-UD-Q3_K_XL.gguf"
  ["gemma4-e2b"]="unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf"
  ["gemma4-31b"]="TOMBSTONE - too large for RTX3050 6GB"
  ["nemotron-3-nano-4b"]="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-Q4_0.gguf"
  ["qwen3.6-27b"]="unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-UD-Q3_K_XL.gguf"
  ["qwen3.6-35b-moe"]="unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
  ["qwopus-glm-18b"]="TOMBSTONE - GGUF deleted, no advantage over qwen3.6-27b"
  ["nemotron3-omni-30b"]="unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q3_K_XL.gguf"
)

# Multimodal projector files (downloaded alongside their vision models)
# Format: "repo filename [local_filename]"
declare -A MMPROJ=(
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF mmproj-LFM2.5-VL-450m-F16.gguf"
  ["qwen3.6-27b"]="unsloth/Qwen3.6-27B-GGUF mmproj-F16.gguf mmproj-Qwen3.6-27B-F16.gguf"
  ["qwen3.6-35b-moe"]="unsloth/Qwen3.6-35B-A3B-GGUF mmproj-F16.gguf mmproj-Qwen3.6-35B-A3B-F16.gguf"
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-4B-F16.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-9B-F16.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF mmproj-F16.gguf mmproj-gemma-4-E4B-F16.gguf"
  ["gemma4-e2b"]="unsloth/gemma-4-E2B-it-GGUF mmproj-F16.gguf mmproj-gemma-4-E2B-F16.gguf"
  ["nemotron3-omni-30b"]="unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF mmproj-F16.gguf mmproj-Nemotron-3-Nano-Omni-30B-A3B-F16.gguf"
)

# Legacy aliases with colons (for backwards compatibility)
declare -A ALIASES=(
  ["qwen3.5:4b"]="qwen3.5-4b"
  ["qwen3.5:9b"]="qwen3.5-9b"
  ["gemma4:e4b"]="gemma4-e4b"
  ["gemma4:e2b"]="gema4-e2b"
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
  
  # Check if model is a tombstone (removed but kept as reference)
  if [[ "$repo_file" == TOMBSTONE* ]]; then
    echo "⚠️  Model '$key' has been removed from disk."
    echo "   Reason: $repo_file"
    echo "   To re-enable: download the model manually and uncomment its config in llama-swap."
    return 1
  fi
  
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
  else
    # Download directly to models dir (avoids tmpfs /tmp for large models)
    hf download "$repo" "$remote_file" --local-dir "$MODELS_DIR"
    
    # Rename if needed
    if [[ "$remote_file" != "$local_file" ]]; then
      echo "  Renaming: $remote_file → $local_file"
      mv "$MODELS_DIR/$remote_file" "$MODELS_DIR/$local_file"
    fi
    
    echo "  ✓ Downloaded: $MODELS_DIR/$local_file"
  fi
  
  # Download mmproj if applicable
  local mmproj_entry="${MMPROJ[$key]}"
  if [[ -n "$mmproj_entry" ]]; then
    local mmproj_repo mmproj_remote_file mmproj_local_file
    read -r mmproj_repo mmproj_remote_file mmproj_local_file <<< "$mmproj_entry"
    mmproj_local_file="${mmproj_local_file:-$mmproj_remote_file}"
    
    echo "Downloading mmproj: $mmproj_remote_file from $mmproj_repo..."
    
    if [[ -f "$MODELS_DIR/$mmproj_local_file" ]]; then
      echo "  ✓ Already exists: $MODELS_DIR/$mmproj_local_file"
    else
      hf download "$mmproj_repo" "$mmproj_remote_file" --local-dir "$MODELS_DIR"
      
      if [[ "$mmproj_remote_file" != "$mmproj_local_file" ]]; then
        echo "  Renaming: $mmproj_remote_file → $mmproj_local_file"
        mv "$MODELS_DIR/$mmproj_remote_file" "$MODELS_DIR/$mmproj_local_file"
      fi
      echo "  ✓ Downloaded mmproj: $MODELS_DIR/$mmproj_local_file"
    fi
  fi
}

show_sizes() {
  echo ""
  echo "Model Sizes (quantization noted):"
  echo "  qwen3.5-4b            2.63 GB  (Q4_K_M) - Fits in VRAM"
  echo "  qwen3.5-9b           ~5.05 GB  (UD-Q3_K_XL) - Fits in VRAM + mmproj"
  echo "  qwen3.5-27b            REMOVED - use qwen3-14b"
  echo "  qwen3-14b              REMOVED - GGUF deleted, use qwen3.6-27b"
  echo "  qwen3-coder-30b        REMOVED - replaced by qwen3.6-35b-moe"
  echo "  qwopus3.5-4b           REMOVED - GGUF deleted"
  echo "  qwopus3.5-9b           REMOVED - GGUF deleted"
  echo "  carnice-9b             REMOVED - GGUF deleted"
  echo "  gemma4-e4b           ~4.50 GB  (UD-Q3_K_XL) - Fits in VRAM + mmproj"
  echo "  gemma4-e2b            3.11 GB  (Q4_K_M) - Fits in VRAM"
  echo "  gemma4-31b             REMOVED - too large for RTX3050 6GB"
  echo "  nemotron-3-nano-4b    2.90 GB  (Q4_K_M) - Fits in VRAM, tool-calling"
  echo "  lfm2.5-vl-450m        0.22 GB  (Q4_0) - Fits in VRAM, vision/OCR + mmproj"
  echo "  qwen3.6-27b        ~14.50 GB  (UD-Q3_K_XL) - Heavy offload, dense vision + coding + mmproj"
  echo "  qwen3.6-35b-moe    ~13.80 GB  (UD-Q3_K_XL) - Heavy offload, MoE coding + tools + mmproj"
  echo "  qwopus-glm-18b        REMOVED - GGUF deleted"
  echo "  nemotron3-omni-30b  ~17.00 GB  (UD-Q3_K_XL) - Heavy offload, Omni + mmproj"
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
  "qwen3.5-4b"|"qwen3.5-9b"|"qwen3-14b"|"qwen3-coder-30b"|"qwopus3.5-4b"|"qwopus3.5-9b"|"carnice-9b"|"gemma4-e4b"|"gemma4-e2b"|"nemotron-3-nano-4b"|"lfm2.5-vl-450m"|"qwen3.6-27b"|"qwen3.6-35b-moe"|"qwopus-glm-18b"|"nemotron3-omni-30b")
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
    echo "Available: qwen3.5-4b, qwen3.5-9b, gemma4-e4b, gemma4-e2b, nemotron-3-nano-4b, lfm2.5-vl-450m, qwen3.6-27b, qwen3.6-35b-moe, nemotron3-omni-30b, all, sizes"
    echo "Removed: qwen3.5-27b, qwen3-14b, qwen3-coder-30b, qwopus-glm-18b, gemma4-31b, qwen3.6-35b-moe (old)"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"