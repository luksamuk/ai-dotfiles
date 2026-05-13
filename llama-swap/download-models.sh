#!/bin/bash
# Download GGUF models for llama-swap
# Usage: ./download-models.sh [model-name]
#
# Available models:
#   qwen3.5-0.8b         - Qwen3.5-0.8B UD-Q3_K_XL (~0.46 GB) + mmproj (~196 MB) - fits in VRAM, vision+text
#   qwen3.5-4b           - Qwen3.5-4B UD-Q3_K_XL (~2.27 GB) - fits in VRAM
#   qwen3.5-9b           - Qwen3.5-9B UD-Q3_K_XL (~5.05 GB) - fits in VRAM
#   gemma4-e4b           - Gemma-4 E4B UD-Q3_K_XL (~4.50 GB) - fits in VRAM
#   gemma4-e2b           - Gemma-4 E2B UD-Q3_K_XL (~2.72 GB) - fits in VRAM
#   [REMOVED] nemotron-3-nano-4b — poor quality, superseded by Qwen3.5-4B/9B
#   lfm2.5-vl-450m       - LFM2.5-VL-450M Q4_0 (0.22 GB) + mmproj F16 - vision/OCR
#   [REMOVED] granite-4.1-3b — tool-calling failed in Pi, removed May 2026
#   [REMOVED] granite-4.1-8b — tool-calling failed in Pi, removed May 2026
#   [REMOVED] glm-4.7-flash — superseded by Qwen3.6 35B MoE
#   qwen3.6-35b-moe      - Qwen3.6-35B-A3B APEX I-Compact (~17.3 GB) - MoE coding + tools
#   gemma4-26b-moe       - Gemma 4 26B-A4B APEX I-Compact (~15.5 GB) - MoE reasoning + coding, text-only
#   gpt-oss-20b          - GPT-OSS 20B Q4_K_M (~11 GB) - Dense coding, text-only
#   ds-r1-distill-14b    - [REMOVED] Dense 14B, poor perf on RTX 3050
#   ds-r1-distill-32b    - [REMOVED] Dense 32B, very slow on limited VRAM
#   qwopus-35b           - Qwopus3.6-35B-A3B-v1 APEX I-Compact (~16.5 GB) - MoE coding+reasoning SFT
#   [REMOVED] qwen3.5-9b-ace — superseded by Qwopus for agentic tasks
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
  ["qwen3.5-0.8b"]="unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-UD-Q3_K_XL.gguf"
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q3_K_XL.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q3_K_XL.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-UD-Q3_K_XL.gguf"
  ["gemma4-e2b"]="unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-UD-Q3_K_XL.gguf"
  # [REMOVED] nemotron-3-nano-4b — poor quality, superseded by Qwen3.5-4B/9B
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-Q4_0.gguf"
  # Granite 4.1 — dense, Apache 2.0, strong tool-calling + code
  # glm-4.7-flash removed
  ["qwen3.6-35b-moe"]="mudler/Qwen3.5-35B-A3B-APEX-GGUF Qwen3.5-35B-A3B-APEX-I-Compact.gguf Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
  ["qwopus-35b"]="mudler/Qwopus3.6-35B-A3B-v1-APEX-GGUF Qwopus3.6-35B-A3B-v1-APEX-I-Compact.gguf"
  ["gemma4-26b-moe"]="mudler/gemma-4-26B-A4B-it-APEX-GGUF gemma-4-26B-A4B-APEX-I-Compact.gguf"
  ["gpt-oss-20b"]="unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf"
  # [REMOVED] ds-r1-distill-14b — Dense 14B, poor perf on RTX 3050, SSD pressure
  # [REMOVED] ds-r1-distill-32b — Dense 32B, very slow on limited VRAM, SSD pressure
  # [REMOVED] qwen3.5-9b-ace — analyzed, worse perplexity than 9B regular (no imatrix)
)

# Multimodal projector files (downloaded alongside their vision models)
# Format: "repo filename [local_filename]"
declare -A MMPROJ=(
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF mmproj-LFM2.5-VL-450m-F16.gguf"
  ["qwen3.6-35b-moe"]="mudler/Qwen3.5-35B-A3B-APEX-GGUF mmproj-F16.gguf mmproj-Qwen3.6-35B-A3B-F16.gguf"
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-4B-F16.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-9B-F16.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF mmproj-F16.gguf mmproj-gemma-4-E4B-F16.gguf"
  ["gemma4-e2b"]="unsloth/gemma-4-E2B-it-GGUF mmproj-F16.gguf mmproj-gemma-4-E2B-F16.gguf"
  ["qwen3.5-0.8b"]="unsloth/Qwen3.5-0.8B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-0.8B-F16.gguf"

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
  
  if [[ -z "$repo_file" ]]; then
    echo "Error: Unknown model '$key'"
    echo "Available: qwen3.5-4b, qwen3.5-9b, gemma4-e4b, gemma4-e2b, nematron-3-nano-4b, lfm2.5-vl-450m, qwen3.6-35b-moe, gemma4-26b-moe, gpt-oss-20b, all"
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
    echo "  qwen3.5-0.8b          ~0.46 GB  (UD-Q3_K_XL) + ~0.20 GB mmproj - Tiny, vision+text"
    echo "  qwen3.5-4b            ~2.27 GB  (UD-Q3_K_XL) - Fits in VRAM"
  echo "  qwen3.5-9b           ~5.05 GB  (UD-Q3_K_XL) - Fits in VRAM + mmproj"
  echo "  gemma4-e4b           ~4.50 GB  (UD-Q3_K_XL) - Fits in VRAM + mmproj"
  echo "  gemma4-e2b            ~2.72 GB  (UD-Q3_K_XL) - Fits in VRAM"
  echo "  [REMOVED] nemotron-3-nano-4b — poor quality"
  echo "  lfm2.5-vl-450m        0.22 GB  (Q4_0) - Fits in VRAM, vision/OCR + mmproj"
  echo "  granite-4.1-3b       ~2.10 GB  (Q4_K_M) - Dense, tool-calling, 128K ctx"
  echo "  granite-4.1-8b       ~4.60 GB  (UD-Q3_K_XL) - Dense, tool-calling, 512K ctx"
  # glm-4.7-flash removed
  echo "  qwen3.6-35b-moe    ~17.30 GB  (APEX I-Compact) - Heavy offload, MoE coding + tools"
  echo "  qwopus-35b        ~16.50 GB  (APEX I-Compact) - Heavy offload, MoE coding+reasoning SFT"
  echo "  gemma4-26b-moe    ~15.50 GB  (APEX I-Compact) - Heavy offload, MoE reasoning + coding text-only"
  echo "  gpt-oss-20b      ~11.00 GB  (Q4_K_M) - Heavy offload, dense coding text-only"
  echo "  ds-r1-distill-14b    [REMOVED] — poor perf on RTX 3050"
  echo "  ds-r1-distill-32b    [REMOVED] — very slow on limited VRAM"
  echo "  [REMOVED] nemotron-3-nano-4b"
  echo "  [REMOVED] qwen3.5-9b-ace — worse perplexity, no imatrix quant"
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
  "qwen3.5-0.8b"|"qwen3.5-4b"|"qwen3.5-9b"|"gemma4-e4b"|"gemma4-e2b"|"lfm2.5-vl-450m"|"qwen3.6-35b-moe"|"qwopus-35b"|"gemma4-26b-moe"|"gpt-oss-20b")
    download_model "$1"
    ;;
  "qwen3.5:4b"|"qwen3.5:9b"|"gemma4:e4b"|"gemma4:e2b")
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
    echo "Available: qwen3.5-0.8b, qwen3.5-4b, qwen3.5-9b, gemma4-e4b, gemma4-e2b, lfm2.5-vl-450m, qwen3.6-35b-moe, qwopus-35b, gemma4-26b-moe, gpt-oss-20b, all"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"