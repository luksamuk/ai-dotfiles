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
#   lfm2.5-1.2b           - LFM2.5-1.2B-Instruct Q8_0 (~1.25 GB) - edge, tool-calling
#   lfm2.5-1.2b-think      - LFM2.5-1.2B-Thinking Q8_0 (~1.25 GB) - edge reasoning, CoT
#   lfm2-24b              - LFM2-24B-A2B Q4_K_M (~14.4 GB) - MoE hybrid, 2.3B active
#   [REMOVED] granite-4.1-3b — tool-calling failed in Pi, removed May 2026
#   [REMOVED] granite-4.1-8b — tool-calling failed in Pi, removed May 2026
#   [REMOVED] glm-4.7-flash — superseded by Qwen3.6 35B MoE
#   minicpm-v-4.6        - MiniCPM-V 4.6 Q5_K_M (~0.54 GB) + mmproj F16 (~1.03 GB) - VLM, video+text, 256K ctx
#   smolllm3-3b           - SmolLM3-3B UD-Q5_K_XL (~2.06 GB) - dense, tool-calling, 128K ctx
#   qwopus-coder-9b       - Qwopus3.5-9B-Coder Q4_K_M (~5.63 GB) + mmproj - agentic coding + tools
#   littlelamb-0.3b-tc   - LittleLamb 0.3B Tool-Calling Q8_0 (~0.30 GB) - ultra-light agentic, 40K ctx
#   webworld-8b          - WebWorld-8B i1-Q5_K_M (~5.9 GB) - web world model, predicts next page state
#   qwen3.6-35b-moe      - Qwen3.6-35B-A3B APEX I-Compact (~17.3 GB) - MoE coding + tools
#   gemma4-26b-moe       - Gemma 4 26B-A4B APEX I-Compact (~15.5 GB) - MoE reasoning + coding, text-only
#   gpt-oss-20b          - GPT-OSS 20B Q4_K_M (~11 GB) - Dense coding, text-only
#   granite-4.0-h-1b-vllm - (vLLM only, auto-downloads from HF: ibm-granite/granite-4.0-h-1b ~2.8 GB BF16)
#   granite-4.0-h-1b      - Granite 4.0 H 1B Nano UD-Q3_K_XL (~708 MB) - hybrid Mamba-2, multilingual (PT), thinking+tools
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
# NOTE: MTP GGUFs available (unsloth/Qwen3.5-XB-MTP-GGUF) but Qwen3.5 SSM architecture
# requires ~1.3GB Gated Delta Net compute buffer per context, making MTP unusable on 6GB VRAM
declare -A MODELS=(
  ["qwen3.5-0.8b"]="unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-UD-Q3_K_XL.gguf"
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q3_K_XL.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q3_K_XL.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-UD-Q3_K_XL.gguf"
  ["gemma4-e2b"]="unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-UD-Q3_K_XL.gguf"
  # [REMOVED] nemotron-3-nano-4b — poor quality, superseded by Qwen3.5-4B/9B
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-Q8_0.gguf"
  ["lfm2.5-1.2b"]="LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q8_0.gguf"
  ["lfm2.5-1.2b-think"]="LiquidAI/LFM2.5-1.2B-Thinking-GGUF LFM2.5-1.2B-Thinking-Q8_0.gguf"
  ["lfm2-24b"]="LiquidAI/LFM2-24B-A2B-GGUF LFM2-24B-A2B-Q4_K_M.gguf"
  # Granite 4.1 — dense, Apache 2.0, strong tool-calling + code
  # NOTE: Gemma 4 MTP assistants available but NOT useful on RTX 3050 6GB
  # E2B: MTP overhead > speedup (10.6 vs 38.3 tok/s). E4B: OOM with MTP.
  # To download+convert manually for other hardware:
  #   hf download google/gemma-4-E2B-it-assistant --local-dir ~/.llama-models/gemma-4-E2B-it-assistant
  #   python3 ~/git/ik_llama.cpp/convert_hf_to_gguf.py ~/.llama-models/gemma-4-E2B-it-assistant --outfile ~/.llama-models/gemma-4-E2B-it-assistant-Q4_K_M.gguf --outtype q4_k_m
  # glm-4.7-flash removed
  ["qwen3.6-35b-moe"]="mudler/Qwen3.5-35B-A3B-APEX-GGUF Qwen3.5-35B-A3B-APEX-I-Compact.gguf Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
  # Granite 4.0 H 1B Nano — hybrid Mamba-2, Apache 2.0, 12 langs (incl. PT), thinking+tools
  # Uses granite-4.0 arch — supported in llama.cpp v546+ and ik_llama.cpp v4504+
  ["granite-4.0-h-1b"]="unsloth/granite-4.0-h-1b-GGUF granite-4.0-h-1b-UD-Q3_K_XL.gguf"
  ["qwopus-35b"]="mudler/Qwopus3.6-35B-A3B-v1-APEX-GGUF Qwopus3.6-35B-A3B-v1-APEX-I-Compact.gguf"
  ["gemma4-26b-moe"]="mudler/gemma-4-26B-A4B-it-APEX-GGUF gemma-4-26B-A4B-APEX-I-Compact.gguf"
  ["gpt-oss-20b"]="unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf"
  # Ministral 3 3B — Mistral edge model, text-only (mmproj crashes CUDA), tool calling
  # Uses mistral3 arch — ONLY llama.cpp upstream, NOT ik_llama.cpp
  ["ministral-3-3b"]="unsloth/Ministral-3-3B-Instruct-2512-GGUF Ministral-3-3B-Instruct-2512-UD-Q5_K_XL.gguf"
  # MiniCPM-V 4.6 — VLM with SigLIP2-400M, Qwen3.5-0.8B backbone, 256K ctx, video+image+text
  # Uses qwen35 arch — supported in both ik_llama.cpp and upstream
  # Q5_K_M: good q/size ratio for a 1.3B model that fits entirely in VRAM
  ["minicpm-v-4.6"]="openbmb/MiniCPM-V-4.6-gguf MiniCPM-V-4_6-Q5_K_M.gguf"
  # SmolLM3-3B — HuggingFace dense model, dual tool-calling (XML+Python), 128K ctx
  # Uses smollm3 arch — supported in both ik_llama.cpp and upstream
  ["smolllm3-3b"]="unsloth/SmolLM3-3B-GGUF SmolLM3-3B-UD-Q5_K_XL.gguf"
  # LittleLamb 0.3B TC — ultra-light tool-calling model compressed from Qwen3-0.6B via CompactifAI
  # Uses qwen3 arch — supported in both ik_llama.cpp and upstream
  # Q8_0 preferred: 290M params tiny, quality matters more than size
  ["littlelamb-0.3b-tc"]="mradermacher/LittleLamb-ToolCalling-GGUF LittleLamb-ToolCalling.Q8_0.gguf"
  # WebWorld-8B — Qwen3-8B web world model, predicts next page state given current state + action
  # Uses qwen3 arch — ik_llama.cpp segfaults, use upstream only
  # i1 (imatrix) quantization for better quality at aggressive compression
  ["webworld-8b"]="mradermacher/WebWorld-8B-i1-GGUF WebWorld-8B.i1-Q5_K_M.gguf"
  # [REMOVED] ds-r1-distill-14b — Dense 14B, poor perf on RTX 3050, SSD pressure
  # [REMOVED] ds-r1-distill-32b — Dense 32B, very slow on limited VRAM, SSD pressure
  # [REMOVED] qwen3.5-9b-ace — analyzed, worse perplexity than 9B regular (no imatrix)
  # Qwopus3.5-9B-Coder — Qwen3.5-9B fine-tuned for agentic coding + tool calling (Trace Inversion + GLM-5.1 traces)
  # Uses qwen35 arch — supported in both ik_llama.cpp and upstream
  # "Coder" name = agentic coding focus, but also does browser/memory/delegation traces
  ["qwopus-coder-9b"]="Jackrong/Qwopus3.5-9B-Coder-GGUF Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf"
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
  # Ministral 3 3B — mmproj downloaded but UNUSED (crashes CUDA on mistral3 arch)
  ["ministral-3-3b"]="unsloth/Ministral-3-3B-Instruct-2512-GGUF mmproj-F16.gguf mmproj-Ministral-3-3B-F16.gguf"
  # MiniCPM-V 4.6 — mmproj includes SigLIP2-400M vision encoder (1.03 GB F16)
  ["minicpm-v-4.6"]="openbmb/MiniCPM-V-4.6-gguf mmproj-model-f16.gguf mmproj-MiniCPM-V-4.6-F16.gguf"
  # Qwopus3.5-9B-Coder — vision model, mmproj renamed for clarity
  ["qwopus-coder-9b"]="Jackrong/Qwopus3.5-9B-Coder-GGUF mmproj.gguf mmproj-Qwopus3.5-9B-coder-F16.gguf"

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
    echo "Available: qwen3.5-0.8b, qwen3.5-4b, qwen3.5-9b, gemma4-e4b, gemma4-e2b, lfm2.5-vl-450m, lfm2.5-1.2b, lfm2-24b, minicpm-v-4.6, smolllm3-3b, littlelamb-0.3b-tc, webworld-8b, qwen3.6-35b-moe, qwopus-35b, gemma4-26b-moe, gpt-oss-20b, ministral-3-3b, all (NOTE: granite-3.3-8b-vllm and granite-4.0-h-tiny-vllm are vLLM-only, auto-downloaded on first serve)"
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
  echo "  qwen3.5-0.8b          ~0.47 GB  (UD-Q3_K_XL) + ~0.20 GB mmproj - Tiny, vision+text"
  echo "  qwen3.5-4b            ~2.27 GB  (UD-Q3_K_XL) - Fits in VRAM"
  echo "  qwen3.5-9b           ~5.05 GB  (UD-Q3_K_XL) - Fits in VRAM + mmproj"
  echo "  gemma4-e4b           ~4.50 GB  (UD-Q3_K_XL) - Fits in VRAM + mmproj"
  echo "  gemma4-e2b            ~2.72 GB  (UD-Q3_K_XL) - Fits in VRAM"
  echo "  [REMOVED] nemotron-3-nano-4b — poor quality"
  echo "  lfm2.5-vl-450m        0.22 GB  (Q4_0) - Fits in VRAM, vision/OCR + mmproj"
  echo "  lfm2.5-1.2b           ~1.25 GB  (Q8_0) - Fits in VRAM, edge instruct tool-calling"
  echo "  lfm2.5-1.2b-think      ~1.25 GB  (Q8_0) - Fits in VRAM, edge reasoning CoT"
  echo "  lfm2-24b            ~14.40 GB  (Q4_K_M) - Heavy offload, MoE hybrid"
  echo "  granite-4.1-3b       ~2.10 GB  (Q4_K_M) - Dense, tool-calling, 128K ctx"
  echo "  granite-4.1-8b       ~4.60 GB  (UD-Q3_K_XL) - Dense, tool-calling, 512K ctx"
  # glm-4.7-flash removed
  echo "  qwen3.6-35b-moe    ~17.30 GB  (APEX I-Compact) - Heavy offload, MoE coding + tools"
  echo "  qwopus-35b        ~16.50 GB  (APEX I-Compact) - Heavy offload, MoE coding+reasoning SFT"
  echo "  gemma4-26b-moe    ~15.50 GB  (APEX I-Compact) - Heavy offload, MoE reasoning + coding text-only"
  echo "  gpt-oss-20b      ~11.00 GB  (Q4_K_M) - Heavy offload, dense coding text-only"
  echo "  minicpm-v-4.6      ~0.54 GB  (Q5_K_M) + 1.03 GB mmproj - VLM video+image+text, 256K ctx"
  echo "  smolllm3-3b         ~2.06 GB  (UD-Q5_K_XL) - Dense, dual tool-calling (XML+Python), 128K ctx"
  echo "  littlelamb-0.3b-tc  ~0.30 GB  (Q8_0) - Ultra-light tool-calling, 40K ctx"
  echo "  webworld-8b         ~5.90 GB  (i1-Q5_K_M) - Web world model, predicts next page state"
  echo "  ds-r1-distill-14b    [REMOVED] — poor perf on RTX 3050"
  echo "  ds-r1-distill-32b    [REMOVED] — very slow on limited VRAM"
  echo "  [REMOVED] nemotron-3-nano-4b"
  echo "  [REMOVED] qwen3.5-9b-ace — worse perplexity, no imatrix quant"
  echo "  qwopus-coder-9b      ~5.63 GB  (Q4_K_M) + mmproj - Dense 9B, agentic coding + tools"
  echo ""
  echo "vLLM-only models (safetensors, auto-downloaded on first serve):"
  echo "  granite-4.0-h-1b-vllm ~2.8 GB  (BF16) - IBM Granite 4.0 H 1B Nano, hybrid Mamba-2, multilingual (PT), thinking+tools"
  echo ""
  echo "GGUF models (also available as llama.cpp backend):"
  echo "  granite-4.0-h-1b      ~708 MB  (UD-Q3_K_XL) - IBM Granite 4.0 H 1B Nano, hybrid Mamba-2, multilingual (PT), thinking+tools"
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
  "qwen3.5-0.8b"|"qwen3.5-4b"|"qwen3.5-9b"|"gemma4-e4b"|"gemma4-e2b"|"lfm2.5-vl-450m"|"lfm2.5-1.2b"|"lfm2.5-1.2b-think"|"lfm2-24b"|"qwen3.6-35b-moe"|"qwopus-35b"|"gemma4-26b-moe"|"gpt-oss-20b"|"ministral-3-3b"|"minicpm-v-4.6"|"smolllm3-3b"|"littlelamb-0.3b-tc"|"webworld-8b"|"granite-4.0-h-1b"|"qwopus-coder-9b")
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
    echo "Available: qwen3.5-0.8b, qwen3.5-4b, qwen3.5-9b, gemma4-e4b, gemma4-e2b, lfm2.5-vl-450m, lfm2.5-1.2b, lfm2.5-1.2b-think, lfm2-24b, webworld-8b, qwen3.6-35b-moe, qwopus-35b, gemma4-26b-moe, gpt-oss-20b, qwopus-coder-9b, all"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"