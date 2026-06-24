#!/bin/bash
# Download GGUF models for llama-swap
# Usage: ./download-models.sh [model-name]
#
# Available models:
#   qwen3.5-0.8b         - Qwen3.5-0.8B UD-Q3_K_XL (~0.46 GB) + mmproj (~196 MB) - fits in VRAM, vision+text
#   qwen3.5-4b           - Qwen3.5-4B MoQ-3.75 (~1.92 GB) - fits in VRAM
#   [REMOVED] lfm2-8b-moe — superseded by LFM2.5-8B-A1B, disabled May 2026
#   qwen3.5-9b           - Qwen3.5-9B MoQ-3.6 (~3.75 GB) - fits in VRAM + mmproj
#   gemma4-e4b           - Gemma-4 E4B Q4_K_M (~4.63 GB) - fits in VRAM + mmproj
#   gemma4-e2b       - Gemma-4 E2B Q4_0 QAT (~3.2 GB) - higher quality than PTQ, text-only
#   gemma4-12b - Gemma-4 12B Q4_0 QAT (~6.5 GB) - dense, ik backend with offload. Won vs PTQ 25/25 Snake.
#   [REMOVED] gemma4-e2b (PTQ) — superseded by QAT, Jun 2026
#   [REMOVED] nemotron-3-nano-4b — poor quality, superseded by Qwen3.5-4B/9B
#   lfm2.5-vl-450m       - LFM2.5-VL-450M Q4_0 (0.22 GB) + mmproj F16 - vision/OCR
#   lfm2.5-vl-1.6b-extract - LFM2.5-VL-1.6B-Extract Q4_K_M (0.70 GB) + mmproj F16 - structured vision extraction (JSON)
#   lfm2.5-vl-450m-extract - LFM2.5-VL-450M-Extract Q4_K_M (0.22 GB) + mmproj F16 - ultra-light structured extraction (JSON)
#   [REMOVED] lfm2.5-1.2b — superseded by LFM2.5-8B-A1B, disabled May 2026
#   lfm2.5-1.2b-think      - LFM2.5-1.2B-Thinking Q8_0 (~1.25 GB) - edge reasoning, CoT
#   lfm2-24b              - LFM2-24B-A2B Q4_K_M (~14.4 GB) - MoE hybrid, 2.3B active
#   [REMOVED] granite-4.1-3b — tool-calling failed in Pi, removed May 2026
#   [REMOVED] granite-4.1-8b — tool-calling failed in Pi, removed May 2026
#   [REMOVED] granite-4.0-h-1b — removed from fleet May 2026
#   [REMOVED] granite-4.0-h-1b-vllm — removed from fleet May 2026
#   [REMOVED] glm-4.7-flash — superseded by Qwen3.6 35B MoE
#   minicpm-v-4.6        - MiniCPM-V 4.6 Q5_K_M (~0.54 GB) + mmproj F16 (~1.03 GB) - VLM, video+text, 256K ctx
#   smolllm3-3b           - SmolLM3-3B UD-Q5_K_XL (~2.06 GB) - dense, tool-calling, 128K ctx
#   qwopus-coder-9b       - Qwopus3.5-9B-Coder Q4_K_M (~5.63 GB) + mmproj - agentic coding + tools
#   qwen3.5-4b-abliterated - Qwen3.5-4B abliterated i1-Q4_K_M (~2.71 GB) - no refusal, adversarial testing
#   glm-ocr               - GLM-OCR Q8_0 (~0.95 GB + 0.48 GB mmproj) - OCR/document specialist
#   nomic-embed-text-v2-moe - Nomic Embed v2 MoE Q4_K_M (~0.33 GB) - embedding, RAG/search
#   littlelamb-0.3b-tc   - LittleLamb 0.3B Tool-Calling Q8_0 (~0.30 GB) - ultra-light agentic, 40K ctx
#   webworld-8b          - WebWorld-8B i1-Q5_K_M (~5.9 GB) - web world model, predicts next page state
#   qwen3.6-35b-a3b      - Qwen3.6-35B-A3B APEX I-Compact (~17.3 GB) - MoE coding + tools
#   [REMOVED] gemma4-26b-a4b — removed from fleet Jun 2026 (replaced by Qwen 3.6 + North Mini Code)
#   gpt-oss-20b          - GPT-OSS 20B Q4_K_M (~11 GB) - Dense coding, text-only
#   [REMOVED] granite-4.0-h-1b-vllm — removed from fleet May 2026
#   [REMOVED] granite-4.0-h-1b — removed from fleet May 2026
#   mellum2-12b-thinking  - Mellum2-12B-A2.5B-Thinking Q4_K_M (~7.6 GB) - JetBrains MoE, reasoning + tools, manual conversion
#   ornstein-36-35b       - Ornstein 3.6 35B SABER Q4_K_M (~21.7 GB) - Qwen3.6 MoE NSC-ACE-SABER fine-tune, test only
#   ds-r1-distill-14b    - [REMOVED] Dense 14B, poor perf on RTX 3050
#   ds-r1-distill-32b    - [REMOVED] Dense 32B, very slow on limited VRAM
#   qwopus-35b           - Qwopus3.6-35B-A3B-v1 APEX I-Compact (~16.5 GB) - MoE coding+reasoning SFT
#   [REMOVED] qwen3.5-9b-ace — superseded by Qwopus for agentic tasks
#   [REMOVED] nanbeige4.1-3b — multi-turn tool calling broken (#22684), GGUF deleted
#   ornstein-36-35b       - Ornstein 3.6 35B SABER Q4_K_M (~21.7 GB) - Qwen3.6 MoE NSC-ACE-SABER fine-tune, test only
#   locate-anything       - LocateAnything-3B Q8_0 (~6.3 GB) - NVIDIA visual grounding, NOT llama-server (subprocess CLI)
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
  ["qwen3.5-4b"]="kaitchup/Qwen3.5-4B-GGUF-MoQ MoQ-3.75.gguf"
  # [REMOVED] lfm2-8b-moe — superseded by LFM2.5-8B-A1B, disabled May 2026
  ["lfm2.5-8b-a1b"]="LiquidAI/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-Q4_0.gguf"
  ["qwen3.5-9b"]="w-ahmad/Qwen3.5-9B-GGUF-MoQ Qwen3.5-9B-MoQ-3.6.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf"
  # [REMOVED] gemma4-e2b — superseded by gemma4-e2b (QAT won benchmark, Jun 2026)
#   [REMOVED] gemma4-e2b (PTQ) — superseded by QAT, Jun 2026
  ["gemma4-e2b"]="google/gemma-4-E2B-it-qat-q4_0-gguf gemma-4-E2B_q4_0-it.gguf"
  ["gemma4-12b"]="google/gemma-4-12B-it-qat-q4_0-gguf gemma-4-12b-it-qat-q4_0.gguf"
  # mmproj for 12B Unified (vision+audio projector, ~168 MB)
  ["gemma4-12b-mmproj"]="google/gemma-4-12B-it-qat-q4_0-gguf mmproj-gemma-4-12b-it-qat-q4_0.gguf"
#   [REMOVED] gemma4-e2b (PTQ) — superseded by QAT, Jun 2026
  # [REMOVED] nemotron-3-nano-4b — poor quality, superseded by Qwen3.5-4B/9B
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-Q8_0.gguf"
  # LFM2.5-VL-1.6B-Extract — structured vision extraction, returns JSON not free-form text
  # lfm2 arch — upstream llama.cpp ONLY (ik_llama.cpp does not support lfm2)
  ["lfm2.5-vl-1.6b-extract"]="LiquidAI/LFM2.5-VL-1.6B-Extract-GGUF LFM2.5-VL-1.6B-Extract-Q4_K_M.gguf"
  # LFM2.5-VL-450M-Extract — ultra-light structured extraction (450M params)
  # Same Extract family as 1.6B but smaller — edge deployment, faster inference
  ["lfm2.5-vl-450m-extract"]="LiquidAI/LFM2.5-VL-450M-Extract-GGUF LFM2.5-VL-450M-Extract-Q4_K_M.gguf"
  # [REMOVED] lfm2.5-1.2b — superseded by LFM2.5-8B-A1B, disabled May 2026
  # Only Instruct variant was in fleet; Thinking variant never downloaded
  # [REMOVED] lfm2.5-1.2b-think — superseded by LFM2.5-8B-A1B, disabled May 2026
  ["lfm2-24b"]="LiquidAI/LFM2-24B-A2B-GGUF LFM2-24B-A2B-Q4_K_M.gguf"
  # Granite 4.1 — dense, Apache 2.0, strong tool-calling + code
  # NOTE: Gemma 4 MTP assistants available but NOT useful on RTX 3050 6GB
  # E2B: MTP overhead > speedup (10.6 vs 38.3 tok/s). E4B: OOM with MTP.
  # To download+convert manually for other hardware:
  #   hf download google/gemma-4-E2B-it-assistant --local-dir ~/.llama-models/gemma-4-E2B-it-assistant
  #   python3 ~/git/ik_llama.cpp/convert_hf_to_gguf.py ~/.llama-models/gemma-4-E2B-it-assistant --outfile ~/.llama-models/gemma-4-E2B-it-assistant-Q4_K_M.gguf --outtype q4_k_m
  # glm-4.7-flash removed
  ["qwen3.6-35b-a3b"]="mudler/Qwen3.5-35B-A3B-APEX-GGUF Qwen3.5-35B-A3B-APEX-I-Compact.gguf Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
  # [REMOVED] granite-4.0-h-1b — removed from fleet May 2026
  ["qwopus-35b"]="mudler/Qwopus3.6-35B-A3B-v1-APEX-GGUF Qwopus3.6-35B-A3B-v1-APEX-I-Compact.gguf"
  # [REMOVED] gemma4-26b-a4b APEX I-Compact — superseded by QAT Q4_0 (faster, smaller, same quality)
  # [REMOVED] gemma4-26b-a4b QAT Q4_0 — removed from fleet Jun 2026 (replaced by Qwen 3.6 + North Mini Code)
  ["gpt-oss-20b"]="unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf"
  # [REMOVED] ministral-3-3b — removed from fleet May 2026
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
  # Hy-MT2-1.8B — Tencent Hunyuan Translation Model v2.0, 33 langs + 5 dialects
  # Uses hunyuan_v1_dense arch — upstream llama.cpp ONLY (no ik support)
  # Potential TranslateGemma alternative for Sprachspiel translate mode
  ["hy-mt2-1.8b"]="tencent/Hy-MT2-1.8B-GGUF Hy-MT2-1.8B-Q4_K_M.gguf"
  # TranslateGemma 4B — Google translation model, 55 langs, Sprachspiel default
  # Uses gemma3 arch — supported in both ik_llama.cpp and upstream
  # Q4_K_M ~2.7GB — fits in 6GB VRAM with room to spare
  ["translategemma-4b"]="mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q4_K_M.gguf"
  # Qwen3.5-4B-abliterated -- refusal-removed variant for adversarial testing
  # Same qwen35 arch as base Qwen3.5-4B, Abliterix orthogonalized steering
  # i1 (imatrix) quant for better quality at same bitrate
  ["qwen3.5-4b-abliterated"]="mradermacher/Qwen3.5-4B-abliterated-i1-GGUF Qwen3.5-4B-abliterated-i1-Q4_K_M.gguf"
  # GLM-OCR -- OCR and document understanding specialist (glm4 arch, mmproj required)
  # #1 on OmniDocBench V1.5 (94.62). Complements LFM2.5-VL-450M.
  # Q8_0 model + Q8_0 mmproj from official ggml-org release
  ["glm-ocr"]="ggml-org/GLM-OCR-GGUF GLM-OCR-Q8_0.gguf"
  # Nomic Embed v2 MoE -- embedding model for RAG/semantic search/similarity
  # 475M params (305M active), 768-dim with Matryoshka, 100+ languages
  # Q4_K_M only 328MB -- runs via /v1/embeddings with --embeddings --pooling mean
  ["nomic-embed-text-v2-moe"]="nomic-ai/nomic-embed-text-v2-moe-GGUF nomic-embed-text-v2-moe.Q4_K_M.gguf nomic-embed-text-v2-moe-Q4_K_M.gguf"
  # Nanbeige4.1-3B — BOSS Zhipin dense 3B reasoning + agentic coding model
  # Architecture: LlamaForCausalLM (llama) — supported in ALL backends
  # Q4_K_M ~1.8 GB — fits entirely in 6GB VRAM with room to spare
  # ALWAYS THINKS (no :think toggle) — uses think/end_think format (same as Qwen3)
  # Tool calling uses XML func_call tags — NOT OpenAI-compatible JSON
  # head_dim=128 → attn_rot ✅, vocab 166K (larger than Qwen3.5's 151K)
  # [REMOVED] nanbeige4.1-3b — multi-turn tool calling broken (#22684), GGUF deleted
  # Mellum2-12B-A2.5B-Thinking — JetBrains MoE 12B/2.5B, reasoning + tool calling
  # Architecture: Qwen3-MoE derivative (MellumForCausalLM alias registered in ik)
  # MANUAL CONVERSION (2026-06-01): no community GGUF available yet
  # Converted from BF16 safetensors → F16 GGUF → Q4_K_M via ik_llama.cpp's convert_hf_to_gguf.py
  # Steps: 1) Register MellumForCausalLM alias 2) Patch rope_parameters 3) Add mellum tokenizer hash 4) Convert F16 5) Quantize Q4_K_M
  # Swap for community GGUF when available
  ["mellum2-12b-thinking"]="LOCAL Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf"
  # Ornstein 3.6 35B SABER — Qwen3.6-35B-A3B NSC-ACE-SABER fine-tune, +2.87pp BFCL tool calling
  # Architecture: qwen35moe (same as Qwen3.6-35B-A3B-APEX), no APEX quant, standard Q4_K_M
  # TEST ONLY: evaluating as potential replacement for Qwen3.6-APEX
  ["ornstein-36-35b"]="GestaltLabs/Qwen3.6-35B-A3B-NSC-ACE-SABER-GGUF Qwen3.6-35B-A3B-NSC-ACE-SABER-Q4_K_M.gguf"
)

# Multimodal projector files (downloaded alongside their vision models)
# Format: "repo filename [local_filename]"
declare -A MMPROJ=(
  ["lfm2.5-vl-450m"]="LiquidAI/LFM2.5-VL-450M-GGUF mmproj-LFM2.5-VL-450m-F16.gguf"
  ["lfm2.5-vl-1.6b-extract"]="LiquidAI/LFM2.5-VL-1.6B-Extract-GGUF mmproj-LFM2.5-VL-1.6B-Extract-F16.gguf"
  ["lfm2.5-vl-450m-extract"]="LiquidAI/LFM2.5-VL-450M-Extract-GGUF mmproj-LFM2.5-VL-450M-Extract-F16.gguf"
  ["qwen3.6-35b-a3b"]="mudler/Qwen3.5-35B-A3B-APEX-GGUF mmproj-F16.gguf mmproj-Qwen3.6-35B-A3B-F16.gguf"
  ["qwen3.5-4b"]="unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-4B-F16.gguf"
  ["qwen3.5-9b"]="unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-9B-F16.gguf"
  ["gemma4-e4b"]="unsloth/gemma-4-E4B-it-GGUF mmproj-F16.gguf mmproj-gemma-4-E4B-F16.gguf"
# [REMOVED] gemma4-e2b mmproj — deleted Jun 2026, vision not needed (dedicated VLMs in fleet)
  ["qwen3.5-0.8b"]="unsloth/Qwen3.5-0.8B-GGUF mmproj-F16.gguf mmproj-Qwen3.5-0.8B-F16.gguf"
  # [REMOVED] ministral-3-3b mmproj — removed from fleet May 2026
  # MiniCPM-V 4.6 — mmproj includes SigLIP2-400M vision encoder (1.03 GB F16)
  ["minicpm-v-4.6"]="openbmb/MiniCPM-V-4.6-gguf mmproj-model-f16.gguf mmproj-MiniCPM-V-4.6-F16.gguf"
  # Qwopus3.5-9B-Coder — vision model, mmproj renamed for clarity
  ["qwopus-coder-9b"]="Jackrong/Qwopus3.5-9B-Coder-GGUF mmproj.gguf mmproj-Qwopus3.5-9B-coder-F16.gguf"
  # GLM-OCR mmproj -- Q8_0 vision projection model
  ["glm-ocr"]="ggml-org/GLM-OCR-GGUF mmproj-GLM-OCR-Q8_0.gguf"

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
  echo "  qwen3.5-4b            ~1.92 GB  (MoQ-3.75) - Fits in VRAM"
  echo "  qwen3.5-9b           ~3.75 GB  (MoQ-3.6) - Fits in VRAM + mmproj"
  echo "  gemma4-e4b           ~4.63 GB  (Q4_K_M) - Fits in VRAM + mmproj"
  echo "  gemma4-e2b       ~3.2 GB   (Q4_0 QAT) - Text-only, higher quality than PTQ"
  gemma4-12b          ~6.67 GB  (Q4_0 QAT + mmproj) - Dense 12B, upstream --fit on, unified multimodal
#   [REMOVED] gemma4-e2b (PTQ) — superseded by QAT, Jun 2026
  echo "  [REMOVED] nemotron-3-nano-4b — poor quality"
  echo "  lfm2.5-vl-450m        0.22 GB  (Q4_0) - Fits in VRAM, vision/OCR + mmproj"
  echo "  lfm2.5-vl-1.6b-extract 0.70 GB  (Q4_K_M) + 0.82 GB mmproj - Structured vision extraction (JSON), lfm2 arch (upstream llama.cpp only)"
  # [REMOVED] lfm2.5-1.2b — superseded by LFM2.5-8B-A1B
  # [REMOVED] lfm2.5-1.2b-think — superseded by LFM2.5-8B-A1B
  echo "  lfm2-24b            ~14.40 GB  (Q4_K_M) - Heavy offload, MoE hybrid"
  echo "  [REMOVED] granite-4.1-3b — tool-calling failed in Pi"
  echo "  [REMOVED] granite-4.1-8b — tool-calling failed in Pi"
  echo "  [REMOVED] granite-4.0-h-1b — removed from fleet May 2026"
  # glm-4.7-flash removed
  echo "  qwen3.6-35b-a3b    ~17.30 GB  (APEX I-Compact) - Heavy offload, MoE coding + tools"
  echo "  qwopus-35b        ~16.50 GB  (APEX I-Compact) - Heavy offload, MoE coding+reasoning SFT"
  # [REMOVED] gemma4-26b-a4b — removed from fleet Jun 2026
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
  echo "  qwen3.5-4b-abliterated   ~2.71 GB  (i1-Q4_K_M) - Abliterated Qwen3.5-4B, no refusal"
  echo "  glm-ocr             ~0.95 GB  (Q8_0) + 0.48 GB mmproj - OCR/document specialist"
  echo "  nomic-embed-text-v2-moe  ~0.33 GB  (Q4_K_M) - Embedding, RAG/search/similarity"
  echo "  mellum2-12b-thinking  ~7.60 GB  (Q4_K_M) - JetBrains MoE 12B/2.5B, reasoning + tools (manual conversion)"
  echo "  ornstein-36-35b       ~21.70 GB (Q4_K_M) - Qwen3.6-35B NSC-ACE-SABER fine-tune, +2.87pp BFCL (test only)"
  echo ""
  echo "vLLM-only models (safetensors, auto-downloaded on first serve):"
  echo "  [REMOVED] granite-4.0-h-1b-vllm — removed from fleet May 2026"
  echo ""
  echo "GGUF models (removed):"
  echo "  [REMOVED] granite-4.0-h-1b — removed from fleet May 2026"
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
  *)
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
    echo "Available: qwen3.5-0.8b, qwen3.5-4b, qwen3.5-9b, gemma4-e4b, gemma4-e2b, lfm2.5-vl-450m, lfm2.5-8b-a1b, lfm2-24b, webworld-8b, qwen3.6-35b-a3b, qwopus-35b, gpt-oss-20b, qwopus-coder-9b, mellum2-12b-thinking, ornstein-36-35b, locate-anything, all"
    exit 1
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
echo "Contents:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no GGUF files yet)"