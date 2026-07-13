#!/bin/bash
# diffuse — Download model weights from HuggingFace
#
# Usage:
#   ./download-model.sh hidream       # HiDream-O1 SDNQ (T2I + editing)
#   ./download-model.sh ideogram4     # Ideogram 4 Q4 (T2I, text rendering)
#   ./download-model.sh all           # All models
#
# Environment:
#   HF_TOKEN          HuggingFace token (if models require auth)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
LLAMA_MODELS_DIR="${HOME}/.llama-models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_hf_cli() {
    local venv_hf="${SCRIPT_DIR}/.venv/bin/hf"
    if [[ -x "$venv_hf" ]]; then
        HF_DOWNLOAD_CMD="$venv_hf download"
    elif command -v hf &>/dev/null; then
        HF_DOWNLOAD_CMD="hf download"
    else
        log_error "hf CLI not found"
        log_info "Run 'diffuse setup' first to install dependencies"
        exit 1
    fi
}

download_hidream() {
    local dir="${LLAMA_MODELS_DIR}/HiDream-O1-Image-Dev-SDNQ-last8"
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        log_info "HiDream already exists at ${dir}"
        return 0
    fi
    log_info "Downloading HiDream-O1 SDNQ from WaveCut/HiDream-O1-Image-Dev-SDNQ-uint4-svd-r32-last8-odown-bf16..."
    mkdir -p "$dir"
    export HF_XET_HIGH_PERFORMANCE=1
    $HF_DOWNLOAD_CMD \
        --local-dir "$dir" \
        WaveCut/HiDream-O1-Image-Dev-SDNQ-uint4-svd-r32-last8-odown-bf16
    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    log_info "HiDream downloaded (${size}) → ${dir}"
}

download_ideogram4() {
    local dir="${MODELS_DIR}/ideogram-4-Q4_0"
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        log_info "Ideogram 4 already exists at ${dir}"
        return 0
    fi
    log_info "Downloading Ideogram 4 components..."
    mkdir -p "$dir"
    export HF_XET_HIGH_PERFORMANCE=1
    # DiT + uncond
    $HF_DOWNLOAD_CMD --local-dir "$dir" \
        leejet/ideogram-4-GGUF \
        ideogram4-Q4_0.gguf ideogram4_uncond-Q4_0.gguf
    # Text encoder
    $HF_DOWNLOAD_CMD --local-dir "$dir" \
        unsloth/Qwen3-VL-8B-Instruct-GGUF \
        Qwen3-VL-8B-Instruct-Q4_K_M.gguf
    mv "$dir/Qwen3-VL-8B-Instruct-Q4_K_M.gguf" "$dir/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    # VAE (gated repo — requires license acceptance)
    mkdir -p "$dir/vae"
    $HF_DOWNLOAD_CMD --local-dir "$dir/vae" \
        black-forest-labs/FLUX.2-dev ae.safetensors
    mv "$dir/vae/ae.safetensors" "$dir/vae/flux2-vae.safetensors"
    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    log_info "Ideogram 4 downloaded (${size}) → ${dir}"
}

cmd_download() {
    check_hf_cli

    local target="${1:-help}"

    case "$target" in
        hidream)
            download_hidream
            ;;
        ideogram4|ideogram)
            download_ideogram4
            ;;
        all)
            download_hidream
            # ideogram4 skipped in 'all' — gated repo, needs manual license acceptance
            log_warn "Ideogram 4 skipped (gated repo). Run: diffuse download ideogram4"
            ;;
        *)
            log_error "Unknown target: $target"
            log_info "Usage: $0 [hidream|ideogram4|all]"
            exit 1
            ;;
    esac
}

cmd_help() {
    cat << EOF
diffuse — Download model weights from HuggingFace

Usage: $0 [target]

Targets:
  hidream       HiDream-O1 SDNQ (T2I + editing, 7.3 GB)
  ideogram4     Ideogram 4 Q4 (T2I, text rendering, ~14.9 GB — gated VAE repo)
  all           Download hidream (skips ideogram4 — gated)

Environment:
  HF_TOKEN  HuggingFace token (needed for gated repos like FLUX.2-dev)

Examples:
  $0 hidream         # Download HiDream weights
  $0 ideogram4       # Download Ideogram 4 (needs HF auth for VAE)
  $0 all             # Download everything (except gated repos)
EOF
}

case "${1:-help}" in
    hidream|ideogram4|ideogram|all)
        cmd_download "$1"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        log_error "Unknown target: $1"
        cmd_help
        exit 1
        ;;
esac