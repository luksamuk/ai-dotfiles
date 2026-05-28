#!/bin/bash
# diffuse — Download model weights from HuggingFace
#
# Usage:
#   ./download-model.sh binary          # 1-bit variant (recommended for RTX 3050 6GB)
#   ./download-model.sh ternary         # 1.58-bit variant (better quality, more VRAM)
#   ./download-model.sh both            # Both variants
#
# Environment:
#   HF_TOKEN          HuggingFace token (if models require auth)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Models registry
BINARY_REPO="prism-ml/bonsai-image-binary-4B-gemlite-1bit"
BINARY_DIR="${MODELS_DIR}/bonsai-image-4B-binary-gemlite"

TERNARY_REPO="prism-ml/bonsai-image-ternary-4B-gemlite-2bit"
TERNARY_DIR="${MODELS_DIR}/bonsai-image-4B-ternary-gemlite"

check_hf_cli() {
    # Prefer the venv's hf if available (avoids system PATH conflicts)
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

download_variant() {
    local variant="$1"
    local repo="$2"
    local dir="$3"

    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        log_info "${variant} model already exists at ${dir}"
        log_info "Skipping download. To re-download, remove the directory first."
        return 0
    fi

    log_info "Downloading ${variant} model from ${repo}..."
    mkdir -p "$dir"

    # Enable fast parallel downloads
    export HF_XET_HIGH_PERFORMANCE=1

    $HF_DOWNLOAD_CMD \
        --local-dir "$dir" \
        "$repo"

    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    log_info "${variant} model downloaded (${size}) → ${dir}"
}

cmd_download() {
    check_hf_cli

    local variant="${1:-binary}"

    case "$variant" in
        binary)
            download_variant "binary-gemlite" "$BINARY_REPO" "$BINARY_DIR"
            ;;
        ternary)
            download_variant "ternary-gemlite" "$TERNARY_REPO" "$TERNARY_DIR"
            ;;
        both)
            download_variant "binary-gemlite" "$BINARY_REPO" "$BINARY_DIR"
            download_variant "ternary-gemlite" "$TERNARY_REPO" "$TERNARY_DIR"
            ;;
        *)
            log_error "Unknown variant: $variant"
            log_info "Usage: $0 [binary|ternary|both]"
            exit 1
            ;;
    esac
}

cmd_help() {
    cat << EOF
diffuse — Download model weights from HuggingFace

Usage: $0 [variant]

Variants:
  binary    1-bit weights (0.93 GB transformer) — recommended for RTX 3050 6GB
  ternary   1.58-bit weights (1.21 GB transformer) — better quality, more VRAM
  both      Download both variants

Environment:
  HF_TOKEN  HuggingFace token (needed if models require auth)

Examples:
  $0 binary         # Download 1-bit variant only
  $0 both           # Download everything
EOF
}

case "${1:-help}" in
    binary|ternary|both)
        cmd_download "$1"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        log_error "Unknown variant: $1"
        cmd_help
        exit 1
        ;;
esac