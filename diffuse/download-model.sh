#!/bin/bash
# diffuse — Download model weights from HuggingFace
#
# Usage:
#   ./download-model.sh hidream       # HiDream-O1 SDNQ (T2I + editing)
#   ./download-model.sh ideogram4     # Ideogram 4 Q4 (T2I, text rendering)
#   ./download-model.sh wan22         # Wan2.2 I2V (image-to-video)
#   ./download-model.sh lingbot       # LingBot T2V (text-to-video)
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

download_wan22() {
    local dir="${MODELS_DIR}/wan22-i2v"
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        log_info "Wan2.2 already exists at ${dir}"
    else
        mkdir -p "$dir"
        export HF_XET_HIGH_PERFORMANCE=1
        log_info "Downloading Wan2.2 I2V DiT..."
        $HF_DOWNLOAD_CMD --local-dir "$dir" \
            desirel/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10 \
            wan2.2-i2v-rapid-aio-v10-nsfw-Q4_K_S.gguf
    fi

    # Shared VAE
    local vae_dir="${MODELS_DIR}/shared/vae/wan"
    if [[ ! -d "$vae_dir" || -z "$(ls -A "$vae_dir" 2>/dev/null)" ]]; then
        log_info "Downloading shared Wan VAE..."
        mkdir -p "$vae_dir"
        $HF_DOWNLOAD_CMD --local-dir "$vae_dir" \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/vae/wan_2.1_vae.safetensors
        mv "$vae_dir/split_files/vae/wan_2.1_vae.safetensors" "$vae_dir/diffusion_pytorch_model.safetensors"
        rm -rf "$vae_dir/split_files"
        curl -sL "https://huggingface.co/robbyant/lingbot-video-dense-1.3b/raw/main/vae/config.json" -o "$vae_dir/config.json"
    fi
    # Link VAE to model dir
    mkdir -p "$dir/vae"
    if [[ ! -L "$dir/vae/wan_2.1_vae.safetensors" ]]; then
        ln -s ../../shared/vae/wan/diffusion_pytorch_model.safetensors "$dir/vae/wan_2.1_vae.safetensors"
    fi

    # Text encoder
    mkdir -p "$dir/text_encoder"
    if [[ ! -f "$dir/text_encoder/umt5-xxl-encoder-Q8_0.gguf" ]]; then
        log_info "Downloading UMT5-XXL text encoder..."
        $HF_DOWNLOAD_CMD --local-dir "$dir/text_encoder" \
            city96/umt5-xxl-encoder-gguf \
            umt5-xxl-encoder-Q8_0.gguf
    fi

    # CLIP vision (need GGUF format — safetensors has 5D tensor sd-cli can't convert)
    mkdir -p "$dir/clip_vision"
    if [[ ! -f "$dir/clip_vision/clip_vision_h.gguf" ]]; then
        log_info "Downloading CLIP Vision..."
        $HF_DOWNLOAD_CMD --local-dir "$dir/clip_vision" \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/clip_vision/clip_vision_h.safetensors
        echo "  NOTE: clip_vision_h.safetensors needs manual conversion to GGUF for sd-cli"
        echo "  See diffuse backends/sd_cpp.py for details"
    fi

    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    log_info "Wan2.2 ready (${size}) → ${dir}"
}

download_lingbot() {
    local dir="${LLAMA_MODELS_DIR}/lingbot-t2v"
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        log_info "LingBot already exists at ${dir}"
        return 0
    fi
    log_info "Downloading LingBot-Video Dense 1.3B..."
    mkdir -p "$dir"
    export HF_XET_HIGH_PERFORMANCE=1
    $HF_DOWNLOAD_CMD \
        --local-dir "$dir" \
        robbyant/lingbot-video-dense-1.3b
    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    log_info "LingBot downloaded (${size}) → ${dir}"

    # Link shared VAE
    local vae_dir="${MODELS_DIR}/shared/vae/wan"
    if [[ ! -d "$vae_dir" || -z "$(ls -A "$vae_dir" 2>/dev/null)" ]]; then
        log_info "Downloading shared Wan VAE..."
        mkdir -p "$vae_dir"
        $HF_DOWNLOAD_CMD --local-dir "$vae_dir" \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/vae/wan_2.1_vae.safetensors
        mv "$vae_dir/split_files/vae/wan_2.1_vae.safetensors" "$vae_dir/diffusion_pytorch_model.safetensors"
        rm -rf "$vae_dir/split_files"
        curl -sL "https://huggingface.co/robbyant/lingbot-video-dense-1.3b/raw/main/vae/config.json" -o "$vae_dir/config.json"
    fi
    # Replace LingBot's VAE with symlink to shared
    if [[ -d "$dir/vae" && ! -L "$dir/vae" ]]; then
        rm -rf "$dir/vae"
    fi
    if [[ ! -L "$dir/vae" ]]; then
        ln -s ../../git/ai-dotfiles/diffuse/models/shared/vae/wan "$dir/vae"
    fi
    log_info "LingBot VAE linked to shared component"

    # Clone the lingbot-video repo if not present
    local repo_dir="${HOME}/git/lingbot-video"
    if [[ ! -d "$repo_dir" ]]; then
        log_info "Cloning lingbot-video repo (pipeline code)..."
        git clone https://github.com/Robbyant/lingbot-video.git "$repo_dir"
    fi
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
        wan22|wan)
            download_wan22
            ;;
        lingbot)
            download_lingbot
            ;;
        all)
            download_hidream
            download_wan22
            download_lingbot
            # ideogram4 skipped in 'all' — gated repo, needs manual license acceptance
            log_warn "Ideogram 4 skipped (gated repo). Run: diffuse download ideogram4"
            ;;
        *)
            log_error "Unknown target: $target"
            log_info "Usage: $0 [hidream|ideogram4|wan22|lingbot|all]"
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
  wan22         Wan2.2 I2V (image-to-video, ~16.4 GB + shared VAE)
  lingbot       LingBot T2V (text-to-video, ~11.7 GB + shared VAE + repo clone)
  all           Download hidream + wan22 + lingbot (skips ideogram4 — gated)

Environment:
  HF_TOKEN  HuggingFace token (needed for gated repos like FLUX.2-dev)

Examples:
  $0 hidream         # Download HiDream weights
  $0 lingbot         # Download LingBot + clone repo + link shared VAE
  $0 all              # Download everything (except gated repos)
EOF
}

case "${1:-help}" in
    hidream|ideogram4|ideogram|wan22|wan|lingbot|all)
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