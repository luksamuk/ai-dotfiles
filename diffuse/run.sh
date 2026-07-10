#!/bin/bash
# diffuse — Local diffusion image generation CLI
#
# Usage: diffuse [command] [options]
#
# Commands:
#   generate    Generate an image (main command)
#   setup       Install dependencies (uv sync + clone vendor deps)
#   download    Download model weights
#   evict       Evict all LLM models from llama-swap (free VRAM)
#   status      Check prerequisites and model availability
#   clean       Remove generated images and caches
#   help        Show help
#
# Environment:
#   DIFFUSE_MODEL   Model variant (default: ternary-gemlite)
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
VARIANT="${DIFFUSE_MODEL:-ternary-gemlite}"

# Vendor SHAs — pin to specific commits for supply chain integrity
IMAGE_STUDIO_SHA="4e26a021abea2e9926900ba49edcef1f05d51241"
MFLUX_PRISM_SHA="bcd13e83b7fcfd76186c98ef322dd9cf28e996c1"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_venv() {
    if [[ ! -f "$VENV_PY" ]]; then
        log_error "Virtual environment not found. Run: diffuse setup"
        return 1
    fi
    return 0
}

check_models() {
    local model_dir="${SCRIPT_DIR}/models/bonsai-image-4B-${VARIANT}"
    if [[ ! -d "$model_dir" ]] || [[ -z "$(ls -A "$model_dir" 2>/dev/null)" ]]; then
        log_warn "Model not found: $model_dir"
        log_info "Download first: diffuse download ${VARIANT%%-*}"
        return 1
    fi
    return 0
}

verify_sha() {
    # Verify a git repo is at the expected commit for supply chain integrity
    local repo_dir="$1"
    local expected_sha="$2"
    local actual_sha
    actual_sha="$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null)" || true
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        log_warn "SHA mismatch in $repo_dir"
        log_warn "  Expected: $expected_sha"
        log_warn "  Actual:   $actual_sha"
        log_warn "  If you intentionally updated, update the SHA in run.sh"
        return 1
    fi
    return 0
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_setup() {
    log_info "Setting up diffuse..."

    # Check for uv
    if ! command -v uv &>/dev/null; then
        log_error "uv not found. Install: https://docs.astral.sh/uv/"
        exit 1
    fi

    local vendor_dir="${SCRIPT_DIR}/vendor"

    # ── Clone image-studio (contains backend_gpu) — pinned SHA ──
    local studio_dir="${vendor_dir}/image-studio"
    if [[ -d "$studio_dir/.git" ]]; then
        log_info "vendor/image-studio already cloned — verifying SHA..."
        if ! verify_sha "$studio_dir" "$IMAGE_STUDIO_SHA"; then
            log_warn "Updating to pinned SHA..."
            git -C "$studio_dir" fetch origin
            git -C "$studio_dir" checkout "$IMAGE_STUDIO_SHA"
        fi
    else
        log_info "Cloning image-studio (backend_gpu) at SHA ${IMAGE_STUDIO_SHA:0:8}..."
        mkdir -p "$vendor_dir"
        git clone https://github.com/PrismML-Eng/image-studio.git "$studio_dir"
        git -C "$studio_dir" checkout "$IMAGE_STUDIO_SHA"
    fi

    # ── Clone mflux-prism — pinned SHA (macOS dep, skipped on Linux but pyproject.toml references it) ──
    local mflux_dir="${vendor_dir}/mflux-prism"
    if [[ -d "$mflux_dir/.git" ]]; then
        log_info "vendor/mflux-prism already cloned — verifying SHA..."
        if ! verify_sha "$mflux_dir" "$MFLUX_PRISM_SHA"; then
            log_warn "Updating to pinned SHA..."
            git -C "$mflux_dir" fetch origin
            git -C "$mflux_dir" checkout "$MFLUX_PRISM_SHA"
        fi
    else
        log_info "Cloning mflux-prism at SHA ${MFLUX_PRISM_SHA:0:8}..."
        mkdir -p "$vendor_dir"
        git clone https://github.com/PrismML-Eng/mflux-prism.git "$mflux_dir"
        git -C "$mflux_dir" checkout "$MFLUX_PRISM_SHA"
    fi

    # ── Patch image-studio pyproject.toml: point mflux to local vendor ──
    local studio_pp="$studio_dir/pyproject.toml"
    if [[ -f "$studio_pp" ]] && grep -q '^mflux = { git = ' "$studio_pp" 2>/dev/null; then
        log_info "Patching image-studio mflux source to local vendor..."
        sed -i.bak 's|^mflux = { git = .*$|mflux = { path = "../mflux-prism", editable = true }|' "$studio_pp"
        rm -f "$studio_pp.bak"
    fi

    # ── Sync dependencies (DISABLE_CUDA=1 to skip hqq_aten build — not needed for gemlite) ──
    log_info "Installing Python dependencies (DISABLE_CUDA=1 for hqq safety)..."
    cd "$SCRIPT_DIR"
    DISABLE_CUDA=1 uv sync
    log_info "Setup complete!"
}

cmd_generate() {
    check_venv || exit 1
    check_models || exit 1

    DIFFUSE_ORIG_CWD="$(pwd)"
    cd "$SCRIPT_DIR"
    DIFFUSE_ORIG_CWD="$DIFFUSE_ORIG_CWD" exec uv run --no-sync -m diffuse "$@"
}

cmd_download() {
    "${SCRIPT_DIR}/download-model.sh" "$@"
}

cmd_status() {
    echo "=== diffuse Status ==="
    echo ""

    # Check venv
    if [[ -f "$VENV_PY" ]]; then
        echo -e "venv: ${GREEN}✓${NC} $VENV_DIR"
    else
        echo -e "venv: ${RED}✗${NC} not found (run: diffuse setup)"
    fi

    # Check vendor
    local vendor_dir="${SCRIPT_DIR}/vendor"
    local studio_dir="${vendor_dir}/image-studio"
    local gpu_dir="${studio_dir}/backend_gpu"
    if [[ -d "$gpu_dir" ]]; then
        local sha
        sha="$(git -C "$studio_dir" rev-parse --short HEAD 2>/dev/null)" || sha="?"
        echo -e "backend_gpu: ${GREEN}✓${NC} $gpu_dir (SHA: $sha)"
        # Verify against pinned SHA
        local full_sha
        full_sha="$(git -C "$studio_dir" rev-parse HEAD 2>/dev/null)" || true
        if [[ "$full_sha" != "$IMAGE_STUDIO_SHA" ]]; then
            echo -e "           ${YELLOW}⚠${NC} SHA mismatch! Expected ${IMAGE_STUDIO_SHA:0:8}, got ${full_sha:0:8}"
        fi
    elif [[ -d "$studio_dir" ]]; then
        echo -e "backend_gpu: ${YELLOW}!${NC} image-studio cloned but backend_gpu not found"
    else
        echo -e "backend_gpu: ${RED}✗${NC} not found (run: diffuse setup)"
    fi

    # Check models
    echo ""
    echo "=== Models ==="
    local found=0
    for model_dir in "${SCRIPT_DIR}/models"/bonsai-image-4B-*/; do
        if [[ -d "$model_dir" ]]; then
            local name=$(basename "$model_dir")
            local size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            echo -e "  ${GREEN}✓${NC} $name ($size)"
            found=1
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo -e "  ${YELLOW}!${NC} no models downloaded (run: diffuse download ternary)"
    fi

    # Check GPU
    echo ""
    echo "=== GPU ==="
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
    else
        echo -e "  ${RED}✗${NC} nvidia-smi not found"
    fi
}

cmd_clean() {
    log_info "Cleaning generated images and caches..."
    rm -rf "${SCRIPT_DIR}/outputs"
    log_info "Done. Models and venv preserved."
}

cmd_evict() {
    local cli="${LLAMA_SWAP_CLI:-${HOME}/git/ai-dotfiles/llama-swap/llama-swap-cli}"
    if [[ ! -x "$cli" ]]; then
        log_error "llama-swap-cli not found at: $cli"
        log_error "Set LLAMA_SWAP_CLI environment variable to override"
        exit 1
    fi

    local running
    running=$(curl -s http://localhost:12434/running 2>/dev/null | jq -r '.running[] | .model' 2>/dev/null)
    if [[ -z "$running" ]]; then
        log_info "No LLM models currently loaded — VRAM already free"
        return 0
    fi

    log_info "Evicting LLM models from llama-swap..."
    echo "$running" | while read -r model; do
        echo "  • $model"
    done

    "$cli" unload
    log_info "VRAM freed for diffusion"
}

cmd_help() {
    cat << EOF
diffuse — Local diffusion image generation CLI

Usage: diffuse [command] [options]

Commands:
  generate    Generate an image (passes args to generate.py)
  setup       Install dependencies (uv sync + vendor deps, pinned SHAs)
  download    Download model weights (passes args to download-model.sh)
  evict       Evict all LLM models from llama-swap (free VRAM for diffusion)
  status      Check prerequisites and model availability
  clean       Remove generated images and caches
  help        Show this help message

Environment:
  DIFFUSE_MODEL   Model variant (default: ternary-gemlite)

Examples:
  diffuse generate -p "A bonsai tree under moonlight"
  diffuse generate -p "Cyberpunk city" --size 1024x1024 --seed 42
  diffuse generate -p "Forest path" --open       # Open result in feh after saving
  diffuse generate -p "Dragon" --evict-llm         # Evict LLMs first to free VRAM
  diffuse generate                       # Interactive prompt mode
  diffuse evict                         # Free VRAM by unloading all LLMs
  diffuse setup                           # First-time setup
  diffuse download ternary                # Download 1.58-bit model weights
  diffuse status                          # Check everything
EOF
}

# ── Main ────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    generate)
        shift
        cmd_generate "$@"
        ;;
    setup)
        cmd_setup
        ;;
    download)
        shift
        cmd_download "$@"
        ;;
    evict)
        cmd_evict
        ;;
    status)
        cmd_status
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac