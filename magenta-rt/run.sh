#!/bin/bash
# magenta-rt — Local Music Generation CLI
#
# Usage: magenta-rt [command] [options]
#
# Commands:
#   generate    Generate audio (main command)
#   setup       Install dependencies (uv sync)
#   download    Download model weights and resources
#   evict       Evict all LLM models from llama-swap (free VRAM)
#   status      Check prerequisites and model availability
#   clean       Remove generated audio and caches
#   help        Show help
#
# Environment:
#   MRT_MODEL       Model variant (default: mrt2_small)
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
VARIANT="${MRT_MODEL:-mrt2_small}"
MRT_DATA_DIR="${MRT_DATA_DIR:-$HOME/Documents/Magenta/magenta-rt-v2}"

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
        log_error "Virtual environment not found. Run: magenta-rt setup"
        return 1
    fi
    return 0
}

check_mrt_cli() {
    if ! "$VENV_PY" -m magenta_rt.cli --help &>/dev/null; then
        log_error "magenta-rt CLI not found in venv."
        log_error "Run: magenta-rt setup"
        return 1
    fi
    return 0
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_setup() {
    log_info "Setting up magenta-rt..."

    # Check for uv
    if ! command -v uv &>/dev/null; then
        log_error "uv not found. Install: https://docs.astral.sh/uv/"
        exit 1
    fi

    # Python 3.12+ required (magenta-rt needs 3.12)
    local py_version
    py_version=$(python3 --version 2>/dev/null | grep -oP '\d+\.\d+' || echo "0.0")
    local py_major=$(echo "$py_version" | cut -d. -f1)
    local py_minor=$(echo "$py_version" | cut -d. -f2)

    if [[ "$py_major" -lt 3 ]] || [[ "$py_major" -eq 3 && "$py_minor" -lt 12 ]]; then
        log_info "Python 3.12+ required (found $(python3 --version 2>/dev/null || echo 'unknown'))"
        log_info "uv will create a 3.12 venv automatically"
    fi

    # Create venv and install
    cd "$SCRIPT_DIR"
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating venv with Python 3.12..."
        uv venv --python 3.12
    fi

    log_info "Installing dependencies (magenta-rt[jax])..."
    uv pip install "magenta-rt[jax]"

    log_info ""
    log_info "Setup complete! Next steps:"
    log_info "  1. magenta-rt download resources   # Download MusicCoCa + SpectroStream"
    log_info "  2. magenta-rt download small        # Download mrt2_small model"
    log_info "  3. magenta-rt generate -p 'disco funk'"
}

cmd_generate() {
    check_venv || exit 1
    check_mrt_cli || exit 1

    cd "$SCRIPT_DIR"
    exec "$VENV_PY" generate.py "$@"
}

cmd_download() {
    check_venv || exit 1
    check_mrt_cli || exit 1

    local target="${1:-small}"

    case "$target" in
        resources)
            log_info "Downloading MusicCoCa + SpectroStream resources..."
            "$VENV_PY" -m magenta_rt.cli models init
            log_info "Resources downloaded to: $MRT_DATA_DIR/resources"
            ;;
        small)
            # Ensure resources first
            if [[ ! -d "$MRT_DATA_DIR/resources" ]] || [[ -z "$(ls -A "$MRT_DATA_DIR/resources" 2>/dev/null)" ]]; then
                log_info "Resources not found — downloading first..."
                cmd_download resources
            fi
            log_info "Downloading mrt2_small model (230M params)..."
            "$VENV_PY" -m magenta_rt.cli models download --model mrt2_small
            log_info "Model downloaded to: $MRT_DATA_DIR/models/mrt2_small"
            ;;
        base)
            if [[ ! -d "$MRT_DATA_DIR/resources" ]] || [[ -z "$(ls -A "$MRT_DATA_DIR/resources" 2>/dev/null)" ]]; then
                log_info "Resources not found — downloading first..."
                cmd_download resources
            fi
            log_info "Downloading mrt2_base model (2.4B params)..."
            log_warn "This model requires ~5-6GB VRAM — may not fit on RTX 3050 6GB"
            "$VENV_PY" -m magenta_rt.cli models download --model mrt2_base
            log_info "Model downloaded to: $MRT_DATA_DIR/models/mrt2_base"
            ;;
        all)
            cmd_download resources
            cmd_download small
            cmd_download base
            ;;
        *)
            log_error "Unknown download target: $target"
            log_info "Usage: magenta-rt download [resources|small|base|all]"
            exit 1
            ;;
    esac
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
    log_info "VRAM freed for music generation"
}

cmd_status() {
    echo "=== magenta-rt Status ==="
    echo ""

    # Check venv
    if [[ -f "$VENV_PY" ]]; then
        echo -e "venv: ${GREEN}✓${NC} $VENV_DIR"
        # Check magenta-rt version
        local mrt_ver
        mrt_ver=$("$VENV_PY" -c "import magenta_rt; print(magenta_rt.__version__)" 2>/dev/null || echo "unknown")
        echo -e "magenta-rt: ${GREEN}✓${NC} v$mrt_ver"
    else
        echo -e "venv: ${RED}✗${NC} not found (run: magenta-rt setup)"
    fi

    # Check JAX + CUDA
    if [[ -f "$VENV_PY" ]]; then
        local jax_info
        jax_info=$("$VENV_PY" -c "
import jax
print(f'JAX {jax.__version__}')
devices = jax.devices()
for d in devices:
    print(f'  {d.device_kind}')
" 2>/dev/null)
        if [[ $? -eq 0 ]]; then
            echo -e "JAX: ${GREEN}✓${NC}"
            echo "$jax_info" | while IFS= read -r line; do
                echo "     $line"
            done
        else
            echo -e "JAX: ${RED}✗${NC} not installed or broken"
        fi
    fi

    # Check resources
    echo ""
    echo "=== Resources ==="
    if [[ -d "$MRT_DATA_DIR/resources" ]] && [[ -n "$(ls -A "$MRT_DATA_DIR/resources" 2>/dev/null)" ]]; then
        local res_size
        res_size=$(du -sh "$MRT_DATA_DIR/resources" 2>/dev/null | cut -f1)
        echo -e "MusicCoCa + SpectroStream: ${GREEN}✓${NC} ($res_size)"
    else
        echo -e "MusicCoCa + SpectroStream: ${YELLOW}!${NC} not downloaded (run: magenta-rt download resources)"
    fi

    # Check models
    echo ""
    echo "=== Models ==="
    for model_dir in "$MRT_DATA_DIR/models"/mrt2_*/; do
        if [[ -d "$model_dir" ]]; then
            local name=$(basename "$model_dir")
            local size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            echo -e "  ${GREEN}✓${NC} $name ($size)"
        fi
    done
    if ! compgen -G "$MRT_DATA_DIR/models/mrt2_*" > /dev/null; then
        echo -e "  ${YELLOW}!${NC} no models downloaded (run: magenta-rt download small)"
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
    log_info "Cleaning generated audio and caches..."
    rm -rf "${SCRIPT_DIR}/outputs"
    log_info "Done. Models and venv preserved."
}

cmd_help() {
    cat << EOF
magenta-rt — Local Music Generation with Magenta RealTime 2

Usage: magenta-rt [command] [options]

Commands:
  generate    Generate audio (passes args to generate.py)
  setup       Install dependencies (uv + magenta-rt[jax])
  download    Download model weights and resources
  evict       Evict all LLM models from llama-swap (free VRAM)
  status      Check prerequisites, models, and GPU
  clean       Remove generated audio and caches
  help        Show this help message

Environment:
  MRT_MODEL       Model variant (default: mrt2_small)
  MRT_DATA_DIR    Model data directory (default: ~/Documents/Magenta/magenta-rt-v2)
  LLAMA_SWAP_CLI   Path to llama-swap-cli (default: ~/git/ai-dotfiles/llama-swap/llama-swap-cli)

Examples:
  magenta-rt setup                          # First-time setup
  magenta-rt download resources              # Download MusicCoCa + SpectroStream
  magenta-rt download small                  # Download 230M model (recommended)
  magenta-rt generate -p 'disco funk'        # Generate 4s of music
  magenta-rt generate -p 'jazz piano' --duration 8.0
  magenta-rt generate -p 'ambient' --evict-llm  # Free VRAM first
  magenta-rt status                          # Check everything
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