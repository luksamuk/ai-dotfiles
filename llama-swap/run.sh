#!/bin/bash
# llama-swap runner script
# Usage: ./run.sh [command] [options]
#
# Commands:
#   start       Start llama-swap server (foreground)
#   stop        Stop llama-swap server (via systemd)
#   status      Check status of llama-swap
#   install     Install systemd user service
#   uninstall   Uninstall systemd user service
#   logs        Show logs from systemd journal
#   download    Download models using download-models.sh
#
# Environment:
#   LLAMA_SWAP_CONFIG  - Config file path (default: ~/.config/llama-swap/config.yaml)
#   LLAMA_SWAP_PORT    - Port to listen on (default: 12434)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${HOME}/.config/llama-swap"
CONFIG_FILE="${LLAMA_SWAP_CONFIG:-$CONFIG_DIR/config.yaml}"
SERVICE_NAME="llama-swap"
SERVICE_FILE="${SCRIPT_DIR}/llama-swap.service"
USER_SERVICE_DIR="${HOME}/.config/systemd/user"
PORT="${LLAMA_SWAP_PORT:-12434}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_llama_server() {
    local LLAMA_SERVER="${HOME}/git/llama.cpp/build/bin/llama-server"
    if [[ ! -x "$LLAMA_SERVER" ]]; then
        log_error "llama-server not found at $LLAMA_SERVER"
        log_info "Build llama.cpp first:"
        log_info "  cd ~/git && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git"
        log_info "  cd llama.cpp && mkdir -p build && cd build"
        log_info "  cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
        return 1
    fi
    return 0
}

check_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        log_info "Copy the config first:"
        log_info "  mkdir -p $CONFIG_DIR"
        log_info "  cp $SCRIPT_DIR/config.yaml $CONFIG_DIR/"
        return 1
    fi
    return 0
}

check_models() {
    local MODELS_DIR="${HOME}/.llama-models"
    if [[ ! -d "$MODELS_DIR" ]] || [[ -z "$(ls -A "$MODELS_DIR"/*.gguf 2>/dev/null)" ]]; then
        log_warn "No models found in $MODELS_DIR"
        log_info "Download models first:"
        log_info "  $SCRIPT_DIR/download-models.sh all"
        return 1
    fi
    return 0
}

# Commands
cmd_start() {
    check_llama_server || exit 1
    check_config || exit 1
    check_models || exit 1
    
    log_info "Starting llama-swap on port $PORT..."
    log_info "Config: $CONFIG_FILE"
    log_info "Press Ctrl+C to stop"
    
    exec llama-swap -config "$CONFIG_FILE" -listen "127.0.0.1:$PORT" -watch-config
}

cmd_stop() {
    if systemctl --user is-active "$SERVICE_NAME" &>/dev/null; then
        log_info "Stopping llama-swap service..."
        systemctl --user stop "$SERVICE_NAME"
        log_info "Service stopped"
    else
        log_warn "Service is not running"
    fi
}

cmd_status() {
    echo "=== llama-swap Status ==="
    echo ""
    
    # Check systemd service
    if systemctl --user is-active "$SERVICE_NAME" &>/dev/null; then
        echo -e "Service: ${GREEN}running${NC}"
        systemctl --user status "$SERVICE_NAME" --no-pager 2>/dev/null | head -5
    else
        echo -e "Service: ${YELLOW}not running${NC}"
    fi
    
    echo ""
    echo "=== Prerequisites ==="
    
    # Check llama-server
    local LLAMA_SERVER="${HOME}/git/llama.cpp/build/bin/llama-server"
    if [[ -x "$LLAMA_SERVER" ]]; then
        echo -e "llama-server: ${GREEN}✓${NC} $LLAMA_SERVER"
    else
        echo -e "llama-server: ${RED}✗${NC} not found"
    fi
    
    # Check config
    if [[ -f "$CONFIG_FILE" ]]; then
        echo -e "Config: ${GREEN}✓${NC} $CONFIG_FILE"
    else
        echo -e "Config: ${RED}✗${NC} not found"
    fi
    
    # Check models
    local MODELS_DIR="${HOME}/.llama-models"
    if [[ -d "$MODELS_DIR" ]]; then
        local count=$(ls -1 "$MODELS_DIR"/*.gguf 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            echo -e "Models: ${GREEN}✓${NC} $count model(s) in $MODELS_DIR"
        else
            echo -e "Models: ${YELLOW}!${NC} no models downloaded"
        fi
    else
        echo -e "Models: ${RED}✗${NC} directory not found"
    fi
    
    echo ""
    echo "=== Available Models ==="
    if command -v llama-swap &>/dev/null; then
        curl -s "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | jq -r '.data[].id' 2>/dev/null || echo "(service not running or jq not installed)"
    fi
}

cmd_install() {
    check_llama_server || exit 1
    check_config || exit 1
    
    log_info "Installing llama-swap as user service..."
    
    # Create directories
    mkdir -p "$USER_SERVICE_DIR"
    mkdir -p "$CONFIG_DIR"
    
    # Copy config if not exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cp "$SCRIPT_DIR/config.yaml" "$CONFIG_FILE"
        log_info "Copied default config to $CONFIG_FILE"
    fi
    
    # Create instantiated service from template
    local SERVICE_TARGET="$USER_SERVICE_DIR/$SERVICE_NAME.service"
    
    # Process template
    sed -e "s|{{HOME}}|$HOME|g" \
        -e "s|{{CONFIG}}|$CONFIG_FILE|g" \
        -e "s|{{PORT}}|$PORT|g" \
        "$SCRIPT_DIR/llama-swap.service.template" > "$SERVICE_TARGET"
    
    log_info "Created service file: $SERVICE_TARGET"
    
    # Reload systemd
    systemctl --user daemon-reload
    
    # Enable service
    systemctl --user enable "$SERVICE_NAME"
    
    log_info "Service installed and enabled"
    log_info ""
    log_info "Commands:"
    log_info "  systemctl --user start $SERVICE_NAME   # Start service"
    log_info "  systemctl --user stop $SERVICE_NAME    # Stop service"
    log_info "  systemctl --user status $SERVICE_NAME  # Check status"
    log_info "  journalctl --user -u $SERVICE_NAME -f  # Follow logs"
}

cmd_uninstall() {
    log_info "Uninstalling llama-swap user service..."
    
    # Stop and disable
    systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true
    
    # Remove service file
    local SERVICE_TARGET="$USER_SERVICE_DIR/$SERVICE_NAME.service"
    if [[ -f "$SERVICE_TARGET" ]]; then
        rm "$SERVICE_TARGET"
        log_info "Removed: $SERVICE_TARGET"
    fi
    
    # Reload systemd
    systemctl --user daemon-reload
    
    log_info "Service uninstalled"
    log_warn "Config and models are preserved in:"
    log_warn "  - $CONFIG_DIR"
    log_warn "  - ~/.llama-models/"
}

cmd_logs() {
    journalctl --user -u "$SERVICE_NAME" -f
}

cmd_download() {
    "$SCRIPT_DIR/download-models.sh" "$@"
}

cmd_help() {
    cat << EOF
llama-swap runner script

Usage: $0 [command] [options]

Commands:
  start       Start llama-swap server (foreground)
  stop        Stop llama-swap server (via systemd)
  status      Check status of llama-swap and prerequisites
  install     Install systemd user service
  uninstall   Uninstall systemd user service
  logs        Show logs from systemd journal (follow mode)
  download    Download models (passes args to download-models.sh)
  help        Show this help message

Environment Variables:
  LLAMA_SWAP_CONFIG   Config file path (default: ~/.config/llama-swap/config.yaml)
  LLAMA_SWAP_PORT      Port to listen on (default: 12434)

Examples:
  $0 install                    # Install as user service
  $0 start                      # Run in foreground
  $0 status                     # Check status
  $0 download nemotron-4b       # Download specific model
  $0 download all               # Download all models

EOF
}

# Main
case "${1:-help}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    install)
        cmd_install
        ;;
    uninstall)
        cmd_uninstall
        ;;
    logs)
        cmd_logs
        ;;
    download)
        shift
        cmd_download "$@"
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