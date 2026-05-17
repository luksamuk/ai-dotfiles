# llama-swap: Usage

## Usage

### Quick Start (No Systemd)

Run directly in foreground - no installation needed:

```bash
# Check prerequisites first
./run.sh status

# Run in foreground (press Ctrl+C to stop)
./run.sh run

# Or use LLAMA_SWAP_PORT to change port
LLAMA_SWAP_PORT=8080 ./run.sh run
```

### Systemd User Service

```bash
# List available models
curl http://127.0.0.1:12434/v1/models

# Chat completion
curl http://127.0.0.1:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Systemd User Service

For running as a background service:

```bash
# Install service (copies config if needed)
./run.sh install

# Start service
systemctl --user start llama-swap

# Check status
systemctl --user status llama-swap

# View logs
./run.sh logs
# or
journalctl --user -u llama-swap -f

# Stop service
systemctl --user stop llama-swap

# Enable/disable autostart
systemctl --user enable llama-swap
systemctl --user disable llama-swap

# Uninstall service (keeps config and models)
./run.sh uninstall
```

The service file is installed at `~/.config/systemd/user/llama-swap.service`
and uses the config from `~/.config/llama-swap/config.yaml`.

### Testing the API

Once running, llama-swap provides an OpenAI-compatible API:

```bash
# List available models
curl http://127.0.0.1:12434/v1/models

# Chat completion
curl http://127.0.0.1:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_SWAP_CONFIG` | Config file path | `~/.config/llama-swap/config.yaml` |
| `LLAMA_SWAP_PORT` | Port to listen on | `12434` |

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:12434/v1",
    api_key="not-needed"  # llama-swap doesn't require auth by default
)

response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Using with Ollama-compatible tools

The API is OpenAI-compatible, so most tools work out of the box:
- [Continue](https://continue.dev/) - VS Code extension
- [Open WebUI](https://github.com/open-webui/open-webui)
- [LibreChat](https://github.com/danny-avila/LibreChat)

## CLI: llama-swap-cli

A companion CLI for managing llama-swap models from the terminal.

```bash
# List configured models
llama-swap-cli list [--pretty]

# Show running models (VRAM, RSS, tok/s, GPU layers)
llama-swap-cli ps [--pretty]

# Detailed metrics (tokens, speed, queue)
llama-swap-cli stats [--pretty]

# Unload all models (or a specific one)
llama-swap-cli unload [MODEL]

# Recent logs
llama-swap-cli logs [N]

# Interactive chat with model selection
llama-swap-cli testchat
```

Supports both llama.cpp and vLLM backends. The `ps` and `stats` commands detect vLLM metrics automatically.

## Interactive Chat (testchat)

An interactive terminal chat with streaming, reasoning display, and tool calling.

```bash
# Via CLI
llama-swap-cli testchat

# Or directly
cd testchat && uv run main.py
```

### Features

- **Model selection** with feature icons (🤔 thinking, 🛠️ tools, 👁️ vision)
- **Tool calling** with mock tools (get_weather, calculator, get_time) — auto-enabled for models with `tools: true`
- **Reasoning panel** — split-screen display for thinking models
- **Streaming** — real-time token display with timing stats

### Tool Calling Flow

When a model with `tools: true` is selected, mock tools are sent automatically. The model decides whether to call a tool, and the testchat provides simulated responses:

1. Model decides to call a tool → displays `🔧 tool_name(args)`
2. Testchat generates mock response → displays inline
3. Model formats final answer using mock data

### Waybar Integration

The `inference-status.py` Waybar module shows loaded models and allows unloading via click:

```json
// ~/.config/waybar/config.jsonc
"custom/inference": {
  "exec": "~/.config/waybar/scripts/inference-status.py llamaswap status",
  "return-type": "json",
  "on-click": "~/.config/waybar/scripts/inference-status.py llamaswap eject_all"
}
```

Uses the llama-swap `/running` API — works with both llama.cpp and vLLM backends.