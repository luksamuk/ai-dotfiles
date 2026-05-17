# Model Selection Guide

### Standard Models (General Use)

| Model | VRAM | Context | Features |
|-------|------|---------|----------|
| **qwen3.5-4b** | ~3GB | 32K-128K | - |
| **qwen3.5-9b** | ~5GB + RAM | 16K-128K | - |
| **gemma4-e4b** | ~4.5GB + RAM | 16K-128K | - |
| **gemma4-e2b** | ~3GB | 32K-128K | - |
| **nemotron-3-nano-4b** | ~3GB | 32K-128K | `tools` |

### Thinking Models (With Reasoning)

| Model | VRAM | Context | Features |
|-------|------|---------|----------|
| **qwen3.5-4b-think** | ~3GB | 32K-128K | `thinking` |
| **qwen3.5-9b-think** | ~5GB + RAM | 16K-128K | `thinking` |
| **gemma4-e4b-think** | ~4.5GB + RAM | 16K-128K | `thinking` |
| **gemma4-e2b-think** | ~3GB | 32K-128K | `thinking` |
| **nemotron-3-nano-4b-think** | ~3GB | 32K-128K | `thinking`, `tools` |

### Tool-Calling Models (Reasoning + Tools)

These models are fine-tuned for function calling and always use reasoning:

| Model | VRAM | Context | Features |
|-------|------|---------|----------|
| **qwopus-4b** | ~3GB | 32K-128K | `thinking`, `tools` |
| **qwopus-9b** | ~5GB + RAM | 16K-128K | `thinking`, `tools` |

### vLLM Backend Models

Experimental vLLM backends for API compatibility testing. Not for daily use — slower startup (~30-90s), higher overhead.

| Model | VRAM | Context | Features | Notes |
|-------|------|---------|----------|-------|
| **qwen3.5-0.8b-vllm** | ~2.5-3GB | 8K | `tools` | vLLM safetensors, auto-download from HF |
| **qwen3.5-2b-vllm** | ~5GB | 2K | `tools` | vLLM safetensors, no vision (`--skip-mm-profiling`) |

**vLLM-specific flags:**
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` — Required for tool calling
- `--skip-mm-profiling` + `--limit-mm-per-prompt '{"image": 0}'` — Skip ViT profiling (2B only)
- `--default-chat-template-kwargs '{"enable_thinking": false}'` — Disable reasoning in output

### Context Size Behavior

Context is **dynamic** - automatically adjusts based on available VRAM:

| Model Type | Minimum | Maximum | Behavior |
|------------|---------|---------|----------|
| Small (4B, E2B) | 32K | 128K | Fits entirely in VRAM |
| Large (9B, E4B) | 16K | 128K | May use RAM offload |

The `--fit` feature ensures:
- **Never crashes** - reduces context if VRAM is tight
- **Maximum utilization** - expands context when VRAM is free
- **Dynamic adjustment** - adapts to current system state

### Feature Flags

| Flag | Description |
|------|-------------|
| `thinking` | Model has reasoning/thinking capability enabled |
| `tools` | Model excels at function calling/tool use |
| `vision` | Model supports image input (not yet available) |

### Inference Parameters

Based on Unsloth recommendations:

| Parameter | Standard | Thinking | Code/Tools |
|-----------|----------|----------|------------|
| `temp` | 0.7 | 0.6 | 0.6 |
| `top_p` | 0.9 | 0.9 | 0.85 |
| `top_k` | 20 | 20 | 40 |
| `min_p` | 0.01 | 0.0 | 0.02 |
| `repeat_penalty` | 1.05 | 1.0 | 1.05 |
| `reasoning` | - | `auto` | - |

### Model Names (Ollama-style)

Models use the format `model:size` for consistency with Ollama:

| Primary Name | Aliases |
|--------------|---------|
| `qwen3.5-4b` | `qwen3.5-4b`, `qwen3.5-4b-q4` |
| `qwen3.5-4b-think` | `qwen3.5-4b-think`, `qwen3.5-4b-reasoning` |
| `qwen3.5-9b` | `qwen3.5-9b`, `qwen3.5-9b-q4` |
| `qwen3.5-9b-think` | `qwen3.5-9b-think`, `qwen3.5-9b-reasoning` |
| `gemma4-e4b` | `gemma4-e4b`, `gemma-4-e4b` |
| `gemma4-e4b-think` | `gemma4-e4b-think`, `gemma-4-e4b-think` |
| `gemma4-e2b` | `gemma4-e2b`, `gemma-4-e2b` |
| `gemma4-e2b-think` | `gemma4-e2b-think`, `gemma-4-e2b-think` |
| `nemotron-3-nano-4b` | `nemotron-4b`, `nemotron-3-nano-4b`, `nemotron` |
| `nemotron-3-nano-4b-think` | `nemotron-4b-think`, `nemotron-think` |
| `qwopus-4b` | `qwopus4b`, `qwopus-4b` |
| `qwopus-9b` | `qwopus`, `qwopus9b` |