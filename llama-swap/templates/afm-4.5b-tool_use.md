# AFM-4.5B Tool Calling Template

**Source**: `NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja` from llama.cpp b766
**Format**: Hermes 2 Pro (auto-detected by llama.cpp PEG parser)

## Why this template exists

AFM-4.5B (Arcee Fusion Model) ships with an **empty `tool_use` chat template** in its
`tokenizer_config.json`. Without a tool-calling template, `llama-server` falls back to
`CONTENT_ONLY` format, which means the model receives no tool schemas and cannot emit
structured tool calls.

This template is the **exact** Hermes 2 Pro template from llama.cpp's model template
collection, which is the format AFM was trained with for tool calling (Arcee confirmed
Hermes 2 Pro compatibility in their documentation).

## How it works

1. `llama-server --jinja --chat-template-file afm-4.5b-tool_use.jinja` loads the template
2. llama.cpp's auto-parser recognizes the Hermes 2 Pro format signature
3. Tool calls are parsed from XML tags (tool_call / tool_response format)
4. The response `finish_reason: "tool_calls"` is correctly emitted

## Template behavior

| Scenario | Behavior |
|----------|----------|
| **With tools** | Emits system prompt: "You are a function calling AI model..." + XML tool schemas |
| **Without tools** | Falls through to standard ChatML (im_start turns) |
| **Tool call output** | JSON inside tool_call XML tags |
| **Tool results** | Wrapped in tool_response XML tags |

## Important notes

- **Do NOT modify** this template unless you understand how llama.cpp's PEG auto-parser
  works. Custom templates that deviate from the Hermes 2 Pro format signature will cause
  `COMMON_CHAT_FORMAT_CONTENT_ONLY` fallback, breaking tool calling entirely.
- **Macro support**: The template uses the `{% macro json_to_python_type %}` Jinja macro.
  llama.cpp's Jinja engine supports this since the PEG parser handles format detection
  independently of macro evaluation.
- **Parallel tool calls**: Supported. Pass `"parallel_tool_calls": true` in the API payload.
- **Without `--chat-template-file`**: AFM uses its built-in default template (AFM persona),
  which has NO tool calling support. Only use the flag when tools are needed.

## Verified test (2026-05-31)

Tested with AFM-4.5B-Q5_K_M on upstream llama.cpp b766:

- Model: AFM-4.5B-Q5_K_M
- Backend: upstream llama.cpp
- Command: `llama-server --model AFM-4.5B-Q5_K_M.gguf --jinja --chat-template-file afm-4.5b-tool_use.jinja`
- Result: `finish_reason=tool_calls`, correct JSON arguments, ~44 tok/s
- Test prompt: weather query -> model correctly emitted `get_weather(city="Diamantina", unit="celsius")`

## Version history

- **1.0** (2026-05-31): Initial version, copied from llama.cpp b766 template collection.
