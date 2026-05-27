# Cline CLI Config

Backup of Cline CLI configuration files.

## Files

- `providers.json` → `~/.cline/data/settings/providers.json`
  - API key replaced with `PLACEHOLDER_USE_LOCAL_KEY` (no PII)
  - After restoring, run: `cline auth -p openai-compatible -k <your-key> -m qwen3.5-4b:think`
- `cline_mcp_settings.json` → `~/.cline/data/settings/cline_mcp_settings.json`
  - MCP servers (searxng + duckduckgo), no PII

## Restore

```bash
mkdir -p ~/.cline/data/settings/
cp providers.json ~/.cline/data/settings/providers.json
cp cline_mcp_settings.json ~/.cline/data/settings/cline_mcp_settings.json
# Then set your API key:
cline auth -p openai-compatible -k <your-key>
```