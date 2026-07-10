# OpenScience — Docker Deploy

Containerized deploy of [OpenScience](https://github.com/synthetic-sciences/openscience) — the open-source AI workbench for scientific research.

## Architecture

```
Browser (http://localhost:4096)
  │
  └── Docker container (network_mode: host)
        ├── openscience binary (Bun-compiled, pre-built from releases)
        ├── Agent runtime (internal, NOT delegated to external CLIs)
        ├── Tool layer (shell, editor, LSP, MCP, science connectors)
        └── State → /data volume (XDG dirs redirected)
              ├── config/   ← ~/.config/openscience
              ├── share/    ← ~/.local/share/openscience (sessions, artifacts)
              ├── cache/    ← ~/.cache/openscience
              ├── state/    ← ~/.local/state/openscience
              └── projects/ ← optional project workspace
```

### Key design decisions

- **Non-root**: runs as `openscience` user (UID 1001), no login shell
- **Lightweight**: `debian:bookworm-slim` base, only runtime deps (curl, git, ca-certificates, libstdc++6). No Node.js, no Bun install — the binary is pre-compiled
- **No database**: all state is on-disk (JSON + files). No SQLite, no Postgres
- **Portable persistence**: single named volume `openscience-data` mounted at `/data`. Survives `docker compose down`. All XDG dirs redirected via env vars
- **Host networking**: the server hardcodes bind to `127.0.0.1` (no `--hostname` flag). Using `network_mode: host` exposes it directly on localhost:4096, same pattern as OpenDesign

### Security notes

- The agent runtime is **not sandboxed** (per upstream docs). Running in Docker provides the container isolation they recommend
- API keys are passed via env vars, go straight to providers, never to Synthetic Sciences
- Credentials are filtered from subprocess environments and redacted from output (upstream behavior)
- Container runs as non-root — the agent cannot escalate within the container
- For stronger isolation: add `--cap-drop ALL --security-opt no-new-privileges` to compose if desired

## Usage

```bash
# 1. Copy .env.example → .env and add your API keys (optional)
cp .env.example .env

# 2. Build
docker compose build

# 3. Start (headless, no browser auto-open)
docker compose up -d

# 4. Access
#    Open http://localhost:4096 in your browser
#    First run: onboarding walks through model setup (BYOK, Atlas, or demo)

# 5. Logs
docker compose logs -f

# 6. Stop (data persists)
docker compose down

# 7. Wipe all data
docker compose down -v
docker volume rm openscience-data
```

## Data management

```bash
# Inspect the volume
docker volume inspect openscience-data

# Backup
docker run --rm \
  -v openscience-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/openscience-data-$(date +%Y%m%d).tar.gz /data

# Restore
docker run --rm \
  -v openscience-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/openscience-data-20260705.tar.gz -C /
```

## SSD usage

- **Image**: ~150MB (debian:bookworm-slim + binary + deps)
- **Volume**: starts small (~1MB), grows with sessions and cached model catalog
- **Build cache**: ~200MB during `docker compose build` (clearable with `docker builder prune`)
- **To clean everything**: `docker compose down -v && docker rmi openscience:latest && docker builder prune -f`

## Version

Pinned to `v1.3.2` (latest as of 2026-07-10). Update via `OPENSCIENCE_VERSION` build arg:

```bash
docker compose build --build-arg OPENSCIENCE_VERSION=1.3.2
docker compose up -d
```

## Local model configuration

OpenScience detects Ollama automatically but the "Local models" settings panel
crashes with `SyntaxError: Unexpected token '<'` (UI bug — fetch returns HTML
from SPA fallback instead of JSON). Configure local providers via CLI instead:

```bash
# Add Ollama (cloud models)
docker exec openscience openscience local add --url http://localhost:11434/v1

# Add llama-swap (local GGUF fleet, OpenAI-compatible)
docker exec openscience openscience local add --url http://localhost:12434/v1

# Add specific models only (recommended — filter out embeddings, ASR, TTS, etc.)
docker exec openscience openscience local add \
  --url http://localhost:12434/v1 \
  --model agents-a1-35b --model agents-a1-35b:think \
  --model ornith-1.0-35b --model ornith-1.0-35b:think \
  --model qwen3.6-35b-a3b --model qwen3.6-35b-a3b:think

# List configured providers
docker exec openscience openscience local list

# Remove a provider
docker exec openscience openscience local remove <provider-id>
```

Config is stored in `/data/config/openscience/openscience.jsonc` (named volume).

## Ports

| Port | Purpose |
|------|---------|
| 4096 | OpenScience web UI + API |

## Comparison with OpenDesign deploy

| | OpenDesign | OpenScience |
|---|---|---|
| Runtime | Spawns external CLIs (Hermes, OpenCode) | Internal agent runtime |
| Binary | Node.js daemon | Bun-compiled native binary |
| Database | SQLite (better-sqlite3) | None (on-disk JSON) |
| Network | `network_mode: host` | `network_mode: host` |
| Non-root | Yes | Yes (UID 1001) |
| Volume | `open-design-data` (SQLite + artifacts) | `openscience-data` (XDG state) |
| Agent safety | CLIs have own permission systems | Container isolation (not sandboxed upstream) |