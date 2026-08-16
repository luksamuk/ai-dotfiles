# Sisyphus (RPi4 8GB) — Restructuring Plan

## Current State (Aug 2026)

### Hardware
- Raspberry Pi 4, 8GB RAM, ARM Cortex-A72 @ 1.5GHz, 4 cores
- Storage: 59GB SD card, 21GB used, 35GB free
- No CUDA — CPU-only inference

### LLM Stack
- **llama-swap** (v223, port 8081) — old binary, not systemd-managed (bare nohup process)
- **llama.cpp** build 9802 (beac5309f, aarch64) — desatualizado vs Navi (b1416)
- **Status server** Python (port 8082) — feeds Homepage widgets
- **Pi Coding Agent** — embedded Node.js, models.json points to localhost:8081

### Active Models (llama-swap config.yml)
| Model | GGUF Size | Quant | Context | Features | Notes |
|-------|-----------|-------|---------|----------|-------|
| gemma4-e2b | 3.2 GB | Q4_0 QAT | 16K | thinking, tools | General reasoning |
| lfm2.5-vl-450m | 362 MB | Q8_0 | 8K | vision | VLM auxiliar |
| lfm2.5-230m | 147 MB | Q4_K_M | 8K | tools | Ultra-light tool calling |
| lfm2.5-8b-a1b | 3.4 GB | APEX I-Mini | 8K | thinking, tools | 8B MoE, 3.82 t/s |
| lfm2.5-2.6b | 976 MB | IQ2_M | 8K | thinking, tools | Deployed today, 2.04 t/s |

### Docker Services (docker-compose.yml)
| Container | Image | RSS (MB) | Port | Status |
|-----------|------|----------|------|--------|
| searxng | searxng:latest | 30 | 8888 | **CRITICAL** — feeds Hermes MCP |
| pihole | pihole:latest | 2 | 53/8080 | Untested, took down network last time |
| uptime-kuma | uptime-kuma:1 | 1 | 3001 | Running but barely used |
| chromadb | chromadb:latest | 1 | 8100 | ChromaDB semantic memory |
| forgejo | forgejo:9 | 1 | 3300/2222 | Git server |
| homepage | homepage:latest | 153 | — | Dashboard |
| karakeep-web | karakeep:release | 0.1 | 3200 | Bookmarks |
| karakeep-chrome | alpine-chrome:124 | 106 | — | Headless Chrome for Karakeep |
| karakeep-meilisearch | meilisearch:v1.41 | 0.5 | 7700 | Search index |

### Other Services
- **llama-swap.service** — systemd unit enabled but runs as bare process (PID-managed)
- No Telegram bot currently running
- No cron jobs beyond system defaults

---

## Proposed Architecture

### Goal
Replace llama-swap with 2-3 bare llama-server processes managed by systemd, always-on, with ngram-mod speculative decoding. Add a lightweight agent (PicoClaw or similar) for Telegram communication with persistent session and context compression.

### Process Map (proposed)

| Service | Model | Port | Context | Quant | Est. RAM | Systemd Unit |
|---------|-------|------|---------|-------|----------|--------------|
| llm-main | LFM2.5-2.6B | 8081 | 32K | IQ2_M (976MB) | ~1.5 GB | sisyphus-llm.service |
| vlm-aux | LFM2.5-VL-450M | 8082 | 4K | Q4_0 (209MB) + mmproj Q8_0 (98MB) | ~370 MB | sisyphus-vlm.service |
| llm-sub | LFM2.5-230M | 8083 | 4K | Q4_0 (142MB) | ~180 MB | sisyphus-sub.service |
| status | Python | 8085 | — | — | ~10 MB | sisyphus-status.service |
| agent | PicoClaw (?) | — | — | — | ~10 MB | sisyphus-agent.service |

### RAM Budget

| Component | RAM |
|-----------|-----|
| System + daemons | ~1.5 GB |
| Docker containers (all) | ~300 MB |
| LLM main (2.6B, 32K) | ~1.5 GB |
| VLM aux (450M, 4K) | ~370 MB |
| LLM sub (230M, 4K) | ~180 MB |
| Status server | ~10 MB |
| Agent (PicoClaw) | ~10 MB |
| **Total** | **~3.87 GB** |
| **Free headroom** | **~3.73 GB** |

With 32K context, total is ~3.87 GB — leaves 3.73 GB headroom. Could push to 64K ctx (+320 MB KV) = ~4.2 GB total, still 3.4 GB free.

### ngram-mod

Build 9802 already supports `--spec-type ngram-mod`. Flags for upstream:
```
--spec-type ngram-mod
--spec-ngram-mod-n-match 24
--spec-ngram-mod-n-min 48
--spec-ngram-mod-n-max 64
--parallel 1
```

**Why ngram helps here (unlike Navi benchmarks):**
- CPU-only dense model — compute is the bottleneck (not GPU transfer)
- Always-thinking model — reasoning_content has repetitive patterns
- Multi-turn agentic — JSON tool calls, code boilerplate repeat
- Long context (32K) — more patterns to match in ngram cache
- Cost: ~16 MB RAM, zero extra compute when no match found

### Quantization Choices

| Model | Current | Proposed | Reason |
|-------|---------|----------|--------|
| LFM2.5-2.6B | IQ2_M (976MB) | IQ2_M (keep) | Most aggressive available, proven working |
| LFM2.5-VL-450M | Q8_0 (362MB) | Q4_0 (209MB) | Save 153MB, VL doesn't need token fidelity |
| LFM2.5-230M | Q4_K_M (147MB) | Q4_0 (142MB) | Marginal, keep current if easier |
| mmproj | F16 (181MB) | Q8_0 (98MB) | Save 83MB, Q8_0 is lightest available |

### Disk Space Recovery

| File | Size | Action |
|------|------|--------|
| gemma-4-E2B_q4_0-it.gguf | 3.2 GB | Delete |
| LFM2.5-8B-A1B-APEX-I-Mini.gguf | 3.4 GB | Delete |
| LFM2.5-VL-450M-Q8_0.gguf | 362 MB | Delete (replaced by Q4_0) |
| mmproj-LFM2.5-VL-450m-F16.gguf | 181 MB | Delete (replaced by Q8_0) |
| **Total freed** | **~7.1 GB** | |

---

## Agent Options (Telegram Communication)

### Requirements
- Single persistent session (always same conversation thread)
- Context compression/compaction (don't lose history, but don't OOM)
- Reports business/status via Telegram
- Connects to local llama-server (OpenAI-compatible API)
- Minimal RAM footprint (<50MB ideally)
- Runs on ARM Cortex-A72

### Candidates

| Agent | Lang | Size | RAM | Telegram | Context Mgmt | Notes |
|-------|------|------|-----|----------|--------------|-------|
| **PicoClaw** | Go | ~400KB binary | <10 MB | Yes (16+ platforms) | Episodic memory, structured | Best fit: ultra-light, Go binary, RPi-native |
| **NanoClaw** | Python | ~800KB | ~30-50 MB | Yes | Episodic memory | Heavier, Python deps |
| **MiniClaw** | Python | — | — | Via gateway | From-scratch runtime | More research needed |
| **OpenClaw** | Node.js | — | 100+ MB | Yes | Full agent loop | Too heavy for RPi |
| **Hermes Agent** | Python | — | 200+ MB | Yes | Full compact system | "Fat" — user's words |
| **Custom Python bot** | Python | ~5KB | ~20 MB | Yes (python-telegram-bot) | Manual | Roll our own, minimal |

### Recommendation: PicoClaw (primary), Custom Python (fallback)

**PicoClaw** is the strongest candidate:
- Go binary, ~400KB, <10MB RAM
- Built for Raspberry Pi (even Pi Zero)
- Native Telegram adapter
- OpenAI-compatible API support (points to llama-server:8081)
- Episodic memory (structured logs, picks up where left off)
- Single process, single binary, systemd-friendly

**Fallback**: A minimal Python script (~100 lines) using `python-telegram-bot` + `requests` to llama-server, with a rolling context window (keep last N messages, summarize older ones with the 230M sub-agent). ~20MB RAM. No external dependencies beyond pip.

---

## Services Audit

### Must Keep
| Service | Reason | Action |
|---------|--------|--------|
| **SearXNG** | Feeds Hermes MCP search, critical infrastructure | Keep as-is, monitor |
| **ChromaDB** | Semantic memory for Hermes | Keep as-is |
| **Forgejo** | Git server (ai-dotfiles remote) | Keep as-is |
| **Homepage** | Dashboard, status monitoring | Update widgets for new architecture |

### Evaluate
| Service | Status | Recommendation |
|---------|--------|----------------|
| **Uptime Kuma** | Running 2 weeks, barely used | Keep for now (low RAM), but evaluate if Pi-hole goes |
| **Pi-hole** | Running but took down network before | Investigate properly OR remove. Needs separate session |
| **Karakeep** | Running, 3 containers (~107MB RSS) | Evaluate usage — if unused, remove 3 containers + images (2.1GB) |

### Remove (if unused)
| Service | Disk Saved | RAM Saved |
|---------|-----------|----------|
| Karakeep (3 containers) | ~2.1 GB images | ~107 MB RSS |
| Uptime Kuma | ~439 MB image | ~1 MB RSS |
| Pi-hole (if problematic) | ~114 MB image | ~2 MB RSS |

---

## Implementation Phases

### Phase 1: Backup & Document (no changes to running system)
- [ ] Version current llama-swap config.yml in ai-dotfiles/sisyphus/
- [ ] Version current status server script
- [ ] Version current homepage services.yaml
- [ ] Version current docker-compose.yml
- [ ] Document current state in PLAN.md (this file)
- [ ] Create systemd unit templates

### Phase 2: Update llama.cpp
- [ ] git pull in ~/llama.cpp
- [ ] Apply 3 patches (assert, UI off, UI stubs)
- [ ] Build: `cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_UI=OFF && cmake --build build -j4 --target llama-server -k`
- [ ] Verify: `llama-server --version`
- [ ] Smoke test with existing models

### Phase 3: Download new GGUFs
- [ ] LFM2.5-VL-450M Q4_0 (209MB) — `wget` from LiquidAI
- [ ] mmproj-LFM2.5-VL-450m Q8_0 (98MB) — `wget` from LiquidAI
- [ ] Keep existing: LFM2.5-2.6B IQ2_M, LFM2.5-230M Q4_K_M

### Phase 4: Create systemd services
- [ ] sisyphus-llm.service (LFM2.5-2.6B, port 8081, 32K ctx, ngram-mod)
- [ ] sisyphus-vlm.service (LFM2.5-VL-450M, port 8082, 4K ctx)
- [ ] sisyphus-sub.service (LFM2.5-230M, port 8083, 4K ctx) — optional
- [ ] sisyphus-status.service (Python, port 8085) — update endpoints
- [ ] Enable + start all services

### Phase 5: Update Pi models.json
- [ ] Point main model to lfm2.5-2.6b on port 8081
- [ ] Update context windows
- [ ] Add VLM model entry (if Pi supports image input to port 8082)

### Phase 6: Update Homepage
- [ ] Update services.yaml: remove llama-swap widget, add per-model widgets
- [ ] Update status server: new model endpoints, new ports
- [ ] Version updated files in ai-dotfiles/sisyphus/

### Phase 7: Deploy agent (Telegram)
- [ ] Evaluate PicoClaw on RPi4 (download, configure, test)
- [ ] If PicoClaw works: configure Telegram bot, point to localhost:8081
- [ ] If not: build minimal Python Telegram bot with context compression
- [ ] Create systemd service for agent
- [ ] Test: send message, get response, verify session persistence

### Phase 8: Cleanup
- [ ] Stop llama-swap, remove from boot
- [ ] Delete old GGUFs (gemma4-e2b, lfm2.5-8b-a1b, old VL Q8_0, old mmproj F16)
- [ ] Remove llama-swap binary and config (keep backup in ai-dotfiles)
- [ ] Evaluate Karakeep/Uptime Kuma/Pi-hole removal
- [ ] Final RAM/disk audit

### Phase 9: Commit & document
- [ ] git add -A in ai-dotfiles/sisyphus/
- [ ] Update rpi-fleet.md skill reference
- [ ] Update memory with new architecture

---

## Open Questions

1. **Context size**: Start with 32K (safe) or 64K (ambitious)? Prefill at 2.5 tok/s makes 64K impractical for cold starts (~7 hours), but ngram-mod + warm sessions could make it viable for multi-turn.
2. **Pi-hole**: Fix or remove? Needs separate debugging session (DNS config, network topology).
3. **Karakeep**: Still using it? 3 containers + 2.1GB images is heavy if unused.
4. **Sub-agent (230M)**: Worth keeping always-on, or spin up on demand only?
5. **Agent choice**: PicoClaw (Go, ultra-light) vs custom Python (more flexible, heavier)?
6. **llama.cpp update**: Do it now (Phase 2) or after the architecture migration?