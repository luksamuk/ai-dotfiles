# dsh (DeepSeek Harness) — config da casa

Harness de agentes da DeepSeek (Cordis), instalado com **node isolado** e
configurado 100% manual (sem web UI). Espelho dos arquivos vivos de `~/.dsh`.

## Topologia (contrato de instalação — não desvirtue)

- **Node isolado**: `~/.local/opt/node-dsh` (v24.22.1, cache própria
  `node-dsh/npm-cache`, corepack/pnpm dentro). O outro node da máquina é o do
  Hermes (`~/.hermes/node`) — NUNCA instalar/rodar dsh com ele.
- **PATH**: só o symlink `~/.local/bin/dsh` (+ `dsh-tui` wrapper e o wrapper
  `pnpm` do node isolado).
- **DSH_HOME** = `~/.dsh`: profiles (patches por profile), sessions, storages,
  `.env`, `.credentials.yaml` (os dois ÚLTIMOS NÃO SÃO BACKUPADOS — segredo).
- Profiles: `tui` (terminal) e `headless` (one-shot) usam o MESMO catálogo.
- Versões (política de supply-chain: `min-release-age=7` no ~/.npmrc):
  - dsh `@deepseek-ai/dsh@0.2.1-rc.3`
  - TUI `@deepseek-harness-tui/dsh-tui@0.18.0`
  - MCP `@deepseek-ai/dsh-mcp-client@0.2.1-rc.3`
- Skill de referência: `llama-swap-fleet` + `deepseek-harness` (Hermes).

## O que tem aqui

- `profiles/{tui,headless}/cordis.patch.yml` — catálogo de models (frota
  llama-swap 12434 + ollama cloud 11434), rota default, reasoning efforts por
  modelo, MCP searxng (wrapper no Sisyphus, `100.72.245.16:8899/mcp`), deepseek
  desativado. Regra da casa: thinking ON por default, sem variantes `:think`.
- `profiles/*/package.json` — pins das dependências (manifest de restore).
- `env.example` — estrutura do `~/.dsh/.env` (dummies + token do MCP searxng).

## Restore

```bash
# 1. node isolado + dsh (pin por política de 7 dias)
mkdir -p ~/.local/opt/node-dsh && tar -xJf node-v24.22.1-linux-x64.tar.xz -C ~/.local/opt
export PATH="$HOME/.local/opt/node-dsh/bin:$PATH" npm_config_cache=$HOME/.local/opt/node-dsh/npm-cache
npm i -g @deepseek-ai/dsh@0.2.1-rc.3
corepack enable --install-directory ~/.local/opt/node-dsh/bin   # pnpm

# 2. profiles + plugins
dsh plugin --profile tui add @deepseek-harness-tui/dsh-tui@0.18.0
dsh plugin --profile tui add @deepseek-ai/dsh-mcp-client@0.2.1-rc.3
dsh plugin --profile headless add @deepseek-ai/dsh-mcp-client@0.2.1-rc.3
cp profiles/tui/cordis.patch.yml ~/.dsh/profiles/tui/
cp profiles/headless/cordis.patch.yml ~/.dsh/profiles/headless/

# 3. env
cp env.example ~/.dsh/.env && $EDITOR ~/.dsh/.env   # preencher SEARXNG_MCP_TOKEN

# 4. symlinks
ln -sf ~/.local/opt/node-dsh/bin/dsh ~/.local/bin/dsh
# wrapper dsh-tui: exec dsh --profile tui "$@"
# wrapper pnpm: export PATH="$HOME/.local/opt/node-dsh/bin:$PATH"; exec .../bin/pnpm "$@"

# 5. valida
dsh --profile headless --dump-config   # deve listar os 19 reasoningEfforts
dsh --profile headless "Say: VIVO"
```

## O que NÃO mora aqui (por quê)

- `~/.dsh/.env` real — token do MCP (segredo).
- `~/.dsh/.credentials.yaml` — credenciais.
- `~/.dsh/sessions/` + `storages/` — conteúdo de conversa (privado).
- `~/.dsh-tui/onboarding.json` — flag de first-run (regenerável).