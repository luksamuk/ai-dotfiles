# Claude Code

## Posição atual (17/nov/2026): uso SOMENTE via `claude-local`

O Claude Code da Anthropic contém código que coleta dados do usuário (timezone,
proxy, conexões de AI Lab) e injeta no prompt sem consentimento, com
direcionamento específico a usuários chineses. A posição histórica deste
documento era "NÃO USE, recuso-me".

Posição atual: uso condicional com o wrapper `claude-local` (espelhado aqui),
que desliga TUDO o que aceita env:

- `ANTHROPIC_BASE_URL` → Ollama (:11434) ou llama-swap (:12434); nada sai pra
  api.anthropic.com
- trio `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` + `SUBAGENT` +
  `ANTHROPIC_DEFAULT_FABLE_MODEL` → o modelo local escolhido no picker (gum)
- `--model sonnet` fixo no exec (alias válido client-side; o estado global do
  CLI restaura o último modelo usado e rejeita ids fora do catálogo dele)
- telemetria: `DISABLE_TELEMETRY`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`,
  error reporting/feedback/Statsig via `NONESSENTIAL_TRAFFIC`, `DO_NOT_TRACK`,
  autopdater off, OTEL exporters `none`, attribution header zerado
- NÃO ligar firewall nos domínios de telemetria (retry storm em `/api/hello`,
  medido 2026, Chaser Systems)

## Roteamento (verificado no fio 17/nov/2026)

- llama-swap v216 expõe `/v1/messages` (Anthropic) — lanes ik E cafe respondem
  com thinking blocks. O claude → `:12434/v1/messages` → backend funciona pra
  frota inteira.
- Listagem de modelos replica o filtro do testchat (`_classify_non_chat`:
  tts/embedding/image-gen/asr/decision-only escondidos, dedupe por id+name,
  contagem de ocultos). `[vision]` marca modelos que aceitam imagem; o claude
  não tem flag client de inibição de visão — recusa em modelo sem vision é 400
  do server.
- Mythos: o CLI 2.1.327 não tem env `ANTHROPIC_DEFAULT_MYTHOS_MODEL` (é
  availability-gated: `isMythosAvailable`); Fable tem env completo e está
  mapeado.
