# h3/scripts — auxiliares do wrapper

Os 5 arquivos aqui são dependências de runtime do `h3` (o wrapper versionado em `h3/h3`).
Cada um tem um destino fixo de deploy:

| Arquivo (repo) | Destino de produção | Papel |
|---|---|---|
| `rerank_catalog.py` | `~/.local/share/h3/` | Pré-ranking do catálogo por embeddings (ColBERT no llama-swap :12434) + boost léxico; fail-open — sem modelo ativo, imprime o catálogo completo |
| `build_catalog_prompt.py` | `~/.local/share/h3/` | Fallback do bloco de catálogo (sem ranking) |
| `nsfw_lora_resolve.py` | `~/.local/share/h3/` | Resolve IDs→arquivos/multipliers/triggers; modos `--ids` (force) e `--out` (extrai `LORAS:` do output do enhance) |
| `h3-output-filter.py` | `~/.local/bin/` | Compacta o chatter de offload do stdout do wgp.py num log vivo; log íntegro em `/tmp/h3_debug.log` |
| `relay_vision_common.py` | `~/.local/bin/` | Utilitário de vision check para o auto-despacho de imagens no enhance |

## O arquivo que NÃO está aqui (de propósito)

`nsfw_catalog_h3.json` — o catálogo de LoRAs — **não é versionado neste repo**
porque contém descrição de conteúdo explícito; em conversa/prose isso é tratado
como seleção técnica de LoRAs, mas o texto integral do catálogo fica fora do
histórico público. Em produção ele vive em `~/.local/share/h3/nsfw_catalog_h3.json`
e TODOS os três scripts Python o leem via env `NSFW_CATALOG_JSON` (default
`~/.local/share/h3/nsfw_catalog_h3.json`).

Deploy: copiar cada arquivo para o destino da tabela (não há symlinks — são
cópias simples). Os defaults de path dos scripts já apontam para produção
(`~/.local/share/h3/`, `~/.local/bin/`), então cópia sem ajuste funciona.