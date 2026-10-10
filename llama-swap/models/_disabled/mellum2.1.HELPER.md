# Re-enable: Mellum2.1-12B (aposentado 10/10/2026 com honras)

GGUF removido em 10/10/2026. Fragmento preservado em este diretório
(`mellum2.1-12b.yaml`) com a receita ik completa (unified memory, 131K SWA).

## Fonte do modelo

- GGUF: `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf` (JetBrains Mellum 2.1,
  MoE 12B total / 2.5B ativos, Q4_K_M ~8.07GB)
- Destino: `~/.llama-models/mellum2.1/`
- Texto-only (sem mmproj). Tool calls: `max_tokens >= 600` obrigatório
  (reasoning consome budget antes do call).

## Reativação (ordem)

1. `mv models/_disabled/mellum2.1-12b.yaml models/`
2. `cd llama-swap && python3 build-config.py` (sync automático pra live)
3. Re-add nos harnesses: `configs/pi/models.json` (schema igual aos outros
   handles llama-swap, entradas `mellum2.1-12b` + `mellum2.1-12b:think`)
   + sync pra `~/.pi/agent/models.json` — e qualquer outro harness desejado
4. Reload do swap (watch-config auto ou SIGHUP)
5. Verificar: `curl -s localhost:12434/v1/models | jq -r '.data[].id'`
   lista AMBOS os handles; `pi --list-models`

## Parâmetros de deploy (vencedores do A/B de 05/10)

- ik: `-ngl 99 -ot "ffn_.*_exps=CPU" + GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`
  → 43.4 t/s (recorde da frota); café `-cmoe` 40.6; vanilla `-ot` 40.2
- KV: 21/28 layers sliding-window @1024 + 7 full (~60MB/16K) → 131K native
  (~7.2GB, decode paridade com 32K)
- Sampling vendor: temp 0.6 / top-p 0.95 / top-k 20
- Detalhes + notas de bench: skill `saluki-mellum-model-notes`

## Legado de benchmark (procedência disk, `FLEET-N5-CANONICAL.md`)

N=5 régua 155: média 121.8 (78.6%), mediana 122, piso 120, teto 124,
CV 1.2% (mais estável da classe pequena), fail_rate 0%. Per bench:
01=22.2 · 02=20.8 · 03=19.4 · 04-ichef=23.4/30 (teto 27) · 05=22.4 ·
06-sec=13.6 (intermitência de tool-calling, dado do modelo).
Runs em `~/projects/ai/benchmarks/agentic/results/mellum2_1-12b_think/`
(score_report.json íntegros, re-score possível).