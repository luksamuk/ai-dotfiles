# _disabled/ — Re-enable guide

> **REATIVADO 20/out/2026.** Fragments de volta em `models/` (principal cafe + `-fast` ik),
> pesos re-baixados (APEX-I-Mini 14,3 GB + mmproj 902 MB, repo `mudler/Qwen3.6-35B-A3B-uncensored-heretic-APEX-GGUF`),
> alias `q36` e `evict_costs.q36` descomentados no footer. Este documento fica como histórico;
> os sets `medium`/`heavy` NÃO foram re-anchorados (decisão: enhance não exige coexistência de matriz).


## qwen3.6-35b-a3b-heretic (REMOVED Oct 8 2026 — weights DELETED, user explicit)

User aposentadoria: ornith-1.5-35b-heretic assumiu todos
os enhance da frota (diffuse/h3/ltx via fleet yaml `enhance-models.yaml`).
Nenhum fluxo da casa exige o handle.

- Weights: DELETADOS (2026-10-08, `~/.llama-models/qwen36-heretic-apex-mini/` — GGUF 14G APEX-I-Mini + mmproj 861M exclusive). Re-download: HF `SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF` I-Compact + mmproj unsloth Q8 (~15G).
- Fragment: `_disabled/qwen3.6-35b-a3b-heretic.yaml`
- Footer edits (comentado, não removido): alias `q36`, evict_costs.q36
- Sets re-anchored: `medium` e `heavy` agora `"(k2 | lag) & (nemb | lfmcol | eg2)"` e heavy `... & qpoll` (footer lines 125/133)
- Re-enable: baixar de volta (path era `${models_dir}/qwen36-heretic-apex-mini/...`), mover fragment pra `models/`, descomentar alias+evict no footer, re-add q36 aos sets SE desejado, `python3 build-config.py`, reload verificado via `/v1/models`.