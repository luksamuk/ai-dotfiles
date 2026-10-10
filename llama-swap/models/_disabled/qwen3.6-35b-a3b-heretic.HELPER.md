# _disabled/ — Re-enable guide

## qwen3.6-35b-a3b-heretic (REMOVED Oct 8 2026 — weights DELETED, user explicit)

User aposentadoria: ornith-1.5-35b-heretic assumiu todos
os enhance da frota (diffuse/h3/ltx via fleet yaml `enhance-models.yaml`).
Nenhum fluxo da casa exige o handle.

- Weights: DELETADOS (2026-10-08, `~/.llama-models/qwen36-heretic-apex-mini/` — GGUF 14G APEX-I-Mini + mmproj 861M exclusive). Re-download: HF `SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF` I-Compact + mmproj unsloth Q8 (~15G).
- Fragment: `_disabled/qwen3.6-35b-a3b-heretic.yaml`
- Footer edits (comentado, não removido): alias `q36`, evict_costs.q36
- Sets re-anchored: `medium` e `heavy` agora `"(k2 | lag) & (nemb | lfmcol | eg2)"` e heavy `... & qpoll` (footer lines 125/133)
- Re-enable: baixar de volta (path era `${models_dir}/qwen36-heretic-apex-mini/...`), mover fragment pra `models/`, descomentar alias+evict no footer, re-add q36 aos sets SE desejado, `python3 build-config.py`, reload verificado via `/v1/models`.