# whittle-35b — DESINSTALADO 20/out/2026 (decisão do user)

Whittle-Qwen-3.8-35B-A3B-ONYX-compact (21,9 GB) saiu da frota para dar lugar
ao `qwen3.6-35b-a3b-heretic` (melhor prompter para os enhance da casa).

- Pesos: **APAGADOS** (`~/.llama-models/whittle-35b/`), decisão explícita do user (liberar 21,9 GB).
- Fragments: `_disabled/whittle-35b.yaml` (cafe, principal) + `_disabled/whittle-35b-fast.yaml` (ik_llama, offload de experts).
- Re-enable: re-baixar o GGUF ONYX-compact 35B, mover os 2 fragments de volta pra `models/`,
  `python3 build-config.py` (~/.config/llama-swap/config.yaml), reload e conferir `/v1/models`.
- Nenhum alias do footer apontava pra ele (entrava só por nome completo).
