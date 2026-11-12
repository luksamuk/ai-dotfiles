# qwen3.5-4b-3d-coder (ccore) — EPITAFIO (aposentado 12/nov/2026 pelo dono)

Probe 2048: REPROVADO (2/7 no instrumento deterministico de slide/merge — sem compactao,
cascata de merge, up/left no-ops — aps declarar "all requirements verified" com a simulao
nunca executada). Round-2 multimodal (imagem de referncia voxel via @attach no pi + pedido
explcito de PATCH): title corrigido e casinha central, mas faroilha/ilha/mar nunca vieram
("Voxel Light" round-1; assets do experimento em references/game-probe-assets/).

Viso: FUNCIONA (mmproj BF16 acoplado, imagem 1024 do qwen-image-2.1-turbo ingerida OK).

RESTORE: mv _disabled/qwen3.5-4b-3d-coder.yaml models/ + re-adicionar var tq3d no
config-footer (vars + small_vision + evict 2) + download-models.sh (2 MODELS entries +
header + Available + size echo) + build-config ORIGINAL_ORDER. Pesos em disco:
~/.llama-models/qwen3.5-4b-3d-coder/ (Q4_K_M 2.7GB + mmproj BF16 0.64GB, INTEGROS).
