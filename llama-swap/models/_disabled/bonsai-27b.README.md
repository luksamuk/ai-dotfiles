# bonsai-27b (1-bit Q1_0 vanilla) — EPITAFIO 2 (aposentado 12/nov/2026 pelo dono)

Segunda aposentadoria no mesmo dia da reativacao (07/nov - 12/nov). Motivo: probe 2048
REPROVADO (Game Over + You Win simultneos no boot, NENHUM board renderizado, js-test.js
de 1 byte criado e nunca escrito) e dono considera trocar por testes de cdigo nica
quando o TPE for ressuscitado (Bonsai-2 27B TPE, ver o epitaph do bonsai2-27b.yaml).
O 1-bit segue tecnicamente slido: 61/75 na rgu July, 9.61 tok/s @131K, 3.6GB.

RESTORE: mv _disabled/bonsai-27b.yaml models/ + footer var b27 + heavy set + evict 4 +
build-config ORIGINAL_ORDER + download-models.sh MODELS (x2: Q1_0 + mmproj) + Available +
header + size echo + build-config. Pesos em disco? CHECAR ~/.llama-models/bonsai-27b/
(podem ter sido preservados; se no: re-download prism-ml/Bonsai-27B-gguf Q1_0 + mmproj Q8_0).
