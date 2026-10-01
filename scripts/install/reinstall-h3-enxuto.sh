#!/usr/bin/env bash
# reinstall-h3-enxuto.sh — H3 versão GGUF Q4 (29/09/2026, ~37,5GB vs ~62GB do original)
# Contexto: LTX 2.3 v1.4 virou motor principal (I2AV/T2V). H3 volta como COEXISTENTE,
#   com o Ref2VA como diferencial (elenco/sheets). User pediu versão enxuta.
#
# O QUE JÁ FOI BAIXADO (29/09, em ~/git/Wan2GP/models/minimax_h3/checkpoints/):
#   minimax_h3_fl2va_pruned-Q4_K.gguf                            10,64GB  (unsloth)
#   minimax_h3_ref2va_pruned-Q4_K.gguf                           10,60GB  (unsloth)
#   MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-Q3_K_S.gguf   10,67GB  (pottokao)
#   vae/minimax_h3_video_vae_fp16.safetensors                     4,85GB  (Abiray)
#   vae/minimax_h3_audio_vae_fp32.safetensors                     0,56GB  (Abiray)
#
# FONTES (verificadas 29/09):
#   - Modelos GGUF: https://huggingface.co/unsloth/MiniMax-H3-GGUF
#       FL2VA: Q2_K/Q3_K/Q4_K/Q5_0/Q6_K/Q8_0 + UD-Q2_K_XL/UD-Q3_K_XL
#       Ref2VA: Q2_K/Q3_K/Q4_K (UD-Q3_K_XL NÃO existe p/ Ref2VA)
#   - TE abliterated: https://huggingface.co/pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF
#       (só Q3_K_S: 10,67GB + variante _vis 11,78GB; Q4_K_M do antigo era 14GB de outro repo que 404ou)
#       alternativa maior: joeygambino/MiniMax-H3-encoder-GGUF Q4_K_M 18,4GB (+ mmproj-F16 1,12GB p/ vision)
#   - VAEs: https://huggingface.co/Abiray/MiniMax-H3-GGUF (pasta vae/) — DmitryDB/MiniMax-H3-Quants GATEOU
#   - Turbo LoRA lightx2v 4step (opcional, enxuga TEMPO): GuangyuanSD/minimax_h3_video_vae_int8_convrot
#       arquivo: minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors
#   - LoRAs (curadoria NSFW em curso): Sentinel7/h3 (297 LoRAs, HM NSFW completo) +
#       EllaPriest45/MinimaxH3_Actions (237 LoRAs, 57 motion/NSFW; 91 mp4 previews em ~/v/preview_h3/)
#
# PENDENTE PÓS-BÁSICO (o que este script NÃO cobriu ainda):
#   1. Atualizar finetunes ~/.hermes/git/Wan2GP/finetunes/minimax_h3_fl2va|ref2va.json p/ apontar GGUF Q4
#      (hoje apontam pros int8_convrot antigos, deletados)
#   2. Symlinks ckpts/ -> models/minimax_h3/checkpoints/ (padrão: cada arquivo, mesmo schema)
#   3. Smoke test: cd ~/git/Wan2GP && h3 "smoke test" --turbo --dry-run
#   4. Curadoria LoRAs NSFW: user assiste ~/v/preview_h3/ e escolhe; depois baixar da EllaPriest45/Sentinel7
#   5. PDD heads (2x 44MB, opcional): DmitryDB/MiniMax-H3-Quants GATEADO — buscar mirror (Abiray?)
#   6. Latent ups. 3D (660MB, opcional): mesmo repo do PDD
#   7. Qwen3-VL-32B-Instruct full (14GB, só se usar --enhance-with-vision H3-native)
#   8. Quality check Q4_K vs int8_convrot antigo: user julga com sheets conhecidos; degradou → Q5_K
#      (FL2VA Q5 13,1GB + Ref2VA Q5 13,1GB = +4,9GB, ainda -12GB vs original)
#
# MANIFEST de LoRAs antigos (perdidos, re-baixáveis de Sentinel7/h3 — NOMES para grep):
#  HMNSFW_AIO_V2, hmpussy_v6, minimax_vag, vagassist_e40, HMInnie_v1_e50,
#  HMPenetration_I2V_stripped, Grindtime, PenisV2M3_000002750,
#  PornMaster_penis_side_view_I2V_V1, moawxx_000002750, jav_moans_jpnmoans_conv,
#  minimax-h3-facial-realism-closeup-cp2000, natural_face_speech_h3_lora_v1_500
#
# Não remove nada do LTX-2.3 (ltx2_uncensored/, loras/ltx2/, finetunes/ltx2_22B*)
exit 0