#!/usr/bin/env bash
# UPDATE 29/09 (post-aposentadoria): user em dúvida sobre manter H3 coexistente c/ LTX
# ("talvez a gente possa deixar o LTX para I2V e ter o H3 ainda instaladinho"; Ref2VA = saudade)
# → este script é o caminho de volta OFICIAL. Se aprovar a volta, rodar este e testar Ref2VA com
#   um caso de elenco (2 pessoas interagindo) - a rota onde o I2AV puro perde identidade.
# reinstall-h3.sh — restauração completa do MiniMax H3 aposentado em 29/09/2026
# Motivo da aposentadoria: LTX-2.3 v1.4 uncensored (motor Wan2GP default)
#   - I2AV+ref: 10min warm vs 43min cold H3; audio sem dream speech; pilha 4 LoRAs @1.0
# Fontes de re-download (todas verificadas em 2026-09-29):
#   - Checkpoints: HF DmitryDB/MiniMax-H3-Quants (INT8 ConvRot, mesmo formato era atual)
#     * FL2VA pruned 21GB: MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors (21GB)
#     * Ref2VA pruned: MiniMax-H3-Ref2VA-pruned_int8_convrot.safetensors (21GB)
#     * video VAE fp16 (4,9GB): MiniMax-H3-video_vae_fp16.safetensors
#     * audio VAE fp32 (605MB): MiniMax-H3-audio_vae_fp32.safetensors
#   - TE GGUF: Qwen3-VL-32B (14GB) — qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf (mesma fonte atual)
#   - Qwen3-VL-32B-Instruct full (14GB): HF Qwen/Qwen3-VL-32B-Instruct (usado pra VLM de
#     referências no --enhance-with-vision; opcional se só gerar)
#   - LoRAs: HF mirror Sentinel7/h3 + EllaPriest45/MinimaxH3_Actions + Civitai (HM NSFW pack)
#     * lista completa em loras/minimax_h3/ (12 arquivos, ~4,8GB — ver MANIFEST abaixo)
#   - PDD heads bf16 (2x 44MB): ckpts/minimax_h3_{fl2va,ref2va}_pdd_8step_heads_bf16.safetensors
#   - Latent upscaler 3D (660MB): ckpts/minimax_h3/minimax_h3_latent_upscaler_3d_bf16.safetensors
#   - Turbo LoRA: minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors
#     + turbo fl2v 8step (ver skill wan2gp seção LoRAs p/ lista + fontes HF)
#
# MANIFEST de LoRAs (aposentados com o H3; re-baixáveis das fontes acima):
#  HMNSFW_AIO_V2, hmpussy_v6, minimax_vag, vagassist_e40, HMInnie_v1_e50,
#  HMPenetration_I2V_stripped, Grindtime, PenisV2M3_000002750,
#  PornMaster_penis_side_view_I2V_V1, moawxx_000002750, jav_moans_jpnmoans_conv,
#  minimax-h3-facial-realism-closeup-cp2000, natural_face_speech_h3_lora_v1_500
#
# Passos (ordem):
# 1. git -C ~/git/Wan2GP log --oneline -3  # confirmar que os 3 WIP locais sobreviveram
# 2. hf download DmitryDB/MiniMax-H3-Quants --include "MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors MiniMax-H3-Ref2VA-pruned_int8_convrot.safetensors MiniMax-H3-video_vae_fp16.safetensors MiniMax-H3-audio_vae_fp32.safetensors MiniMax-H3-latent-upscaler-3d-bf16.safetensors" --local-dir ~/git/Wan2GP/models/minimax_h3/checkpoints/
# 3. hf download <fonte-do-TE> --include "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf" --local-dir ~/git/Wan2GP/models/minimax_h3/checkpoints/
# 4. hf download Qwen/Qwen3-VL-32B-Instruct --local-dir ~/git/Wan2GP/models/minimax_h3/checkpoints/Qwen3-VL-32B-Instruct/
# 5. LoRAs conforme skill wan2gp (seções NSFW LoRAs / nafasp / cp2000)
# 6. Verificar symlinks: ckpts/ -> models/minimax_h3/checkpoints/ (4K cada, mesmo schema)
# 7. cd ~/git/Wan2GP && h3 "smoke test" --turbo --dry-run  (valida schema headless)
#
# Não remove nada do LTX-2.3 (ltx2_uncensored/, loras/ltx2/, finetunes/ltx2_22B*)
# Para retornar ao H3 como default: h3 funciona direto após reinstall (wrapper intacto)
exit 0