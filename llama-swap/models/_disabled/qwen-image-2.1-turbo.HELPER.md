# Re-enable: qwen-image-2.1-turbo

Aposentado 19/10/2026 (user: backend trocado; a rota nova do qwen-image não serve API como o sd.cpp servia — o fragment era puro sd-server).

1. GGUF ausente: `diffuse/models/qwen-image-2.1/` só tem `lora/` e `SHA256SUMS`. Re-baixar o triple (diffusion Q4_K_M + vae bf16 + text encoder Sakura-8bit) — o SHA256SUMS valida os nomes/tamanhos.
2. `mv models/_disabled/qwen-image-2.1-turbo.yaml models/` (sd.cpp precisa estar buildado: `git/stable-diffusion.cpp/build/bin/sd-server`).
3. `python3 build-config.py` (deploy sincroniza) e conferir `/v1/models`.
4. Nada de footer: nunca teve alias em vars/evict_costs/groups. Harnesses nunca o receberam.
