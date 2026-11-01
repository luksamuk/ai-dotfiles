# TwIL-LM3-Pro — APOSENTADO A SECO 31/10/2026

Pedido do dono. Dir `~/.llama-models/twil-lm3-pro/` deletado (~2.8 GB, Q6_K;
o Q8_0 já tinha saído do disco no A/B de 01/10). Especialista de lógica formal
(webAI, base Granite 4.2 da IBM), track A com :think (handle morre junto),
temp 0 no think inject; card pede max_tokens >=2048.

Fonte HF verificada HTTP 200 no momento do delete: `webAI-Official/TwIL-LM3-Pro` (Q6_K).

Reviver:
```sh
hf download webAI-Official/TwIL-LM3-Pro TwIL-LM3-Pro-Q6_K.gguf \
  --local-dir ~/.llama-models/twil-lm3-pro/
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/twil-lm3-pro.yaml \
       ~/git/ai-dotfiles/llama-swap/models/twil-lm3-pro.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
# + reverter comentários: var twp, evict twp (twp: 2), e pequenos sets (small_vision/
#   small_embed/small_only perderam o twp em 31/10 — comentário da trilha diz como era)
```