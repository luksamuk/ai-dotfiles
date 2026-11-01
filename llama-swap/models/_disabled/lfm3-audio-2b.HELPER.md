# LFM3-Audio-2B (stock Q8_0) — APOSENTADO A SECO 31/10/2026

Substituto ATIVO: `lfm3-audio-2b-pollard` (Q6_K, mesma arch, smoke provado
27/10 + duplex realtime do dono 31/10). O mmproj-F16 oficial (570846400
bytes) foi MOVIDO pra ~/.llama-models/lfm3-audio-2b-pollard/mmproj-F16.gguf
(o lane Pollard reusa — repo Pollard não inclui mmproj). O Q8_0 stock foi
DELETADO (~1.72 GB) pra economizar disco, a pedido do dono.

Fonte HF (verificada HTTP 200 no momento do delete):
`LiquidAI/LFM3-Audio-2B-GGUF` (Q8_0 = 1846236704 bytes exatos)

Reviver:
```sh
hf download LiquidAI/LFM3-Audio-2B-GGUF LFM3-Audio-2B-Q8_0.gguf --local-dir ~/.llama-models/lfm3-audio-2b/
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/lfm3-audio-2b.yaml \
       ~/git/ai-dotfiles/llama-swap/models/lfm3-audio-2b.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
# decidir mmproj: devolver p/ este dir (mv de volta + reverter path no
# fragment pollard) ou duplicar — o Pollard aponta pro arquivo onde estiver
```