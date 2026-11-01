# LFM2.5 Embedding 350M — APOSENTADO A SECO 31/10/2026

Pedido do dono. GGUF `~/.llama-models/lfm2.5-embed/LFM2.5-Embedding-350M-Q8_0.gguf`
deletado (~350 MB). ATENÇÃO: o dir `lfm2.5-embed/` é COMPARTILHADO com o
**lfm2.5-colbert-350m (ATIVO)** — o ColBERT-350M Q8_0 que lá mora NÃO foi
tocado. O lfm2.5-embedding-350m cobria só /v1/embeddings (CPU-only); lfmcol
(ColBERT) + eg2 (EmbeddingGemma 2) permanecem cobrindo.

Nome corrigido no fragment (31/10): "LFM2.5 Embed 350M" → **"LFM2.5 Embedding 350M"**
(nome certo do card da LiquidAI) — pedido do dono junto da aposentadoria.

Fonte HF verificada HTTP 200 no momento do delete: `LiquidAI/LFM2.5-Embedding-350M-GGUF` (Q8_0).

Reviver:
```sh
hf download LiquidAI/LFM2.5-Embedding-350M-GGUF LFM2.5-Embedding-350M-Q8_0.gguf \
  --local-dir ~/.llama-models/lfm2.5-embed/
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/lfm2.5-embedding-350m.yaml \
       ~/git/ai-dotfiles/llama-swap/models/lfm2.5-embedding-350m.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
# + reverter comentários: var lfmemb, evict lfmemb (twp: 1), sets embed_only/rerank_embed
#   (aí as expressões de 31/10 voltam pro estado com (lfmemb | eg2) / lfmemb & ...)
```