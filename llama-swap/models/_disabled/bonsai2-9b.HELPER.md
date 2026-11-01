# Ternary Bonsai 2 9B (PTQ1_0 + PQ2_0-MTP fast) — APOSENTADOS A SECO 31/10/2026

Pedido do dono: desinstalar o "Ternary Bonsai 9B" — os DOIS deploys saíram
(base PTQ1_0 @131K e a variante fast PQ2_0-MTP @64K, dir `~/.llama-models/bonsai2/`
deletado inteiro, ~5.3 GB: 2.0 + 2.68 + mmproj Q8_0 0.62).

Fontes HF verificadas HTTP 200 no momento do delete: `prism-ml/Ternary-Bonsai-2-9B-gguf`.
Sempre-thinking (sem :think; samplers oficiais: temp 1.0/0.95/20/presence 1.0).
História relevante: migram pro cafe (kernel PTQ1_0 nativo do Prism, 22/10);
variante sempre-think sem handle dual. NOTA: o bonsai2-27b-heretic SEGUE ATIVO
(outro modelo, outra dir).

Reviver:
```sh
hf download prism-ml/Ternary-Bonsai-2-9B-gguf Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf \
  --local-dir ~/.llama-models/bonsai2/
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/bonsai2-9b.yaml \
       ~/git/ai-dotfiles/llama-swap/models/bonsai2-9b.yaml
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/bonsai2-9b-fast.yaml \
       ~/git/ai-dotfiles/llama-swap/models/bonsai2-9b-fast.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
# + reverter os comentários no config-footer.yaml: var bs2, set bs2_embed, evict bs2
```