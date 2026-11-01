# Nemotron 3.5 Bolt E4B — APOSENTADO A SECO 31/10/2026

Pedido do dono. Dir `~/.llama-models/nemotron-bolt-e4b/` deletado (~5.1 GB:
UD-Q4_K_XL + dflash draft Q8_0). Arch gemma4 (pos-treino NVIDIA), 64K em Q4
por escolha do dono, text-only.

Fonte: `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` — **GATED** (API 401:
existe, precisa aprovação de acesso pra re-baixar).

Reviver:
```sh
hf download unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF \
  NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf \
  dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf \
  --local-dir ~/.llama-models/nemotron-bolt-e4b/     # requer acesso (gated)
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/nemotron-bolt-e4b.yaml \
       ~/git/ai-dotfiles/llama-swap/models/nemotron-bolt-e4b.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
# + reverter comentários: var nbe4, evict nbe4, sets small_embed/small_only (twp/nbe4 removidos 31/10)
```