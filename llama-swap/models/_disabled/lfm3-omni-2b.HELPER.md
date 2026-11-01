# LFM3-Omni-2B (stock Q8_0) — APOSENTADO A SECO 31/10/2026

Substituto ATIVO: `lfm3-omni-2b-pollard` (Q6_K + mmproj-BF16 do repo
Pollard — MESMO TAMANHO (949841248 bytes) mas sha256 DIVERGE do F16
stock: a9938f9f vs 361bc951; o lane Pollard usa o par testado do repo
Pollard INTEIRO — Q6_K + mmproj BF16). O Q8_0 stock (~1.72 GB) e o
mmproj-F16 stock (~905 MB) foram DELETADOS (~2.6 GB), a pedido do dono.
Frag do Pollard: models/lfm3-omni-2b-pollard.yaml (nova, 31/10, SMOKE
PENDENTE — aguardando janela de GPU do dono).

Fonte HF (verificada HTTP 200 no momento do delete):
`LiquidAI/LFM3-Omni-2B-GGUF` (Q8_0 = 1846240768 bytes; mmproj-F16 =
949841248 bytes)

Reviver:
```sh
hf download LiquidAI/LFM3-Omni-2B-GGUF LFM3-Omni-2B-Q8_0.gguf \
  mmproj-F16.gguf --local-dir ~/.llama-models/lfm3-omni-2b/
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/lfm3-omni-2b.yaml \
       ~/git/ai-dotfiles/llama-swap/models/lfm3-omni-2b.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
```

Notas herdadas (ver o HELPER/frag do pollard): doom loop na descricao de
IMAGEM (aberto 27/10, investigar mtmd do lfm3), voz PT sotaque EN
(checkpoint), nao-thinking (dual-mode e do 1.7B apenas).