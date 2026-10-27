# Flint 40B A2.5B — aposentado em 27/out/2026 a pedido do dono

Motivo do dono: "ele não é nem metade de bom igual o Ornith" (Ornith-1.5-35B
heretic = code do sprach). Desativado antes do teste de aposentadoria formal
via testchat; nunca chegou a disputar N=5.

## Origem (re-download)

- Fragment: `mudler/flint-APEX-GGUF` — arquivo `Flint-APEX-I-Compact.gguf`,
  `18424602016 bytes exatos` (~18.42 GB), arquitetura `flint` (40B MoE A2.5B,
  256 experts top-6 + 2 shared, 49 blocos, 2 dense leading, híbrida full+SWA-513).
- Destination era `~/.llama-models/flint-40b-a2b/Flint-APEX-I-Compact.gguf`.

```sh
hf download mudler/flint-APEX-GGUF Flint-APEX-I-Compact.gguf --local-dir ~/.llama-models/flint-40b-a2b/
```

## Reativar

1. Baixar o GGUF acima (comando), conferir tamanho == 18424602016.
2. `git mv` os dois fragments de volta (`flint-40b-a2b.yaml`, `flint-40b-a2b-fast.yaml`).
3. Descomentar no `config-footer.yaml`: aliases `flint`/`flintf` + `evict_costs` 9/9
   (blocos comentados com "APOSENTADO 27/out/2026").
4. `build-config.py` e smoke curto nos dois ids.

## Estado do disco

- Desinstalação total (rm de `~/.llama-models/flint-40b-a2b/` = +18.42 GB livres)
  era PENDENTE em 27/out (destrava o experimento Groot-65B-A6B Q2_K no cafe).
- Café ganhou parser dedicado flint.cpp (commits 5d261b305/1dd699d35/02dda80a9)
  — o parser fica no fork, não nos pesos; nada a fazer para reativar.
- Histórico/métricas do smoke (25-26/10): decode 36,3 t/s @112K cafe / 47 t/s @64K ik,
  tool calling OK, reasoning always-on — detalhes nos fragments movidos.