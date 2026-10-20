# laya — DESINSTALADO 20/out/2026 (decisão do user)

O laya (421M ModernBERT decision model) saiu da frota por falhar com número nos
dois papéis da casa:

- **Seletor de LoRA (harness 20/out, 10 casos rotulados):** recall@3 21% (d1-3b: 93%),
  3/10 casos, 4 FALSOS no controle negativo (latex 0.88 pra paisagem sem gente),
  prior fixo (latex/analogcore sempre no topo) — não lê o contexto do pedido.
- **Juiz do laya-snake (smoke 02/10):** ruído ~0.5 no payload real (0.48 vs 0.53);
  o julia-1 discriminou ~400x melhor no mesmo payload.
- Risco ativo: o scanner do laya_client pega qualquer `decision: true` — o seletor
  da UI podia escolher o pior juiz da frota pro jogo.

Pesos: **APAGADOS** (`~/.llama-models/decision-models/Laya-GGUF/`, 805MB).
Re-enable: re-baixar GGUF (ggml-org/Laya-GGUF, BF16), mover o fragment de volta
pra `models/`, descomentar alias+evict no footer e a linha do build-config.py,
`python3 build-config.py`, reload. A frota segue com 3 decision models:
julia-1 (juiz do snake), d1-3b (seletor diffuse), d1-omni (reserva com visão).
