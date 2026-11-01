# Spark-X2.6-1.7B (+DSpark draft) — APOSENTADO A SECO 01/11/2026

Vida total: ~2 horas. Instalado a pedido do dono (1M papel, plain + draft
BF16 345M), julgado no playground do dono: o thinker do modelo (thinking
HIGH é o default) estourou o cap de resposta da sessão — "puta thinker" só
que na régua errada. GGUFs deletados ~2.4 GB (Q8_0 1.82 GB + draft 696 MB).

Réguas medidas aqui (01/11, fork xhtoken 22c0ae1 + frota):
- plain @262K (fit manteve os 262K c/ desktop 1.2GB): 60.1 t/s cru, VRAM 4631 total
- chat via template (temp 1.0, thinking ON): 57 t/s — **o DSpark draft foi
  REPROVADO P/ CHAT**: aceitação 16.8-21%, 1.16-1.46 tok/verificação; o passe
  do draft (~21ms) + alvo (~17ms) exigiria >2.27 tok/verif pra vencer.
- cru temp 0 (sem template): 184 t/s com 92% de aceitação (3× REAL) — mas
  nenhum uso da casa é continuação crua.
- Benchmarks do card: Average 49.6 / BFCL 55.2 / τ² 85.3 / SWE-V 59.8 /
  MRCR-256k 56.1 / Graphwalks-1M 25.7 (thinking HIGH).

Fontes HF verificadas 200: XHToken/Spark-X2.6-1.7B-GGUF (Q8_0 1820113440 bytes)
e XHToken/Spark-X2.6-1.7B-DSpark-GGUF (DSpark 695644928 bytes). O gate do
draft (#31372, tracker vigiando) segue valendo para re-testes.

Reviver:
```sh
hf download XHToken/Spark-X2.6-1.7B-GGUF Spark-X2.6-1.7B-Q8_0.gguf --local-dir ~/.llama-models/spark-x2.6-1.7b/
hf download XHToken/Spark-X2.6-1.7B-DSpark-GGUF Spark-X2.6-1.7B-DSpark.gguf --local-dir ~/.llama-models/spark-x2.6-1.7b/
git mv ~/git/ai-dotfiles/llama-swap/models/_disabled/spark-x2.6-1.7b.yaml \
       ~/git/ai-dotfiles/llama-swap/models/spark-x2.6-1.7b.yaml
cd ~/git/ai-dotfiles/llama-swap && env -u PYTHONPATH /usr/bin/python3 build-config.py
```