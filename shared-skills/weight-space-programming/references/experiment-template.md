# Pré-registro de experimento WSP/WSR

Copie este arquivo para a pasta do experimento e preencha antes da seleção
de candidatos. Não preencha resultados do teste final antecipadamente.

## 1. Claim

- Objetivo do usuário:
- Base exata e caminho/revision:
- Modo: READ / DIAGNOSE / EDIT / SYNTHESIZE / MINIMIZE / MERGE:
- Comportamento alvo e direção desejada:
- Mecanismo hipotético falsificável:
- Non-claims (o que este experimento não demonstrará):
- Condição explícita para manter a base:

## 2. Recursos e autorização

- Host, RAM/VRAM/disco livres, working set e GPU ocupada por outro trabalho?:
- Escrita streaming/sharded cabe? Ground truth em alvo menor, offload medido ou
  compute/API externa autorizada?:
- Downloads autorizados e tamanho:
- Budget de tempo/API/tokens/forwards:
- O agente reader pode ou não rodar o target? Como isso é imposto?:
- Deploy não ocorrerá sem qual confirmação?:

## 3. Reprodutibilidade

- Config/tokenizer/chat template e hashes:
- Engine, commit, Transformers/vLLM/llama.cpp e backend:
- Dtype de cálculo/materialização e quantização final (ou N/A):
- Sampling: temperatura/top-p/top-k/penalties/seed/cap/timeout:
- Código do edit e commit:
- Dados e revisões; filtros de contaminação:
- Uso de dados, gradientes, forwards, checkpoints treinados/metamodelos:

## 4. Splits

| Split | Itens/entidades | Pode ver outputs? | Pode escolher hiperparâmetros? |
|---|---|---|---|
| Descoberta | | | |
| Seleção/validação adaptativa | | | |
| Teste final intocado | | | |

Se houver minimização, faça toda seleção em discovery/selection antes de abrir
o teste final. Se o original já consumiu esse teste, registre aqui um novo
teste externo independente para o minimizado:

- Unidade estatística e clusters (problema/parent edit/dose):
- Seeds e emparelhamento:
- Gerações por problema e braço (n=5 quando viável) e número separado de
  problemas independentes; justificativa/poder:

## 5. Braços e controles

- [ ] Base
- [ ] No-op serialização/conversão/quantização
- [ ] Candidato
- [ ] Random norm-matched, mesmas matrizes/update rule
- [ ] Random seeds/direção ortogonal/shuffled, se aplicável
- [ ] Prompting/decoding/thinking baseline
- [ ] Runtime steering vs compiled, se aplicável
- [ ] Adapter/fine-tune/merge baseline, se aplicável
- [ ] Doses zero/negativa/positivas
- [ ] WSR: text-only com mesma informação e floors same/change/random

Descreva cada braço:

## 6A. Campos adicionais para WSR

- Reader/editor: identidade, revision, provider, sampling e harness:
- Prompt, ferramentas, duração/nudges e número de sessões repetidas:
- Contenção de rede/forward, mounts, segredos ausentes e limites de recursos:
- Ground truth por comportamento, seeds da base e do editado:
- Threshold estatístico e efeito mínimo para cada label up/down/same:
- Regra de destrutividade e tratamento de respostas ausentes:
- Transformação de confiança em score/AUROC:
- Text-only com mesma informação de tokens/metadata; controles delta-only,
  wrong/shuffled/statistics-only/no-tokenizer quando aplicáveis:

## 6B. Operação

- Fórmula e orientação dos tensores:
- Tensors, shapes, índices, layers/components/experts:
- Direções/máscaras/projeções e normalização:
- Alpha/rank/density e norma relativa:
- Suporte BF16 exato, `sum(abs(DeltaW))`, `||DeltaW||_F` e relativos:
- Pós-GGUF, se aplicável: diferenças lógicas edit vs no-op dequantizados da
  mesma forja e expansão por scales/blocos; não é suporte armazenado:
- Visão/MTP/router/embeddings congelados ou alterados, e por quê:

## 7. Outcomes e gates

- Target e efeito mínimo:
- Generalização/paráfrases/locality/multi-hop:
- Guards e margens de não inferioridade:
- IF, fluência, repetição e safety:
- Tokens, cap hits, thinking closure e latência:
- Critério de destrutividade:
- IC, bootstrap/McNemar, cluster e multiplicidade:
- Critério de minimização e revalidação:

Não use p>.05 como prova de equivalência.

## 8. Artefatos e release

- Base imutável:
- Delta/edit recipe e manifest:
- Checkpoint no dtype/formato final e hash; BF16 se aplicável:
- GGUF/quant/calibração e hash, ou N/A com motivo:
- Logs brutos e script que gera os números:
- Reload/smoke do checkpoint editado, com dtype/formato:
- Reload/smoke do artefato servido:
- Rollback testado:
- Deploy autorizado? Quem/quando?:

## 9. Resultado

- Aceito / rejeitado / inconclusivo / base mantida:
- Resultados com IC e contagens melhores/piores:
- Regressões e resultados negativos:
- Screening versus validação adaptativa versus teste final:
- Efeito residual após quantização, ou N/A:
- Custo real:
- Limitações e próximos testes:
- Papers relacionados; oferecer PDF/link/browser ao usuário:
