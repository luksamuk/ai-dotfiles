# Evidência central: WSP, WSR, steering e minimização

Corte da pesquisa: 17/nov/2026. Esta referência separa resultados reportados
de validação local. As URLs completas e mais literatura estão em `papers.md`.
Nenhum número abaixo foi produzido por esta casa, exceto a reanálise local dos
artefatos do alphaXiv explicitamente marcada.

## 1. Sparks of Weight-Space Programming (MIT CSAIL, 02/nov/2026)

Fonte primária: https://rikiyat.github.io/blog/weight-space-programming/
Código/logs: https://github.com/RikiyaT/lm_understand-logs

O termo é um guarda-chuva: um LLM geral lê/escreve pesos por raciocínio e
ferramentas. Não há um algoritmo WSP único nem um encoder de pesos especial.
O editor do post é GPT-6 Astra via Codex CLI, com harness externo de Python,
forwards, jogos ou atenções conforme o experimento.

### Leitura em redes pequenas

- MLPs de quatro jogos, 17.253 pesos em texto, sem ferramentas: 4/4 jogos
  nomeados. Os canais espaciais e ações eram descritos; primeira/última camada
  já oferecem pistas fortes.
- DDPMs pequenos de círculo/quadrado/triângulo: 3/3 formas; Astra desenhou
  contornos em vez de formas preenchidas, evidência de leitura incompleta.
- Char Transformers de 7,16M, apenas quatro blocos intermediários, Python sem
  forward: três línguas corretas em duas tentativas e copying heads corretas,
  sem falsos positivos. As confianças de língua ficaram em torno de 0,48–0,62.

Esses resultados não mostram leitura arbitrária de checkpoints nem descarta a
influência de priors. Ahmed encontrou aritmética local real, mas classes
lembradas erradas após permutação de labels e confusão de SAT-6 com CIFAR-10:
https://www.alphaxiv.org/abs/2611.08070 . É outro setup, não refutação direta.

### Escrita/reparo

- Carro Kinetix: base 0/256; edit 1.024/1.024, em uma única tentativa sem
  ferramentas, com pesos, descrição da cena e um episódio; alterou 6.073
  entradas, aproximadamente 3,5% dos pesos. Um nível e uma política.
- MinAtar do zero: supera PPO-100M nos três jogos; contra PPO-500M, vence
  Breakout, fica próximo em Asterix e atinge cerca de 60% em Seaquest. Não
  resumir como “supera PPO” sem especificar o baseline.
- Go 9x9 do zero: sem forwards vence PPOs menores; com 50 forwards supera o
  PPO de mesmo tamanho reportado. É programa heurístico compilado, com seleção
  iterativa quando há feedback.
- Sycophancy em Gemma 3 270M: pesos sozinhos falharam em quatro tentativas;
  com atenções em textos escolhidos, 3/3 pares. Sucesso veio de ativações, não
  leitura estática pura.

### Qwen3.5-0.8B: remoção de looping

Com 50 forward-tool requests, prática em GSM8K/MATH train e thinking on:

| Métrica | Base | Edit |
|---|---:|---:|
| MATH-500 | 7,9% | 48,1% |
| GSM8K | 13,8% | 67,1% |
| IFEval | 21,0% | 62,3% |
| Respostas no cap 32k | 93,0% | 33,1% |

O edit subtrai uma direção de repetição por 64 neurônios de MLP nas camadas 15
e 19: nominalmente 131.072 valores, 130.188 alterados após BF16. O run levou
20,3h, 158 turns, 80 execuções Python e quatro handoffs. O prompt mencionava
looping como possibilidade e o model card já alertava para o defeito.

No follow-up Gemma 3 270M, que raramente loopava, MATH-500 ficou 6,8 na base e
7,0/7,6/7,7/7,0 nos orçamentos testados: nenhum ganho estabelecido. O limite
prático é central: o edit pode remover uma obstrução e revelar capacidade
existente, não instalar capacidade matemática ausente.

## 2. Reinterpretação UCSD: steering contextual compilado

Post: https://zhi0467.github.io/posts/2026/post6/
Código: https://github.com/Zhi0467/astra-weight-edit-as-steering

A forma rank-one do edit é equivalente a activation steering dependente de
contexto: uma direção determina para onde empurrar; um sensor linear decide a
intensidade. Isso não invalida o reparo nem a utilidade do agente; muda a
interpretação de “native weight editing”.

Em 1.152 problemas de prática GSM8K/MATH:

| Configuração | Acurácia | Loop | Cap hit |
|---|---:|---:|---:|
| Base, thinking on | 12,3% | 87,0% | 88,5% |
| Base, thinking off | 53,3% | 4,5% | 5,3% |
| Steering Astra, thinking on | 57,7% | 29,5% | 23,7% |

A baseline patológica exige comparar thinking off e decoding alternativo.
Com 32 pares, treinar apenas o sensing vector por Unlikelihood reduziu loop de
27,2% para 6,5%; Corrective NTP chegou a 10,9%. Essa variante usa Adam e não é
mais gradient-free. Pares sintéticos curtos funcionaram melhor que rollouts
naturais longos no setup relatado; mudança de temperatura exige recalibração.

## 3. WSP-Scope: generalização e variância

Paper: https://arxiv.org/abs/2611.09382
Código: https://github.com/WSP-Scope/WSP-Scope-Project

16 trajetórias Astra, Qwen3.5-0.8B/9B, sete benchmarks e até 50 requests:

- 0.8B MATH: +29,45pp com 1.023 valores em uma trajetória; outras duas deram
  +5,00 e +27,30. As trajetórias variam substancialmente; contagem isolada não
  ordena nem explica o efeito.
- Seis de 16 trajetórias mantiveram a base por falta de validação confiável.
- 9B Math: +29,69pp em dev, mas -0,35pp no held-out do candidato avaliado.
- Math edit transferiu para GPQA +3,03, IFEval +12,18 e SciBench +20,10; não
  melhorou SciCode. Transferência existe em alguns casos, não é garantida.

## 4. Concept Country: activation-to-weight, não replicação da suíte

Paper: https://concept.country/research/weight-space-programming
Código/agregados: https://github.com/conceptcountry/weight-space-programming
Pesos: https://huggingface.co/ConceptCountry/qwen-concept-27b

Receita determinística em Qwen3.8-27B, sem um LLM escolhendo a pesquisa:
48 problemas MATH level-5 x 6 amostras, 288 traces, ativações residuais e
contrastes `sl`, `vf`, `cc`. Triagem de 64 problemas frescos escolheu `cc`,
alpha=0,1, update aditivo e down-projections das camadas pares 32..48.

```text
W' = W + alpha * n_w * d * u^T / ||u||^2
```

Nessa receita, `d` é a direção residual de contraste `cc`, `u` é o input médio
dos tokens de thinking no writer e `n_w` é a norma média da saída. Preserve a
normalização exata de cada vetor no manifest; não a copie para outro modelo.
Perturbação relativa aproximada: 0,5% por matriz. Controle random `r` perp a
`d`, mesmas matrizes e norma, com uma direção random por camada. Confirmação
com prompts/seeds emparelhados e bootstrap 10k:

| Benchmark | Base -> edit | Controle vs base | Delta tokens: edit-base / controle-base |
|---|---|---|---|
| GPQA Diamond | 81,8 -> 86,1; +4,3 [+1,8,+7,1] | -1,0 [-3,3,+1,3] | -30,0% vs +0,2% |
| LiveCodeBench | 58,0 -> 71,0; +13,0 [+6,1,+20,6] | +3,0 [-3,8,+9,9] | -29,2% vs -2,6% |
| IFBench loose | 79,3 -> 82,7; +3,3 [-1,0,+7,7] | -1,3 [-6,0,+3,3] | -54,1% vs -0,1% |

GPQA e LCB sustentam o efeito nesse protocolo; IFBench é borderline contra o
controle. Dose-response mostrado apenas entre alpha .05 e .1 na triagem.
Todos os braços foram avaliados em Q8_0, embora o edit seja construído em
BF16; isso não valida BF16 separadamente, Q4 ou outros quants.

Limites de artefato: o repositório público tem agregados e scripts, mas a
auditoria de 17/nov não encontrou todos os brutos, código end-to-end de
descoberta e os `r2_*.safetensors` nomeados pelo `EDIT_SPEC.json`. Pesos finais
continuam avaliáveis; não chamar o checkout de reprodução completa.

## 5. Weight-Space Reading (Kocabay/Poindexter, 13/nov/2026)

Paper/PDF: https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models
Código: https://github.com/suayptalha/wsr
Commit auditado pelo subagente: b5251bfa6342bf9a3eb43481fffeb683173490c5.
Preprint alphaXiv Exclusive; README ainda marcava arXiv pending moderation.

### O que é medido

Reader CPU recebe base Safetensors e tensores pós-edit; pode calcular
`W_edit-W0`, estatísticas, SVD etc. Não escolhe o edit. Prevê cinco agregados
up/down/same: thinking length, MATH, MMLU-Pro, sycophancy e refusal. Ground
truth usa 288 probes por checkpoint, cinco seeds de base, Welch p<.01 e efeito
mínimo de 10% tokens ou .05 em taxas. O target precisa ser executado.

Modelos: Qwen3.8-27B, Qwen3.5-9B e Muse-Glimmer-30B. Full-Dose é majoritariamente
destrutivo; Subtle mantém geração. Cada célula reader-target-condition rodou
uma vez. Há 10/15 intervenções changed sutis e 18/22 labels changed nos dois
Qwen, com doses relacionadas e pools não totalmente matched.

### Resultado principal e resposta bias

BA / change recall / specificity no regime Subtle. Aqui change recall exige a
**direção correta**, e `BA = (change recall direcional + unchanged
specificity)/2`:

| Reader -> target | Weights+Tools | Text-only | Delta BA [IC95] |
|---|---|---|---|
| 27B -> 27B | .730/.667/.793 | .578/.278/.878 | +.152 [.056,.256] |
| 9B -> 27B | .541/.278/.805 | .451/.000/.902 | +.090 [-.049,.206] |
| 9B -> 9B | .554/.409/.699 | .492/.091/.893 | +.062 [-.063,.175] |
| 27B -> 9B | .628/.364/.893 | .648/.364/.932 | -.019 [-.109,.081] |

Somente o primeiro delta tem IC acima de zero. Recall sobe nas três primeiras
células e permanece .364 na quarta; specificity cai em todas. Assim, pesos
aumentam falsos alarmes em todas, mas nem sempre aumentam acertos direcionais.
Sempre-same tem BA .5; random uniforme fica em torno de .33 nessa definição.
Não chamar .730 de raw accuracy nem de capacidade geral.

### Self-reading e arquitetura

Contraste cruzado de self-reading: Full-Dose +.085 [-.078,.243], Subtle +.114
[-.093,.315]. Não confirmado. Framing “estes são seus pesos” não mostrou efeito
detectável. Em edits destrutivos no 27B, Muse externo empata com self:
.594 vs .593, sugerindo damage detection.

No Muse, 34 edits padrão e seis no-ops foram inertes. Readers previram mudança
em 85% de 40 (Muse self), 88% de 40 (9B) e 53% dos 30 avaliados pelo 27B em
três batches. Pós-RMSNorm explica apenas inércia de scaling positivo uniforme,
não todas as outras operações. No banco norm/gate,
27B externo supera Muse self em BA .615 vs .468, mas também o supera no
text-only; reader skill/tool use é explicação mais plausível que self-access.

### Quatro regras do paper

1. Edits destrutivos medem damage detection.
2. Raw accuracy e recall isolados são exploráveis; usar BA ou threshold-free.
3. Descrição entrega detecção de mudança; pesos precisam acrescentar direção.
4. Validar o banco na arquitetura; uma receita pode ser inerte em outra.

### Limitações e auditoria do harness

- Não há delta-only, wrong/shuffled-weights, statistics-only, no-tokenizer ou
  ablação separada de ferramentas. Text-only recebe IDs sem tokenizer enquanto
  weights+tools pode decodificá-los.
- Sessões únicas, contrasts múltiplos sem correção e bootstrap sem cluster por
  parent tornam intervalos otimistas. Probes repetidos entre seeds não devem
  ser tratados como independentes.
- AUROC de detectar mudança: mediana .76 com pesos vs .73 text-only; boa parte
  de “se muda” já vem da descrição.
- Custo reportado: 475 avaliações, 136.800 gerações, quase 500M tokens em
  GH200, fora readers/judge. Leitura barata não torna ground truth barato.
- O repositório não publica os resultados brutos/logs do paper nem oito edits
  do piloto; testes de equivalência usam modelos sintéticos e não replicam as
  tabelas. Não havia CI verde útil no snapshot auditado.
- O código de labels usa referência normal via `erfc`, não o quantil t com
  Welch–Satterthwaite. Não reproduzir esse atalho sem declarar.
- Prompt/sandbox não garantem universalmente ausência de rede/forward: o modo
  OpenCode/CLI precisa endpoint e config de contenção; firewall e logs reais
  são necessários. Isso não prova violação, apenas limita a garantia.
- `wsr write/program` é funcionalidade do toolkit, não evidência confirmatória.
  O “held-out” de aceitação é reusado em tentativas/rodadas; é validação
  adaptativa. Adicione teste final externo intocado.

O piloto de self-edit reduziu thinking 39% ao custo de 1,5pp MATH e 2,5pp GPQA;
não é confirmatório e os oito edits não foram distribuídos. Propor e prever
são capacidades separadas.

## 6. Minimização Inkling/alphaXiv (17/nov/2026)

Post enviado pelo usuário:
https://x.com/askalphaxiv/status/2122744324354097172
Notebook/código/artefatos:
https://github.com/alphaXiv/weight-edit-minimization
Log OpenResearch: https://openresearch.sh/p/EdsyLoFUfY40
Commit reanalisado localmente: ac65cc5e72c102fdb83f6437b2f163a8f99c96ae.

### Car-ramp

O edit Astra de 6.073 entradas foi decomposto em erasures de isolamento e
functional writes. Backward elimination chegou a 12/11 entradas do edit; um
design novo chegou a 10. O relatório registra 1.024/1.024 em quatro sampled
seeds + argmax para o dez-pesos. Contagens são contra base arredondada a três
dígitos significativos. É evidência forte para aquele MLP/nível, não mínimo
global nem transferência para LLM.

### Qwen D4k: o que a reanálise local confirmou

Os JSONL brutos foram recomputados localmente, sem rodar o modelo:
256 MATH-train, n=1, temp .6, cap 32k.

| Candidato | Entradas | Acurácia | Cap hit |
|---|---:|---:|---:|
| Base G000 | 0 | 23,83% | 91,4% |
| Full A64 | 130.188 | 58,59% | 30,1% |
| D4k top-|Delta| por camada | 8.192 | 58,20% | 38,7% |
| D8k | 16.384 | 58,98% | 35,2% |
| Layer 15 only full | 65.260 | 56,25% | 39,5% |
| Layer 19 only full | 64.928 | 43,36% | 63,7% |

D4k-full = -0,39pp; 20/21 discordantes; McNemar p=1 no pareamento. Isso é
ausência de evidência de diferença nesse screen, não equivalência ou não
inferioridade. O cap hit foi numericamente pior no D4k (38,7% vs 30,1%).
D1k/D2k ficaram 55,47/54,30: ruído e não monotonicidade impedem anunciar cliff
limpo. A seleção por entradas top-|Delta| foi melhor que colunas inteiras no
screen.

Held-out parcial D4k: 500 MATH-500 x 4, 1.084/2.000 = 54,20%, recomputado
localmente. O notebook diz que o braço A64 pareado, GSM8K e IFEval não
terminaram. Portanto 54,2 vs 48,1 publicado não é comparação controlada;
equivalência/não inferioridade e guards não foram estabelecidos. O post só
mostra acurácia numericamente próxima no screening de treino para Qwen.

A análise de discordância mostra que todos os 93 rescues full-vs-base vieram
de problemas em que a base atingiu o cap; zero finished-wrong -> correct. O
screen isola terminação/looping, não correção de raciocínio terminado.

## 7. iWSP, WeightBench e arquitetura esparsa

### iWSP

Paper: https://arxiv.org/abs/2611.18114
Código: https://github.com/csinva/weight-space-programming-autoresearch

Agentes escrevem modelos pequenos e interpretáveis em código, com 30–50 ou
centenas de avaliações visíveis. Em 13 tarefas algorítmicas, Opus obteve 26/26
runs com 100% visível/oculto; NLP foi acima do majority em 3/4, MRPC falhou.
Descrições de até 200 palavras permitiram rebuild dos 13 algorítmicos a 100%.
Isso mede compressibilidade/legibilidade operacional, não compreensão humana
completa. Houve violações detectadas pela auditoria; algumas regras dependem
de prompt+auditoria, não contenção física.

Em brain encoding, o evaluator ajusta ridge regression: sem backprop nos
features escritos não significa pipeline inteiro sem treino. Modelos black-box
continuam superiores em vários domínios; o valor principal é explicabilidade e
diagnóstico de confounds. Custo reportado total aproximado: US$17.500 em API.

### WeightBench

Paper: https://arxiv.org/abs/2611.18553
Código: https://github.com/Aayush-Marasini/WeightBench

42 tarefas em reading, fixing, writing, edits impossíveis e compressão sem
dados, redes de 2K a 7B. Frontier agents leem redes pequenas e escrevem
políticas, mas ainda falham em patch de LMs, compressão sem dados e detecção
de impossibilidade. Macro scores: Opus 5.5 .535; Sonnet .466; Haiku .402;
DeepSeek v4-flash .215; GPT-6.1-sol .568 em 38 tarefas vs Opus .546 nessas 38.
É evidência de fronteira/abstenção, não de que toda edição LM falha sempre.

### WSP Doesn't Need Transformers

Paper: https://www.alphaxiv.org/abs/2611.weight-space-programming-associative-memory
Código: https://github.com/effl-lab/Weight-Space-Programming

Quando três writers podem escolher a arquitetura, todos compilam fatos em
tabelas esparsas, até 54,4% ARC. Um modelo de tabela supera o melhor Transformer
escrito pelo mesmo LLM com cerca de 1.800x menos parâmetros; pretrained
fine-tuned chega a 82,1%, muito acima. A comparação é entre modelos escritos
do zero, não contra os pesos pretrained usados normalmente. Falha em pixels
crus sem features e não demonstra multi-step reasoning geral. Serve como
lembrete: em síntese do zero, escolha uma representação nomeável e editável.

## 8. Frase-síntese permitida

“A evidência atual sustenta leitura estrutural parcial, reparos localizados de
falhas concentradas e síntese de modelos pequenos/legíveis. Não sustenta uma
receita universal de edição, previsão confiável sem ground truth, vantagem de
self-reading ou criação geral de capacidades ausentes. O protocolo de controles,
splits, causalidade, guards e avaliação do artefato final é a parte portável.”
