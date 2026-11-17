# Métodos relacionados e validação rigorosa

Esta referência ajuda a escolher a ferramenta sem chamar toda modificação de
pesos de WSP. Links e contexto bibliográfico: `papers.md`.

## 1. Taxonomia operacional

| Família | Pré-condição | Força | Limite principal |
|---|---|---|---|
| Leitura estrutural: normas, deltas, SVD, espectro | Conhecer layout/orientação | Triagem CPU e hipóteses | Função invariante pode ter pesos diferentes; estatística não prova comportamento |
| ROME/r-ROME | Causal tracing e estatísticas no próprio modelo; valor-alvo otimizado | Update rank-one factual persistente | Não é editor geral de skill/raciocínio; sequential edits interferem |
| MEMIT/EMMET | Mesmas premissas, update distribuído em camadas | Batch de muitas associações | Batch grande não é lifelong learning nem preservação garantida |
| PMET | Otimiza hidden states MHSA/FFN, escreve via FFN | Pode melhorar valor impreciso de ROME/MEMIT | Não é inteiramente sem gradiente; ganhos específicos do setup |
| WilKE/AlphaEdit/REVIVE | Escolha de camada, null space ou subespaço preservado | Atacam interferência/instabilidade concreta | Preservação linear estimada não garante todo comportamento |
| SERAC/MeLLo/GRACE/WISE/adapters | Runtime de memória/adapter e roteamento | Rollback e updates frequentes | Overhead/complexidade; não internaliza necessariamente |
| Task arithmetic/soups/merging | Checkpoints compatíveis, idealmente mesma base | Sem novo gradiente, composição modular | Interferência, negative transfer e transfer de misalignment |
| Activation steering | Pares contrastivos e hook por camada/posição | Descoberta barata e reversível | Variância, anti-steering, hook permanente; prompting pode vencer |
| Steering compilado | Direção e writer locais; distribuição de inputs | Persistência sem hook | Aproximação dependente de contexto; evidência após quantização estreita |
| LoRA/SVD/low-rank | Delta/adapters existentes | Materialização, rollback, compressão e merge | Fatoração não cria conhecimento; PiSSA e variants ainda treinam |
| WSR/metanetworks/probes | Model zoo e encoder/probe treinado, ou WSR zero-shot com edit dado | Predizer/triar propriedades | Predição não implica causalidade nem fornece edit; specialist não é plug-and-play |

## 2. Knowledge editing

ROME tem update final algébrico, mas o valor que representa o novo objeto é
otimizado. Não descrever todo o método como closed-form ou gradient-free.
Causal tracing em MLPs intermediários de GPT é evidência local, não lei para
Qwen/Llama/MoE/multimodal.

Métricas mínimas:

- efficacy no prompt alvo;
- generalização em paráfrases;
- locality em vizinhos e fatos não relacionados;
- portabilidade/multi-hop e raciocínio com o fato;
- reappearance do conhecimento antigo em geração;
- fluência, repetição, downstream e retenção de edits anteriores.

Evidências negativas úteis: Model Editing at Scale encontra perda progressiva
de editabilidade/esquecimento; Butterfly Effect mostra colapsos em casos
difíceis, enquanto r-ROME identificou artefatos de implementação em parte
desses colapsos; MQuAKE mostra que acerto factual não garante multi-hop;
Superficial Editing mostra métricas altas coexistindo com conhecimento antigo.

EasyEdit (https://github.com/zjunlp/EasyEdit) é ponto de partida prático.
Pine o commit, hparams, estatísticas de preservação e revisões dos dados.

Em MoE, não atualize todos os experts por reflexo. Meça expert coupling e
routing distribution shift antes/depois; MoEEdit endereça esse problema.
Em multimodal, separe texto, visão, projector e transferência cross-modal;
MMEdit, VLKEB e MMKE-Bench são referências de avaliação.

## 3. Task arithmetic e merging

```text
tau_i = theta_i - theta_base
theta_new = theta_base + sum_i lambda_i * tau_i
```

Use por default apenas com mesma base exata, arquitetura, tokenizer e
parâmetros alinhados. Model soups servem melhor a runs da mesma tarefa e
inicialização, não a skills arbitrárias.

- Task Arithmetic: composição/negação de vectors de checkpoints treinados.
- TIES: trim, eleição de sinal e merge de valores compatíveis.
- DARE: drop aleatório e rescaling; 90–99% reportado em alguns SFT não é
  default e falha quando deltas grandes mudam o regime.
- AdaMerging: aprende coeficientes usando entropia/dados; não é data-free.
- KnOTS: SVD para alinhar LoRA updates; coseno/ortogonalidade isolados não
  preveem compatibilidade. Avalie a junta sem fornecer task ID.
- One Bad Model Spoils the Bunch: merging pode propagar misalignment; safety
  precisa de suíte própria.

Ferramenta: https://github.com/arcee-ai/mergekit . Preserve config, revisions,
base e cada constituinte; inclua best-single e linear merge como baselines.

## 4. Activation steering e compilação

ActAdd/CAA extraem diferenças de ativações. HPR lembra que updates aditivos
mudam norma. AxBench encontrou prompting acima dos métodos de steering
avaliados no seu setup; prompting e fine-tuning são baselines obrigatórias
quando práticas. Generalisation and Reliability of Steering Vectors documenta
variância, efeitos invertidos e brittleness OOD.

Para `y=Wx`:

```text
DeltaW = beta * d * u^T / ||u||^2
DeltaW x = beta * d * (u^T x / ||u||^2)
```

O hook equivalente é `y -> y + beta*d*(u^T x/||u||^2)`, na saída do writer.
O shift vale exatamente `beta*d` quando `u^T x = ||u||^2`, inclusive `x=u+z`
com `z` ortogonal a `u`. Em geral, ele não reproduz steering constante
`y -> y + beta*d`; esse caso requer outra construção. Teste compiled versus o
mesmo hook contextual, com hook removido no checkpoint final.

Weight Updates as Activation Shifts (https://arxiv.org/abs/2603.00425) discute
equivalência de primeira ordem e complementaridade, mas seus adapters são
treinados. Não citar como conversão universal sem treino.

## 5. LoRA e SVD: o que é realmente sem novo treino

Permitido descrever como sem novo treino:

- materializar `W' = W + sBA` de adapter existente;
- guardar um delta denso como `U_r Sigma_r V_r^T`;
- alinhar/combinar deltas existentes;
- analisar espectro/truncamento e validar o erro funcional.

Não descrever como sem treino:

- aprender adapter novo;
- PiSSA, que muda inicialização e depois faz fine-tuning;
- treinar metamodelo/probe WSR;
- AdaMerging/evolutionary search sem contabilizar dados/forwards/otimização.

LoRA possui simetrias próprias; coordenadas A/B equivalentes podem parecer
diferentes. LoRA vs Full Fine-tuning relaciona intruder dimensions a forgetting;
rank baixo e accuracy no target não provam equivalência funcional.

## 6. Leitura de pesos mais ampla

Predicting Neural Network Accuracy from Weights (2002.11448) já reportava
predição alta em model zoos de CNNs; DWSNets formalizou simetrias. O survey
Weight Space Learning (2603.10090) separa understanding, representation e
generation. Essa literatura geralmente treina outro modelo em model zoos e
usa redes pequenas/médias ou pesos parciais.

Não misturar:

- WSR zero-shot com `W0` e edit concreto;
- probe que usa respostas do modelo a inputs;
- metamodelo treinado em LoRAs/checkpoints;
- auditoria mecanística com ativações;
- previsão causal validada por intervenção real.

HuggingSANE, Learning on LoRAs e Weight Oracles exigem especialização/dados.
Não os apresente como leitor pronto para qualquer LLM local.

## 7. Checklist completo de experimento

### Claim e pré-registro

- [ ] Target, comportamento e mecanismo hipotético são falsificáveis.
- [ ] O claim distingue leitura, diagnóstico, reparo, síntese e minimização.
- [ ] Declarados dados, gradientes, forwards, checkpoints treinados e
      metatreinamento; “sem gradiente” não vira “sem dados”.
- [ ] Base revision/hash, tokenizer/template, engine, dtype, decoding, cap,
      quantização e hardware fixados.
- [ ] Efeito mínimo, margens de não inferioridade, guards, critério de parada
      e opção de reter base registrados antes do teste final.

### Splits e controles

- [ ] Descoberta, seleção e confirmação separadas por problema/entidade/tempo.
- [ ] Nenhuma camada, direção, alpha, rank ou densidade escolhida no teste.
- [ ] Base intacta e no-op de serialização/conversão/quantização.
- [ ] Random norm-matched nas mesmas matrizes, mesma update rule e norma
      relativa; vários seeds, orthogonal/shuffled quando aplicável.
- [ ] Todos os braços passam pela mesma forja/calibração.
- [ ] Prompting, decoding, runtime steering, adapter/fine-tune ou merge linear
      comparados conforme o objetivo.
- [ ] Em WSR: text-only com mesma informação, no-ops, edits sutis, inertes,
      always-same/change/random e sessões repetidas.
- [ ] Em leitura semântica: labels permutadas, datasets desconhecidos e
      wrong/shuffled/delta-only weights para detectar priors.

### Causalidade e arquitetura

- [ ] Clean/corrupt/restore ou ablation/restoration no próprio modelo.
- [ ] Camada, componente, posição, expert e writer isolados.
- [ ] Orientação e layout verificados por shape e forward de smoke.
- [ ] Normalizações que anulam scaling identificadas antes do banco de edits.
- [ ] Em MoE, routing e experts medidos; em multimodal, modalidades/projector
      testados; vision/MTP congelados por default.
- [ ] Runtime steering comparado ao checkpoint compilado sem hooks.
- [ ] Norma relativa, espectro, condition number e magnitudes L1/Frobenius do
      delta reportados como diagnósticos, não outcomes.

### Comportamento e estatística

- [ ] Doses zero, negativa se interpretável e várias positivas; monotonicidade
      não presumida.
- [ ] Target, paráfrases/generalização, locality, portabilidade/multi-hop,
      downstream, IF, fluência, repetição e safety medidos.
- [ ] Tokens, truncamentos, thinking closure, cap hits e latência junto de
      accuracy; resposta incompleta não conta como eficiência.
- [ ] Prompts e seeds emparelhados; mesma régua para todos os modelos/editores,
      com timeout maior de thinkers anotado.
- [ ] Na casa, quando viável, n=5 gerações por problema e braço; registre
      separadamente o número de problemas independentes, média, mínimo
      observado (piso) e fail_rate. Use CV só em métricas positivas de escala
      proporcional com média não próxima de zero.
- [ ] McNemar com uma observação pareada por problema ou bootstrap clusterizado
      por problema; cluster por parent edit/dose; não inflar N com amostras
      correlacionadas.
- [ ] ICs, contagens melhores/piores e correção de multiplicidade reportados.
- [ ] p>.05 não é equivalência; usar margem de não inferioridade pré-fixada.
- [ ] Em WSR, BA + recall + specificity + response rates/AUROC; nunca apenas
      raw accuracy ou recall.

### Minimização e composição

- [ ] Sparsification/drop-one e seleção de subsets ficam em discovery/selection;
      original, minimizado e controles são congelados antes do teste final.
      Se o original já consumiu o teste, usar novo teste externo independente.
- [ ] Suporte BF16 exato (`W_edit != W_base`), `sum(abs(DeltaW))`,
      `||DeltaW||_F` e versões relativas registrados separadamente.
- [ ] Se houver GGUF, dequantizar edit e no-op da mesma forja e contar
      diferenças lógicas, relatando expansão por scales/blocos; não chamar
      isso de suporte armazenado ou esparsidade real.
- [ ] Edits sequenciais testados para retenção/forgetting, não só último alvo.
- [ ] “Mínimo” significa ponto fixo do algoritmo/candidato, não mínimo global.

### Persistência e release

- [ ] Base nunca sobrescrita; delta/recipe e manifest preservados.
- [ ] Reload, shapes/dtypes, hashes e smoke generation do BF16 editado.
- [ ] Se o destino for GGUF: converter/quantizar, fazer reload/smoke e rodar a
      bateria no artefato servido; nos demais runtimes, materializar no formato
      apropriado, testar esse artefato e marcar GGUF/quantização como N/A.
- [ ] Efeito residual após conversão/quantização medido quando aplicável;
      não extrapolar Q8 para Q4.
- [ ] Rollback real ensaiado; deploy/lane/restart só com autorização.
- [ ] Resultados negativos, contaminação e falhas de harness arquivados.
- [ ] Números gerados programaticamente de brutos; caminhos e hashes entregues.

## 8. Manifest mínimo

```text
claim e non-claims
base: repo/path, revision, sha256, config/tokenizer/template
runtime: engine+commit, hardware, RAM/working set/disco, dtype, decoding,
      cap, timeout; writer streaming/sharded e estratégia de ground truth
cost: dados, gradientes, forwards, chamadas, tempo, tokens, API
split: discovery/selection/final, filtros de contaminação
edit: código+commit, ops, tensors, shapes, índices, direções, masks,
      alpha/rank/density, seeds, delta e hashes
controls: no-op/random/shuffled/text-only/runtime/baselines
metrics: target+guards, unidade estatística, IC, multiplicidade
artifacts: BF16, GGUF/quant, calibração, logs, resultados brutos
release: validações, deploy autorizado?, rollback e owner
```
