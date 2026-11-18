---
name: weight-space-programming
description: "Use ao ler/editar pesos; exige controles e validação."
version: 2.1.0
author: Hermes Agent
license: CC0
metadata:
  hermes:
    tags: [interpretability, weight-editing, weight-reading, steering]
    related_skills: [calib-forja, llama-swap-fleet]
---

# Weight-Space Programming e Reading

## When to Use

Use para ler, diagnosticar, escrever, editar, combinar ou minimizar pesos de
uma rede; prever o efeito de um edit sem rodar o alvo; compilar steering em
pesos; ou avaliar um artefato desse tipo. Serve para Safetensors/BF16, GGUF e
modelos densos, MoE, híbridos e multimodais, mas o protocolo é portável, não a
receita numérica.

Não use WSP como sinônimo de fine-tuning. ROME/MEMIT, task arithmetic, LoRA e
metamodelos entram como métodos relacionados, com suas próprias premissas.

## Definições que não podem ser misturadas

- **WSR / reading:** dado `W0` e um edit concreto `DeltaW`, prever se
  comportamentos sobem, descem ou ficam iguais. O reader pode ficar sem GPU e
  sem forward do alvo; o ground truth ainda exige rodar o alvo editado.
- **Diagnóstico:** formular hipóteses sobre mecanismo usando pesos, traces,
  logprobs e/ou ativações. Uma explicação plausível continua sendo hipótese.
- **WSP / writing/editing:** um agente escolhe ou escreve os pesos para um
  objetivo, sem presumir uma update rule única. Pode haver dados, ferramentas,
  forward passes e seleção iterativa; declare todos.
- **Síntese do zero:** escrever uma rede nova, inclusive escolhendo sua
  arquitetura. Não é evidência de que seja possível reparar um LLM pretrained.
- **Minimização:** remover ou esparsificar partes de um edit já descoberto.
  Menos entradas não significa menor norma, maior generalização ou mínimo
  global.

“Sem gradiente” descreve apenas uma restrição de otimização. Não diga que é
zero-data, zero-feedback, barato ou sem treino em todo o pipeline sem provar.

## Escolha do editor: não confunda com o alvo

Editor, target e evaluator são papéis separados. O editor pode ser local ou
remoto, maior ou menor que o target; self-reading não é default nem vantagem
comprovada. Não escolha por parâmetros, coding benchmark ou reputação isolados:
escrever tooling não prova leitura causal de pesos.

Antes de confiar num editor, faça uma escada mínima:

1. **Tooling read-only:** shapes, nomes, índices, código seguro e resultados
   reproduzíveis, sem inventar outputs.
2. **Controles de leitura:** text-only, no-ops, edits impossíveis e inertes;
   meça BA, recall direcional, specificity e falsos alarmes.
3. **Reparo conhecido em target pequeno:** base, random norm-matched, no-op e
   guards, sem deixar o editor aprovar o próprio edit.
4. **Confirmação no target pretendido:** sessões repetidas e teste final
   externo. Só então promova o editor para essa classe de tarefa.

O evaluator externo, com gates pré-registrados, é dono da aceitação. O editor
propõe hipóteses e código, nunca concede o próprio deploy. Para editor e target
grandes na mesma máquina, planeje avaliação sequencial ou externa; evitar
evictions/thrashing por alternância de lanes a cada iteração.

## Fontes e leitura para o usuário

A bibliografia anotada está em `references/papers.md`; resultados e limites
dos trabalhos centrais, em `references/evidence-wsp-wsr.md`; métodos clássicos
e detalhes de validação, em `references/methods-and-validation.md`. Para um
guia curto de leitura e minimização, use `references/reading-and-validation.md`.
Resolva esses caminhos relativos a este `SKILL.md`: no Hermes, use
`skill_view(name="weight-space-programming", file_path=...)`; nos demais
harnesses, use a ferramenta de leitura equivalente. Não presuma que uma skill
relacionada esteja instalada.

Quando o usuário demonstrar interesse num trabalho, ofereça o paper/PDF ou
abra seu link primário no browser mediante confirmação. Ex.: “Quer o PDF ou
que eu abra o paper no browser?”. No TUI, entregue a URL em texto puro.
Priorize paper, repositório e artefatos; X e resumos automáticos são pistas,
não evidência final. Diga se é preprint, blog, working paper ou revisado.

## Regras inegociáveis

1. **Base imutável.** Nunca sobrescreva o único checkpoint. Trabalhe numa nova
   árvore, preserve delta/recipe e valide reload antes de qualquer deploy.
2. **Escopo e recurso antes de agir.** Confirme target, objetivo, orçamento e
   se o usuário autoriza downloads, GPU, gravação e deploy. Não carregue uma
   lane paralela na GPU ocupada; leitura estrutural costuma bastar em CPU.
3. **Nenhuma receita universal.** Não copie camadas, tokens, direções, alpha,
   rank, densidade ou tensores de outro checkpoint sem redescoberta e teste.
4. **Predição não é validação.** Nunca aplique/deploye um edit apenas porque
   um reader previu o efeito ou porque a narrativa mecanística é convincente.
5. **Manter o original é um resultado válido.** Se nenhum candidato passar nos
   gates, entregue a base e os resultados negativos; não force um edit.
6. **Artefato real ou nada.** Se o destino for quantizado/GGUF, edite o master
   BF16 antes de quantizar e avalie o arquivo efetivamente servido. Em outro
   runtime, materialize no dtype/formato apropriado e teste esse artefato.
   Um edit que desaparece na conversão não existe no produto final.

## Roteamento: escolha o método pelo contrato

| Objetivo | Primeira rota | Não faça |
|---|---|---|
| Prever efeito de edits já especificados | WSR com weights vs text-only, no-ops e ground truth | Chamar previsão de compreensão geral do checkpoint |
| Entender falha difusa, como raciocínio errado | Diagnóstico comportamental + ativações; considerar fine-tune | Supor que exista um único neurônio/direção culpável |
| Remover loop/overthinking ou outro defeito concentrado | Baseline de decoding/thinking, steering runtime, depois compilar em pesos | Chamar terminação de loop de nova capacidade matemática |
| Corrigir associação factual | ROME/MEMIT/AlphaEdit/EasyEdit, com paráfrases e locality | Usar WSP genérico se um editor validado já resolve |
| Combinar checkpoints da mesma base | Task arithmetic/TIES/DARE/mergekit | Subtrair modelos de bases/tokenizers diferentes |
| Atualização frequente ou rollback imediato | Adapter/memória externa/GRACE/SERAC | Internalizar à força no checkpoint |
| Já existe LoRA/delta | Materializar, alinhar ou fatorar o delta existente | Dizer que SVD/LoRA criou conhecimento sem treino |
| Criar política para jogo/controle | WSP do zero ou reparo localizado + simulador externo | Supor que um scorer pretrained aprende a estratégia por edição |
| Criar modelo pequeno e legível do zero | Closed-form WSP/iWSP; considerar arquitetura esparsa | Extrapolar para substituir LLMs pretrained |
| Só há GGUF quantizado | Localizar fonte BF16; editar e requantizar | Tratar dequantização como recuperação da precisão |

Detalhes e referências dessas rotas: `references/methods-and-validation.md`.

### Síntese de políticas para jogos e controle

Este é um caso de síntese ou reparo, não substituição automática de fine-tune.
Use-o quando as regras/observações permitem especificar uma política pequena ou
quando uma política existente tem falha concentrada. Não assuma que um scorer
pretrained aprenderá uma estratégia espacial difusa só por weight editing.

- Fixe primeiro o contrato `observação -> ações`, se o estado é completo ou se
  há histórico, e a arquitetura mínima da política.
- Use simulador headless como evaluator externo, com seeds/episódios held-out.
  O editor pode receber métricas e traces sem controlar o aceite.
- Inclua política aleatória, algoritmo conhecido/oracle e, quando pertinentes,
  RL/imitação ou busca evolutiva. WSP precisa vencer em qualidade ou custo, não
  apenas ser mais sofisticado.
- Declare o que o harness calcula: features, caminhos, candidatos e safety.
  Se um algoritmo externo escolhe a ação e a rede apenas a copia, a inteligência
  não está nos pesos.
- Separe **autonomia** de **vitória**. Depois de compilada, a política roda sem
  o editor no loop; isso não prova win rate, generalização ou ausência de loops.
- Meça score, colisões, timeouts/loops, truncamento, latência e transferência a
  seeds/mapas/tamanhos não usados na descoberta. Materialize no runtime real;
  GGUF é N/A quando a política usa outro formato.

## Fluxo operacional

### 0. Escreva o claim e o pré-registro

Antes de editar, preencha `references/experiment-template.md`:

- comportamento alvo, mecanismo hipotético e o que o edit NÃO pretende fazer;
- base revision/hash, tokenizer/chat template, engine, dtype, decoding, cap e
  quantização final;
- uso de dados, gradientes, forwards, checkpoints treinados e metamodelos;
- splits de descoberta, seleção e teste final intocado;
- efeito mínimo, margens de não inferioridade, guards e critério de rejeição;
- orçamento de RAM/VRAM/tempo/API e condição de rollback.

Faça primeiro baselines triviais: zero edit, no-op de serialização, prompting,
decoding/thinking alternativo, steering runtime ou merge linear quando fizer
sentido. Se configuração resolve, não venda weight editing como necessário.

### 1. Inspecione sem carregar o modelo inteiro

Leia `config.json`, índice de shards, shapes, dtypes e layout. Use
`safetensors.safe_open(..., framework="pt", device="cpu")`, `keys()`,
`get_slice()` e lazy loading; carregue só os tensores necessários.

Confirme:

- orientação `nn.Linear` versus Conv1D e onde cada writer atua;
- pre/post-norm, QK-norm, residual streams e ganhos armazenados como `1+w`;
- tied embeddings, lm_head e tokens especiais;
- experts separados/fused, shared expert e router em MoE;
- blocos híbridos de full/linear attention;
- vision tower, projector, mmproj e MTP, congelados por default.

Um 27B BF16 ocupa cerca de 54 GB decimais antes de saída, delta, GGUF e logs;
não cabe nos 31 GB da casa e seu ground truth não é default na RTX 3050 6GB.
Meça RAM, working set e disco livre; processe shard/tensor por vez, acumule em
FP32 quando couber e prove que o writer streaming/sharded cabe antes de editar.
Lazy reading não garante lazy writing. Escolha explicitamente alvo menor,
offload medido ou compute/API externa autorizada para o ground truth.

### 2. READ: teste se os pesos acrescentam informação

Dê ao reader `W0`, os tensores editados/recuperáveis e ferramentas CPU. Se a
pergunta é leitura pura, impeça tecnicamente forward do alvo e rede externa;
não confie só no prompt. Use container/VM sem credenciais, mounts read-only
só dos artefatos necessários, saída isolada, limites de RAM/CPU/processos/tempo
e logs de todo código/resposta. Valide shapes/tamanhos antes de alocar;
proíba pickle/checkpoints executáveis não confiáveis e revise código de
repositórios antes de executá-lo.

Use no mínimo:

- **Weights+Tools vs Text-Only**, com a mesma informação de tokens e metadata;
- edits sutis, não apenas destrutivos;
- no-ops, edits inertes por arquitetura e controles de familiaridade/labels;
- sempre-same, sempre-change e random como floors;
- leitores self e outros, sem presumir vantagem de self;
- sessões repetidas para estimar variância do agente.

Para cada comportamento, registre `up/down/same`, confiança e ground truth.
Reporte balanced accuracy **junto com** change recall, specificity, taxas de
resposta e, se houver scores contínuos, AUROC. Raw accuracy recompensa dizer
“same”; recall isolado recompensa dizer “changed”.

Quatro filtros do paper WSR:

1. Edits destrutivos medem damage detection, não leitura fina.
2. Base rates podem ser exploradas nos dois sentidos.
3. A descrição do edit já entrega muita coisa; o valor dos pesos é o delta
   sobre text-only, sobretudo a direção do efeito.
4. O banco de edits precisa ser validado em cada arquitetura.

Não apresente 0,730 como taxa geral: é a melhor célula de um preprint, com
pool pequeno, sessões únicas e intervalos otimistas. Leia as limitações em
`references/evidence-wsp-wsr.md` antes de citar.

### 3. DIAGNOSE: separe falha de formato, obstrução e capacidade

Classifique as respostas base antes de localizar pesos:

- correta e terminada; errada e terminada; correta apesar de truncada;
- loop/repetição; thinking sem fechamento; cap hit; recusa/formato inválido.

Se todo ganho vier de cap-hit -> resposta terminada, o claim é remoção de
obstrução. Não afirme que o modelo aprendeu matemática, código ou conhecimento.

Construa contrastes curtos e pareados que mudem só a propriedade de interesse.
Traces naturais longos misturam conteúdo, dificuldade e posição. Capture a
posição de token e a residual stream e o writer corretos no modelo alvo.

Pesos sozinhos podem bastar para pistas estruturais, mas falharam em casos
como sycophancy; use ativações, logprobs e ablações quando necessário. Confirme
causalidade com clean/corrupt/restore, isolamento de camada/componente e
dose-resposta; correlação, norma, SVD e coseno são diagnósticos, não outcomes.

### 4. EDIT: torne a intervenção explícita e mínima

Salve para cada candidato: operação, tensors, índices/máscaras, direções,
alpha/rank, unidade de normalização, seeds, código e hash. Calcule updates em
FP32 e materialize no dtype combinado; conte entradas realmente alteradas após
o rounding.

Para writer linear `y = W x`, uma compilação rank-one típica é:

```text
DeltaW = beta * d * u^T / ||u||^2
DeltaY = beta * d * (u^T x / ||u||^2)
```

O hook equivalente é `y -> y + beta*d*(u^T x/||u||^2)`, na saída desse writer.
O shift vale exatamente `beta*d` quando `u^T x = ||u||^2`, inclusive `x=u+z`
com `z` ortogonal a `u`; não apenas em `x=u`. Em geral, isso NÃO reproduz o
steering constante `y -> y + beta*d`. Prove que o checkpoint sem hooks reproduz
o mesmo hook contextual, não um hook constante diferente. Se a arquitetura usa
outra orientação, transponha conscientemente.

Não assuma que remover uma direção de unembedding é semanticamente limpo, que
null-space preservation cobre toda a distribuição ou que edits independentes
são composáveis. Em MoE, meça routing antes/depois e faça slicing correto de
experts; em multimodal, teste todas as modalidades e conexões.

### 5. VERIFY: seleção não é confirmação

Use descoberta/seleção em train/dev e rode uma única confirmação final em
problems/entidades/tempos nunca usados para escolher camada, alpha ou edit. Se
rodadas veem os scores de validação, isso é validação adaptativa, não teste
final. O toolkit `wsr program` também precisa desse teste externo intocado.

Braços mínimos:

- base e no-op passando pela mesma serialização/conversão/quantização;
- candidato e random norm-matched nas mesmas matrizes/update rule, idealmente
  com vários seeds e direção ortogonal/shuffled quando aplicável;
- runtime steering versus compiled quando essa for a hipótese;
- baselines de prompting/decoding/adapter/fine-tune/merge pertinentes;
- doses zero, negativa quando interpretável e várias positivas, sem presumir
  monotonicidade.

Avalie target, paráfrases/generalização, locality, tarefas não alvo, IF,
fluência, repetição, safety, tokens, thinking closure, cap hits e latência.
Brevidade comprada com resposta incompleta é regressão.

Use prompts e seeds emparelhados. Quando viável na régua da casa, faça n=5
gerações por problema e braço, registrando separadamente o número de problemas
independentes e a justificativa/poder. Reporte média, mínimo observado (piso) e
fail_rate; use CV só para métricas positivas de escala proporcional com média
não próxima de zero. Anote o timeout maior de thinkers. Para acurácia binária,
use McNemar com uma observação pareada por problema ou bootstrap clusterizado
por problema; para contínuas, bootstrap por problema. Clusterize doses do mesmo
parent e corrija multiplicidade quando houver vários contrasts. Não trate
samples do mesmo problema como replicações independentes.

Pré-registre não inferioridade: por exemplo, exigir que o intervalo do delta
de guard fique acima da margem negativa definida. “p>0,05” não prova
ausência de efeito. Preserve contagens de problemas melhores/piores e os
resultados negativos.

Checklist completo: `references/methods-and-validation.md`.

### 6. MINIMIZE só depois de provar o edit

Duas operações diferentes:

- **Sparsification:** manter top-k entradas/colunas/rows de `DeltaW`, por
  magnitude ou critério mecanístico, e reavaliar.
- **Backward elimination/drop-one:** reverter cada entrada/grupo e manter a
  remoção só se o comportamento continuar válido; repetir até ponto fixo.

Faça screening e toda a seleção de subsets em discovery/selection, sem consumir
o teste final. Congele original, minimizado e controles e avalie-os uma única
vez no teste final. Se o original já consumiu esse teste, o minimizado precisa
de um novo teste externo independente. Conte suporte BF16 exato e registre
`sum(abs(DeltaW))`, `||DeltaW||_F` e versões relativas separadamente.
Um empate em treino single-sample não demonstra equivalência geral.

O experimento Inkling/alphaXiv reduziu o Qwen de 130.188 para 8.192 entradas
no screening, mas não terminou os braços pareados de held-out/GSM8K/IFEval.
O carro dez-pesos tem evidência muito mais forte, mas é um MLP de controle e
não licencia a mesma expectativa em LLMs. Detalhes e links na evidência.

### 7. Materialize, quantize quando aplicável e proponha deploy

**Se o destino for LLM/GGUF quantizado:**

```text
base BF16/Safetensors imutável
  -> edit BF16 + manifest + delta
  -> reload e smoke funcional
  -> conversão GGUF
  -> quantização/calibração
  -> reload e smoke do GGUF
  -> bateria completa no artefato de deployment
  -> proposta explícita de deploy ou retenção da base
```

Se não houver fonte editável adequada, pare e peça o master; dequantização não
recupera precisão. Passe base, random e edit pela mesma forja. Para medir o
efeito residual, dequantifique edit e no-op da mesma forja e conte diferenças
lógicas, relatando expansão por scales/blocos. Essa contagem não é suporte
armazenado nem prova de esparsidade real. Não extrapole Q8_0 para Q4_K_M,
NVFP4 ou outros i-quants.

**Nos demais formatos/runtimes:** materialize no dtype apropriado, valide
reload/smoke e rode a bateria no artefato efetivamente servido; marque
GGUF/quantização como N/A. Não force uma conversão que o runtime não usa.

Para quantização da casa, use `calib-forja` se disponível; para serving,
`llama-swap-fleet` se disponível. Sem elas, siga os controles genéricos desta
skill. Não altere a lane nem reinicie serviço sem autorização.

## Portabilidade nesta casa

Copie o diretório inteiro, incluindo `references/`. Caminhos de instalação:
Hermes default: `~/.hermes/skills/mlops/weight-space-programming/`; Hefesto:
`~/.hermes/profiles/hefesto/skills/mlops/weight-space-programming/`; DSH/Pi:
`~/.agents/skills/weight-space-programming/`. Profiles Hermes não herdam
automaticamente a skill default. Após copiar, recarregue/reinicie o harness e
valide a descoberta; YAML válido não prova que a skill foi descoberta.

## Relatório obrigatório

Entregue sem inflar:

1. claim exato e status da evidência;
2. base/editor, versões, dados, forwards, custo e tempo;
3. fórmula/operações, tensores, entradas/norma/densidade;
4. todos os braços, seeds, métricas, ICs e regressões;
5. diferença entre screening, validação adaptativa e teste final;
6. resultado no artefato final efetivamente servido, com quantização quando
   aplicável;
7. caminhos do delta, checkpoint, logs, manifest e rollback;
8. limitações e o que NÃO foi testado;
9. papers relacionados, oferecendo PDF/link ou abertura no browser.

Nunca transforme “removeu loop”, “previu um edit” ou “passou num benchmark” em
“entende pesos”, “aprendeu uma capacidade” ou “funciona em qualquer modelo”.
