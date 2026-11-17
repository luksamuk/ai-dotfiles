# Bibliografia anotada e artefatos

Corte: 17/nov/2026. Links primários primeiro. “Preprint/blog” não é demérito;
serve para não confundir evidência nova com consenso revisado por pares.
Quando o usuário quiser estudar, ofereça o PDF/link ou abra no browser.

## Núcleo WSP/WSR de novembro de 2026

1. **Takehi, Teoh, Zhang, Khattab — Sparks of Weight-Space Programming**.
   MIT CSAIL, blog técnico com logs, 02/nov/2026.
   https://rikiyat.github.io/blog/weight-space-programming/
   https://github.com/RikiyaT/lm_understand-logs
   Pesos Qwen: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
   Oito experimentos + follow-up Gemma; leitura, escrita e reparo sem uma
   update rule única. O harness/forward server completos não estão todos no repo.

2. **Kocabay — Weight-Space Reading**. Poindexter Labs, preprint 13/nov/2026.
   https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models
   PDF: https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models.pdf
   Código: https://github.com/suayptalha/wsr
   Predição up/down/same de edits dados, não editor geral. Melhor célula .730 BA;
   self-reading inconclusivo; harness requer teste final externo para program.

3. **Concept Country — Gradient-Free Weight Editing Reduces Overthinking in
   Qwen3.8-27B**. Working paper, 08/nov/2026.
   https://concept.country/research/weight-space-programming
   https://github.com/conceptcountry/weight-space-programming
   https://huggingface.co/ConceptCountry/qwen-concept-27b
   Activation-to-weight determinístico com random norm-matched; não replica os
   nove setups do MIT. Todos os resultados reportados em Q8_0.

4. **Hu et al. — Exploring Weight-Space Programming: Capabilities,
   Generalization, and Limitations**. arXiv preprint, 06/nov/2026.
   https://arxiv.org/abs/2611.09382
   https://github.com/WSP-Scope/WSP-Scope-Project
   16 trajetórias, dois tamanhos e sete benchmarks; alta variância e seis
   trajetórias que retêm a base.

5. **Mehta et al. — Interpretable weight-space programming builds transparent
   transformers and brain encoding models**. arXiv preprint, 13/nov/2026.
   https://arxiv.org/abs/2611.18114
   https://github.com/csinva/weight-space-programming-autoresearch
   Síntese iterativa legível; brain pipelines ainda ajustam ridge readouts.

6. **Marasini, Khadka, Rahimi — WeightBench: Evaluating Language Models as
   Weight-Space Programmers**. arXiv preprint, 13/nov/2026.
   https://arxiv.org/abs/2611.18553
   https://github.com/Aayush-Marasini/WeightBench
   42 tarefas; inclui edits impossíveis e compressão sem dados, com OS sandbox.

7. **Ahmed — Can an LLM Actually Reason and Program about Neural Network's
   Weights or is it just memorization on steroids?** Preprint exploratório,
   05/nov/2026.
   https://arxiv.org/abs/2611.08070
   Local arithmetic real, mas labels/dataset priors; útil para desenhar
   permutação de labels e testes de familiaridade. Não refuta o setup MIT.

8. **Wang, Hsu, Wang, Belkin — A closer look at Sparks of WSP**.
   Análise técnica/blog + código, 13/nov/2026.
   https://zhi0467.github.io/posts/2026/post6/
   https://github.com/Zhi0467/astra-weight-edit-as-steering
   Interpreta o edit Qwen como steering contextual; variantes com sensor
   treinado usam Adam e deixam de ser gradient-free.

9. **alphaXiv/Inkling — Weight-edit minimization**. Experimento/notebook,
   17/nov/2026, não paper revisado.
   https://github.com/alphaXiv/weight-edit-minimization
   https://openresearch.sh/p/EdsyLoFUfY40
   Post: https://x.com/askalphaxiv/status/2122744324354097172
   Car-ramp 10 pesos com vários rollouts; Qwen D4k 8.192 tem screen forte e
   held-out parcial, mas faltam braços pareados de confirmação/guards.

10. **Ok, Lee — Weight-Space Programming Doesn't Need Transformers**.
    Preprint alphaXiv, 17/nov/2026.
    https://www.alphaxiv.org/abs/2611.weight-space-programming-associative-memory
    https://github.com/effl-lab/Weight-Space-Programming
    Tabelas esparsas escritas do zero; não evidência contra pretrained
    Transformers nem receita de edit de LLM.

### Comentários e replicação limitada

- Simon Willison, teste do Experimento 1 com modelos abertos, 03/nov/2026:
  https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
- Matthew Honnibal, crítica à agregação de condições muito diferentes:
  https://honnibal.dev/blog/astra-read-write-misleading
- Sean Goedecke, contextualização das capacidades versus pesquisadores:
  https://seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/

Esses comentários são análise secundária; não substituem os logs/papers.

### Fila de leitura, ainda não sintetizada nesta skill

Não usar estes tópicos como evidência até ler o trabalho:

- IWIBench: https://www.alphaxiv.org/abs/2611.iwibench-intra-weight-injection
- Weight Space Misfolding: https://www.alphaxiv.org/abs/2611.weight-space-misfolding
- Auto-WSP part 1: https://www.alphaxiv.org/abs/2611.auto-weight-space-programming-part-1-defining-the-problem-and-initial-experiments
- Clinical decision networks: https://www.alphaxiv.org/abs/2611.05341
- Gradient-free diffusion edit: https://www.alphaxiv.org/abs/2611.gradient-free-diffusion-weight-editing

## Knowledge editing

- **ROME**, Meng et al., NeurIPS 2022: https://arxiv.org/abs/2202.05262
  Localização causal + rank-one update factual; valor-alvo é otimizado.
- **MEMIT**, Meng et al., ICLR 2023: https://arxiv.org/abs/2210.07229
  Batch editing distribuído por camadas.
- **PMET**: https://arxiv.org/abs/2308.08742
  Otimiza hidden states MHSA/FFN e escreve via FFN.
- **EMMET**, Findings EMNLP 2024:
  https://aclanthology.org/2024.findings-emnlp.903/
- **WilKE**, Findings ACL 2024:
  https://aclanthology.org/2024.findings-acl.207/
  Escolhe camada por pattern matching em vez de índice fixo.
- **AlphaEdit**: https://arxiv.org/abs/2410.02355
  Projeta updates no null space de conhecimento preservado.
- **r-ROME**, EMNLP 2024: https://aclanthology.org/2024.emnlp-main.1210/
  Corrige instabilidade/artefatos de implementação do ROME.
- **REVIVE**, ACL 2026: https://aclanthology.org/2026.acl-long.1384/
  Protege subespaços dominantes e avalia muitas edições sequenciais.
- **Model Editing at Scale**, Findings ACL 2024:
  https://aclanthology.org/2024.findings-acl.902/
  Editabilidade, forgetting e downstream em escala.
- **The Butterfly Effect of Model Editing**, 2024:
  https://arxiv.org/abs/2402.09656
  Colapsos após edits difíceis; ler junto com r-ROME.
- **MQuAKE**, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.971/
  Multi-hop após edits; usar versão/auditoria corrigida quando disponível.
- **Superficial Editing**, ACL 2025:
  https://aclanthology.org/2025.acl-long.868/
  Métricas convencionais altas podem esconder conhecimento antigo em geração.
- **MoEEdit**, 2026: https://arxiv.org/abs/2602.10965
  Routing distribution shift e coupling em MoE.
- Ferramenta **EasyEdit**: https://github.com/zjunlp/EasyEdit
- Avaliação multimodal: https://aclanthology.org/2023.emnlp-main.856/ ,
  https://github.com/VLKEB/VLKEB e https://arxiv.org/abs/2502.19870

## Task arithmetic, merging e adapters

- **Model Soups**: https://arxiv.org/abs/2203.05482
  Média de runs da mesma tarefa/inicialização.
- **Task Arithmetic**: https://arxiv.org/abs/2212.04089
  Soma/subtração de vectors de checkpoints fine-tuned compatíveis.
- **TIES-Merging**: https://arxiv.org/abs/2306.01708
  Trim, sign election e disjoint merge.
- **DARE**: https://arxiv.org/abs/2311.03099
  Drop/rescale de delta; taxas extremas são específicas do regime.
- **AdaMerging**: https://arxiv.org/abs/2310.02575
  Coeficientes aprendidos por entropia; não data-free.
- **KnOTS**: https://arxiv.org/abs/2410.19735
  SVD para alinhar/mergir LoRAs.
- **One Bad Model Spoils the Bunch**: https://arxiv.org/abs/2406.14563
  Merging pode transferir misalignment.
- **Cross-Model Transfer**: https://arxiv.org/abs/2505.12021
  Bases diferentes exigem alinhamento; não é subtração gratuita.
- **Task Vector Quantization**: https://arxiv.org/abs/2503.06921
  Quantiza deltas para armazenamento, não prova preservação em GGUF Q4.
- **mergekit**: https://github.com/arcee-ai/mergekit
- **PiSSA**: https://arxiv.org/abs/2404.02948
  Inicialização espectral seguida de fine-tuning; não é sem treino.
- **LoRA vs Full Fine-tuning**: https://arxiv.org/abs/2410.21228
  Estrutura espectral, intruder dimensions e forgetting.

## Activation/representation steering

- **ActAdd**: https://arxiv.org/abs/2308.10248
- **Contrastive Activation Addition (CAA)**: https://arxiv.org/abs/2312.06681
- **Householder Pseudo-Rotation (HPR)**:
  https://aclanthology.org/2024.emnlp-main.761/
- **Generalisation and Reliability of Steering Vectors**:
  https://arxiv.org/abs/2407.12404
  Variância, anti-steering e OOD brittleness.
- **AxBench**: https://arxiv.org/abs/2501.17148
  No setup publicado, prompting superou os steering methods avaliados.
- **Weight Updates as Activation Shifts**: https://arxiv.org/abs/2603.00425
  Relação de primeira ordem; adapters são treinados.
- **Compilação com null-space constraints**, ACL 2026:
  https://aclanthology.org/2026.acl-long.1206/
  Viabilidade e risco de comportamento condicionado; ler antes de aplicar.

## Weight-space learning, representações e metamodelos

- **Predicting Neural Network Accuracy from Weights**:
  https://arxiv.org/abs/2002.11448
  Predição em model zoos, antecedente do rótulo WSR atual.
- **DWSNets**, ICML 2023: https://proceedings.mlr.press/v202/navon23a.html
  Simetrias de weight space.
- **Weight Space Learning survey**, 2026: https://arxiv.org/abs/2603.10090
  Understanding, representation e generation; escopo e limitações.
- **Learning on LoRAs**: https://arxiv.org/abs/2410.04207
  Simetria GL(r) de adapters.
- **ProbeGen**: https://arxiv.org/abs/2410.10811
  Usa respostas a probes, não leitura estritamente estática.
- **HuggingSANE**: https://arxiv.org/abs/2510.02096
  Backbone treinado em hubs heterogêneos; não leitor plug-and-play local.
- **Weight Oracles**: https://arxiv.org/abs/2610.07334
  Especialista treinado + código externo; comparar separadamente de zero-shot.

## Formatos e operação

- **Safetensors docs**: https://huggingface.co/docs/safetensors/index
  Lazy/zero-copy e formato seguro; evitar pickle de origem não confiável.
- **GGUF specification**:
  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
  Layout/metadata; GGUF dequantizado não recupera precisão perdida.

## Regras de citação nesta skill

- Cite o trabalho pelo que o setup mede, não pela manchete do X.
- Ao dar um número, nomeie modelo, split, dose/budget e braço de comparação.
- Diga “reportado” quando não houve replicação local.
- Se só houver agregados, não chame de reprodução end-to-end.
- Não converta alphaXiv AI Overview em fonte primária: abra o PDF/repo.
