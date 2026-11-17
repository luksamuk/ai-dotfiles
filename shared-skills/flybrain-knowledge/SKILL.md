---
name: flybrain-knowledge
description: Conhecimento consolidado sobre o cérebro da Drosophila melanogaster — anatomia e circuitos (corpo fungiforme/Mushroom Body, sistema olfatório, neurônios MBON/KC/PN/DAN, lateral horn), connectomas disponíveis (FAFB, L1EM, hemibrain), neurotransmissores, IDs e terminologia VFB/FBbt, e o vocabulário exato do Virtual Fly Brain (Term Info, Circuit Browser, pesos forward/reverse, Min Weight, # Paths). Use quando o usuário tirar dúvidas "de mosca": interpretar resultados do VFB, explicar circuitos, neurônios, genes, conectividade, templates ou dados do Virtual Fly Brain.
---

# Cérebro da mosca (Drosophila) — base de conhecimento

Consolidado durante exploração real do Virtual Fly Brain (VFB). O que está
marcado como [VERIFICADO] foi observado funcionando na UI; o resto é
conhecimento estabelecido da literatura. Não invente IDs — confirme pela API
(veja seção 7).

## 1. Como o cérebro da mosca está organizado (mapa mental)

- Neuropilos nomeados (FBbt ontology): cérebro adulto (adult brain,
  FBbt_00003624) ~120+ compartimentos nomeados: **antennal lobe,
  mushroom body, lateral horn, optic lobe, central complex** etc.
- **JRC2018Unisex** (Fasciculation/Joanelia 2018; VFB_00101567) é o template
  padrão em que tudo é registrado/alinhado. Imagens e reconstruções são
  mapeadas nesse espaço ("JRC2018U").
- Templates de imagem: confocal LM (Luz de luz) para linhas de driver GAL4 e
  EM (microscopia eletrônica) para reconstruções de neurônios.

## 2. Sistema olfatório: a arquitetura central (pense neste fluxo)

```
ORN (antenna, OR/IR/CO receptors) → antennal lobe (glomérulos) → UNIGLOMERULAR PN
   → mushroom body calyx (KCs) e lateral horn (LH)
   → KC (Kenyon cells) → MBONs → descendentes/comportamento
DANs (PAM/PPL1) → modulação dopaminérgica sobre KCs (aprendizado/recompensa)
APL → inibição lateral global dentro do MB
```

- O córtex do corpo fungiforme ("mushroom body") é o circuito de
  aprendizado: KCs (Kenyon) são esparso-codificadores; cada MBON lê uma
  "compartimentalização" dos lobos (α, β, γ, pedúnculo).
- VFB tem diagramas prontos: "Anatomy Diagrams → Circuit diagram of the
  mushroom body", "Olfactory system neuroanatomy", "Lateral horn".

## 3. Gerações de neurônios com que trabalhamos [VERIFICADO no grafo]

- **MBON** = Mushroom Body Output Neuron (classe FBbt_00047953); saída do
  corpo fungiforme. Exemplos reais usados: **MBON-B1>a_L (FAFB:1385339,
  MBON06, Glutamatergic)** e **MBON-a2sc_R (FAFB:346114, MBON18,
  Cholinergic)**.
- Nomes tipo "MBON18"/"MBON06" nesses nós = designação do reconstrutor
  (numeração do FAFB/CATMAID), não classe oficial — sempre abrir a Term Info
  para ver o nome curado.
- **KC** (Kenyon cells) recebem de PNs no calyx; **PN** (projection neurons)
  saem do antennal lobe; **DAN** (dopaminergic) modulam aprendizado.

## 4. Conectividade e pesos [VERIFICADO]

- O grafo do **Circuit Browser** desenha aresta com rótulo `forward [reverse]`
  = contagem de **sinapses EM** em cada direção. Exemplo real:
  MBON06 ⇄ MBON18 com `10 [0]` (10 sinapses forward, 0 reverse).
- Legenda de cores dos nós (neurotransmissor): marrom "Neuron", azul
  "GABAergic", lilás "Cholinergic", verde "Glutamatergic".
- Semântica da consulta (da doc oficial):
  - **Direcional**: campo 1 = source (origem), campo 2 = target (destino).
  - `# Paths` = máximo de caminhos retornados, do mais forte ao mais fraco
    (na imagem: mais forte embaixo).
  - `Min Weight` (padrão 10) = mínimo de sinapses **por conexão**: arestas
    mais fracas são cortadas — caminhos atravessados por elas desaparecem.
  - Algoritmo e fórmulas: github.com/VirtualFlyBrain/graph_queries
    (weighted_path.md).
- Consequência prática [VERIFICADO]: com Min Weight 10, um par fortemente
  conectado direto (10+) retorna só a aresta direta mesmo com # Paths=6;
  pares com hops fracos podem retornar vazio. Para ver multi-hop, use
  Min Weight menor ou pares conectados por intermediários fortes.
- "Neuron connectivity" no VFB é agregado por par de neurônios inteiros
  (somatório de sinapses); "synapse-level" fica nas ferramentas CATMAID/
  neuPrint.

## 5. Datasets disponíveis no VFB

| Dataset | Escala | O que tem |
|---|---|---|
| **FAFB** (Full Adult Fly Brain, TEM; Otoo et al. 2020/curation) | adulto, EM | reconstruções nomeadas "FAFB:NNNNNNN" com conectividade |
| **Janelia hemibrain** | adulto, EM | núcleo central denso |
| **L1EM** (larva) | larva | connectoma completo do CNS larval |
| **FlyCircuit / FlyLight** | adulto, LM | linhas de driver GAL4, split-GAL4 |
| **Templates**: JRC2018U (unisex/adult), JRC2013, larval L3 | | alvos de registro |

- Term Info lista **"Query For"** (ex.: "Neurons connected to X",
  "Expression patterns with similar morphology... [NBLAST]"), **"Graphs
  For"** ("Show location of X", "Show classification of X") e **"Add [term]
  to Circuit Browser Query"** — o link que preenche os campos do Circuit
  Browser (primeiro campo vazio; troca o último se cheios).

## 6. Vocabulário de identificadores (não erre)

- `FBbt_NNNNNNN` = classe (ontologia FBbt). `VFB_NNNNNNN` = instância/indivíduo
  no VFB (ex.: VFB_0010141p). `FAFB:NNNNNNN`, `L1EM:NNNNNNN` = IDs de
  reconstrução do dataset. `skid:NNN` = skeleton id CATMAID.
- Facetas úteis na `/search`: "has_neuron_connectivity" (tem conectividade!),
  "Individual"/"Class", "Adult"/"Larva", "FAFB", "L1EM", "NBLASTexp".

## 7. APIs para resposta programática (confie nelas, não na memória)

```
Busca por nome:
  GET https://v3-cached.virtualflybrain.org/search?query=MBON&limit=10
Term Info (metadados + queries prontas com preview):
  GET https://v3-cached.virtualflybrain.org/get_term_info?id=<VFB_...>
```
- O preview de "Neurons connected to X" já vem com rows (id/label/outputs/
  inputs) — dá para montar a resposta sem abrir a UI.
- Existe também um servidor MCP do próprio VFB (vfb3-mcp.virtualflybrain.org)
  e a biblioteca Python `vfb-connect`.

## 8. Cuidados ao responder

- Diferencie claramente classe (MBON, FBbt…) vs instância específica
  (FAFB:1385339) vs template (JRC2018U) nos relatos.
- Pesos de conectividade são do connectoma EM específico (adulto FAFB vs
  larva L1EM) — não misture estágios.
- Transmissores: MBONs de α/β costumam ser glutamatérgicos/GABAérgicos etc.
  — confirme o neurotipo pelo nó/cor no grafo ou pela Term Info (tags
  Cholinergic/Glutamatergic/GABAergic) em vez de assumir.
- Para dúvidas anatômicas profundas, o diagrama oficial em
  virtualflybrain.org/docs/anatomy-diagrams/ é a referência visual.
