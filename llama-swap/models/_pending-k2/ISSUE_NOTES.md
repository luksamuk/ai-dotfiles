# Dossiê K2-Horizon-MoVA-36B-A4B — material pra resposta da issue

Issue: https://github.com/luksamuk/ai-dotfiles/issues/1 (kassane, 06/09/2026)
Autor: Hermes Agent (sessões 06-09/09/2026). Hardware de teste: RTX 3050 6GB, R7, 31GB RAM.

## 1. Bug de merge no vocab (com patch pronto)

`src/llama-vocab.cpp` do branch k2-horizon: branch `solar-open` ficou com corpo VAZIO
e o corpo dele (que seta `LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN`) caiu DENTRO do branch
`k2-horizon`, sobrescrevendo `LLAMA_VOCAB_PRE_TYPE_K2HORIZON`.

Efeito: o regex custom do K2 (dígitos agrupados 1-3, ZWJ/combining marks) é código
morto — todo modelo k2-horizon tokeniza com o regex do solar-open (quebra dígitos
um a um). Erro de qualidade vs tokenizer oficial do IFM em números/unicode.

Patch: `k2-horizon-vocab-fix.patch` (neste diretório) — commit cf03d82 no clone local.
Testado: build com patch, contagem determinística exata.

## 2. Hadamard K/V corrompe output no k2-horizon (reprodução determinística)

`--k-cache-hadamard` (e a combinação K+V) produz glitchs de tokens REPRODUTÍVEIS
a temp 0 ("33 33, 36 36, 44 44" na contagem 1-50). Sem hadamard, a mesma sequência
saio exata. V-hadamard isolada saiu limpa (amostra única @temp 1.0) — o K é o culpado.

Tese: a de-transformação do K ("fa_h") mora nos kernels IQK FA — e o k2-horizon
EVITA o IQK FA (comentário do próprio build_k2horizon.cpp: K quantizado @ head_dim 128
não suportado; troca por llm_build_kv). A de-transform não cobre esse caminho.
A hadamard precisa ser portada pro caminho llm_build_kv, ou desabilitada p/ k2-horizon.

Matriz (temp 0, determinística, ctx 8192):
| config | resultado |
|---|---|
| K+V hadamard | glitch ("33 33, 36 36, 44 44") |
| sem hadamard | 1..50 exato |
| K isolada | glitch ("44 44") |
| V isolada | 1..50 EXATO (determinístico) — MoVA→V-cache→de-transform funciona end-to-end |

Atribuição final: corrupção é 100% K-side. O caminho do V (onde mora o MoVA) lida
corretamente com hadamard + de-transform; o K (projeção padrão, sem MoVA) perde a
de-transformação no caminho alternativo llm_build_kv. Bug de cobertura de caminho,
não conflito arquitetural com MoVA. (Curiosidade: V-only marcou 21.7 t/s vs 20.4
baseline — possível micro-ganho, não adotado.)

## 3. --fit não funciona na arquitetura (com workaround medido)

`--fit` trata só `ffn_*_exps` como MoE tensors — os `attn_v_exps` do MoVA (3.2GB)
ficam na GPU; resíduo de 5.835 MiB > 6.144 MiB do card → "Unable to auto-fit".
E `--override-tensor` não pode ser combinado com `--fit` (erro explícito).

Workaround (mapeamento manual, medido @131K):
```
-ngl 99
--override-tensor "ffn_gate_exps=CPU,ffn_down_exps=CPU,ffn_up_exps=CPU,tok_embd=CPU"
--no-kv-offload          # KV q4_0 @131K = 6.4GB em RAM
--ubatch-size 256
env: GGML_CUDA_ENABLE_UNIFIED_MEMORY=1   # seguro anti-OOM, doutrina ornith
```
VRAM 5.3GB (86%), RAM 24GB, decode 20.4 t/s, prefill 67.2 t/s.
Output tensor na GPU importa: manter `output` no CUDA = +30% decode (15.2→20.9 @8K).

KV real do modelo: ~49KB/token (48 layers, 8 kv-heads × 128, q4_0) → 432MiB @8K,
6.4GB @131K. 512K nativo seria impossível no card.

## 4. O alias "base" da config de exemplo PENSA

Sem `enable_thinking:false` no filtro do alias base, o template cai no default
`reasoning_effort=high` — o "base" queima os tokens em thinking. Fix: chat_template_kwargs
enable_thinking:false no alias base (ou --reasoning off). Adicionalmente recomendado:
--reasoning-format deepseek (thinking em reasoning_content) + budget 16384 como teto
(8192 trunca o modo high com frequência).

## 4b. Temperatura: temp 1.0 do model card quebra leitura agêntica (nossa régua: 0.7)

Descoberta empírica (08/09): com temp 1.0 (`:think` seguindo a recomendação "always high"
do model card), o modelo leu "London" como "Tokyo" 3x seguidas no mesmo prompt de tools.
Mesma task com temp 0.7 (thinking ON, tudo igual): 4 calls paralelas perfeitas, cidades
corretas. A régua 1.0/top_p 0.95 do IFM é para BENCHMARK (máxima diversidade p/ scoring);
para agentes que precisam ler o prompt, 0.7 é o correto. Ambos os configs (o do kassane
0.7/0.6 @ top_p 0.9 e o nosso 0.7/0.7 @ 0.95) rodam ABAIXO de 1.0 na prática — o
--temp 1.0 do cmd é sobrescrito pelo stripParams+setParamsByID em ambos.

## 5. Tool calls: parser engole as boas, vaza as malformadas

Com o template da abenzerps + --parallel-tool-calls, calls bem-formadas são parseadas
em estruturadas (testado: 2 e 4 tool_calls paralelos num turno). MAS: call malformada
(args trocados, comum em :think temp 1.0) = tags cruas `<ifm|tool_call>` vazam no
content. Candidato a melhoria: parser tolerante ou fallback strip; testar também
tool_call_format json via --chat-template-kwargs.

Achado adicional de harness (não é do fork/modelo): num cliente de chat com loop
de tools, o round-trip PRECISA manter as tools no payload com tool_choice "auto".
Com tool_choice "none" o parser server-side é PULADO (server-common.cpp:814) e
qualquer re-emissão do modelo vaza tags cruas; sem tools no payload, o template
perde as definições de ferramenta do system e o modelo re-planeja em vez de responder.
Simulador de agente corrigido no ai-dotfiles (testchat, commits b0a3b3c..5969107):
ciclo ReAct natural — mesmo contexto do round 1 (system incluído), tools sempre
presentes, loop termina quando o modelo responde sem calls.

## 6. Template: proveniência

GGUF abenzerps: `chat_template.jinja` = adaptada pro Jinja limitado do llama.cpp
(sem dict()/sameas, think-tags extraídas do content, escape enable_thinking=false).
`chat_template.upstream.jinja` = original do IFM. A embutida do IFM pressupõe o
parser k2_horizon do fork MBZUAI-IFM (inexistente no ik_llama) → usar a da abenzerps.

## 7. Números de referência (RTX 3050 6GB, 31GB RAM)

| config | decode | prefill | VRAM | RAM |
|---|---|---|---|---|
| baseline kassane (--fit, @8K, falhou load) | — | — | — | — |
| manual R1 @8K | 16.3 t/s | 48.8 t/s | 3.3GB (54%) | ~17GB |
| manual + output GPU @8K | 20.9 t/s | 49.2 | 3.7GB | ~24GB |
| v3 @131K (final): ffn_*_exps CPU + MoVA GPU + UMA insurance | 20.4 t/s | 67.2 | 5.3GB (86%) | 24GB (77%) |
| v2 @131K (intermediário, p/ referência) | 19.6 t/s | 45.4 | 3.2GB (52%) | 27GB (87%) |

Snake test (curses, score, aceleração): 180 linhas, compila limpo.
Contagem determinística: exata (sem hadamard). SHA256 Q3_K_M verificado (662610e0).

Samplers EFETIVOS finais (após stripParams+filters; cmd declara 1.0 mas filters
sobrescrevem): base 0.7/0.95, :think 0.7/0.95 (revertido de 1.0 — ver 4b),
top_k default 40, min_p 0, repeat_penalty 1. Fragmento baseline do kassane
(k2-horizon-36b-r1) mantém a régua DELE exata (0.7/0.6 @ 0.9, top_k 20) p/ A/B.

## 8. Paisagem de quants (07-08/09)

- NANI-Nithin GGUF: escada completa i-quants IQ1_M (8.1GB) → IQ4_XS (18.7GB) + MXFP4_MOE (20.2GB) + BF16
- vincespeed APEX-GGUF: i-quality 23.9GB / i-balanced 26.3GB / i-compact 17.6GB
- abenzerps (usada nos testes): Q3_K_M 16.4GB … Q8_0 37.1GB
- ONYX: nada ainda. APEX configs do mudler: nada ainda. PR de arch no ggml-org/llama.cpp: nada ainda.
- hermitdave oQ4e: safetensors p/ runtime Python do IFM — não GGUF, fora do escope llama.cpp
- MXFP4 suportado nos tipos ggml do fork (R8) — MXFP4_MOE 20.2GB é candidato de qualidade futura
- IFM lançou irmãos menores: 7B, 3.7B, 0.9B (+32B denso) — candidatos p/ outros devices
- FAMÍLIA COMPLETA (08/09): K2-Horizon-375B-A23B existe (flagfrontier; GGUF Baekpica mixed-quant) —
  26 repos de GGUF na busca. Quantizers ativos do MoVA: NANI-Nithin, abenzerps, darioooooo0o,
  SAIFIINDUSTRIES, aj9o9, vincespeed (APEX), kingjones777 (ROCmFP4), primitive-ai (NVFP4)