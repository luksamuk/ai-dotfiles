# Pesquisa técnica: K2-Horizon-MoVA-36B-A4B no ik_llama.cpp (fork kassane, branch k2-horizon, patch cf03d82)

Repositório local: `~/git/kassane-ik_llama.cpp` (fork de ikawrakow/ik_llama.cpp).
Data: 2026-09-06. Pesquisa documental (código + docs + HF), sem builds.

---

## 1. REASONING BUDGET

**CONCLUSÃO PRÁTICA:** `--reasoning-budget` e `reasoning_effort` são mecanismos complementares, não alternativos: o esforço (via `--chat-template-kwargs '{"reasoning_effort":"high"}'`) escolhe **qual** bloco de pensamento o template abre (`<ifm|think>` / `<ifm|think_fast>` / `<ifm|think_faster>`), enquanto o budget é um **sampler** que corta tokens dentro de qualquer bloco de pensamento. Recomendação: mapear `reasoning_effort` via `chat_template_kwargs` no `setParamsByID` do llama-swap (é o que realmente controla profundidade) **e** deixar `--reasoning-budget` (ou `thinking_budget_tokens` por request) como capa de segurança — não usar budget fixo 8192 como único mecanismo, pois ele não muda o comportamento do modelo, apenas executa o corte.

### (a) Como `--reasoning-budget` interage com o template

- O budget **não é** um parâmetro do template: é um sampler de sampling-chain (`common/reasoning-budget.h:19-46`, `common/reasoning-budget.cpp`). Máquina de estados: `IDLE → COUNTING → WAITING_UTF8 → FORCING → DONE`. Em COUNTING ele conta tokens entre `start_tokens` e `end_tokens`; quando o budget zera, FORCING força a sequência `reasoning_budget_message + end_tag` token-a-token (todos os outros logits → -inf).
- Os tags start/end não vêm de flag: o servidor injeta os `thinking_start_tag`/`thinking_end_tag` detectados pelo parser do template em `llama_params["reasoning_budget_start_tag"/"end_tag"]` (`examples/server/server-common.cpp:907-916`). Se `chat_params.thinking_end_tag` ficar vazio, **o budget silenciosamente não é ativado** (o bloco inteiro fica dentro do `if` de server-common.cpp:912).
- Portanto, para um modelo que controla thinking via `reasoning_effort` no template, o fluxo correto é: template escolhe a tag de abertura conforme o esforço → sampler conta tokens dentro dessa tag → ao esgotar, força o fechamento. Os dois mecanismos coexistem sem conflito.
- Overrides por request: campo `thinking_budget_tokens` no body (`examples/server/server-common.cpp:908-910`) e `reasoning_budget_tokens`/`reasoning_budget_start_tag`/`reasoning_budget_end_tag`/`reasoning_budget_message` por slot (`examples/server/server-context.cpp:1505-1522`).
- O pensamento sai em `reasoning_content` com `--reasoning-format deepseek` (`docs/parameters.md:182` — "puts thoughts in `message.reasoning_content`"), que reproduz o contrato da API do IFM.

### (b) Equivalente de `reasoning_effort` em llama.cpp/ik_llama.cpp

- Não existe flag `--reasoning-effort` nativa. O caminho documentado é `--chat-template-kwargs JSON` (`docs/parameters.md:183`: "Example for gpt-oss: `--chat-template-kwargs '{\"reasoning_effort\": \"medium\"}'`").
- No código: `common/common.cpp:2709-2716` popula `params.default_template_kwargs`; `examples/server/server-common.cpp:825-830` faz merge de `body.chat_template_kwargs` (por request) sobre os kwargs de linha de comando; `common/chat.cpp:2737-2739` injeta tudo como `extra_context` no render Jinja. O template da abenzerps consome `reasoning_effort` com default `high` (seção 3), então o kwarg funciona.
- `enable_thinking` via kwargs está deprecado em favor de `--reasoning on/off` (`common/common.cpp:2718-2721`, warning explícito; `docs/parameters.md:185`, PR 1376).
- O template embutido também suporta `enable_thinking=false` (emite `<ifm|think>\n</ifm|think>` vazio) — ver seção 3.

### (c) Recomendação para o setParamsByID do llama-swap

1. `--chat-template-kwargs '{"reasoning_effort":"high"}'` (ou `medium`/`low` conforme o caso) — controla a profundidade real; IFM recomenda sempre `high` (model card, Best Practices item 1).
2. `--reasoning-format deepseek` para o pensamento sair em `reasoning_content`.
3. `--reasoning-budget N` apenas como teto duro anti-runaway (ex.: 16384 em vez de 8192, dado que o modo `high` do IFM gera cadeias longas; 8192 tende a truncar o modo `high` com frequência). Alternativa por request: `thinking_budget_tokens` no body.
4. Verificar no primeiro boot (log verboso) que `chat_params.thinking_end_tag` foi detectado — sem isso o budget não engaja (server-common.cpp:912).

**Flags exatos documentados:** `--reasoning-format FORMAT` (docs/parameters.md:182), `--chat-template-kwargs JSON` (:183), `-rea/--reasoning on|off|auto` (:185), `--reasoning-budget N` (:186, "-1 unrestricted, 0 immediate end, N>0 token budget", PR 1376), `--reasoning-budget-message` (:187). Docs upstream: https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md

---

## 2. HADAMARD × MoVA

**CONCLUSÃO PRÁTICA:** A Hadamard no V-cache é aplicada **depois** da rota MoVA (sobre o V já roteado e somado pelos value-experts) e **antes** da escrita no cache, por bloco de head_dim=128, com de-transformação automática na leitura — o esquema é matematicamente consistente (transformada ortogonal, H·H=I). Não há risco teórico de amplificação de erro (a rotação preserva energia); o risco real é **empírico**: nenhum precedente MoVA+hadamard e o benefício tende a ser menor justamente porque a soma de 4 experts já difunde energia entre canais. Recomendação: não habilitar por padrão; só adotar após AB-test de perplexity com KV quantizado.

### (a) Onde a Hadamard é aplicada no pipeline

- O V roteado é produzido em `k2_horizon_routed_value()` (`src/graphs/build_k2horizon.cpp:39-127`): router `attn_v_gate` → softmax/sigmoid → `ggml_top_k` (n_used=4) → `llm_build_lora_mm_id` sobre `attn_v_exps` → `ggml_silu` → peso × silu → **soma** dos 4 experts (linhas 106-125). O retorno `value_out` é o V pós-rota.
- No loop (`build_k2horizon.cpp:183-194`), `Vcur` = `k2_horizon_routed_value(...)` nas camadas MoVA, depois reshape em heads e chamada de `llm_build_kv(..., Kcur, Vcur, Qcur, ...)` (linha ~213).
- Dentro de `llm_build_kv` (`src/llama-build-context.cpp:2274-2277`): `if (cparams.v_cache_hadamard) { if (block_size = hadamard_size_v(il)) v_cur = ggml_hadamard(ctx, v_cur, block_size); }` — aplicada sobre o **V já roteado/somado**, imediatamente antes de `llm_build_kv_store(...)` (linha ~2296) escrever no cache quantizado. Ou seja: **antes da escrita no cache, depois da rota MoVA**. O V bruto (por expert, antes da soma) nunca é transformado.
- Leitura: no caminho FA (`llm_build_kqv`, `src/llama-build-context.cpp:2096-2100`), após `ggml_flash_attn_ext` aplica-se `cur = ggml_hadamard(ctx, cur, block_size)` ("fa_h") — como atenção sobre V' = H·V dá S·(H·V) = H·(S·V), a segunda Hadamard (auto-inversa) recupera S·V no espaço original. O esquema fecha matematicamente.

### (b) Por-head/per-head-dim? head_dim 128 elegível?

- Sim, por head-dim: `hadamard_size_v(il)` (`src/llama-model.h:696-699`) = `hadamard_size(hparams.n_embd_head_v(il))`; `hadamard_size` (:681-689) retorna o próprio head_size se for potência de 2 (senão o maior múltiplo de 64–512). Com head_dim 128 do K2 → **bloco de 128**, ou seja, a transformada roda sobre cada vetor de cabeça de 128 dims individualmente (kernel CUDA `hadamard_f32/f16` opera em blocos `nh` dentro de `ne[0]`, `ggml/src/ggml-cuda/hadamard.cu:29-77`; CPU `iqk_hadamard`, `ggml/src/iqk/iqk_cpu_ops.cpp:522-577`).
- Modelos MLA forçam bloco 64 fixo (`src/llama-model.h:692` e `:697`: `if (is_mla_model()) return 64;` — `is_mla_model()` = DEEPSEEK2/GLM_DSA/MISTRAL4/BAILINGMOE3, :643-645). O K2 não é MLA, então usa 128.
- O kernel CPU (`iqk_cpu_ops.cpp:533`) exige `nh` potência de 2; 128 ✓. O caminho do Q/K é análogo em `llm_build_kv` (:2264-2273) e no build genérico (:3223-3235).

### (c) Riscos de interação com a rota MoVA

- **A hipótese de "amplificar erro de quantização" não se sustenta matematicamente**: a transformada de Hadamard é ortogonal e normalizada (scale 1/√n, `fast_ht` em iqk_cpu_ops.cpp:510-519 e `hadamard_butterfly` em hadamard.cu:14-27), preserva a norma do vetor. O erro absoluto de quantização pós-rotação não cresce; o que muda é a distribuição por canal — a rotação espalha outliers concentrados.
- **Risco real — benefício marginal**: o V do MoVA já é uma mistura ponderada (softmax normalizado por `expert_weights_norm`) de 4 saídas silu-ativadas; essa soma tende a ter energia mais difusa entre canais do que um `W_v·x` denso, que é o caso onde a hadamard brilha (outliers por canal estáveis). Como o router muda por token, os outliers do MoVA são token-dependentes e não há estrutura por canal estável para "concentrar e espalhar". Resultado esperado: ganho pequeno ou nulo, nunca regressão estrutural.
- **Risco operacional**: `v_cache_hadamard` exige flash-attn (`src/llama.cpp:8448-8451`: "V cache quantization requires flash_attn") e é auto-desativada se o cache não for quantizado (:8468-8470, "no point in Hadamard transforms with not quantized V-cache"). Como o grafo K2 atualmente evita IQK FA com K-type quantizado no head_dim 128 (comentário em `src/graphs/build_k2horizon.cpp:209-211`), a hadamard só é relevante **depois** de existir uma config de KV quantizado que funcione.
- **Risco residual**: nenhuma validação numérica MoVA+hadamard existe no repo; a de-transformação na leitura ("fa_h") está no caminho FA de `llm_build_kqv` — caminho que o K2 usa via `llm_build_kv` — mas qualquer desvio futuro no build do K2 que não passe por `llm_build_kqv` deixaria V rotacionado no output. Testar antes de adotar.

### (d) Precedentes com arquitetura exótica

- **DeepSeek2 (MLA)**: usa hadamard com bloco fixo 64 com sucesso (`src/llama-model.h:692/:697`) — no MLA o "V" é o conteúdo latente comprimido, e a hadamard é aplicada sobre as projeções absorvidas; precedente positivo de hadamard sobre valores não-canônicos.
- **DeepSeek4**: contra-exemplo — K-hadamard **proibido** com erro fatal (`src/llama.cpp:8450-8454`: "DeepSeek4 K-cache Hadamard is not supported") e V-hadamard **silenciosamente ignorado** (:8464-8467: "DeepSeek4 has no independent V-cache; ignoring -vhad"). Motivo: ausência de cache V independente, não estatística da distribuição.
- Não há nenhum comentário no código de hadamard mencionando attention-MoE/MoVA; o caminho do K2 é genérico via `llm_build_kv`.

**Evidência-chave (arquivo:linha):**
- Aplicação pré-store sobre V roteado: `src/llama-build-context.cpp:2274-2277` (hadamard) → `:2296-2298` (kv_store).
- De-transformação pós-FA: `src/llama-build-context.cpp:2096-2100`.
- Tamanho de bloco por head_dim / MLA=64: `src/llama-model.h:681-699`.
- Exigência de FA + auto-off sem quant: `src/llama.cpp:8448-8478`.
- V roteado (silu + soma de experts): `src/graphs/build_k2horizon.cpp:106-127`.

---

## 3. ORIGEM DO CHAT TEMPLATE

**CONCLUSÃO PRÁTICA:** Cadeia de proveniência: template original do IFM (checkpoint/tokenizer) → abenzerps converteu para GGUF e manteve o original como `chat_template.upstream.jinja`, gerando uma variante compatível com o Jinja do llama.cpp/ik em `chat_template.jinja` (sem `is sameas`/`dict()`, parsing de think-tags pelo conteúdo, escape `enable_thinking=false`). A template "oficial" do GGUF do IFM não é exposta como arquivo e o runtime alvo dela é o fork MBZUAI-IFM (parser `k2_horizon`), inexistente no fork kassane — por isso a variante da abenzerps é a correta para o ik_llama.cpp.

### (a) O que o README da abenzerps documenta

- `https://huggingface.co/abenzerps/K2-Horizon-MoVA-36B-A4B-GGUF/raw/main/README.md`, seção "Chat template" (linhas 45-49): *"Each GGUF embeds the llama.cpp-compatible chat template. `chat_template.jinja` is a matching external copy for tools that require one. The original source template is retained as `chat_template.upstream.jinja` for runtimes with full Jinja support."* E: *"`xml` is the default tool-call format. Use `--chat-template-kwargs` to select `json` or `xml_typed` when required."*
- Ou seja: a origem declarada é o template do modelo fonte (IFM/K2-Horizon-MoVA-36B-A4B, commit fonte `05cab0a`, README linha 64-65), adaptado para o Jinja limitado do llama.cpp. A adaptação não está atribuída a um PR/issue específico — é documentação do próprio repo.
- O README também reafirma o runtime alvo: bloco IMPORTANT (linhas 19-20) aponta para o fork MBZUAI-IFM (`https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon`) até o suporte subir no upstream.

### (b) `chat_template.jinja` vs `chat_template.upstream.jinja` (ambos em `/tmp/k2-template-analysis/`)

Tamanhos: `.jinja` 883 linhas vs `.upstream.jinja` 994 linhas; diff = 173 linhas alteradas. Diferenças essenciais:

1. **`$ref`/`$defs` inlining de schemas de tools**: o upstream usa `namespace` + construtor `dict(...)` para expandir `$ref` (blocos inteiros removidos no `.jinja`); o `.jinja` comenta explicitamente (linhas 188-190): *"llama.cpp's Jinja has no dictionary constructor, so $ref inlining stays template-local by falling back to the exact JSON presentation"* — ou seja, ferramentas com `$ref` caem para JSON verbatim.
2. **Testes Jinja não suportados**: upstream usa `is sameas true/false` e `is not boolean`; `.jinja` troca por `is true/false`/`is not boolean` (llama.cpp minja não tem `sameas`).
3. **Mensagens de assistant com thinking**: o upstream **exige** campo tipado (`raise_exception` se a mensagem assistant não tiver `think`/`reasoning`/`reasoning_content`/`think_fast`/`think_faster`, linhas 958-959) e rejeita conteúdo não-string. O `.jinja` **extrai os think-tags do próprio `content`** (split em `<ifm|think>`/`<ifm|think_fast>`/`<ifm|think_faster>`, linhas 833-845) — compatível com como o ik_llama.cpp devolve histórico (ele reconstrói thinking como campo `thinking`, `common/chat.cpp:853-857`, que o template upstream **não** aceita — incompatibilidade concreta).
4. **Escape `enable_thinking`**: o `.jinja` adiciona `if enable_thinking is defined and enable_thinking is false → emite '<ifm|think>\n</ifm|think>\n'` (linhas 872-874), alinhado com o mecanismo `--reasoning off`/`enable_thinking` do ik_llama.cpp (server-common.cpp:829-838). O upstream não tem esse ramo.
5. **Igual nos dois**: `tool_call_format` json/xml/xml_typed (default **xml**), `reasoning_effort` high/medium/low mapeado para `<ifm|think>`/`<ifm|think_fast>`/`<ifm|think_faster>` (`.jinja` linhas 871-883; upstream equivalente), tags de turno `<|ifm|im_start|>`/`<|ifm|im_end|>` e `{% generation %}`.

### (c) Por que a template do IFM não serve

- O repo oficial `IFM/K2-Horizon-MoVA-36B-A4B-GGUF` **não publica nenhum `.jinja`** — a lista de arquivos via API do HF (`https://huggingface.co/api/models/IFM/K2-Horizon-MoVA-36B-A4B-GGUF`) contém apenas GGUFs + `README.md`. O README do IFM (linha 27) diz apenas: *"The GGUF files include the tokenizer metadata and a `llama.cpp`-compatible chat template"* — template embutida no GGUF, sem cópia externa.
- O runtime de referência do IFM é o fork MBZUAI-IFM: o README oficial manda usar `--tool-call-parser k2_horizon` (linhas 112 e 135-136 do README capturado em `/tmp/k2-template-analysis/ifm-gguf-readme-check.md`). **O fork kassane/ik_llama.cpp não tem parser `k2_horizon`** (grep por `k2_horizon` em `common/` e `examples/server/*.cpp`: 0 resultados) — a template embutida do IFM pressupõe esse parser e o comportamento do fork IFM, que o ik_llama.cpp não reproduz.
- Incompatibilidade técnica objetiva da versão upstream com o ik_llama.cpp: (i) usa `is sameas`/`dict()` que o Jinja do llama.cpp não suporta (o `.jinja` da abenzerps documenta isso em comentário); (ii) exige campo tipado de thinking em mensagens de assistant e não aceita o campo `thinking` que o servidor ik reconstrói (`common/chat.cpp:853-857`), o que quebraria multi-turno com histórico de raciocínio; (iii) não tem o ramo `enable_thinking=false` usado pelo `--reasoning off` do ik.
- O model card do IFM documenta o contrato de API que a template serve: `chat_template_kwargs {"reasoning_effort": "high", "tool_call_format": "xml"}`, resposta com `reasoning_content` + `content` (`ifm-gguf-readme-check.md` linhas 92-112; best practices linha 135: "Reasoning effort: always `high`"). URLs: https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-GGUF , https://huggingface.co/abenzerps/K2-Horizon-MoVA-36B-A4B-GGUF , https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon

---

## Arquivos gerados

- `/tmp/k2-review/pesquisa-reasoning-hadamard-template.md` (este relatório)
- `/tmp/k2-template-analysis/`: `chat_template.jinja`, `chat_template.upstream.jinja`, `README.md` (abenzerps), `ifm-gguf-readme-check.md` (IFM), `ifm-gguf-api.json` (lista de arquivos do repo IFM), `template.diff`, `ifm-gguf-readme.md`, `ifm-readme.md` (de sessão anterior)