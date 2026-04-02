# Pesquisa de Modelos Alternativos com Tool Calling
**Data:** 2 de abril de 2026
**Objetivo:** Encontrar modelos GGUF <=14B com function calling de empresas não-mainstream

---

## Resumo Executivo

| Categoria | Quantidade |
|-----------|------------|
| Modelos pesquisados (HF + Ollama) | 428 |
| Modelos <=8B com FC confirmado | 113 |
| Modelos <=14B recomendados | 15 |
| Modelos testados neste benchmark | 7 |

---

## Benchmarks Realizados

### Teste: "Qual o tamanho em KB dos arquivos .toml no diretório atual?"

| Modelo | Params | Tokens | Tempo | Resultado | Status |
|--------|--------|--------|-------|-----------|--------|
| qwen3.5:4b | 4B | ~6K | Rápido | Correto (~15KB) | ✅ PASS |
| nemotron-3-nano:4b | 4B | ~5K | Rápido | Correto (~15KB) | ✅ PASS |
| qwen3.5:9b | 9B | ~15.5K | Rápido | Correto (~15KB) | ✅ PASS |
| qwen2.5-coder:7b | 7B | ~13K | Rápido | JSON bruto | ❌ FAIL (template) |
| ministries-3:3b (temp 0.3) | 3B | ~14.5K | Médio | Correto (~15KB) | ✅ PASS |
| nanbeige4.1:3b | 3B | ~16.5K | Lento | Correto (~15KB) | ⚠️ OK (lento) |
|| nanbeige4-thinking:3b | 3B | ∞ | Loop | Falhou | ❌ FAIL |

### Novos Testes (Abril 2026) - Modelos Alternativos

|| Modelo | Params | Tamanho | Resultado | Status |
||--------|--------|---------|-----------|--------|
|| lfm2.5-nova-fc:1.2b | 1.2B | 730MB | Tools não detectado | ❌ TEMPLATE |
|| hermes3-llama3.2:3b | 3B | 2.0GB | Tools não detectado | ❌ TEMPLATE |
|| bonsai:4b | 4B | 572MB | Erro ao carregar | ❌ LOAD ERROR |

**Problema identificado:** Modelos do HuggingFace GGUF nem sempre incluem templates de tool calling no metadata do arquivo. Ollama detecta tools automaticamente apenas se o template incluir blocos `<tools></tools>`.

**Solução necessária:** Criar modelfiles customizados com templates de tool calling para cada modelo.

### Vereditos por Modelo

| Modelo | Veredito | Notas |
|--------|----------|-------|
| **qwen3.5:4b** | ⭐ EXCELLENTE | Melhor custo-benefício, rápido, eficiente |
| **qwen3.5:9b** | ⭐ EXCELLENTE | Maior qualidade, mais contexto, thinking visível |
| **nemotron-3-nano:4b** | ⭐ EXCELLENTE | NVIDIA, muito eficiente em tokens |
| **ministral-3:3b** | ✅ BOM (temp=0.3) | Funciona com temp baixa, senão falha |
| **nanbeige4.1:3b** | ⚠️ OK | Funciona, mas lento e consome muitos tokens |
| **nanbeige4-thinking:3b** | ❌ NÃO RECOMENDADO | Loop infinito de thinking |
| **qwen2.5-coder:7b** | ❌ FALHA | Template não converte tools para JSON |

---

## Modelos Recomendados para Instalação

### Alta Prioridade (Tool Calling Confirmado)

| Modelo | Params | Empresa | GGUF | FC Support | Tamanho |
|--------|--------|---------|------|------------|---------|
| **Hermes-3-Llama-3.2-3B** | 3B | NousResearch | Sim | FORTE (13 menções) | ~2GB |
| **Hermes-2-Pro-Mistral-7B** | 7B | NousResearch | Sim | FORTE (17 menções) | ~4GB |
| **Bonsai-4B** | 4B | PrismML | Sim | Sim (chat_template) | ~2.5GB |
| **Bonsai-8B** | 8B | PrismML | Sim | Sim (chat_template) | ~5GB |
| **LFM2.5-1.2B-Nova-FC** | 1.2B | NovelChronoAI | Sim | ESPECIALIZADO | ~0.8GB |
| **Ministral-3-3B-Instruct** | 3B | Mistral | Sim | Nativo (19K DL) | ~2GB |

### Prioridade Média

| Modelo | Params | Empresa | Notas |
|--------|--------|---------|-------|
| **cogito:8b** | 8B | Deep Cogito | Hybrid reasoning + tools |
| **granite3.3:8b** | 8B | IBM | Enterprise-grade, 128K ctx |
| **deepseek-r1:7b** | 7B | DeepSeek | Excelente reasoning + tools |
| **command-r** | ~7B | Cohere | Especializado em RAG + tools |
| **smollm2:1.7b** | 1.7B | HuggingFace | Ultra-leve para testes |
| **Falcon-H1:7B** | 7B | TII | Nova geração Falcon |

### Já Instalados - Ranking

1. **qwen3.5:4b** - Melhor custo-benefício
2. **qwen3.5:9b** - Melhor qualidade geral
3. **nemotron-3-nano:4b** - Alternativa NVIDIA eficiente
4. **ministral-3:3b** - Funciona com ajuste de temperatura
5. **nanbeige4.1:3b** - Funcional, mas lento

---

## Empresas Alternativas Descobertas

| Empresa | Modelos | Foco | Notas |
|---------|---------|------|-------|
| **PrismML** | Bonsai 1.7B/4B/8B | Function calling | Baseados em Qwen3 |
| **NovelChronoAI** | LFM2.5-1.2B-Nova | FC especializado | Ultra-pequeno |
| **NousResearch** | Hermes series | Tool calling | Especialistas em FC |
| **Deep Cogito** | Cogito 8B/14B | Hybrid reasoning | Novo player |
| **IBM** | Granite 3.3 | Enterprise | 128K contexto |
| **Cohere** | Command-R | RAG + tools | Especializado |
| **TII (Falcon)** | Falcon H1/H1R | Reasoning | Nova geração |
| **Nanbeige LLM Lab** | Nanbeige4 | Reasoning + Tools | China |

---

## Configurações Recomendadas (após ajustes)

```toml
# ===== MODELOS COM PROBLEMAS CORRIGIDOS =====

# ministral-3:3b - temp 0.3 funciona bem (era 0.5)
[models."ministral-3"]
model_id = "ministral-3:3b"
num_ctx = 65536
temperature = 0.3  # Reduzido de 0.5 para determinismo
top_p = 0.9
top_k = 40
repeat_penalty = 1.0

# ===== MODELOS FUNCIONANDO BEM =====

# qwen3.5:4b - Melhor custo-benefício
[models."qwen3.5"]
model_id = "qwen3.5:4b"
num_ctx = 65536
temperature = 0.7
top_p = 0.95
top_k = 40
repeat_penalty = 1.05

# qwen3.5:9b - Melhor qualidade
[models."qwen3.5:9b"]
model_id = "qwen3.5:9b"
num_ctx = 65536
temperature = 0.7
top_p = 0.95
top_k = 40
repeat_penalty = 1.05

# nanbeige4.1:3b - temp 0.6 do fabricante
[models."nanbeige4.1"]
model_id = "nanbeige4.1:3b"
num_ctx = 65536
temperature = 0.6
top_p = 0.95
top_k = 40
repeat_penalty = 1.0

# ===== NÃO RECOMENDADOS =====
# nanbeige4-thinking - Loop infinito de thinking
# qwen2.5-coder:7b - Template não converte tools
```

---

## Próximos Passos

### Instalação de Novos Modelos

```bash
# PrismML Bonsai (via HuggingFace)
ollama run hf.co/prism-ml/Bonsai-4B-gguf:Q4_K_M
ollama run hf.co/prism-ml/Bonsai-8B-gguf:Q4_K_M

# NousResearch Hermes (via HuggingFace)
ollama run hf.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF:Q4_K_M
ollama run hf.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF:Q4_K_M

# Outros (via Ollama Library)
ollama pull cogito:8b
ollama pull granite3.3:8b
ollama pull deepseek-r1:7b
```

### Criação de Modelfiles

1. Verificar se os templates suportam tools
2. Criar modelfiles com parâmetros otimizados
3. Testar tool calling em cada modelo
4. Benchmark comparativo completo

---

## Fontes

- **HuggingFace**: 428 modelos GGUF pesquisados
- **Ollama Library**: Modelos com tag `tools`
- **Testes locais**: 7 modelos avaliados
- **Arquivos salvos**: `/home/alchemist/gguf_models_research/`

---

## Arquivos Relacionados

- `~/git/ai-dotfiles/docs/model-research-2026-04-02.md` - Este relatório
- `~/git/ai-dotfiles/configs/ask-ai/models.toml` - Configurações de modelos
- `~/git/ai-dotfiles/modelfiles/` - Modelfiles personalizados
- `~/gguf_models_research/` - Dados da pesquisa HuggingFace