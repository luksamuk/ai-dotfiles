<p align="center">
  <h1>🤖 Testchat — Chat Interativo com Reasoning, Tools e Streaming</h1>
</p>

## 📋 Descrição

Aplicação de chat com interface rica no terminal para interação com modelos de linguagem via API OpenAI-compatível (llama-swap, vLLM, Ollama, etc).

## ✨ Recursos

- 🎯 **Seleção de modelos**: Menu navegável com metadados (thinking, tools, vision, VRAM)
- 🧠 **Detecção automática**: Identifica modelos com reasoning (thinking) e tool calling via metadata
- 🛠️ **Tool calling mockado**: Testa capacidade de tool calling com ferramentas simuladas (get_weather, calculator, get_time)
- 📐 **Layout adaptativo**: Split-screen para reasoning+response, ou tela cheia apenas response
- 📝 **Markdown formatado**: Syntax highlighting e renderização rica
- 📜 **Histórico persistente**: Acesse prompts anteriores com ↑/↓
- ⚡ **Streaming em tempo real**: Acompanhamento do raciocínio, tool calls e resposta

## 🚀 Instalação

### Pré-requisitos

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) instalado
- Servidor OpenAI-compatível rodando (llama-swap com vLLM e/ou llama.cpp)
- Porta padrão: `12434`

### Setup

```bash
# Instalar dependências
uv sync
```

### Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `LLAMA_SWAP_HOST` | `127.0.0.1` | Host do servidor (use IP Tailscale para acesso remoto) |
| `LLAMA_SWAP_PORT` | `12434` | Porta do servidor |

**Acesso remoto via Tailscale (ex: Termux no celular):**

```bash
# No Termux ou outra máquina na rede Tailscale
export LLAMA_SWAP_HOST=100.65.187.74
uv run main.py
```

## 🛠️ Tool Calling

Modelos com `tools: true` nos metadados recebem automaticamente as ferramentas mock de teste:

| Ferramenta | Descrição | Exemplo |
|------------|-----------|---------|
| 🌡️ `get_weather` | Clima atual por cidade | "Qual o clima em Tokyo?" |
| 🧮 `calculator` | Expressões matemáticas | "Quanto é 2**10 + sqrt(144)?" |
| 🕐 `get_time` | Data/hora por fuso | "Que horas são em São Paulo?" |

As respostas são simuladas — servem pra testar se o modelo sabe chamar ferramentas corretamente, não pra obter dados reais.

**Fluxo de tool calling:**
1. O modelo decide chamar uma ferramenta → exibe `🔧 nome(args)` no buffer
2. Testchat gera a resposta mockada → exibe resultado inline
3. O modelo recebe o resultado e formula a resposta final

### Configuração de modelos vLLM com tool calling

Para que modelos vLLM suportem `tool_choice: "auto"`, adicione as flags:

```yaml
cmd: |
  ${vllm_bin} serve Qwen/Qwen3.5-0.8B
  --port ${PORT}
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
```

Sem essas flags, vLLM retorna erro 400 ao receber `tool_choice: "auto"`.

## 🔌 Configuração do Servidor

### llama-swap

Configure o `config.yaml` com modelos que tenham features nos metadados:

```yaml
models:
  "qwen3.5-9b-ace":
    cmd: |
      ${llama_server} --port ${PORT} -m qwen3.5-9b-ace.Q4_K_M.gguf ...
    metadata:
      llamaswap:
        features:
          thinking: true
          tools: true

  "qwen3.5-0.8b-vllm":
    cmd: |
      ${vllm_bin} serve Qwen/Qwen3.5-0.8B
      --port ${PORT}
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      ...
    metadata:
      llamaswap:
        features:
          tools: true
```

## 🖥️ Uso

```bash
# Via llama-swap-cli
llama-swap-cli testchat

# Ou direto
uv run main.py
```

### Fluxo

1. **Menu de modelos**: Selecione o modelo usando ↑/↓ (features marcadas com ícones 🤔 🛠️ 👁️)
2. **Prompt**: Digite sua pergunta (use ↑/↓ para histórico)
3. **Streaming**: Acompanhe reasoning, tool calls e resposta em tempo real
4. **Resultado**: Visualize o painel final com timing, tokens, e tool calls

### Controles

| Tecla | Ação |
|-------|------|
| ↑ / ↓ | Navegar no menu de modelos ou histórico de prompts |
| Enter | Selecionar modelo ou enviar prompt |
| Ctrl+C | Sair a qualquer momento |

## 📦 Dependências

| Pacote | Versão | Uso |
|--------|--------|-----|
| rich | >=13.0.0 | Interface rica no terminal |
| openai | >=1.0.0 | Cliente API OpenAI |
| questionary | >=2.0.0 | Menus interativos |
| prompt-toolkit | >=3.0.0 | Histórico de comandos |
| requests | >=2.31.0 | Busca de modelos via API |

### Estrutura de Arquivos

```
testchat/
├── main.py              # Aplicação principal (chat, streaming, UI)
├── tools.py             # Mock tools (get_weather, calculator, get_time)
├── pyproject.toml       # Configuração do projeto
├── uv.lock              # Lock de dependências
├── .venv/               # Ambiente virtual
└── README.md            # Este arquivo
```

## 🔧 Customização

O endpoint do servidor é controlado por variáveis de ambiente (veja tabela acima). Não é necessário editar o código.

- **Servidor local**: rode sem configurar nada (usa `127.0.0.1:12434`)
- **Servidor remoto**: exporte `LLAMA_SWAP_HOST` com o IP do servidor
- **Arquivo de histórico**: modifique `~/.rich_chat_history` (padrão)

## 📄 Licença

MIT License - Sinta-se livre para usar e modificar!

---

<p align="center">
  Feito com ❤️ usando <a href="https://github.com/Textualize/rich">Rich</a> e <a href="https://github.com/astral-sh/uv">uv</a>
</p>