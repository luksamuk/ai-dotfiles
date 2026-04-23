<p align="center">
  <h1>🤖 Chat com Reasoning e Streaming Markdown</h1>
</p>

## 📋 Descrição

Aplicação de chat com interface rica no terminal para interação com modelos de linguagem via API OpenAI-compatível (LM Studio, llama-swap, Ollama, etc).

## ✨ Recursos

- 🎯 **Seleção de modelos**: Menu navegável com metadados (thinking, tools, VRAM)
- 🧠 **Detecção automática**: Identifica modelos com suporte a reasoning via API
- 📐 **Layout adaptativo**: Split-screen para reasoning+response, ou tela cheia apenas response
- 📝 **Markdown formatado**: Syntax highlighting e renderização rica
- 📜 **Histórico persistente**: Acesse prompts anteriores com ↑/↓
- ⚡ **Streaming em tempo real**: Acompanhamento do raciocínio e resposta

## 🚀 Instalação

### Pré-requisitos

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) instalado
- Servidor OpenAI-compatível rodando (llama-swap, LM Studio, etc)
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

## 🔌 Configuração do Servidor

### LM Studio

1. Baixe e instale o [LM Studio](https://lmstudio.ai/)
2. Carregue um modelo com suporte a thinking (ex: Qwen 3.5, Gemma 4)
3. Ative o **"Local Inference Server"** na aba **"Developer"**
4. Certifique-se de que está acessível em `LLAMA_SWAP_HOST:LLAMA_SWAP_PORT` (default: `127.0.0.1:12434`)

### llama-swap

Configure o `config.yaml` com modelos que tenham a flag `thinking: true` nos metadados:

```yaml
models:
  "gemma4-e4b-think":
    cmd: ./llama-server ... -m gemma-4b-q4_k_m.gguf
    meta:
      llamaswap:
        features:
          thinking: true
```

## 🖥️ Uso

```bash
# Rodar a aplicação
uv run main.py

# Ou via script
uv run chat
```

### Fluxo

1. **Menu de modelos**: Selecione o modelo usando ↑/↓
2. **Prompt**: Digite sua pergunta (use ↑/↓ para histórico)
3. **Streaming**: Acompanhe o raciocínio e resposta em tempo real
4. **Resultado**: Visualize o prompt, modelo, raciocínio (se houver) e resposta

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
lmstudio/
├── main.py              # Aplicação principal
├── pyproject.toml       # Configuração do projeto
├── uv.lock             # Lock de dependências
├── .venv/              # Ambiente virtual
├── README.md           # Este arquivo
└── DEPENDENCIES.md     # Documentação de dependências
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
