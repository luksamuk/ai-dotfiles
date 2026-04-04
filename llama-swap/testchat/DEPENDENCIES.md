# Dependências do LM Studio Chat

## Visão Geral

Aplicação de chat com interface rica no terminal, suporte a streaming de reasoning/resposta, seleção navegável de modelos via API, e histórico persistente de prompts.

## Pacotes Principais

As dependências são gerenciadas pelo `pyproject.toml` e instaladas automaticamente via `uv sync`:

| Pacote | Versão | Descrição |
|--------|--------|-----------|
| rich | >=14.0.0 | Renderização de interface rica no terminal (panels, layouts, markdown, cores) |
| openai | >=1.0.0 | Cliente para APIs OpenAI-compatíveis |
| questionary | >=2.0.0 | Menus interativos navegáveis com setas |
| prompt-toolkit | >=3.0.0 | Histórico persistente de comandos (↑/↓ no prompt) |
| requests | >=2.32.0 | Requisições HTTP para buscar modelos da API |

## Instalação

```bash
uv sync
```

## Uso

```bash
# Rodar diretamente
uv run main.py

# Ou via script definido no pyproject.toml
uv run chat
```

## Fluxo da Aplicação

1. **Busca de modelos**: Consulta `GET /v1/models` para listar modelos disponíveis
2. **Seleção**: Menu navegável com ↑↓ mostrando ID, descrição e metadados (thinking, tools, VRAM)
3. **Detecção automática**: Verifica `meta.llamaswap.features.thinking` para adaptar layout
4. **Prompt**: Campo com histórico persistente (`~/.rich_chat_history`)
5. **Streaming**: Layout adaptativo - split-screen para reasoning+response, ou tela cheia apenas response

## Controles

| Tecla | Ação |
|-------|------|
| ↑ / ↓ | Navegar no menu de modelos ou histórico de prompts |
| Enter | Selecionar modelo ou enviar prompt |
| Ctrl+C | Sair a qualquer momento |

## Layout Adaptativo

- **Modelos com thinking**: Painel de raciocínio (esquerda) + resposta (direita)
- **Modelos sem thinking**: Apenas painel de resposta em tela cheia

## Requisitos

- Python >= 3.12
- Servidor OpenAI-compatível rodando em `127.0.0.1:12434` (ex: LM Studio, llama-swap)
