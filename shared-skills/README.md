# Backup de skills compartilhadas

Snapshot de 17/nov/2026 dos pacotes usados por harnesses que leem
`~/.agents/skills/`, incluindo DSH e Pi. O WSP também está instalado no Hermes
default e no profile Hefesto; esta cópia preserva o mesmo pacote completo.

## Conteúdo

- `weight-space-programming/`: WSP/WSR v2.0.0, protocolo, evidências,
  bibliografia e template de pré-registro.
- `cua-driver/`: conteúdo real do symlink compartilhado, snapshot da versão
  0.51.5; o host tinha uma instalação filtrada por plataforma.
- `computer-use/`, `find-skills/`, `flybrain-knowledge/`, `orca-cli/`,
  `orchestration/`, `org-mode-files/`, `reverse-engineer-anything/` e
  `window-driving/`: snapshot dos pacotes presentes no diretório compartilhado.
- `upstream-skill-lock.json`: referência de origem/hash dos pacotes registrados
  pelo instalador. Não sobrescreva o lock vivo automaticamente ao restaurar.

Não contém sessões, histórico, credenciais, dotfiles de agentes ou configuração
de endpoints. Na cópia versionada de `org-mode-files`, uma referência ao handle
local foi generalizada para “usuário”; o arquivo ativo não foi alterado.

## Restore seletivo

Copie apenas o pacote desejado para `~/.agents/skills/<nome>/`, preservando os
arquivos de referência. Para WSP em profiles Hermes, use também o caminho de
skills do profile correspondente. Recarregue ou reinicie o harness e verifique a
descoberta da skill; não presuma herança entre profiles.

## Privacidade e validação

Os arquivos incluídos foram varridos por padrões de credenciais, chaves,
contatos e identificadores pessoais antes do commit. O diff staged passa por
nova checagem no fluxo de backup. WSP teve YAML estrito, referências internas,
hashes e cópias verificados; os demais pacotes são snapshots, não uma promessa
de nova validação funcional de suas operações.
