# wan2gp-patches — backup dos patches locais do Wan2GP

Backup versionado da camada de modificações locais aplicada sobre o
`deepbeepmeep/Wan2GP`. Serve para recuperar o estado após reinstalação ou
depois de uma atualização que dê errado.

## Conteúdo

```
wan2gp-patches/
├── patches-locais.patch      # diff unificado: 1e1dd275 → camada WIP local
├── arquivos/                 # cópia literal dos 8 arquivos modificados
├── profiles/                 # perfis de geração do MiniMax H3
├── testes-h3/                # JSONs de configuração de teste
├── base_origin_main.txt      # SHA do upstream no momento do dump
├── head_local.txt            # SHA do commit local mais recente
└── branch_backup.txt         # SHA do branch backup-pre-dlss5
```

## O que a camada local contém

Três commits sobre o upstream:

| SHA | assunto |
|---|---|
| `8aadcb44` | patches locais do MiniMax H3 (enhancer + correção de text_encoder_variant) |
| `2b01d2d2` | defaults PDD W4A8 assimétrico + ajustes no enhancer |
| `ee1dc18b` | enhancer: descrições VLM anti-leak + tratamento de painéis de colagem |

E 8 arquivos tocados, sendo 7 em `defaults/` e `shared/qtypes/` (ajustes de
quantização e defaults de modelo) e um no prompt enhancer do H3.

## Como restaurar

### Opção 1 — reaplicar o patch

```bash
cd ~/git/Wan2GP
git stash -u                      # ou commit do que estiver pendente
git checkout <base de origin/main>
git apply --3way ~/git/ai-dotfiles/wan2gp-patches/patches-locais.patch
```

### Opção 2 — copiar os arquivos literais

```bash
cd ~/git/Wan2GP
cp -r ~/git/ai-dotfiles/wan2gp-patches/arquivos/* .
```

Útil quando o upstream mexeu nos mesmos arquivos e o patch não aplica limpo.

### Perfis e testes

```bash
cp -r ~/git/ai-dotfiles/wan2gp-patches/profiles/* profiles/
cp ~/git/ai-dotfiles/wan2gp-patches/testes-h3/*.json .
```

## Atualizando o backup

Depois de qualquer mudança na camada local:

```bash
cd ~/git/Wan2GP
git diff <base-upstream> HEAD --stat      # conferir o que mudou
git diff <base-upstream> HEAD -- <arquivos> > ~/git/ai-dotfiles/wan2gp-patches/patches-locais.patch
```

E atualizar os SHAs em `base_origin_main.txt` / `head_local.txt`.

## Notas de manutenção

- **Nunca usar `git add -A` neste repositório.** Os checkpoints ficam em
  `models/` e `ckpts/` (dezenas de GB sem ignore) e o scan trava.
- O merge/rebase com o upstream foi verificado sem conflito textual em
  2026-09-15 (`git merge-tree --write-tree` retornou árvore limpa). As
  mudanças locais vivem em regiões distintas do arquivo do enhancer.
- Ao atualizar do upstream, validar com `--dry-run` antes de gerar de verdade.
