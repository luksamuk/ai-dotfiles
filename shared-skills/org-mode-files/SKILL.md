---
name: org-mode-files
description: "Referência e verificação de sintaxe de arquivos Org (Org mode / Emacs). Como escrever e inspecionar .org corretamente: hierarquia de títulos com asteriscos (`*`), blocos de propriedades, palavras-chave de cabeçalho (`#+TITLE`, `#+FILETAGS`, `#+TODO`, `#+STARTUP`, `#+SETUPFILE`), formatação inline, links (URL, `id:`, âncoras, arquivo), blocos de código/quoted/exemplo, listas, tabelas, notas de rodapé, checkboxes, e convenções org-roam (`:ID:`, `ROAM_REFS: cite:`, `NOTER_DOCUMENT:`). Use quando alguém escreve ou revisa um arquivo .org — especialmente ao converter de Markdown (onde `#` é título e `**` é negrito) ou ao validar se um .org está bem formado. Armadilha central: em Org, `#` no início de linha é COMENTÁRIO, não título."
---

# Arquivos Org (Org mode / Emacs) — referência e verificação

Consolidado da sintaxe oficial do Org (orgmode.org) e dos padrões usados no
zettelkasten/org-roam do usuário. O que é [VERIFICADO] foi confirmado na UI do
Emacs; o resto é sintaxe estabelecida do Org mode.

> **Erro comum que vale ouro:** em Markdown, `# Heading` / `## Sub` são títulos.
> Em Org, `#` no início de linha é **comentário** (texto cinza, ignorado como
> estrutura). Título Org é `*` (1 a 6 asteriscos) + espaço. Escrever
> `## Anotações:` num arquivo .org cria um *comentário*, não uma seção — o
> correto é `** Anotações:`.

## 1. Hierarquia de títulos (outline)

Títulos usam de 1 a 6 asteriscos seguidos de um espaço:

```org
* Título (nível 1 / documento)
** Seção (nível 2)
*** Subseção (nível 3)
**** ...
```

- O primeiro `*` do documento é o título do nível 1.
- Níveis 4–6 também são válidos (só os 3 primeiros viram cabeçalhos em export
  ASCII; os demais viram listas — mas ainda são títulos legítimos).
- `org-inlinetask` permite níveis extras como tarefas inline (não usar sem
  carga ativa).

## 2. Bloco de propriedades (`:PROPERTIES:` … `:END:`)

Usado por org-roam e NOTER. Propriedades são `:NAME: value` alinhados:

```org
:PROPERTIES:
:ID:       75fb1dda-262e-494c-ba8c-1457710edcdb
:ROAM_REFS: cite:shiu2023
:NOTER_DOCUMENT: pdfs/cited.pdf
:ADDED:      [2026-11-16]
:END:
```

- `:ID:` — UUID único da nota (org-roam).
- `:ROAM_REFS:` — relações; `cite:KEY` liga a uma entrada BibTeX.
- `:NOTER_PAGE:` — página de uma anotação (NOTER).
- Repita o bloco por seção quando cada uma precisar de propriedades próprias.

## 3. Palavras-chave de cabeçalho (`#+KEY: value`)

Antes do primeiro título, com `#+`:

```org
#+TITLE:      Título do documento
#+FILETAGS:   :cognição:computação:neurociência:
#+TODO:       PRÓXIMO LENDO PARADO | LIDO REVISANDO
#+STARTUP:    showall latexpreview
#+SETUPFILE:  ../setupfile.org
#+DATE:       [2026-11-16]
```

- `#+FILETAGS:` usa `:` em volta de cada tag (não `#tag`).
- `#+STARTUP:` controla colapsar/mostrear árvores, preview LaTeX, etc.

## 4. Formatação inline

| Efeito | Org | Markdown |
|--------|-----|----------|
| Negrito | `*texto*` | `**texto**` |
| Itálico | `/texto/` | `*texto*` |
| Monospace | `=texto=` | `` `texto` `` |
| Ênfase | `__texto__` | `_texto_` |
| Tachado | `+texto+` | `~texto~` |

⚠️ `**texto**` **no início de linha** é título nível 2 em Org, não negrito.
Negrito inline é `*texto*` (um asterisco). Confundir isso gera cabeçalhos
acidentais — a armadilha Markdown→Org mais frequente.

## 5. Links

```org
[[https://exemplo.org][texto]]        # URL externa
[[id:75fb1dda-262e-494c-ba8c-1457710edcdb][nota]]   # interna (org-roam)
[[#section]]                             # âncara p/ seção deste arquivo
[[#Secão Dois]]                          # âncara p/ seção (case-sensitive)
[[file:../outro.pdf][pdf]]              # arquivo local
[[https://doi.org/10.1038/s41586-024-07763-9][doi]]
```

- Links internos por `id:` exigem que o UUID exista (senão fica quebrado).
- Âncoras por nome de seção são case-sensitive.

## 6. Blocos de código, citação, exemplo

```org
#+begin_src python
print("highlight por linguagem")
#+end_src

#+begin_quote
texto citado, sem interpretação.
#+end_quote

#+begin_example
monospace sem interpretação (índexe não precisa de tab).
#+end_example
```

- `#+begin_src` **deve** ter linguagem (`python`, `shell`, `text`, `org`…).
  `#+begin_src` sem lang perde syntax-highlight.
- `#+begin_example` / `#+begin_quote` não interpretam nada.

## 7. Listas, tabelas, checkboxes

```org
- item            # ou * item  (tábua)
- [ ] pendente     # checkbox aberta
- [x] feito        # checkbox cheia

+------------+-------------+
| Coluna A   | Coluna B    |
+------------+-------------+
| célula     | outra       |
+------------+-------------+
```

- Alinhamento de tabela: `|...||...|` (coluna spanning), `^` (cabeçalho),
  `:` (alinhamento de célula).

## 8. Notas de rodapé

```org
Um texto com uma ref[fn:1] aqui.

#footnotes
[fn:1] Texto da footnote.
```

- `#footnotes` marca a seção; não renomear (reservado).

## 9. Convenções org-roam (zettelkasten)

Ao criar/relacionar uma nota `.org`:

- Nome: `YYYYMMDDHHMMSS-slug.org` (ou `YYYYMMDD-slug.org`).
- `:ID:` UUID único; `:ROAM_REFS: cite:KEY` liga ao BibTeX.
- `#+FILETAGS:` com tags `:sepadas:por:dois:dois:`.
- Seção `* KEY` com `:NOTER_DOCUMENT: pdfs/arquivo.pdf` apontando o PDF.
- Seção `* Conexões org-roam` com `[[id:UUID][label]]` p/ notas relacionadas.
- Entrada BibTeX correspondente (ex. `@article{KEY, ...}`) deve existir, senão
  `cite:KEY` fica sem resolução no ivy-bibtex.

## 10. Armadilhas Markdown → Org (cheque sempre)

- `#`/`##`/`###` no início de linha em .org = **comentário**, não título. Título
  Org é `*`.
- `**texto**` no início de linha = título nível 2, não negrito.
- `#+begin_src` sem linguagem = sem syntax-highlight.
- Links `[[id:UUID]...]` sem o UUID no repositório = link quebrado.
- `#tag` (hashtag) ≠ `:tag:` (org-mode tag). Tags de arquivo são `:a:b:c:`.
- Tabela Markdown `|...|` não é reconhecida; Org precisa da linha `+---+`.

## 11. Comandos de verificação (grep)

```bash
# .org com título-Comentário acidental (começa com # mas quer ser seção)
grep -nE '^#+[[:space:]]' *.org

# blocos de código sem linguagem (perdem highlight)
grep -nE '^#+begin_src$' *.org

# links internos org-roam p/ conferir UUIDs existentes
grep -nE '\[\[id:[0-9a-f-]+\]' *.org

# tags de arquivo mal formadas (hashtag em vez de :tag:)
grep -nE '#[a-zA-Z]' *.org | grep -vE '^#+'
```

## 12. Validação final

Ant de "entregar" um `.org`:

1. Todos os títulos usam `*` (1–6), nunca `#`.
2. Blocos `#+begin_src` têm linguagem.
3. `[[id:UUID]]` apontam para UUIDs que existem no repositório.
4. `#+FILETAGS` usa `:tag:`; links `cite:` têm entrada BibTeX.
5. Propriedades ficam dentro de `:PROPERTIES:` … `:END:`.

Referência: The Org Manual (orgmode.org/manual/Headlines.org), Org Syntax
(orgmode.org/worg/org-syntax.html).
