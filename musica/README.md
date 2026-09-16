# musica — gerador de música local, agnóstico de modelo

Irmão do `diffuse` (imagem) e do `h3` (vídeo). Um binário só — o **audio.cpp** —
com várias famílias de música; a interface é sempre a mesma e o modelo se
escolhe com `--modelo`.

## Instalação

```bash
ln -s ~/git/ai-dotfiles/musica/musica ~/.local/bin/musica
```

Depende de: `~/git/audio.cpp/build/linux-cuda-release/bin/audiocpp_cli` com a
família compilada.

## Estrutura

```
musica/
├── musica      # wrapper bash (a ferramenta)
└── README.md   # este arquivo
```

O conhecimento é separado: a ferramenta aqui, a **metodologia na skill**
`musica` (agnóstica) e os **detalhes por modelo** nas referências dela.

## Uso

```bash
# gerar 30s com ACE-Step
musica --modelo ace --letra-file letra.txt --legenda-file legenda.txt --dur 30

# com controle simbólico (ACE-Step)
musica --modelo ace --letra-file L.txt --legenda-file C.txt \
       --dur 30 --bpm 140 --tom "D minor" --compasso 4/4 --idioma pt

# instrumental
musica --instrumental --legenda-file C.txt --modelo ace --dur 60

# ver famílias disponíveis / validar sem executar
musica --lista
musica ... --dry-run
```

| Flag | Default | Função |
|---|---|---|
| `--modelo` | `ace` | família (ver `--lista`) |
| `--letra` / `--letra-file` | — | letra com tags de seção |
| `--legenda` / `--legenda-file` | — | descrição do som |
| `--dur` | 30 | duração alvo em segundos |
| `--seed` | 42 | reprodutibilidade |
| `--steps` | por modelo | passos de denoise |
| `--bpm` / `--tom` / `--compasso` | — | controle simbólico (ACE-Step) |
| `--idioma` | pt | idioma do vocal |
| `--instrumental` | off | sem voz |
| `--specs` | — | dir de override de `model_specs` |
| `--dry-run` | off | imprime o comando e encerra |

## Regras operacionais (aprendidas na marra)

- **Sempre via `tu run`**, nunca terminal em background puro. Background +
  unified memory = morte silenciosa sem traceback (5/5 reproduções).
- **Verificar o WAV em disco depois.** Marcador de fim ≠ sucesso — um download
  já "concluiu" com exit 0 tendo baixado 84MB de 6,18GB.
- **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` é obrigatório** para os modelos que não
  cabem nos 6GB. Sem UMA o ACE-Step nem passa da alocação do DiT. Já está
  exportado dentro do wrapper.
- **`pgrep -x audiocpp_cli`**, nunca `pgrep -f` (auto-match do wrapper do shell).
- **Nunca `pkill -f 'padrão'` genérico** — o padrão casa com a própria linha de
  comando e mata o shell. Use `pkill -x <nome-exato>`.

## Estado das famílias

| apelido | família | pesos instalados | nota |
|---|---|---|---|
| `ace` | ace_step | sim (9,4GB, bf16) | default |
| `base` | ace_step | sim | variante de 50 steps |
| `xl` | ace_step | não | exige ~12GB VRAM |
| `stable` | stable_audio | não | sem letra |
| `heart` | heartmula | não | adesão por tag |

Aposentados: **MiniMax Music 3** (2026-09-15) — indomável, biblioteca reduzida
(entregou salsa quando pedimos MPB), e ~35x mais lento que o ACE-Step no mesmo
binário. Ver a skill `musica` → `references/music3-retired.md`.
