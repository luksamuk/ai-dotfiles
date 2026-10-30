# musica — gerador de música local, agnóstico de modelo

Irmão do `diffuse` (imagem) e do `h3` (vídeo). Um binário só — o **audio.cpp** —
com várias famílias de música; a interface é sempre a mesma e o modelo se
escolhe com `--modelo`.

## Instalação

```bash
ln -s ~/git/ai-dotfiles/musica/musica ~/.local/bin/musica
```

Depende de: `~/git/audio.cpp/build/linux-cuda-release/bin/audiocpp_cli` com a
família compilada (ver seção Recuperação). O wrapper é symlink — a cópia daqui
**é** a ferramenta viva; não existe espelhamento.

## Estrutura

```
musica/
├── musica      # wrapper bash (a ferramenta)
├── specs/      # overrides de model_specs (OBRIGATÓRIO para ace_step)
└── README.md   # este arquivo
```

O conhecimento é separado: a ferramenta aqui, a **metodologia na skill**
`musica` (agnóstica) e os **detalhes por modelo** nas referências dela.

## Uso

```bash
# inventário: famílias + instalado/ausente + tamanho em disco
musica list

# gerar 30s com ACE-Step
musica --modelo ace --letra-file letra.txt --legenda-file legenda.txt --dur 30

# com controle simbólico (ACE-Step)
musica --modelo ace --letra-file L.txt --legenda-file C.txt \
       --dur 30 --bpm 140 --tom "D minor" --compasso 4/4 --idioma pt

# instrumental (Stable Audio não tem letra — não pede --letra)
musica --instrumental --legenda-file C.txt --modelo ace --dur 60
musica --instrumental --legenda "lo-fi, chuva, piano" --modelo stable --dur 30

# validar sem executar
musica ... --dry-run
```

| Flag | Default | Função |
|---|---|---|
| `--modelo` | `ace` | família (estado real: `musica list`) |
| `--letra` / `--letra-file` | — | letra com tags de seção (obrigatória só nas famílias com vocal) |
| `--legenda` / `--legenda-file` | — | descrição do som |
| `--dur` | 30 | duração alvo em segundos |
| `--seed` | 42 | reprodutibilidade |
| `--steps` | por família | passos de denoise (ace 8, base 50) |
| `--bpm` / `--tom` / `--compasso` | — | controle simbólico (ACE-Step) |
| `--idioma` | pt | idioma do vocal |
| `--instrumental` | off | sem voz |
| `--out` / `--out-dir` | `musica_{modelo}_{ts}_seed{seed}` em `$PWD` (env `MUSICA_OUT_DIR`) | caminho de saída — nunca sobrepõe |
| `--enhance` | off | LLM da frota reescreve a LEGENDA (config: fleet cfg, secao `musica:`) |
| `--enhance-with` | do fleet cfg | handle alternativo de enhance |
| `--enhance-only` | — | roda só enhance e encerra |
| `--specs` | `specs/` do wrapper (env `MUSICA_SPECS`) | override de model_specs |
| `--dry-run` | off | imprime o comando e encerra |

## Enhance e decisão de parâmetros (30/10/2026)

- `--enhance` reescreve **só a legenda** via frota (`localhost:12434/v1/chat/completions`,
  handle da seção `musica:` em `~/.local/share/fleet/enhance-models.yaml`, sem
  `max_tokens`). A letra entra como **contexto somente-leitura** — o prompt é
  explícito: proibido reescrever/ecoar lyrics.
- **BPM do ace_step** fica com `decision_model` (d1-2b) via `/v1/systemone`
  (N julgamentos `noul` binários, doutrina do snake — NUNCA `choice` com N
  opções, colapsa na posição). BPMs sugeridos pelo caption enhanced viram
  candidatos; corte cru em `decision_threshold`. Abaixo do corte: descarta e
  o motor usa o default. `--bpm` explícito na CLI pula a decisão inteira.
- Após o enhance o wrapper desembaça a frota (`llama-swap-cli unload`) — o
  LLM residente briga pelos mesmos 6GB.
- Backup do fleet config versionado em `llama-swap/enhance-models.yaml`
  (arquivo vivo: `~/.local/share/fleet/enhance-models.yaml` — sem PII).

## Regras operacionais (aprendidas na marra)

- **Sempre via `tu run`**, nunca terminal em background puro. Background +
  unified memory = morte silenciosa sem traceback (5/5 reproduções).
- **Verificar o WAV em disco depois.** Marcador de fim ≠ sucesso — um download
  já "concluiu" com exit 0 tendo baixado 84MB de 6,18GB.
- **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` é obrigatório** para os modelos que não
  cabem nos 6GB. Sem UMA o ACE-Step nem passa da alocação do DiT. Já está
  exportado dentro do wrapper.
- **O override de `model_specs` é obrigatório para o ACE-Step** (spec stock do
  upstream exige `dit_base_*` que não existe no repo oficial). O wrapper já
  defaulta pro `specs/` versionado aqui — não remova esse diretório.
- **Modelo é passado por caminho ABSOLUTO** (`$AUDIOCPP_HOME/models/...`). O
  dir do Stable (`Stable-Audio-3-Small-Music-GGUF`) é o nome que o instalador
  cria — o antigo `models/stable-audio-3` nunca existiu em disco.
- **`pgrep -x audiocpp_cli`**, nunca `pgrep -f` (auto-match do wrapper do shell).
- **Nunca `pkill -f 'padrão'` genérico** — o padrão casa com a própria linha de
  comando e mata o shell. Use `pkill -x <nome-exato>`.
- Antes de rodar, `llama-swap-cli unload` se houver LLM residente brigando
  pelos mesmos 6GB de VRAM.

## Estado das famílias

| apelido | família | pesos instalados | nota |
|---|---|---|---|
| `ace` | ace_step | sim (9,5G) | default, turbo 8 steps |
| `base` | ace_step | sim (mesmo dir do ace) | variante de 50 steps |
| `xl` | ace_step | não | exige ~12GB VRAM |
| `stable` | stable_audio | sim (1,6G, q8_0) | instrumental/loops/SFX, sem letra |
| `heart` | heartmula | não | adesão à letra por tag |

Estado em tempo real: `musica list`. Pesos moram em `~/.llama-models/audio/`
(symlink `models/` do audio.cpp). `phonon-*` e `Qwen3-TTS-12Hz` naquele diretório
**não são do musica** — são a stack do TTS (llama-swap/ptts), não mexer.

Aposentados: **MiniMax Music 3** (2026-09-15) — indomável, biblioteca reduzida
(entregou salsa quando pedimos MPB), e ~35x mais lento que o ACE-Step no mesmo
binário. O wrapper `music3` e o diretório `ai-dotfiles/music3/` foram removidos
em 2026-10-30; backup dos scripts em
`~/projects/ai/canteiros/arquivo/music3-wrapper-legacy-20260813.tar.gz`.
Ver a skill `musica` → `references/music3-retired.md`.