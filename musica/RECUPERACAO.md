# musica — repositório de ferramentas de música local

Este repositório guarda as **ferramentas** de geração de música local, no mesmo
padrão do `diffuse/` (imagem) e do `h3/` (vídeo): um binário, várias famílias,
interface estável.

## Estrutura

```
musica/
├── musica              # wrapper bash (a ferramenta)
└── README.md           # este arquivo
```

Symlink: `ln -s ~/git/ai-dotfiles/musica/musica ~/.local/bin/musica`

## Modelos suportados (via audio.cpp)

| apelido | família | status |
|---|---|---|
| `ace` | ace_step (turbo) | **default**, instalado |
| `base` | ace_step (base, 50 steps) | instalado |
| `xl` | ace_step XL | não instalado (exige ~12GB VRAM) |
| `stable` | stable_audio | não instalado |
| `heart` | heartmula | não instalado |

## Recuperação após reinstalação

### 1. O binário

O `audio.cpp` precisa ser compilado **com a família**. O build padrão do repo
pode não incluir todas:

```bash
cd ~/git/audio.cpp
scripts/build_linux.sh --backend cuda --cuda-arch native \
  --model-set custom --models ace_step --target audiocpp_cli -j 8
```

Só os `.cpp` da família entram no link se o ggml já estiver compilado — medido
**14 segundos**. Sempre fazer backup do binário antes:
`cp -a bin/audiocpp_cli bin/audiocpp_cli.bak`.

### 2. Pesos do ACE-Step

**NÃO usar o repo de pacotes `audio-cpp/audio.cpp-gguf`** — está quebrado no CDN
(~5 KB/s; e pior, "conclui" com exit 0 sem baixar nada). Usar o repo oficial:

```bash
DEST=~/git/audio.cpp/models/Ace-Step1.5
BASE=https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/main
mkdir -p "$DEST"
for f in \
  "acestep-v15-turbo/model.safetensors" \
  "acestep-5Hz-lm-1.7B/model.safetensors" \
  "Qwen3-Embedding-0.6B/model.safetensors" \
  "vae/diffusion_pytorch_model.safetensors"; do
  mkdir -p "$DEST/$(dirname $f)"
  curl -sSL --retry 5 -C - -o "$DEST/$f" "$BASE/$f" &
done
wait
```

Medido a **8-11 MB/s** (~10GB em ~20 min baixando os 4 em paralelo). O `hf` CLI
trava no protocolo Xet nessa rota — usar curl.

### 3. Override de spec (obrigatório)

O `model_specs/ace_step.json` do upstream exige dois artefatos que o repo
oficial não tem. Sem o override o carregador não sobe.

O override pronto está versionado junto do projeto da campanha:
`~/projects/ai/canteiros/specs/ace_step.json`. Ele remove:
- `dit_base_config`, `dit_base_weights`, `dit_base_silence_latent` — o DiT
  `base` não existe no repo oficial (nem consta como opcional na doc upstream)

E o repo só publica `silence_latent.pt`, não `.safetensors`. Converter:

```python
import torch
from safetensors.torch import save_file
d = torch.load('.../acestep-v15-turbo/silence_latent.pt', map_location='cpu', weights_only=False)
save_file({'silence_latent': d}, '.../acestep-v15-turbo/silence_latent.safetensors')
```

Passar com `--model-spec-override <dir>` (ou `--specs` no wrapper).

### 4. Variável de ambiente crítica

`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — o planner LM (3,7GB) fica residente e o
DiT pede mais 4,3GB, o que não cabe em 5,8GB. UMA deixa transbordar pra RAM.
**Sem ela o ACE-Step não roda em 6GB**, mesmo quantizado. O wrapper já exporta.

## Requisitos de sistema

- `~/git/audio.cpp` com o patch CCCL 3.4 no `top-k.cu` (CUDA 13.3 + CCCL 3.4)
- `ffmpeg`/`ffprobe` para medir e converter
- `tu` para rodar jobs (nunca terminal background puro)

## Ver também

Skill `musica` — método, protocolo de teste e referências por modelo:
`references/ace-step.md` (default), `references/music3-retired.md` (aposentado).
