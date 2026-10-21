# kokoro-82m — DESATIVADO 21/10/2026

## Veredito

Substituído pelo **Phonon 40M Portuguese** (`phonon-40m-pt`). Em PT-BR o Kokoro é
claramente inferior, e o usuário fechou o veredito: "Ele é claramente inferior"
(21/out/2026).

Números que sustentam (card da Gradium, PT-BR, 171 avaliadores / 2.000 votos em
escuta cega pareada):

| Modelo | Elo | Hard set WER | Seed-TTS WER |
|---|---:|---:|---:|
| phonon-40m-pt | **2144** | **2,9%** | **1,1%** |
| Kokoro 82M | 1633 | 4,1% | 1,3% |

Medido aqui na Navi (RTX 3050 6GB), mesma frase em PT-BR:

| Modelo | Áudio | Tempo | VRAM |
|---|---:|---:|---:|
| Kokoro 82M (audio.cpp, CUDA, q8_0) | 4,00 s | 2,48 s (wall, inclui load) | ~0,4 GB |
| Phonon 40M (ptts, CPU, q8_0) | 5,44 s | 0,52 s de geração | 0 |

Ou seja: o substituto é mais rápido, não ocupa VRAM e soa melhor. Não sobrou
argumento para manter o Kokoro.

## Como era servido

- Fragment: `models/kokoro-82m.yaml` (este arquivo ao lado, na `_disabled/`).
- Backend: `audiocpp_server` (audio.cpp, C++/CUDA) com
  `~/git/audio.cpp/server-kokoro.json`, `--model-spec-override model_specs`,
  porta fixa 8890, `cmdStop: pkill -f server-kokoro.json`.
- 4 presets: `pf_dora` (fem PT-BR), `pm_alex`, `pm_santa`, `af_heart`.
- G2P: eSpeak-ng (system lib, `pacman espeak-ng`) + dados eSpeak no pacote.

## Ressurreição

1. `mv models/_disabled/kokoro-82m.yaml models/`
2. Pesos: **mantidos em disco** em `~/.llama-models/audio/Kokoro-82M-GGUF/`
   (não foram apagados — decisão de limpeza ficou pendente com o usuário).
   Se um dia faltarem, re-baixar de `audio-cpp/audio.cpp-gguf`
   (`Kokoro-82M-GGUF/`, entrada `kokoro` em `model_specs/kokoro_tts.json`;
   `tools/model_manager_v2.py install kokoro_82m --models-root ~/.llama-models`).
3. Requer build do audio.cpp (`build/linux-cuda-release/bin/audiocpp_server`) e a
   lib de sistema `espeak-ng`.
4. `build-config.py` e confirmar que `kokoro-82m` volta em `/v1/models`.

## O que não era

Não tinha clonagem de voz (só presets paramétricos) e não passava por
`llama-server` — era o primeiro modelo de áudio nativo do audio.cpp na frota.
