# LFM2.5-Audio (audio.cpp) — DESTRUÍDO 27/out/2026, pedido do dono ("fode com o fragmento")

Substituído pelo LFM3-Audio-2B via realtime WS /v1/realtime no llama.cpp
(branch fleet-liquid, PR ggml-org/llama.cpp#31269). O experimento provou o
pipeline s2s: single-request audio->audio verificado (voz PT 45s -> text+audio
em 1.0s) e o modo duplex do testchat nasceu dele.

## O que foi destruído
- Weights (~1.83 GB): ~/.llama-models/lfm2.5-audio/ (backbone, mmproj,
  vocoder, tokenizer — quant Q8_0, 4 arquivos)
- Lane fora do fleet (fragment em _removed/)

## Reviver (se um dia doer)
```sh
hf download LiquidAI/LFM2.5-Audio-1.5B-GGUF LFM2.5-Audio-1.5B-Q8_0.gguf \
  mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf \
  tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf --local-dir ~/.llama-models/lfm2.5-audio/
# + git mv do fragment de _removed/ p/ models/ (o git history tem o YAML inteiro:
# commit f26971f/8552851) + config server-lfm25audio.json (git mostrar audio.cpp)
```
