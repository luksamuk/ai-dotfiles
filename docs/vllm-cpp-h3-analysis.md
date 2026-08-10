# vllm.cpp + MiniMax H3 — Análise Paralela

## O que é

[mudler/vllm.cpp](https://github.com/mudler/vllm.cpp) — port C++20 do vLLM, pelo time do LocalAI.
Sem Python, sem PyTorch em runtime. 66 MiB de binário. Continuous batching, paged KV, 25+ arquiteturas.

## MiniMax H3 no vllm.cpp

H3 é um **diffusion transformer omni-modal** (33.1B params) — não é LLM autoregressivo.
Gera vídeo (24fps) + áudio stereo (32kHz) num único modelo. 50 steps de flow-matching.

### Componentes implementados (todos gated contra vLLM-Omni upstream)

| Componente | Status |
|---|---|
| DiT (33.1B) forward | DONE — CPU reference, parity-gated |
| Packed layout (fl2va + ref2va) | DONE — bit-exact |
| Scheduler (Euler ancestral) | DONE |
| Denoise loop | DONE |
| H3-Encoder (Qwen3-VL-32B-derived) | DONE — text tower + vision tower |
| Video VAE (FL2VA) | DONE — decoder ported, 8.9e-8 |
| Audio VAE (DAC/BigVGAN) | DONE — reimplemented, 4.2e-9 |
| Pipeline/Serving (/v1/videos) | DONE — async + sync endpoints |
| GGUF loader (ComfyUI format) | DONE — 535-tensor manifest verified |
| NVFP4 loader (safetensors) | DONE — 1051-tensor manifest verified |

### Endpoint /v1/videos

- `POST /v1/videos` — async, retorna job ID
- `POST /v1/videos/sync` — síncrono, retorna MP4 no body
- `GET /v1/videos/{id}` — status do job
- `GET /v1/videos/{id}/content` — MP4 bytes (video/mp4)
- Compatível com API da OpenAI Sora
- Suporta: text-to-video (t2va), first/last-frame (fl2va), reference video (ref2va)
- Input: prompt + opcional imagem de referência + opcional vídeo de referência + opcional áudio
- Output: MP4 com vídeo + áudio stereo

### Checkpoints quantizados disponíveis

| Formato | Repo HF | Tamanho | Notas |
|---|---|---|---|
| GGUF Q3_K_M (DiT) | realrebelai/MiniMax-H3_GGUFs | 15.6 GB DiT + 14.6 GB encoder + ~10 GB VAEs = **~41 GB** | ComfyUI format, 535 tensors |
| NVFP4 (safetensors) | lilcheaty/MiniMax-H3-NVFP4 | ~77.2 GB repo (3 variantes DiT) | FP4 packed, precisa sm_121+ nativo |

### Viabilidade na RTX 3050 6GB (sm_86 Ampere)

**Honestidade**: o projeto foi desenvolvido num Jetson Thor (GB10, 119 GiB unified, sm_121).
O tweet mostra 5s de vídeo em 28min a 4-bit no Jetson Thor.

**Na RTX 3050 6GB**:
- sm_86 é BUILD-supportado (portable kernels only — sem fast-path FP4/cutlass/marlin/FA2)
- GGUF Q3_K_M: ~41 GB working set → NÃO cabe em 6GB VRAM, mas pode usar RAM (31GB)
- NVFP4: sem tensor cores FP4 nativos (sm_121 only), vai dequantizar pra f32 = lento
- O DiT é 33.1B — mesmo a Q3_K_M (15.6GB) não cabe na VRAM, vai pra RAM com offload
- Estimativa: extremamente lento (ordens de magnitude mais lento que Jetson Thor)
- **Praticamente inviável pra uso produtivo na 3050**, mas tecnicamente possível como experimento

### Comparação com setup atual (Wan2GP + MiniMax H3)

- Wan2GP: pipeline Python, usa torch, ComfyUI/workflow
- vllm.cpp: C++ puro, sem torch, endpoint HTTP direto (/v1/videos)
- Vantagem vllm.cpp: sem dependências Python, binário único, API REST limpa
- Desvantagem: projeto muito novo (movimento rápido, breakage esperado), H3 foi landado hoje
- Wan2GP já funciona e está otimizado pra sua GPU

### Quando faria sentido migrar

1. **Nunca na 3050 6GB** — H3 precisa de muita memória, o ganho do C++ não compensa
2. **Se upgraded pra GPU com 24GB+** (VRAM suficiente pra Q3_K_M sem offload massivo)
3. **Se quiser servir H3 via API** (Wan2GP é workflow, não servidor)
4. **Se quiser integrar com LocalAI** (vllm.cpp é o backend do LocalAI)

### Repo

- `~/git/vllm.cpp/` — clonado (depth 50, main branch)
- 189 stars, 16 forks, 1816 commits, 160 branches
- Commit mais recente: 74b2ac9 (11 min atrás — extremamente ativo)
- Build: `cmake -B build -DGGML_CUDA=ON` (ou `VLLM_CPP_CUDA=ON`)
- CUDA arch: `-DVLLM_CPP_CUDA_ARCHITECTURES=86` pra RTX 3050
- Exemplos: `examples/minimax_h3_gen/` (CLI driver), `examples/minimax_h3_mux/` (muxer)
- Spec completa: `.agents/specs/minimax-h3.md`

### Próximos passos (se decidir explorar)

1. Buildar o vllm.cpp com CUDA sm_86
2. Baixar H3 GGUF Q3_K_M (~41GB total — DiT + encoder + VAEs)
3. Rodar `minimax-h3-gen` com `--dry-run` pra validar loader
4. Tentar render 480p, 5 frames, 2 steps (mínimo absoluto) pra ver se roda
5. Se rodar, testar via endpoint /v1/videos (precisa do server, não só o CLI)