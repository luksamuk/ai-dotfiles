# TOMBSTONE — retirado da frota 17/nov/2026 ~23:20 (dono: "hora de aceitar derrota").

## Veredito (com numeros medidos)

Arquitetura genuinamente interessante (looped block + mHC 2 streams + Gated DeltaNet hibrido 3:1,
ctx nativo 512K YaRN) mas NA 3050 6GB do dono so brilha em GPU — e a GPU e disputada pelo
EffectCraft (que e o uso real). Rotas tentadas e medidas (Q3_K_M, fork GuideboardLabs CUDA sem UI):
- full GPU -ngl 99: decode 17.8-17.9 t/s INVARIANTE 2K-16K, prefill 105-142 t/s, POREM degrau fixo
  de compute do grafo loopeado: 16K @94% VRAM, 24K+ OOM (+168 MB), e convive mal com desktop.
- offload parcial -ngl 28/30: 7.0-7.2 t/s e prefill colapsa p/ 14 t/s (sync GPU-CPU por token
  sobre 56 execucoes; 2 camadas na CPU JA custam quase o full-CPU) — omelete que nao compensa.
- CPU-only (-ngl 0, CUDA_VISIBLE_DEVICES=): decode 6.06 t/s (20 threads; 6 threads = 5.91),
  prefill 380-382 t/s @3.8K tokens — mas prefill e O(n^2) nas camadas full-attention: com
  ctx 131072 aberto, prompt ~26K tok levou >5 min (wall de attention em CPU no ctx longo).
  O overhead do harness pi (~20K tokens de system prompt + tools) nao cabia em 16K e em
  131K custa minutos por turno — agente lento por fisica, nao por config.
- Sem escada de effort (template sem reasoning_effort; dual via enable_thinking bool;
  single handle always-thinking). Tools nativas PASS (2 chamadas paralelas, finish=tool_calls).
- Text-only: visao e MTP fora do export do release final (guideboardlabs); GGUF com visao
  so existiu na era Preview (fragment _disabled/sensenova-loop-9b-vision, aposentado 31/out).

## Onde estao as coisas

- Pesos (MANTIDOS no disco): ~/.llama-models/sensenova-loop-9b/SenseNova-Loop-9B-Q3_K_M.gguf
  (4.642.406.752 B byte-checked). Apagar = decisao separada do dono (4.6 GB).
- Fork build: ~/projects/ai/sensenova-loop-fork (branch sensenova-loop, CUDA, server SEM UI
  embutida — receita no skill llama-swap-fleet backend-smoke.md). Var `sensenova_server` no
  config-base aponta la. DISPARADOR DE DESCARTE: ik_llama.cpp PR #2936 MERGEAR (tracker vigia
  a entrada SenseNova-Loop-9B) = fork fora; ai reposor sensenova-loop-9b pela FROTA stock.
- Re-download do GGUF: hf download guideboardlabs/SenseNova-Loop-9B-GGUF
  SenseNova-Loop-9B-Q3_K_M.gguf --local-dir ~/.llama-models/sensenova-loop-9b/
- Fragment de retorno: copiar o bloco abaixo do epitaph anterior (git history 6046f21) e
  devolver refs sn9 no config-footer (vars + evict_costs 2). Historico do Preview
  (aposentado 31/out, 16.2 t/s @16K ngl 88 Vulkan) no git log e no header do fragment antigo.

## Referencias

- https://github.com/ggml-org/llama.cpp/issues/31571 (FR upstream)
- https://github.com/ikawrakow/ik_llama.cpp/pull/2936 (PR ik, draft — trigger de retorno)
- Fork: mission-deny-the-mission/llama.cpp-sensenova-loop (espelhado no GuideboardLabs)
- EffectCraft: dirigido pelo Hermes + frota via skill artcraft-crafting-apps (NAO por este modelo)
