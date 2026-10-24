# Manifest de re-instalação — desinstalação de 19/10/2026 (Strata + hiatus H3)

Estado inicial do disco: 441G usados / 29G livres / 94% em btrfs nvme0n1p2.
Fontes verificadas em 19/10/2026 via API HF (HTTP 200 = re-baixável).

## FROTA — 4 modelos ocultados com `unlisted: true` + GGUFs deletados

Re-visibilizar = remover a linha `unlisted: true` do bloco correspondente em
`~/.config/llama-swap/config.yaml` + re-baixar o arquivo (comando abaixo). O
llama-swap v263 recarrega a config sozinho (-watch-config).

| id no fleet | bloco(s) | arquivo deletado | tamanho | fonte (verificada 200) |
|---|---|---|---|---|
| laguna-xs-2.1 | ~linha 308 | `.llama-models/laguna-xs-2.1/Laguna-XS-2.1-Q4_K_M.gguf` | 19G | `poolside/Laguna-XS-2.1-GGUF` |
| lfm2.5-24b-a2b + -fast | ~1484, ~1555 | `.llama-models/lfm25-24b-a2b/LFM2.5-24B-A2B-APEX-I-Compact.gguf` | 11G | `mudler/LFM2.5-24B-A2B-APEX-GGUF` |
| whittle-35b + fast | ~1711, ~1783 | `.llama-models/whittle-35b/Whittle-Qwen-3.8-35B-A3B-ONYX-compact.gguf` | 22G | `el4/Whittle-Qwen-3.8-35B-A3B-ONYX-GGUF` |
| ornith-1.5-35b-heretic | ~2011 | `.llama-models/ornith-heretic/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Compact.gguf` + `mmproj-Ornith-1.5-35B-A3B-Heretic-MTP-BF16.gguf` | 17G + 861M | `SC117/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF` |

Comando padrão de re-download:
```sh
hf download <repo> "<nome-do-arquivo>" --local-dir ~/.llama-models/<subdir>/
```
Obs: Ornith compact tem sha anotado na própria config do fleet (37f67861…e0e7).
Importante para ornith: re-baixar TAMBM o mmproj junto.

## H3 (Wan2GP) — hiatus: pesos deletados, TODOS re-baixáveis

### Deletado (recuperável, fontes vivas)
- `~/git/Wan2GP/models/minimax_h3/checkpoints/minimax_h3_fl2va_pruned-Q4_K.gguf` (11G) ← `unsloth/MiniMax-H3-GGUF`
- `~/git/Wan2GP/models/minimax_h3/checkpoints/minimax_h3_ref2va_pruned-Q4_K.gguf` (11G) ← `unsloth/MiniMax-H3-GGUF`
- `.../checkpoints/vae/{MiniMax-H3-video_vae_fp16.safetensors 4,9G, MiniMax-H3-audio_vae_fp32.safetensors 578M}` ← `Abiray/MiniMax-H3-GGUF`
- Te ativo: `~/git/Wan2GP/ckpts/Qwen3-VL-32B-Instruct/` (14G, contém o `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf`) e `~/git/Wan2GP/ckpts/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` ← `pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF` (+ dump oficial `Qwen/Qwen3-VL-32B-Instruct`)
- LoRAs verbatim re-baixáveis de `loras/minimax_h3/`: fontes `EllaPriest45/MinimaxH3_Actions`, `Sentinel7/h3` (mirrors HM), `lightx2v/Minimax-h3-Turbo`
- Receita de volta: `bash ~/git/ai-dotfiles/scripts/install/reinstall-h3-enxuto.sh` (+ passo 4 do `reinstall-h3.sh` se quiser o dump oficial do Qwen3-VL-32B). Doutrina no skill wan2gp.

### MANTIDO (fonte morta ou sintetizado — NÃO re-baixável)
- `~/git/Wan2GP/ckpts/minimax_h3/` (3,3G): latent upscaler 3D bf16 + ups — fonte `DmitryDB/MiniMax-H3-Quants` foi APAGADA/gateada da HF (401/404 com token em 19/10). Única cópia é esta.
- `loras/minimax_h3/` mantidos: `h3-cumshot-final-step16600-r16-k3nk` (custom k3nk), `minimax_h3_PDMD_DMAD_4step_r256` (fonte civitai não verificável), `..._fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16` (default da casa, sintetizado por resize local) e `..._alpha16_v0.1` (variante sintetizada).
- Aparato H3 inteiro: `~/.local/share/h3/` (catálogos, rerank, triggers — 328K), wrapper `~/.local/bin/h3`, catálogo `nsfw_catalog_h3.json` (41 entradas).

## STRATA
- Terminal + engine + modelo Qwen3.6-35B-A3B UD-Q4_K_XL (22,9G): baixados por conta do próprio strata (`strata launch --yes --no-tui`). Para desinstalar depois: apagar a pasta de install no home + os models.
- Teste é gate para o Gyro-S (58,5G).
## RESULTADO FINAL (19/10, pós-execução)
- LoRAs: 45 deletados (fontes em `h3_loras_manifest_2026-10-19.json` ao lado), 18 mantidos (custom/sintetizados/fonte incerta).
- TE: mantido em ckpts/Qwen3-VL-32B-Instruct/ (fonte nao confirmada).
## RESTOCK 19/10 (mesma tarde, a decisão mudou)
- Strata desistido (ver qwen36); engine+modelo vanilla removidos.
- Os 4 GGUFs da frota RE-DOWNLOAD em 19/10 (~20 min): tamanhos idênticos aos deletados (23.49/20.27/11.21/17.33+0.90 G) e `unlisted: true` removidos — fleet de volta ao estado pré-experimento.
- H3 permanece em hiatus (pesos fora; mantidos: latent upscaler, TE 14.6G, PDMD, k3nk, sintetizados, aparato).

## SACRIFÍCIO 24/10/2026 — Gyro-S (2º sacrifício dos mesmos 4 blocos)
- Motivo: acomodar o Gyro-S (85GB do plano: arquivo 58.5 + experts.bin ~26 + margem) — disco em 102G livres pós-deleção.
- Deletados (fontes re-verificadas HTTP 200 em 24/10/2026): LFM2.5-24B-A2B-APEX-I-Compact.gguf (11G) e
  Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Compact.gguf + mmproj-...-BF16.gguf (18G).
- Diferença da rotina de 19/10: NÃO movido fragment nem editado rodapé — os blocos ficaram no config.yaml
  com `unlisted: true` inline + comentário de re-download em cada um (3 blocos: lfm2.5-24b-a2b, lfm2.5-24b-a2b-fast, ornith-1.5-35b-heretic).
- Re-baixável: mesmos comandos `hf download` anotados inline no config (mudler/APEX-GGUF e SC117/APEX-GGUF).
- whittle-35b e laguna-xs-2.1: blocos e arquivos JÁ não existem (sacrifício anterior limpou - verificar antes de re-ocultar).

## DESINSTALAÇÃO À FORÇA — Gyro-S (rotina de emergência, pedida pelo user 24/10)
Pesos moram TODOS em ~/.local/share/strata/Strata-data/ (btrfs single, root nvme0n1p2):
- ~/.local/share/strata/Strata-data/models/gyro-Gyro-S/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf  (58,5G; marca .done ao completar)
- ~/.local/share/strata/Strata-data/packs/gyro-gyro-s/ — dense.bin (~1,9G) + tokenizer/ + hadamard.txt + index.txt;
  experts.bin ~26G ESCRITO pelo prepare 24/10 (experts.bin.tmp→experts.bin; confirmado em tempo real — deletável depois:
  -26G e o motor roda --mmap-experts in-place, custa ~15-25% de throughput; regenera rodando o pack de novo).
  experts ficam no GGUF também, o pack copia+reordena (1 leitura por expert em vez de 3). dense.bin é cópia dos densos (1,54G).
- config do launch: ~/.local/state/strata/launch/gyro-s-32k.json | log: gyro-s-32k*.log
- engine 0.1.48: ~/.local/share/strata/components/engine/0.1.48-linux-x86_64-cuda-cuda13 (~1-2G)
One-shot (devolve ~85-87G INSTANTÂNEO — btrfs sem snapshots, provado na deleção de 24/10):
  systemctl --user stop strata-gyro-test 2>/dev/null
  rm -rf ~/.local/share/strata/Strata-data/models/gyro-Gyro-S ~/.local/share/strata/Strata-data/packs/gyro-s
  rm -f ~/.local/state/strata/launch/gyro-s-32k.json
Terminal intocado = strata (--model gyro-s) relança tudo; para sair do Gyro DE VEZ: remover também ~/.local/share/strata (Terminal+engine+python 3.12 do Terminal, ~2-3G) e o symlink ~/.local/bin/strata.
Se btrfs mentir no ENOSPC depois disso: `sudo btrfs balance start -dusage=50 /` (pedir o user).

## PONTO OPERACIONAL (24/10, engine 0.1.48) — o que abre e o que morre
- ABRE 2/2 (preset `gyro-32k`): `strata launch --model gyro-s --context 32768 --kv k8v4
  --set vram-reserve-mib=200 --set prefill=256 --set kv-resident=2048 --parallel 1
  --vision off --yes --no-tui --tune skip --port 18123` + env CUDA_MODULE_LOADING=LAZY GGML_CUDA_FORCE_MMQ=1
  ≈ 5.77G VRAM (94%), 60-62°C, cache 513 slots, KV pinada 0.30G.
- ABRE (sólido, benchado N=5): 16K (482 slots) — chat 17.87 t/s (piso 15.9, CV 3.9%), 4K prompt 257 t/s, agent decode 16.3, hit 70.8%, 0 falhas.
- MORRE em qualquer variação dos 3 knobs (ladder 131K/64K/caíram; -vram-elastic/-expert-vram não curam):
  wall = alocador do prompt path no raso do VMM (free-at-failure constante ~44-48 MiB; buffers 9/18/36 por ctx).
  Detalhes completos na skill strata-gyro-6gb. Bench em gyro-bench-{1,2,32k} (scratch).


## VEREDITO DO EXPERIMENTO GYRO-S (24/10/2026) - NAO OPERACIONAL NA NAVI
- RODA (world-datapoint 1-bit numa 6GB laptop): 17-21 t/s decode, tools paralelas OK,
  vision via reader CPU, prefill 120-257 t/s. Engine Strata 0.1.48 / Terminal 0.2.4.2.
- NAO OPERA PRO USO DA CASA (3 inaceitaveis do user):
  (1) VRAM 93-94% = mata coexistencia (o guard 512 depende de ninguem encostar);
  (2) RAM 97% (21G page-locked + zram pegando a bolha do sistema);
  (3) CONTEXTO 16K = o teto estructural (a parede do alocador no raso do VMM:
      free >=550-600 no arranque do request = passa, abaixo = corte nas trocas de
      fase; 64K/128K NUNCA carregam e 16K nao serve agente - piso da casa = 64K).
- RECEITA FINAL (a que rodava melhor, se um dia rolar re-teste): fragment
  unlisted em ai-dotfiles/llama-swap/models/qwen3.8-flash-next.yaml + json em
  llama-swap/strata/gyro-s-16k.json: cache capado 200, reserve 512, resident-budget
  21.0, kv k8v4, kv-resident 2048, prefill 256 + envs LAZY/MMQ. O modo residente
  mata o thrash de kernel (4 faults vs 10547) a custo de RAM; o multi-turno so
  passa com a regra do free.
- PESOS (60 GB em ~/.local/share/strata/Strata-data/): re-downloadavel
  (agentionai/Qwen3.8-Flash-Next-Gyro-GGUF, HTTP 200 verificado 24/10), mmproj e
  draft incluidos. DECISAO DE DELECAO: pendente com o user.
- MATERIAIS DE ISSUE (Strata repo): a parede de contexto 6GB + o thrash
  do file-tier + o #2065 paging - forense completa na skill strata-gyro-6gb.
