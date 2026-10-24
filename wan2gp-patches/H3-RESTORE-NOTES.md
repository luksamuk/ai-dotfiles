# H3 — estado e reinstalação (23/out/2026)

Os **pesos** foram removidos para liberar ~59 GiB. Código, patches, catálogo e finetunes ficam.
Reinstalar é rodar `scripts/install/h3-restore.sh`.

## O que fica no repo (nada disso é apagado)

| caminho | o que é |
|---|---|
| `h3/h3` | wrapper (`~/.local/bin/h3` é cópia dele) |
| `h3/build_h3_catalog.py` | gera o catálogo a partir do disco + curadoria (descrições, papéis, downloads) |
| `h3/lora_selector.py` | seletor: gate de ato no modelo, eixo de papel, capacity 1, pins |
| `wan2gp-patches/arquivos/models/minimax_h3/prompt_enhancer.py` | system prompts do enhance (refatorados: 3 campos, regras estruturais) |
| `wan2gp-patches/arquivos/finetunes/minimax_h3_*.json` | finetunes do WanGP para as modalidades |
| `scripts/install/h3_manifest_2026-10-23.json` | pesos: repo HF, nome exato, tamanho, sha256 oficial |
| `scripts/install/h3-restore.sh` | reinstala pesos + LoRAs, confere hash, recria symlinks, rebuilda catálogo |
| `scripts/install/h3-uninstall-pesos.sh` | remove só o que está no manifesto (dry-run por padrão) |

## Config do WanGP (aplicar após reinstalar)

`~/git/Wan2GP/wgp_config.json`:

```json
"video_output_codec": "libx264_lossless",
"video_container": "mkv",
"audio_output_codec": "flac"
```

Lossless: FFV1 (matematicamente sem perda) + FLAC. Backup do original em
`wgp_config.json.bak-lossless`. Custo: ~7x o tamanho do mp4.

## Manifestos locais (NÃO versionados — contêm nomes de LoRAs NSFW)

- `~/.local/share/h3/h3_loras_manifest_2026-10-23.json` — 15 LoRAs com URL Civitai + sha256
- `~/.local/share/h3/loras_minimax_h3.json` — catálogo vivo
- `~/.local/share/h3/nsfw_catalog_h3.json` — espelho pro enhancer (mesmos IDs)

Requisitos pra restauração: `~/.cache/huggingface/token` (HF) e `~/.civitai_key` (Civitai).

## Orfão conhecido

`MM-H3 - Sydney Sweeney.safetensors` (148 MB) não existe no Civitai com base H3 — sem
referência de download. Removido em 23/out por decisão do usuário (dispensável).

## Pegadinhas aprendidas (custaram tempo)

1. **Symlinks importam**: o pre-flight do wrapper procura `MiniMax-H3-FL2VA-pruned_int8_convrot`
   (sem `rank8`) e `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` na raiz de `ckpts/`. O restore recria os três.
2. **Nunca editar o wrapper com run ativo**: bash lê o script incrementalmente; a reescrita
   trunca o inode e a execução morre com erro de sintaxe em linha válida. Use
   `~/.hermes/scripts/h3_safe_patch.py` (escrita atômica + trava).
3. **O evict do LLM acontece antes do seletor**: sem evictar o decision model depois da
   seleção, ele ocupa ~2,7 GB de VRAM durante todo o render.
4. **Turbo**: só o `resized rank_21` funciona (314 MB, GuangyuanSD). Os turbos originais do
   lightx2v carregam por nome e geram ruído — falha silenciosa.
5. **`sliding_window_size`** default é 362 frames (15,08s). Acima disso o H3 quebra em janelas
   e o prompt precisa ser paragraph-per-window (`--window-size` exposto no wrapper).
