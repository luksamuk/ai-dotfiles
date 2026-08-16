#!/usr/bin/env python3
"""
MiniMax Music 3 — Gerador de música via diffusers (low-VRAM)

Uso:
  python generate.py --lyrics-file lyrics.txt --prompt "descrição musical"
  python generate.py --lyrics "[Verse]\n..." --prompt "genre..." --duration 60
  python generate.py --lyrics-file lyrics.txt --prompt-file caption.txt --seed 7
  python generate.py --help

Requer:
  venv: ~/git/minimax-music3/.venv
  modelo: ~/git/minimax-music3/models/
  diffusers PR: huggingface/diffusers#14456 (commit dafe3733)

Estratégia de VRAM/RAM (RTX 3050 6GB, 31GB RAM):
  1. enable_auto_cpu_offload — move componentes pequenos pra GPU quando ativos
  2. apply_group_offloading leaf_level + stream + disk offload nos modelos grandes
     (transformer 9GB + language_model 16GB) — pesos vão pro disco, não pra RAM
  3. memory_reserve_margin baixo (1GB) pra maximizar uso de VRAM
  4. bf16 em tudo
  5. Só o grupo de camadas ativo fica na VRAM; o resto está em disco

Importante: lento é esperado (disk I/O), mas VRAM não fica ociosa — margin baixo
garante que o máximo possível de camadas cabe na GPU de cada vez.
"""

import argparse
import os
import sys
import time

MODEL_DIR = os.path.expanduser("~/git/minimax-music3/models")
OUTPUT_DIR = os.path.expanduser("~/git/minimax-music3/output")


def parse_args():
    p = argparse.ArgumentParser(
        description="MiniMax Music 3 — text-to-music generation (low-VRAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python generate.py --lyrics "[Verse]\\nMorning light\\n[Chorus]\\nSoftly breathing" --prompt "acoustic pop, warm female vocals, fingerpicked guitar"
  python generate.py --lyrics-file lyrics.txt --prompt-file caption.txt --duration 120 --seed 42
  python generate.py --lyrics-file lyrics.txt --prompt "dark synthwave, 128 BPM, male vocals" --duration 30 --dry-run
        """,
    )
    p.add_argument("--lyrics", type=str, default=None, help="Lyrics text (com tags [Verse], [Chorus], etc.)")
    p.add_argument("--lyrics-file", type=str, default=None, help="Arquivo com lyrics")
    p.add_argument("--prompt", type=str, default=None, help="Descrição musical (Structured Caption)")
    p.add_argument("--prompt-file", type=str, default=None, help="Arquivo com descrição musical")
    p.add_argument("--duration", type=float, default=60.0, help="Duração em segundos (default: 60, max: 360)")
    p.add_argument("--seed", type=int, default=7, help="Seed (default: 7)")
    p.add_argument("--output", type=str, default=None, help="Arquivo de saída (default: output/<timestamp>.wav)")
    p.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Diretório do modelo")
    p.add_argument("--dry-run", action="store_true", help="Validar config sem gerar")
    p.add_argument("--no-offload", action="store_true", help="Desativar group offloading (só auto CPU offload)")
    p.add_argument("--offload-dir", type=str, default="/home/alchemist/git/minimax-music3/offload_cache", help="Diretório para disk offload (default: ~/git/minimax-music3/offload_cache — NO SSD, não tmpfs)")
    p.add_argument("--memory-margin", type=str, default="1GB", help="Margem de VRAM livre (default: 1GB, baixo pra maximizar uso)")
    p.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    return p.parse_args()


def load_text(source: str, file_path: str, name: str) -> str:
    if source is not None:
        return source
    if file_path is not None:
        with open(file_path, "r") as f:
            return f.read().strip()
    raise ValueError(f"Erro: {name} obrigatório. Use --{name.lower().replace(' ', '-')} ou --{name.lower().replace(' ', '-')}-file")


def main():
    args = parse_args()

    # Validar inputs
    lyrics = load_text(args.lyrics, args.lyrics_file, "Lyrics")
    prompt = load_text(args.prompt, args.prompt_file, "Prompt")

    if args.duration > 360:
        print("Aviso: duração máxima é 360s (6 min). Limitando.")
        args.duration = 360.0
    if args.duration < 1:
        print("Aviso: duração mínima é 1s. Limitando.")
        args.duration = 1.0

    # max_new_tokens: 25 frames por segundo
    max_tokens = int(args.duration * 25)

    # Output path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_music3_s{args.seed}_{int(args.duration)}s.wav")

    # Dry run
    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Model dir: {args.model_dir}")
        print(f"Lyrics ({len(lyrics)} chars):")
        print(f"  {lyrics[:200]}{'...' if len(lyrics) > 200 else ''}")
        print(f"Prompt ({len(prompt)} chars):")
        print(f"  {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"Duration: {args.duration}s ({max_tokens} tokens @ 25fps)")
        print(f"Seed: {args.seed}")
        print(f"Output: {output_path}")
        print(f"Offload: {'auto CPU offload only' if args.no_offload else 'auto CPU offload + group offloading (leaf_level, stream, disk)'}")
        print(f"Disk offload: {args.offload_dir}")
        print(f"Memory margin: {args.memory_margin}")
        print("=== Encerrando sem gerar (--dry-run) ===")
        return

    # Verificar modelo
    if not os.path.exists(args.model_dir):
        print(f"Erro: modelo não encontrado em {args.model_dir}")
        print("Baixe com: hf download MiniMaxAI/MiniMax-Music3 --local-dir <path>")
        sys.exit(1)

    # Imports pesados (só quando vai gerar de verdade)
    print("=== Carregando dependências ===", flush=True)
    import torch
    import soundfile as sf
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.hooks import apply_group_offloading

    # Carregar pipeline com auto CPU offload
    print("=== Carregando MiniMax Music 3 ===", flush=True)
    print(f"Model: {args.model_dir}", flush=True)
    print(f"Offload: auto CPU offload + {'NO group offload' if args.no_offload else 'group offloading (leaf_level, stream, disk)'}", flush=True)
    print(f"Memory margin: {args.memory_margin} (baixo = mais VRAM usada)", flush=True)

    # Criar diretório de disk offload
    os.makedirs(args.offload_dir, exist_ok=True)

    # Estratégia: group offloading SEM disk offload e SEM auto_cpu_offload
    # - auto_cpu_offload compete com group offloading (device mismatch)
    # - disk offload tem bug com Qwen3 sub-modules (layernorm fica na CPU)
    # - group offloading leaf_level sozinho: pesos na RAM (26GB), só leaf ativo na VRAM
    # - 31GB RAM total - 26GB modelo = 5GB pro resto. Apertado mas viável.
    if args.no_offload:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=args.device, memory_reserve_margin=args.memory_margin)
    else:
        manager = ComponentsManager()  # sem auto offload

    pipe = ModularPipeline.from_pretrained(args.model_dir, components_manager=manager, local_files_only=True)

    # Carregar e offloadar componente por componente — evita ter 26GB na RAM ao mesmo tempo
    if not args.no_offload:
        print("=== Carregando componentes com disk offload individual ===", flush=True)
        print(f"Disk offload dir: {args.offload_dir}", flush=True)

        # Componentes pequenos primeiro (carregam todos de uma vez, ~1.5GB total)
        small_components = [n for n in pipe.pretrained_component_names
                           if n not in ('transformer', 'language_model')]
        if small_components:
            print(f"  Carregando componentes pequenos: {small_components}", flush=True)
            pipe.load_components(names=small_components, dtype=torch.bfloat16)
            # Mover componentes pequenos pra GPU (cabeem ~5GB sobrando)
            for name in small_components:
                comp = getattr(pipe, name, None)
                if comp is not None and hasattr(comp, 'to'):
                    comp.to(torch.device(args.device))
                    print(f"    {name} -> {args.device}", flush=True)

        # Transformer (9GB) — low_cpu_mem_usage evita pico de RAM durante offload
        if 'transformer' in pipe.pretrained_component_names:
            print("  Carregando transformer (9GB)...", flush=True)
            pipe.load_components(names='transformer', dtype=torch.bfloat16)
            print("  Group offloading transformer (leaf_level, stream, low_mem)...", flush=True)
            apply_group_offloading(
                pipe.transformer,
                onload_device=torch.device(args.device),
                offload_type="leaf_level",
                use_stream=True,
                low_cpu_mem_usage=True,
            )
            print("  Transformer offloaded (leaf_level, low_mem)", flush=True)

        # Language model (16GB) — low_cpu_mem_usage evita pico de RAM durante offload
        if 'language_model' in pipe.pretrained_component_names:
            print("  Carregando language_model (16GB)...", flush=True)
            pipe.load_components(names='language_model', dtype=torch.bfloat16)
            print("  Group offloading language_model (leaf_level, stream, low_mem)...", flush=True)
            apply_group_offloading(
                pipe.language_model,
                onload_device=torch.device(args.device),
                offload_type="leaf_level",
                use_stream=True,
                low_cpu_mem_usage=True,
            )
            print("  Language model offloaded (leaf_level, low_mem)", flush=True)

        print(f"  Disk offload completo", flush=True)

    else:
        pipe.load_components(dtype=torch.bfloat16)

    # Gerar
    print(f"=== Gerando música ===", flush=True)
    print(f"Duration: {args.duration}s ({max_tokens} tokens)", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    start_time = time.time()

    audio = pipe(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=args.duration,
        generator=torch.Generator(args.device).manual_seed(args.seed),
        output="audios",
    )[0]

    elapsed = time.time() - start_time
    print(f"=== Geração concluída em {elapsed:.0f}s ({elapsed/60:.1f}m) ===", flush=True)

    # Salvar
    if hasattr(audio, 'cpu'):
        audio = audio.float().cpu().numpy()
    sf.write(output_path, audio.T, pipe.sampling_rate)
    print(f"=== Salvo: {output_path} ===", flush=True)
    print(f"    Sample rate: {pipe.sampling_rate} Hz", flush=True)


if __name__ == "__main__":
    main()