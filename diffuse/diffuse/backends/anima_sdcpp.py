"""sd.cpp backend — Anima (CircleStone Labs × Comfy Org) via sd-cli.

30/10/2026. Pesos em ~/.llama-models/anima/ (layout de frota — um só jogo servindo
o diffuse E os fragments anima-* do llama-swap, que rodam o sd-server de
examples/server com a mesma árvore). sd-server = OpenAI-compatible
(/v1/images/generations) — por isso a Anima é CIDADE DE PRIMEIRA CLASSE na frota.

Receitas do card do criador (Civitai #2458426): turbo CFG 1 @8-12 steps;
aesthetic/base/preview 30-50 steps CFG 4-6 (aesthetic melhor em 3). Sampler
default da casa: er_sde (neutro, flat, sharp); euler_a = linhas mais suaves
2.5D. Resoluções 512²-1536². Uncensored por default (tier Unbound).
TE Qwen3-0.6B-Base Q8_0 + VAE qwen_image compartilhados (anima/common).
"""
from __future__ import annotations

from pathlib import Path

from diffuse.models import MODELS
from diffuse.backends.sdcpp import SD_CLI, _require_sdcli

ANIMA_ROOT = Path.home() / ".llama-models" / "anima"


def anima_paths(model_name: str) -> dict[str, Path]:
    info = MODELS[model_name]
    paths = {
        "transformer": ANIMA_ROOT / info["weights_subdir"] / info["diffusion_file"],
        "llm": ANIMA_ROOT / info["te_file"],
        "vae": ANIMA_ROOT / info["vae_file"],
    }
    faltando = [k for k, p in paths.items() if not p.exists()]
    if faltando:
        raise FileNotFoundError(
            f"pesos do {model_name} faltando: {', '.join(faltando)}"
            f"\n  raiz: {ANIMA_ROOT}"
        )
    return paths


def build_command(model_name, prompt, seed, steps, width, height, output_path,
                  cfg_scale=None, sampler=None):
    sd = _require_sdcli()
    p = anima_paths(model_name)
    info = MODELS[model_name]
    cfg = cfg_scale if cfg_scale is not None else info.get("cfg_scale", 1.0)
    sm = sampler or info.get("sampler", "er_sde")
    return [
        str(sd),
        "--diffusion-model", str(p["transformer"]),
        "--llm", str(p["llm"]),
        "--vae", str(p["vae"]),
        "-p", prompt,
        "-W", str(width), "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "--sampling-method", sm,
        "--seed", str(seed),
        "-o", str(output_path),
        "--offload-to-cpu",
    ]


def generate_image_anima(model_name, prompt, seed, steps, width, height, output_path):
    """Roda o sd-cli e devolve (caminho do PNG, segundos de parede)."""
    import subprocess
    import time
    from diffuse.backends.sdcpp import shlex_quote_part

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(model_name, prompt, seed, steps, width, height, output_path)
    print(f"     cmd: {' '.join(shlex_quote_part(c) for c in cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0 or not output_path.exists():
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(
            f"sd-cli falhou (exit {proc.returncode}) ao gerar {output_path}\n{tail}"
        )
    return output_path, wall
