"""sd.cpp backend — Fibo 2.0 (BRIA) via subprocess no sd-cli.

Contexto (24/out/2026): o sd.cpp tinha sido aposentado do diffuse na F5
(`7eebd12`, "Wan2GP canônico pra sempre"), mas o Wan2GP não tem handler de Fibo:
o transformer do Fibo 2.0 é família Flux 2 (~8B, 44 blocos, dim 3584) e o
`flux3` do WanGP é amarrado às dims do Klein. O sd.cpp upstream mergeou suporte
nativo ao Fibo 2.0 (`2e2ba3f`, PR #2180) e o sd-cli reconhece:
`fibo2: dim = 3584, depth = 44, head_dim = 128, text_dim = 7680`.

Então este backend é a exceção contida: só Fibo 2.0, nada mais da frota volta
pro sd.cpp. Se o Wan2GP ganhar model_def de Fibo, este arquivo morre.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from diffuse.models import MODELS

log = logging.getLogger("diffuse")

SD_CLI = Path.home() / "git" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"


def _require_sdcli() -> Path:
    if not SD_CLI.exists():
        raise FileNotFoundError(
            f"sd-cli não encontrado em {SD_CLI}\n"
            "  Build: cd ~/git/stable-diffusion.cpp && git checkout -B fibo2 origin/master && "
            "cmake -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
        )
    return SD_CLI


def _model_paths(model_name: str) -> dict[str, Path]:
    """Resolve transformer/TE/VAE/mmproj a partir do registry do diffuse."""
    info = MODELS[model_name]
    root = Path(__file__).resolve().parent.parent.parent / "models" / info["dir"]
    paths = {
        "transformer": root / "diffusion_models" / info["transformer"],
        "llm": root / "te" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        "llm_vision": root / "te" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
        "vae": root / "vae" / "flux2-vae.safetensors",
    }
    faltando = [k for k, p in paths.items() if not p.exists() and k != "llm_vision"]
    if faltando:
        raise FileNotFoundError(
            f"pesos do {model_name} faltando: {', '.join(faltando)}\n  raiz: {root}"
        )
    return paths


def build_command(
    model_name: str,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    output_path: Path,
    ref_images: list[str] | None = None,
    cfg_scale: float | None = None,
    sampler: str = "euler",
) -> list[str]:
    """Monta a linha do sd-cli. Edição exige --llm_vision (mmproj) e TE na CPU."""
    sd = _require_sdcli()
    p = _model_paths(model_name)
    info = MODELS[model_name]
    is_edit = bool(ref_images)

    cfg = cfg_scale if cfg_scale is not None else info.get("cfg_scale", 1.0)

    cmd = [
        str(sd),
        "--diffusion-model", str(p["transformer"]),
        "--llm", str(p["llm"]),
        "--vae", str(p["vae"]),
        "-p", prompt,
        "-W", str(width),
        "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "--sampling-method", sampler,
        "--seed", str(seed),
        "-o", str(output_path),
        "--offload-to-cpu",
    ]

    if is_edit:
        # o mmproj (0,84 GB) + TE (2,4 GB) não cabem na 3050 junto do runtime do DiT:
        # TE vai pra CPU e o VAE decodifica em tiles.
        cmd += ["--llm_vision", str(p["llm_vision"]), "--backend", "te=cpu", "--vae-tiling"]
        for ref in ref_images:
            cmd += ["-r", str(ref)]
    return cmd


def generate_image_sdcpp(
    model_name: str,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    output_path: Path | str,
    ref_images: list[str] | None = None,
    cfg_scale: float | None = None,
) -> tuple[Path, float]:
    """Roda o sd-cli e devolve (caminho do PNG, segundos de parede)."""
    info = MODELS[model_name]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        model_name, prompt, seed, steps, width, height,
        output_path, ref_images, cfg_scale,
    )
    log.info("sd-cli (%s): %s", info.get("transformer"), " ".join(cmd[1:]))
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


def shlex_quote_part(s: str) -> str:
    return f'"{s}"' if " " in s else s


def parse_torun(model_name: str) -> dict:
    """Metadados para o `diffuse list` (instalado?)."""
    info = MODELS[model_name]
    root = Path(__file__).resolve().parent.parent.parent / "models" / info["dir"]
    return {
        "installed": (root / "diffusion_models" / info["transformer"]).exists(),
        "sd_cli": SD_CLI.exists(),
    }
