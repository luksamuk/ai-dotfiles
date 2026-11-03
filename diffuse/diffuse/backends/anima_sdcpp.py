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
                  cfg_scale=None, sampler=None, lora_dir=None,
                  init_img=None, img2img_strength=0.7, ref_image=None):
    sd = _require_sdcli()
    p = anima_paths(model_name)
    info = MODELS[model_name]
    cfg = cfg_scale if cfg_scale is not None else info.get("cfg_scale", 1.0)
    sm = sampler or info.get("sampler", "er_sde")
    cmd = [
        str(sd),
        "--diffusion-model", str(p["transformer"]),
        "--llm", str(p["llm"]),
        "--vae", str(p["vae"]),
    ]
    if lora_dir:
        # tags <lora:NOME:mult> vem NO prompt; o sd-cli extrai e resolve aqui
        cmd += ["--lora-model-dir", str(lora_dir)]
    if ref_image:
        # condicionamento por imagem (cosmos_reference é o preset default da arch anima)
        cmd += ["-r", str(ref_image)]
    if init_img:
        # img2img: denoise parcial a partir da foto
        cmd += ["-i", str(init_img), "--strength", str(img2img_strength)]
    cmd += [
        "-p", prompt,
        "-W", str(width), "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "--sampling-method", sm,
        "--seed", str(seed),
        "-o", str(output_path),
        "--offload-to-cpu",
    ]
    return cmd


# ── preprocessamento automático das imagens de entrada (img2img/ref) ────────
# rota validada por sd-cli cru (31/out): régua magick "resize ^ + extent center"
# para o size do render; canny SEM negar (linhas escuras em fundo claro —
# a polaridade invertida, estilo cv2/ComfyUI, deu ruído puro no smoke).

def _magick_or_die(cmd: list):
    import subprocess
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("preprocessamento precisa do ImageMagick (binário 'magick' não encontrado no PATH)")
    if proc.returncode != 0:
        raise RuntimeError(f"magick falhou (exit {proc.returncode}):\n{(proc.stderr or proc.stdout or '').strip()[-400:]}")


def fit_to(src, width: int, height: int) -> str:
    """Auto-fit da imagem de partida pro size do render (crop central). Cache ao lado."""
    src = Path(src)
    if not src.exists():
        raise RuntimeError(f"--init-img: arquivo não encontrado: {src}")
    out_dir = src.parent / ".diffuse_pre"
    out = out_dir / f"{src.stem}_{width}x{height}.png"
    if out.exists():
        return str(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _magick_or_die(["magick", str(src), "-resize", f"{width}x{height}^",
                    "-gravity", "center", "-extent", f"{width}x{height}", str(out)])
    return str(out)


def canny_map(src, width: int, height: int) -> str:
    """Mapa de bordas na geometria do render. Cache ao lado (reusa se já existe)."""
    src = Path(src)
    if not src.exists():
        raise RuntimeError(f"--ref-image: arquivo não encontrado: {src}")
    out_dir = src.parent / ".diffuse_pre"
    out = out_dir / f"{src.stem}_canny_{width}x{height}.png"
    if out.exists():
        return str(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _magick_or_die(["magick", str(src), "-colorspace", "gray",
                    "-resize", f"{width}x{height}^", "-gravity", "center",
                    "-extent", f"{width}x{height}",
                    "-canny", "0x1+10%+30%", str(out)])
    return str(out)


def generate_image_anima(model_name, prompt, seed, steps, width, height, output_path,
                         lora_dir=None, init_img=None, img2img_strength=0.7, ref_image=None):
    """Roda o sd-cli e devolve (caminho do PNG, segundos de parede)."""
    import subprocess
    import time
    from diffuse.backends.sdcpp import shlex_quote_part

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(model_name, prompt, seed, steps, width, height, output_path,
                        lora_dir=lora_dir, init_img=init_img,
                        img2img_strength=img2img_strength, ref_image=ref_image)
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
