"""Wan2GP backend — FLUX 3 Image Klein via DeepBeepMeep WanGP (mmgp).

Replaces the old sd_cpp route (fp8r DiT + GGUF TE, denoise bound to CPU at
~67 s/it): the WanGP pipeline offloads blocks one-by-one (mmgp Profile 5,
the same regime that carries Ming/H3 in the 6 GB fleet), so the denoise can
run on GPU and edits gain the full WanGP reference stack (KI main-ref sizing,
10 ordered references, bounding boxes).

Weights live in ~/git/Wan2GP/models/flux3/ (transformer int8_convrot +
Qwen3-VL-4B int8_convrot TE + flux2 VAE), downloaded from DeepBeepMeep/Flux3.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import logging

log = logging.getLogger("diffuse")

WAN2GP_DIR = Path.home() / "git" / "Wan2GP"
PROFILE = 5  # VeryLowRAM_LowVRAM — the fleet standard for 6 GB


def load_pipeline_wan2gp(model_name: str, model_root: Path | None = None) -> tuple:
    """Verify WanGP weights for FLUX 3 Klein. Returns (config_dict, 0.0).

    WanGP stores weights under ckpts/ (auto-downloaded by the finetune
    json); diffuse only verifies their presence.
    """
    wgp_dir = WAN2GP_DIR
    transformer = wgp_dir / "ckpts" / "flux3_image_klein_sd_int8_convrot.safetensors"
    te = wgp_dir / "ckpts" / "Qwen3-VL-4B-Instruct" / "qwen3vl_4b_int8_convrot.safetensors"
    vae = wgp_dir / "ckpts" / "flux3" / "flux2-vae.safetensors"
    wgp_py = wgp_dir / "wgp.py"

    for label, path in [
        ("wgp.py", wgp_py),
        ("transformer int8_convrot", transformer),
        ("TE int8_convrot", te),
        ("VAE", vae),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"WanGP FLUX 3 {label} not found: {path}")

    return {
        "wgp_py": str(wgp_py),
        "wan2gp_dir": str(wgp_dir),
        "transformer": transformer.name,
    }, 0.0


def generate_image_flux3_wan2gp(
    config: dict, prompt: str, seed: int, width: int, height: int,
    output_path: Path, ref_images: list[str] | None = None,
    steps: int = 16, cfg_scale: float = 1.0, cpu_fallback: bool = False,
) -> tuple:
    """Generate (or edit) via WanGP flux3_klein. Returns (path, wall, 0.0).

    Distilled recipe: 16 steps, guidance 1.0. References: up to 10 ordered
    images ("I" mode) or main-image sizing ("KI"). Runs wgp.py in a
    systemd-run scope (MemoryMax=20G — the fleet doctrine for heavy jobs)
    with profile 5 block offload.
    """
    is_edit = bool(ref_images)
    refs = ref_images or []
    ref_mode = "KI" if is_edit else ""
    log.info(
        "Generating via WanGP FLUX 3 Klein %s: seed=%d size=%dx%d steps=%d",
        "edit" if is_edit else "T2I", seed, width, height, steps,
    )

    settings = {
        "model_type": "flux3_klein",
        "prompt": prompt,
        "resolution": f"{width}x{height}",
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "seed": seed,
        "batch_size": 1,
    }
    if refs:
        settings["video_prompt_type"] = ref_mode
        settings["image_refs"] = [str(r) for r in refs]

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    import tempfile
    settings_file = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="diffuse_flux3_", delete=False
    )
    json.dump(settings, settings_file)
    settings_file.close()

    cmd = [
        "systemd-run", "--user", "--scope", "-p", "MemoryMax=20G",
        "--unit=diffuse-flux3",
        str(WAN2GP_DIR / ".venv" / "bin" / "python"),
        config["wgp_py"],
        "--process", settings_file.name,
        "--profile", str(PROFILE),
        "--output-dir", str(out_dir),
        "--verbose", "2",
    ]

    env = os.environ.copy()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=config["wan2gp_dir"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    peak_hbm = 0.0
    try:
        for line in proc.stdout or []:
            print("     wgp:", line.rstrip()[:180])
            m = re.search(r"([\d.]+)\s*GB.*VRAM|VRAM.*?([\d.]+)\s*GB", line)
            if m:
                peak_hbm = max(peak_hbm, float(m.group(1) or m.group(2)))
        rc = proc.wait()
    finally:
        try:
            os.unlink(settings_file.name)
        except OSError:
            pass

    wall = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"wgp exited with code {rc} after {wall:.0f}s")

    # wgp writes {timestamp}_flux3_klein_{seed}.png in out_dir — pick the
    # newest file born after start, then move to the requested output path.
    candidates = sorted(
        glob.glob(str(out_dir / "*.png")) + glob.glob(str(out_dir / "*.jpg")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise RuntimeError(f"wgp produced no image in {out_dir}")
    produced = Path(candidates[-1])

    if produced.resolve() != output_path.resolve():
        shutil.move(str(produced), str(output_path))

    return output_path, wall, peak_hbm