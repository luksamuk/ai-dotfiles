"""sd-cli / stable-diffusion.cpp backend — Ideogram 4, Qwen-Image 2.1, Mage-Flow."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from diffuse.paths import SD_CLI_PATH
from diffuse.models import MODELS
from diffuse.backends import require_model_dir

log = logging.getLogger("diffuse")


def _resolve_sd_cli() -> str:
    """Find sd-cli binary."""
    sd_cli = str(SD_CLI_PATH)
    if not Path(sd_cli).exists():
        alt = Path.home() / "git" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
        if alt.exists():
            sd_cli = str(alt)
        else:
            raise FileNotFoundError(f"sd-cli not found at {SD_CLI_PATH}. Run: diffuse build-sd-cpp")
    return sd_cli


def _resolve_sd_cli_zimage() -> str:
    """Find sd-cli-zimage binary (master branch with native Z-Image support)."""
    base = Path(SD_CLI_PATH).parent
    zimage_cli = base / "sd-cli-zimage"
    if zimage_cli.exists():
        return str(zimage_cli)
    raise FileNotFoundError(
        f"sd-cli-zimage not found at {zimage_cli}\n"
        f"  Build it: cd ~/git/stable-diffusion.cpp && git checkout master && "
        f"cd build && make -j$(nproc) sd-cli && cp bin/sd-cli ~/git/ai-dotfiles/diffuse/bin/sd-cli-zimage"
    )


def load_pipeline_sd_cpp(model_name: str) -> tuple:
    """Prepare sd-cli configuration. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    sd_cli = _resolve_sd_cli()

    # Z-Image-Turbo uses different model files than Ideogram 4
    backend_type = model_info.get("backend_type", "sd_cpp")
    if backend_type == "zimage_sd_cpp":
        return load_pipeline_sd_cpp_zimage(model_name, model_root, sd_cli)
    if backend_type == "mageflow_sd_cpp":
        return load_pipeline_sd_cpp_mageflow(model_name, model_root, sd_cli)
    if backend_type == "qwen21_sd_cpp":
        return load_pipeline_sd_cpp_qwen21(model_name, model_root, sd_cli)

    lora_dir = model_root / "lora"
    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(model_root / "ideogram4-Q4_0.gguf"),
        "uncond_diffusion_model": str(model_root / "ideogram4_uncond-Q4_0.gguf"),
        "llm": str(model_root / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "vae": str(model_root / "vae" / "flux2-vae.safetensors"),
        "lora_dir": str(lora_dir) if lora_dir.exists() else None,
    }

    # Verify all files exist
    for key, path in config.items():
        if key == "sd_cli" and not Path(path).exists():
            raise FileNotFoundError(f"{key} not found: {path}")

    return config, 0.0


def load_pipeline_sd_cpp_zimage(model_name: str, model_root: Path, sd_cli: str) -> tuple:
    """Prepare sd-cli config for Z-Image-Turbo. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]

    # Use the sd-cli binary (master branch has native Z-Image support since #1020)
    sd_cli = _resolve_sd_cli_zimage()

    # Find the best GGUF for our VRAM
    import torch
    vram_gb = torch.cuda.mem_get_info()[1] / 1e9 if torch.cuda.is_available() else 999
    if vram_gb <= 6:
        preferred = ["Q3_K_S", "Q3_K", "Q3_K_M", "Q4_K_S", "Q4_K_M"]
    else:
        preferred = ["Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K", "Q3_K_S"]

    gguf_files = list(model_root.glob("z_image_turbo-*.gguf"))
    dit_gguf = None
    for pref in preferred:
        matches = [f for f in gguf_files if pref.lower() in f.name.lower()]
        if matches:
            dit_gguf = str(matches[0])
            break
    if not dit_gguf and gguf_files:
        dit_gguf = str(gguf_files[0])
    if not dit_gguf:
        raise FileNotFoundError(f"No Z-Image GGUF found in {model_root}")

    # Find Qwen3-4B text encoder GGUF
    llm_files = list(model_root.glob("Qwen3-4B-*.gguf"))
    llm_gguf = str(llm_files[0]) if llm_files else None
    if not llm_gguf:
        raise FileNotFoundError(f"No Qwen3-4B GGUF found in {model_root}")

    # VAE: use the Z-Image pipeline VAE (same as Flux)
    vae_path = str(model_root / "pipeline" / "vae" / "diffusion_pytorch_model.safetensors")
    if not Path(vae_path).exists():
        raise FileNotFoundError(f"Z-Image VAE not found at {vae_path}")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": dit_gguf,
        "llm": llm_gguf,
        "vae": vae_path,
        "is_zimage": True,
    }

    return config, 0.0


def load_pipeline_sd_cpp_qwen21(model_name: str, model_root: Path, sd_cli: str) -> tuple:
    """Prepare sd-cli config for Qwen-Image 2.1. Returns (config_dict, 0.0).

    Qwen-Image 2.1 uses:
    - DiT GGUF (Q4_K_M) as diffusion model
    - Qwen3-VL-8B GGUF as text encoder (--llm)
    - Qwen3-VL-8B mmproj F16 as vision encoder (--llm_vision, only needed for --edit)
    - Its OWN VAE — not interchangeable with Qwen-Image 1.0 or Wan 2.2
    - Reference images via -r (native editing, up to 10 images)
    - Native resolution up to 2048x2048; dimensions must be multiples of 32

    The text encoder is 4.68 GiB, which does not fit alongside the DiT on a 6 GB
    card, so the text encoder must run on CPU.
    """
    # Viggle turbo variant: model_name ending in "-turbo" swaps the DiT for the
    # distilled Viggle Turbo GGUF (6 steps, cfg=1.0, custom sigmas) when present.
    turbo = model_name.endswith("-turbo")
    dit_gguf = model_root / ("qwen_image_2.1_turbo_Q6_K.gguf" if turbo else "qwen-image-2.1-Q4_K_M.gguf")
    vae_path = model_root / "vae" / "qwen_image_2.1_vae_bf16.safetensors"
    # Text encoder: Heretic (pottokao, abliterated via directional ablation, KL 0.022)
    # is the official TE since 25/set — A/B won over the RLHF'd original (attenuated
    # sensitive prompts). mmproj heretic too (vision encoder for --edit).
    llm_gguf = model_root / "text_encoder_heretic" / "qwen3vl_8b_heretic-Q4_K_M.gguf"
    mmproj_gguf = model_root / "text_encoder_heretic" / "mmproj-qwen3vl_8b_heretic-f16.gguf"

    for label, path in [("DiT", dit_gguf), ("VAE", vae_path), ("LLM", llm_gguf)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(dit_gguf),
        "llm": str(llm_gguf),
        "vae": str(vae_path),
        "vae_model": str(vae_path),
        "is_qwen21": True,
    }
    if turbo:
        # NOTE: the Viggle README's sigma list is for the diffusers pipeline; sd-cli's
        # custom-sigma path produces green/magenta noise with them (measured 24/set).
        # Default scheduler at 6 steps + cfg 1.0 yields clean images — no custom sigmas.
        config["is_turbo"] = True

    # LoRA: aplica qualquer safetensors/gguf/pt em models/qwen-image-2.1/lora/
    lora_dir = model_root / "lora"
    if lora_dir.exists():
        config["lora_dir"] = str(lora_dir)

    # Vision encoder is optional — only required for reference-image editing
    if mmproj_gguf.exists():
        config["llm_vision"] = str(mmproj_gguf)

    return config, 0.0


def generate_image_qwen21_sd_cpp(
    config: dict, prompt: str, seed: int, width: int, height: int,
    output_path: Path, ref_images: list[str] | None = None,
    cpu_fallback: bool = False, steps: int = 28, cfg_scale: float = 4.0,
) -> tuple:
    """Generate (or edit) an image using sd-cli with Qwen-Image 2.1.

    Qwen-Image 2.1 is NOT a Turbo/distilled model — it needs real CFG and a
    reasonable step count. Editing is native: pass one or more reference images
    with -r plus an instruction in -p, no masks required.
    """
    is_edit = bool(ref_images)
    log.info(
        "Generating via sd-cli Qwen-Image 2.1 %s: seed=%d size=%dx%d steps=%d cfg=%.1f",
        "edit" if is_edit else "T2I", seed, width, height, steps, cfg_scale,
    )

    if is_edit and "llm_vision" not in config:
        raise FileNotFoundError(
            "Qwen-Image 2.1 editing requires the vision encoder (mmproj F16). "
            "Download: hf download Qwen/Qwen3-VL-8B-Instruct-GGUF "
            "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        )

    cmd = [
        config["sd_cli"],
        "--diffusion-model", config["diffusion_model"],
        "--llm", config["llm"],
        "--vae", config["vae"],
        "-p", prompt,
        "--cfg-scale", str(cfg_scale),
        "--steps", str(steps),
        "--sampling-method", "euler",
        "--diffusion-fa",
        "--offload-to-cpu",
        "-H", str(height),
        "-W", str(width),
        "--seed", str(seed),
        "-o", str(output_path),
    ]

    # Distilled turbo variants pin custom sigma schedules (--sigmas, comma-separated)
    if config.get("sigmas"):
        cmd += ["--sigmas", config["sigmas"]]

    # Editing: attach reference image(s) + vision encoder
    if is_edit:
        cmd += ["--llm_vision", config["llm_vision"]]
        for ref in ref_images:
            cmd += ["-r", str(ref)]

    # LoRA: aplica qualquer safetensors/gguf/pt em models/qwen-image-2.1/lora/
    # tag <lora:nome_sem_ext:0.6> injetada no prompt se o usuário não colocou nenhuma
    # Turbo destilado: SEM auto-injeção (Pruna/Fix/Detailer treinados na base 40-step
    # bagunçam a receita few-step); tags manuais continuam funcionando — o dir é o
    # MESMO lora/ do base, e tags aceitam subdiretório (<lora:lora_nsfw/nome:0.7>).
    if config.get("is_turbo"):
        if config.get("lora_dir"):
            cmd += ["--lora-model-dir", config["lora_dir"]]
    elif lora_dir := config.get("lora_dir"):
        import os as _os
        loras = [f for f in _os.listdir(lora_dir) if f.endswith((".safetensors", ".gguf", ".pt"))]
        if loras:
            cmd += ["--lora-model-dir", lora_dir]
            if "<lora:" not in prompt:
                prompt_lora = " ".join(f"<lora:{f.rsplit('.', 1)[0]}:0.6>" for f in loras)
                cmd[cmd.index("-p") + 1] = prompt + " " + prompt_lora

    # 6 GB VRAM budget. VAE runs on GPU: measured at 1024x1024 it takes 15.3s
    # there against 105.6s on CPU (7x), and pixel-identical output (99.7% of
    # pixels within 2/255, purely backend float noise). The text encoder stays
    # on CPU because its 4.7 GB does not fit beside the 4.3 GB DiT.
    if not cpu_fallback:
        cmd += ["--backend", "te=cpu", "--max-vram", "5.1"]

    if cpu_fallback:
        cmd += ["--backend", "cpu"]
        log.warning("Retrying with CPU-only backend — this will be very slow")

    t0 = time.perf_counter()
    rc, output_text = _run_sd_cli_streaming(cmd)
    wall_time = time.perf_counter() - t0

    if rc != 0:
        stderr_lines = output_text.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(
            f"sd-cli failed (rc={rc}). "
            f"Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}"
        )

    if not output_path.exists():
        raise FileNotFoundError(f"sd-cli did not produce output: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("sd-cli Qwen-Image 2.1 completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0


def load_pipeline_sd_cpp_mageflow(model_name: str, model_root: Path, sd_cli: str) -> tuple:
    """Prepare sd-cli config for Mage-Flow-Edit-Turbo. Returns (config_dict, 0.0).

    Mage-Flow-Edit uses:
    - DiT GGUF (NVFP4) as diffusion model
    - Qwen3-VL-4B GGUF as text encoder (--llm)
    - Qwen3-VL-4B mmproj F16 as vision encoder (--llm_vision)
    - Mage-VAE GGUF as VAE
    - Reference image via -r flag (instruction-based editing, no masks)
    - Turbo: 4 steps, cfg=1.0
    """
    dit_gguf = str(model_root / "mageflow-edit-turbo-nvfp4.gguf")
    vae_gguf = str(model_root / "pig_mageflow_vae_fp32-f16.gguf")
    llm_gguf = str(model_root / "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
    mmproj_gguf = str(model_root / "mmproj-Qwen3VL-4B-Instruct-F16.gguf")

    for label, path in [("DiT", dit_gguf), ("VAE", vae_gguf), ("LLM", llm_gguf), ("mmproj", mmproj_gguf)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": dit_gguf,
        "llm": llm_gguf,
        "llm_vision": mmproj_gguf,
        "vae": vae_gguf,
        "is_mageflow": True,
    }

    return config, 0.0


def generate_image_mageflow_sd_cpp(
    config: dict, prompt: str, seed: int, width: int, height: int,
    output_path: Path, ref_image: str, cpu_fallback: bool = False,
) -> tuple:
    """Generate edited image using sd-cli with Mage-Flow-Edit-Turbo.

    Mage-Flow-Edit is instruction-based: it takes a reference image and a text
    instruction (e.g. "change the background to a beach") and produces an
    edited image. No masks needed. Turbo = 4 steps, cfg=1.0.
    """
    log.info("Generating via sd-cli Mage-Flow-Edit: seed=%d ref=%s", seed, ref_image)

    cmd = [
        config["sd_cli"],
        "--diffusion-model", config["diffusion_model"],
        "--llm", config["llm"],
        "--llm_vision", config["llm_vision"],
        "--vae", config["vae"],
        "-r", ref_image,
        "-p", prompt,
        "--cfg-scale", "1.0",
        "--steps", "4",
        "--sampling-method", "euler",
        "--diffusion-fa",
        "--offload-to-cpu",
        "-v",
        "--seed", str(seed),
        "-o", str(output_path),
    ]

    if cpu_fallback:
        cmd += ["--backend", "cpu"]
        log.warning("Retrying with CPU-only backend — this will be very slow")

    t0 = time.perf_counter()
    rc, output_text = _run_sd_cli_streaming(cmd)
    wall_time = time.perf_counter() - t0

    if rc != 0:
        stderr_lines = output_text.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(f"sd-cli failed (rc={rc}). Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}")

    if not output_path.exists():
        raise FileNotFoundError(f"sd-cli did not produce output: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("sd-cli Mage-Flow completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0
def _run_sd_cli_streaming(cmd: list, label: str = "sd-cli") -> tuple:
    """Run sd-cli, streaming its progress bar to the console while capturing output.

    sd-cli already renders a progress bar via pretty_progress(), but a plain
    subprocess.run(capture_output=True) swallows it — which is why generation
    looked frozen for minutes. This streams stderr line-by-line, redrawing the
    progress lines in place and passing all other lines through to the logger.

    Returns (returncode, stderr_text) so callers keep their existing error handling.
    """
    import sys

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # sd-cli writes progress to stdout; merge to keep order
        text=True,
        bufsize=1,
    )

    captured = []
    in_progress = False
    last_len = 0

    try:
        for line in proc.stdout:
            captured.append(line)
            # A progress line looks like: " |====>     | 12/40 - 5.34s/it" and is
            # carriage-return delimited by the C side, so it arrives without a
            # trailing newline. Redraw it in place.
            if "/" in line and ("s/it" in line or "it/s" in line):
                text = line.strip("\r\n ")
                if text:
                    pad = max(0, last_len - len(text))
                    sys.stdout.write("\r  " + text + " " * pad)
                    sys.stdout.flush()
                    last_len = len(text)
                    in_progress = True
                continue
            if in_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
                in_progress = False
                last_len = 0
            # Surface meaningful lines without flooding the console.
            #
            # The biggest silent gap is the VAE decode: on this 6GB card it runs
            # on CPU and takes ~106s at 1024x1024, and sd-cli only logs it AFTER
            # it finishes. "decoding 1 latents" is emitted when it STARTS, so
            # surfacing it (plus a hint) tells the user where the wait went.
            # Deliberately NOT matched: "loading tensors" (~40 occurrences) and
            # other per-tensor chatter, which would flood the console.
            stripped = line.rstrip()
            if not stripped:
                continue
            if "decoding 1 latents" in stripped or "decoding latents" in stripped:
                print(f"  {stripped}")
                print("     VAE decode — runs on CPU, usually the slowest single step")
                continue
            if any(k in stripped for k in
                   ("ERROR", "WARN", "sampling completed", "generate_image completed",
                    "decode_first_stage", "latent 1 decoded", "latents decoded",
                    "save result image", "images saved", "generating image:")):
                print(f"  {stripped}")
        proc.wait()
    finally:
        if in_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return proc.returncode, "".join(captured)


def generate_image_sd_cpp(config: dict, prompt: str, seed: int, width: int, height: int, output_path: Path, cpu_fallback: bool = False, nsfw: bool = False) -> tuple:
    """Generate image using sd-cli. Returns (output_path, wall_time_seconds, 0.0)."""
    log.info("Generating via sd-cli: seed=%d size=%dx%d cpu_fallback=%s nsfw=%s", seed, width, height, cpu_fallback, nsfw)

    is_zimage = config.get("is_zimage", False)

    # Z-Image-Turbo: no uncond model, CFG=0, 9 steps, flux_flow prediction
    if is_zimage:
        cmd = [
            config["sd_cli"],
            "--diffusion-model", config["diffusion_model"],
            "--llm", config["llm"],
            "--vae", config["vae"],
            "-p", prompt,
            "--diffusion-fa",
            "--offload-to-cpu",
            "--clip-on-cpu",
            "--vae-on-cpu",
            "--vae-tiling",
            "-H", str(height),
            "-W", str(width),
            "--seed", str(seed),
            "-o", str(output_path),
        ]
        # Add steps and cfg from config if provided
        if "steps" in config:
            cmd += ["--steps", str(config["steps"])]
        if "cfg" in config:
            cmd += ["--cfg-scale", str(config["cfg"])]
    else:
        cmd = [
            config["sd_cli"],
            "--diffusion-model", config["diffusion_model"],
        ]
        if not nsfw and "uncond_diffusion_model" in config:
            cmd += ["--uncond-diffusion-model", config["uncond_diffusion_model"]]
        cmd += [
            "--llm", config["llm"],
            "--vae", config["vae"],
            "-p", prompt,
            "--diffusion-fa",
            "--offload-to-cpu",
            "--clip-on-cpu",
            "--vae-on-cpu",
            "--max-vram", "5.1",
            "-H", str(height),
            "-W", str(width),
            "--seed", str(seed),
            "-o", str(output_path),
        ]
        # LoRA: aplica qualquer safetensors/gguf/pt em models/<model>/lora/
        # (antes era incondicional ao --nsfw; agora plug-and-play sempre)
        if lora_dir := config.get("lora_dir"):
            import os as _os
            loras = [f for f in _os.listdir(lora_dir) if f.endswith((".safetensors", ".gguf", ".pt"))]
            if loras:
                cmd += ["--lora-model-dir", lora_dir]
                if "<lora:" not in prompt:
                    prompt_lora = " ".join(f"<lora:{f.rsplit('.', 1)[0]}:0.6>" for f in loras)
                    cmd[cmd.index("-p") + 1] = prompt + " " + prompt_lora

    # CPU fallback: remove VRAM limits and force everything on CPU
    if cpu_fallback:
        log.warning("Retrying with CPU-only backend — this will be very slow (~30+ minutes)")
        if is_zimage:
            cmd = [
                config["sd_cli"],
                "--diffusion-model", config["diffusion_model"],
                "--llm", config["llm"],
                "--vae", config["vae"],
                "-p", prompt,
                "--backend", "cpu",
                "-H", str(height),
                "-W", str(width),
                "--seed", str(seed),
                "-o", str(output_path),
            ]
            if "steps" in config:
                cmd += ["--steps", str(config["steps"])]
            if "cfg" in config:
                cmd += ["--cfg-scale", str(config["cfg"])]
        else:
            cmd = [
                config["sd_cli"],
                "--diffusion-model", config["diffusion_model"],
            ]
            if not nsfw and "uncond_diffusion_model" in config:
                cmd += ["--uncond-diffusion-model", config["uncond_diffusion_model"]]
            cmd += [
                "--llm", config["llm"],
                "--vae", config["vae"],
                "-p", prompt,
                "--backend", "cpu",
                "-H", str(height),
                "-W", str(width),
                "--seed", str(seed),
                "-o", str(output_path),
            ]
            # LoRA no CPU-fallback: mesmo comportamento plug-and-play do caminho CUDA
            if lora_dir := config.get("lora_dir"):
                import os as _os
                loras = [f for f in _os.listdir(lora_dir) if f.endswith((".safetensors", ".gguf", ".pt"))]
                if loras:
                    cmd += ["--lora-model-dir", lora_dir]
                    if "<lora:" not in prompt:
                        prompt_lora = " ".join(f"<lora:{f.rsplit('.', 1)[0]}:0.6>" for f in loras)
                        cmd[cmd.index("-p") + 1] = prompt + " " + prompt_lora

    t0 = time.perf_counter()
    rc, output_text = _run_sd_cli_streaming(cmd)
    wall_time = time.perf_counter() - t0

    if rc != 0:
        # Print last 20 lines of stderr for debugging
        stderr_lines = output_text.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(f"sd-cli failed (rc={rc}). Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}")

    if not output_path.exists():
        raise FileNotFoundError(f"sd-cli did not produce output: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("sd-cli completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0