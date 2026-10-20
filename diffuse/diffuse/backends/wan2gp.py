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
    wgp_py = wgp_dir / "wgp.py"
    if not wgp_py.exists():
        raise FileNotFoundError(f"WanGP wgp.py not found: {wgp_py}")

    if model_name == "flux-3":
        checks = {
            "transformer int8_convrot": wgp_dir / "ckpts" / "flux3_image_klein_sd_int8_convrot.safetensors",
            "TE Sakura bf16 uncensored": wgp_dir / "ckpts" / "Qwen3-VL-4B-Instruct" / "Sakura-TextEncoder-Qwen3VL-4B-Uncensored-bf16-8.27GiB.safetensors",
            "VAE": wgp_dir / "ckpts" / "flux3" / "flux2-vae.safetensors",
        }
    elif model_name == "qwen-image-2.1-turbo":  # WanGP route (official distill)
        checks = {
            "transformer turbo int8_convrot": wgp_dir / "ckpts" / "qwen_image_21_turbo_7B_int8_convrot.safetensors",
            "TE Sakura 8B GGUF uncensored": wgp_dir / "ckpts" / "Qwen3-VL-8B-Instruct" / "Sakura-TextEncoder-Qwen3VL-8B-Uncensored-8bit-8.11GiB.gguf",
            "VAE": wgp_dir / "ckpts" / "qwen_image_21" / "qwen_image_21_vae.safetensors",
        }
        missing_processor = [
            name for name in ("config_legacy.json", "tokenizer_legacy.json",
                              "tokenizer_config_legacy.json", "preprocessor_config.json",
                              "merges.txt", "vocab.json")
            if not (wgp_dir / "ckpts" / "Qwen3-VL-8B-Instruct" / name).exists()
        ]
        if missing_processor:
            raise FileNotFoundError(
                f"WanGP qwen21 processor files missing in ckpts/Qwen3-VL-8B-Instruct/: {missing_processor}"
            )
    elif model_name == "moss-image":
        checks = {
            "transformer SFT int8_convrot": wgp_dir / "ckpts" / "moss_image_sft_int8_convrot.safetensors",
            "TE Qwen3.5-9B int8_convrot": wgp_dir / "ckpts" / "Qwen3.5-9B" / "qwen3_5_9b_text_int8_convrot.safetensors",
            "VAE flux2 small decoder": wgp_dir / "ckpts" / "moss_image" / "flux2_small_decoder_bf16.safetensors",
        }
        missing_processor = [
            name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            if not (wgp_dir / "ckpts" / "Qwen3.5-9B" / name).exists()
        ]
        if missing_processor:
            raise FileNotFoundError(
                f"WanGP moss processor files missing in ckpts/Qwen3.5-9B/: {missing_processor}"
            )
    elif model_name == "moss-image-base":
        checks = {
            "transformer Base int8_convrot": wgp_dir / "ckpts" / "moss_image_base_int8_convrot.safetensors",
            "TE Qwen3.5-9B int8_convrot": wgp_dir / "ckpts" / "Qwen3.5-9B" / "qwen3_5_9b_text_int8_convrot.safetensors",
            "VAE flux2 small decoder": wgp_dir / "ckpts" / "moss_image" / "flux2_small_decoder_bf16.safetensors",
        }
        missing_processor = [
            name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            if not (wgp_dir / "ckpts" / "Qwen3.5-9B" / name).exists()
        ]
        if missing_processor:
            raise FileNotFoundError(
                f"WanGP moss processor files missing in ckpts/Qwen3.5-9B/: {missing_processor}"
            )
    elif model_name == "moss-image-kromoss":
        checks = {
            "transformer Kromoss int8_convrot": wgp_dir / "ckpts" / "kromoss_int8_convrot.safetensors",
            "TE Qwen3.5-9B int8_convrot": wgp_dir / "ckpts" / "Qwen3.5-9B" / "qwen3_5_9b_text_int8_convrot.safetensors",
            "VAE flux2 small decoder": wgp_dir / "ckpts" / "moss_image" / "flux2_small_decoder_bf16.safetensors",
        }
        missing_processor = [
            name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            if not (wgp_dir / "ckpts" / "Qwen3.5-9B" / name).exists()
        ]
        if missing_processor:
            raise FileNotFoundError(
                f"WanGP moss processor files missing in ckpts/Qwen3.5-9B/: {missing_processor}"
            )
    else:
        raise ValueError(f"load_pipeline_wan2gp: unsupported model {model_name}")

    for label, path in checks.items():
        if not path.exists():
            raise FileNotFoundError(f"WanGP {model_name} {label} not found: {path}")

    return {
        "wgp_py": str(wgp_py),
        "wan2gp_dir": str(wgp_dir),
    }, 0.0


def _parse_lora_filename_tag(tag: str) -> tuple[str, str]:
    """<lora:nome:mult> -> (nome, mult). Nome pode vir com ou sem .safetensors."""
    body = tag[len("<lora:"):-1]  # strip prefix and trailing '>'
    name, sep, mult = body.rpartition(":")
    if not sep:  # <lora:nome> sem mult
        name, mult = body, "1.0"
    return name, mult


def _resolve_prompt_lora_tags(prompt: str, root_name: str = "qwen21") -> tuple[str, list[str], list[str]]:
    """Extract <lora:...> tags from prompt; return (clean_prompt, filenames, mults).

    WanGP has NO prompt-tag parser (that is an sd.cpp affordance) — LoRAs go
    exclusively through activated_loras/loras_multipliers settings. Filenames
    are resolved against loras/<root_name>/ (filename stem or filename with ext).
    Unknown tags are dropped with a warning (fail-open, never blocks).
    """
    tags = re.findall(r"<lora:[^>]+>", prompt)
    if not tags:
        return prompt, [], []
    clean = re.sub(r"<lora:[^>]+>", "", prompt).strip()
    lora_root = WAN2GP_DIR / "loras" / root_name
    known = {p.stem: p.name for p in lora_root.glob("*.safetensors")}
    filenames, mults = [], []
    for tag in tags:
        name, mult = _parse_lora_filename_tag(tag)
        fname = known.get(name) or (name if name in known.values() else None)
        if fname is None:
            log.warning("LoRA tag %r not found in loras/%s/ — ignored", name, root_name)
            continue
        filenames.append(fname)
        mults.append(mult)
    return clean, filenames, mults


def generate_image_wan2gp(
    config: dict, prompt: str, seed: int, width: int, height: int,
    output_path: Path, ref_images: list[str] | None = None,
    steps: int = 16, cfg_scale: float = 1.0, cpu_fallback: bool = False,
    model_type: str = "flux3_klein", negative_prompt: str = "",
    activated_loras: list[str] | None = None, loras_multipliers: list[str] | None = None,
    unit_name: str = "diffuse-flux3", lora_root_name: str = "qwen21",
    flow_shift: int | None = None,
) -> tuple:
    """Generate (or edit) via WanGP. Returns (path, wall, 0.0).

    model_type flux3_klein: BFL distilled recipe (16 steps, guidance 1).
    model_type qwen_image_21_turbo_7B: Alibaba distill (8 steps, guidance 1,
    qwen21_turbo scheduler with the released fixed sigmas — set by the
    model_def defaults, do not override the solver here).
    model_type moss_image_sft: OpenMOSS SFT (50 steps, cfg 3, shift 5, SLG
    layer 22 skip built-in) — T2I ONLY: no image_refs (the moss_image handler
    declares no reference/mask inputs; structured editing goes through the
    layout-JSON prompt format instead).
    References: up to 10 ordered images ("I" mode) or main-image sizing ("KI").
    Runs wgp.py in a systemd-run scope (MemoryMax=20G — the fleet doctrine
    for heavy jobs) with profile 5 block offload.
    """
    is_edit = bool(ref_images)
    if is_edit and model_type.startswith("moss"):
        raise ValueError(
            "MOSS-Image no Wan2GP é T2I puro — sem image_refs. "
            "Edição estruturada = reformular o prompt como layout JSON."
        )
    refs = ref_images or []
    ref_mode = "KI" if is_edit else ""
    log.info(
        "Generating via WanGP %s %s: seed=%d size=%dx%d steps=%d",
        model_type, "edit" if is_edit else "T2I", seed, width, height, steps,
    )

    settings = {
        "model_type": model_type,
        "prompt": prompt,
        "resolution": f"{width}x{height}",
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "seed": seed,
        "batch_size": 1,
    }
    if negative_prompt:
        # Only effective at guidance > 1 (CFG); harmless no-op for distilled recipes.
        settings["negative_prompt"] = negative_prompt
    if flow_shift is not None:
        settings["flow_shift"] = flow_shift
    if refs:
        settings["video_prompt_type"] = ref_mode
        settings["image_refs"] = [str(r) for r in refs]

    # LoRAs: WanGP-idiom = activated_loras/loras_multipliers settings (the parser
    # merges the two: any <lora:...> tags embedded in the prompt are ALSO
    # extracted and converted — sd.cpp-style prompts keep working).
    merged_names: list[str] = list(activated_loras or [])
    merged_mults: list[str] = list(loras_multipliers or [])
    clean, tag_names, tag_mults = _resolve_prompt_lora_tags(prompt, root_name=lora_root_name)
    if tag_names:
        settings["prompt"] = clean
        merged_names += tag_names
        merged_mults += tag_mults
    if merged_names:
        settings["activated_loras"] = merged_names
        settings["loras_multipliers"] = merged_mults or ["1.0"] * len(merged_names)
        log.info("WanGP activated_loras: %s (multipliers %s)", merged_names, merged_mults)

    # Text encoder (16/out): the Wan2GP handler's ONLY TE is the local Sakura
    # abliterated bf16 (BFL ConvRot deleted per user). No selection needed here
    # anymore; DIFFUSE_FLUX3_TE left as no-op remnant removed.

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
        f"--unit={unit_name}",
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

    # wgp writes {timestamp}_{model_type}_{seed}.png in out_dir — pick the
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


# Backward-compatible alias (flux-3 call sites / any external tooling).
generate_image_flux3_wan2gp = generate_image_wan2gp