"""CLI — argument parsing, interactive prompt, and main() orchestration."""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from diffuse.paths import setup_environment, DEFAULT_VISION_MODEL
from diffuse.models import MODELS
from diffuse.backends import load_pipeline, unload_pipeline, require_model_dir
from diffuse.backends.gemlite import generate_image_gemlite
from diffuse.backends.agate import generate_image_agate
from diffuse.backends.framepack import generate_video_framepack
from diffuse.llm import evict_llm, llama_swap_running_models
from diffuse.enhance import (
    enhance_prompt,
    enhance_vision_prompt,
    enhance_qwen21_prompt,
    enhance_agate_prompt,
    enhance_moss_prompt,
    analyze_and_enhance_edit,
    _check_model_vision,
)
from diffuse.output import (
    resolve_output_path,
    save_metadata,
    print_debrief,
    get_previous_runs,
    open_image,
)

log = logging.getLogger("diffuse")

# Preserve <lora:...> tags through prompt enhancement.
def _extract_lora_tags(prompt):
    import re
    tags = re.findall(r"<lora:[^>]+>", prompt)
    clean = re.sub(r"<lora:[^>]+>", "", prompt).strip()
    return clean, " ".join(tags)

def _reapply_lora_tags(enhanced, tags):
    if tags and "<lora:" not in enhanced:
        return enhanced + " " + tags
    return enhanced


# ── Argument parsing ───────────────────────────────────────────────────────
def parse_size(s: str) -> tuple[int, int]:
    """Parse 'WxH' (e.g. '1024x1024') into (width, height)."""
    s = s.lower().replace("×", "x")
    try:
        w_str, h_str = s.split("x", 1)
        w, h = int(w_str), int(h_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--size must be 'WxH' (e.g. 1024x1024), got {s!r}")
    for dim, name in ((w, "width"), (h, "height")):
        if not 256 <= dim <= 4096:
            raise argparse.ArgumentTypeError(f"--size {name} {dim} out of range — must be 256–4096")
        if dim % 16:
            raise argparse.ArgumentTypeError(f"--size {name} {dim} must be a multiple of 16")
    return w, h


def _build_model_help() -> str:
    """Build a brief model list for --help. Escape %% for argparse."""
    lines = []
    for name in sorted(MODELS):
        lines.append(f"  {name}")
    return "\n".join(lines).replace("%", "%%")


def print_models() -> None:
    """Print detailed model info to stdout."""
    print()
    print("  ═══ diffuse — Available Models ═══")
    print()
    for name in sorted(MODELS):
        info = MODELS[name]
        bits = info.get("bits", "?")
        desc = info.get("description", "")
        backend = info.get("backend_type", "gemlite")
        size = info.get("default_size")
        size_str = f"{size[0]}×{size[1]}" if size else "512×512"
        enhance = info.get("enhance_model", "")
        enhance_type = info.get("enhance_type", "")
        print(f"  {name}")
        print(f"    {bits}  |  {backend}  |  default {size_str}")
        if enhance:
            et = f" ({enhance_type})" if enhance_type else ""
            print(f"    enhance: {enhance}{et}")
        print(f"    {desc}")
        print()


def print_list() -> None:
    """Print models grouped by category with dependencies, sizes, and shared components."""
    from diffuse.models import MODELS
    from diffuse.paths import MODELS_DIR

    # Group by category
    categories = {"image": [], "video": []}
    for name, info in MODELS.items():
        cat = info.get("category", "image")
        categories.setdefault(cat, []).append((name, info))

    def _fmt_size_gb(gb: float) -> str:
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        return f"{int(gb * 1024)} MB"

    def _check_installed(info: dict) -> bool:
        model_dir = info.get("dir", "")
        backend = info.get("backend_type", "")
        if backend == "flux3_wan2gp":
            # Weights live in the WanGP ckpts/ tree, not in diffuse models/
            wgp = Path.home() / "git" / "Wan2GP" / "ckpts"
            return (wgp / "flux3_image_klein_sd_int8_convrot.safetensors").exists()
        elif backend == "moss_wan2gp":
            wgp = Path.home() / "git" / "Wan2GP" / "ckpts"
            if info.get("dir") == "moss-image-base":
                return (wgp / "moss_image_base_int8_convrot.safetensors").exists()
            if info.get("dir") == "moss-image-kromoss":
                return (wgp / "kromoss_int8_convrot.safetensors").exists()
            if info.get("dir") == "moss-imagine":
                return (wgp / "moss_imagine_int8_convrot.safetensors").exists()
            if info.get("dir") == "moss-image-photoreal":
                return (wgp / "moss_image_photoreal_int8_convrot.safetensors").exists()
            return (wgp / "moss_image_sft_int8_convrot.safetensors").exists()
        elif backend == "qwen21_turbo_wan2gp":
            wgp = Path.home() / "git" / "Wan2GP" / "ckpts"
            return (wgp / "qwen_image_21_turbo_7B_int8_convrot.safetensors").exists()
        else:
            p = MODELS_DIR / model_dir
        return p.exists() and any(p.iterdir()) if p.exists() else False

    # Track total disk usage
    total_installed = 0.0
    total_pending = 0.0

    print()
    # Installed models render full detail inline; not-installed collapse into a
    # summary block at the end so the working inventory stays scannable.
    line_missing = []
    for cat_label, cat_key in [("IMAGE GENERATION", "image"), ("VIDEO GENERATION", "video")]:
        models = categories.get(cat_key, [])
        if not models:
            continue
        print(f"  {cat_label}")
        for name, info in sorted(models):
            installed = _check_installed(info)

            total_gb = 0.0
            for comp in info.get("components", []):
                total_gb += comp["size_gb"]

            if installed:
                total_installed += total_gb
                print(f"    {name:20s} {info.get('description', ''):60s}")
                for comp in info.get("components", []):
                    comp_name = comp["name"]
                    shared_tag = " [shared]" if "[shared]" in comp_name else ""
                    print(f"      ├── {comp_name:55s} {_fmt_size_gb(comp['size_gb'])}")
                print(f"      └── {'Total':55s} {_fmt_size_gb(total_gb)}")
                print()
            else:
                total_pending += total_gb
                line_missing.append(f"    {name:20s} {_fmt_size_gb(total_gb):8s} {info.get('description', '')}")

    if line_missing:
        print("  NOT INSTALLED (weights off disk — download via `diffuse download <model>`):")
        print("\n".join(line_missing))
        print()

    print(f"  TOTAL DISK: {_fmt_size_gb(total_installed + total_pending)} ({_fmt_size_gb(total_installed)} installed + {_fmt_size_gb(total_pending)} pending)")
    print()


def print_loras() -> None:
    """Print the LoRA inventory per model family: cataloged entries x files on disk.

    Cross-checks the unified catalog (~/.local/share/diffuse/loras_<root>.json —
    com nsfw:bool e compat por variante) against the .safetensors installed under
    ~/git/Wan2GP/loras/<root>/, and flags:
      - orphans: file installed with no catalog entry (invisible to the selector)
      - ghosts:  catalog entry with no file on disk (the selector would fail on it)
    """
    import json as _json
    from diffuse.models import MODELS
    from diffuse.paths import MODELS_DIR

    share = Path.home() / ".local" / "share" / "diffuse"
    wgp_loras = Path.home() / "git" / "Wan2GP" / "loras"

    fams: dict = {}
    for name, info in MODELS.items():
        root = info.get("lora_root")
        if not root:
            continue
        fam = fams.setdefault(root, {"models": [], "dirs": []})
        fam["models"].append(name)
        for cand in (wgp_loras / root, MODELS_DIR / info.get("dir", "") / "lora"):
            if cand.exists() and cand not in fam["dirs"]:
                fam["dirs"].append(cand)

    if not fams:
        print("No model declares lora_root — nothing to list.")
        return

    total_files = total_cat = 0
    for root in sorted(fams):
        fam = fams[root]
        # o lora_root nem sempre casa com o nome do catálogo (moss_image -> moss)
        aliases = {root, root.split("_")[0]}
        cats = []
        loaded = set()
        # catálogo unificado (F0): loras_<alias>.json, com nsfw:bool e compat por variante
        for alias in sorted(aliases):
            p = share / f"loras_{alias}.json"
            if p in loaded or not p.exists():
                continue
            loaded.add(p)
            try:
                cats.append(("unificado", _json.load(open(p)).get("loras", [])))
            except Exception as e:
                print(f"  [warn] {p.name}: {e}")
        # F6: catálogos legados (catalog_/nsfw_catalog_) removidos do disco;
        # o único formato agora é o unificado loras_<alias>.json

        files: dict = {}
        for d in fam["dirs"]:
            for f in sorted(d.glob("*.safetensors")):
                files.setdefault(f.name, f)

        cataloged = {e.get("safetensors", "") for _, entries in cats for e in entries}
        orphans = [n for n in files if n not in cataloged]
        ghosts = [e for _, entries in cats for e in entries
                  if e.get("safetensors", "") not in files]

        n_cat = sum(len(e) for _, e in cats)
        total_files += len(files)
        total_cat += n_cat

        print(f"\n=== {root} ===")
        print(f"  models: {', '.join(sorted(fam['models']))}")
        for d in fam["dirs"]:
            print(f"  dir:    {d}")
        print(f"  files: {len(files)} .safetensors | catalog: {n_cat} "
              f"| orphans: {len(orphans)} | ghosts: {len(ghosts)}")

        for label, entries in cats:
            if not entries:
                continue
            print(f"\n  Cataloged / {label} ({len(entries)}):")
            for e in sorted(entries, key=lambda x: str(x.get("id"))):
                nm = e.get("safetensors", "?")
                mark = "" if nm in files else "  [MISSING]"
                st = e.get("strength_autor")
                st = f" @{st}" if st not in (None, "") else ""
                trig = (e.get("trigger_instr_arquivo") or "").strip()
                tr = f"  trigger: {trig[:40]}" if trig else ""
                compat = ",".join(e.get("compat") or []) or "-"
                nsfwtag = " [NSFW]" if e.get("nsfw") else ""
                print(f"    id {str(e.get('id', '?')):>2}  {nm[:44]:<44} "
                      f"{e.get('categoria', '')[:18]:<18}{st}{tr}{nsfwtag}  compat:{compat}{mark}")

        if orphans:
            print(f"\n  Orphans — on disk, not in any catalog ({len(orphans)}):")
            for n in orphans:
                print(f"    {n}  ({files[n].stat().st_size / 1e6:.0f} MB)")

        if ghosts:
            print(f"\n  Ghosts — catalog entry, no file on disk ({len(ghosts)}):")
            for e in ghosts:
                print(f"    id {str(e.get('id', '?')):>2}  {e.get('safetensors', '?')}")

    print(f"\nTotal: {total_files} files on disk, {total_cat} catalog entries.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffuse",
        description="diffuse — Local diffusion image generation CLI for NVIDIA RTX 3050 6GB",
        epilog=(
            "Examples:\n"
            "  diffuse -m ternary-gemlite -p 'a cat on the moon'\n"
            "  diffuse -m ternary-gemlite --enhance -p 'a rainy day at a coffee shop'\n"
            "  diffuse -m ternary-gemlite --enhance-with laguna-xs2 --evict-llm -p 'cyberpunk city'\n"

            "  diffuse --list                        # show model details\n"
            "\n"
            "Model capabilities:\n"
            "  ternary-gemlite  — 1.58-bit Bonsai, T2I only\n"
            "  binary-gemlite   — 1-bit Bonsai, T2I only\n"

            "\n"
            "Resolution guide (RTX 3050 6GB):\n"
            "  Bonsai:     512×512 max (OOM above this)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-m", "--model",
        choices=sorted(MODELS),
        default="ternary-gemlite",
        help="Model variant (default: ternary-gemlite). Use --list for details.",
    )
    p.add_argument("-p", "--prompt", help="Text prompt. If omitted, prompted interactively.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (random if not set).")
    p.add_argument("--steps", type=int, default=None, help="Denoising steps (default per backend: bonsai=4, qwen21-turbo=8, flux3=16, qwen21=40, moss=50, agate=50).")
    p.add_argument(
        "--size", type=parse_size, default=None,
        help="Image size as WxH (default: 512x512 for bonsai; agate fixed 256x256; others 1024x1024).",
    )
    p.add_argument("--output", type=Path, default=None, help="Output PNG path (auto-generated in cwd if not set).")
    p.add_argument("--open", action="store_true", help="Open the generated image with feh after saving.")
    p.add_argument(
        "--list", action="store_true",
        help="List available models with details and exit.",
    )
    p.add_argument(
        "--lora", action="store_true",
        help="List the LoRA inventory (catalog x files on disk, orphans/ghosts) and exit.",
    )
    p.add_argument(
        "--lora-selector", choices=["llm", "reranker", "decision"], default=None,
        help="LoRA selection mode: llm (rerank + LLM, default) or decision (systemone model).",
    )
    p.add_argument(
        "--decision-model", default=None,
        help="Decision model handle for --lora-selector decision (default: d1-3b).",
    )
    p.add_argument(
        "--decision-threshold", type=float, default=None,
        help="Cutoff score for the decision selector (default 0.5).",
    )
    p.add_argument(
        "--decision-max", type=int, default=None,
        help="Max LoRAs the decision selector may pick (default 4).",
    )
    p.add_argument(
        "--evict-llm", action="store_true",
        help="Evict all running LLM models (via llama-swap) to free VRAM before generating. "
             "Automatically done when --enhance is used (double-evict: before and after LLM call).",
    )
    p.add_argument(
        "--enhance", action="store_true",
        help="Expand prompt via LLM. "
             "For qwen21/moss/bonsai: natural English description with details. "
             "Uses model's 'enhance_model' or qwen3.5-4b by default.",
    )
    p.add_argument(
        "--enhance-with", metavar="MODEL",
        help="Enhance prompt using a specific model. Any llama-swap model works "
             "(e.g. qwen3.5-4b, laguna-xs2). Prefix with 'ollama:' to use the local "
             "Ollama daemon instead (port 11434), e.g. ollama:glm-5.3-flash:cloud. "
             "Implies --enhance.",
    )
    p.add_argument(
        "--enhance-edit-with", metavar="MODEL",
        help="Edit-only: refine the edit instruction with a VISION-capable model "
             "that also sees the reference image (one-shot analyze+enhance). "
             "Any llama-swap vision model (e.g. qwen3.6-35b-a3b-heretic:think) or "
             "'ollama:' prefix. Implies --enhance; only meaningful with --edit.",
    )
    p.add_argument(
        "--show-enhanced", action="store_true",
        help="(no-op, kept for compatibility) enhanced prompt is ALWAYS shown right after enhancement, before the render starts",
    )
    p.add_argument(
        "--cpu-fallback", action="store_true",
        help="If CUDA generation fails, automatically retry on CPU (very slow: ~30+ min).",
    )
    p.add_argument(
        "--edit", metavar="IMAGE", type=Path, default=None, nargs="+",
        help="Reference image for editing (qwen-image-2.1 / flux-3). Pass an image path to use instruction-based editing.",
    )
    p.add_argument(
        "--input-image", metavar="IMAGE", type=Path, default=None,
        help="Input image for I2V (image-to-video), used by framepack-i2v.",
    )
    p.add_argument(
        "--seconds", type=float, default=None,
        help="Video length in seconds for FramePack I2V (default: 5.0, max: 120).",
    )
    p.add_argument(
        "--cfg", type=float, default=None,
        help="CFG scale (default: 1.0 for FramePack, 7.0 for other image gen).",
    )
    p.add_argument(
        "--agate-autoguide", type=float, default=None,
        help="Agate autoguidance strength (001 only; 003 has no autoguidance). "
             "Card recipe for sharper faces: 1.0 with cfg 4.0 (~1.5x time). "
             "ON by default for agate-preview-001 (pass --agate-autoguide 0 to disable); "
             "0 = off.",
    )
    p.add_argument(
        "--gs", type=float, default=4.5,
        help="Distilled guidance scale for FramePack I2V (default: 4.5).",
    )
    p.add_argument(
        "--no-teacache", action="store_true",
        help="Disable TeaCache for FramePack I2V (slower but potentially better quality).",
    )
    p.add_argument(
        "--nsfw", action="store_true",
        help="qwen-image-2.1: smart LoRA selection via catalog (strength via DIFFUSE_NSFW_STRENGTH, default 0.7); skipped when the prompt has a manual <lora:...> tag. Other models: NSFW enhance prompts.",
    )
    p.add_argument(
        "--turbo", action="store_true",
        help="moss-image: attach the DMD 8-step distillation LoRA at strength 1.0 and default to 8 steps / cfg 1.0 (50->8 steps, ~4x faster, not part of --enhance LoRA selection).",
    )
    return p.parse_args()


# ── Interactive prompt ──────────────────────────────────────────────────────
def get_prompt_interactive() -> str:
    """Prompt the user for a text prompt interactively."""
    print()
    print("  🎨 diffuse — Enter your prompt (Ctrl+C to cancel)")
    print("  ─────────────────────────────────────────────────")
    try:
        prompt = input("  Prompt: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        sys.exit(0)
    if not prompt:
        print("  Empty prompt — exiting.")
        sys.exit(0)
    return prompt


# ── Main ────────────────────────────────────────────────────────────────────
def _load_enhance_config(model_info: dict | None = None) -> str:
    """Resolve o modelo de enhance via config externa da frota.

    Ordem: chave 'diffuse' em enhance-models.yaml → fallback declarado →
    registry (enhance_model do models.py) → qwen3.5-4b. Valida os handles
    contra /v1/models do llama-swap (evita o 404 silencioso do episódio
    f210454). Fail-open se o swap estiver fora (aceita o primeiro).

    Módulo-level (não aninhada) de propósito: os runners por modelo
    (_run_qwen21_*, _run_moss_*, etc.) precisam do MESMO handle resolvido
    pela config — inclusive o sufixo :think. Quando esta função vivia
    aninhada em main(), cada runner re-resolvia do models.py e o :think
    do fleet se perdia no payload.
    """
    cfg_path = os.path.expanduser("~/.local/share/fleet/enhance-models.yaml")
    candidates: list[str] = []
    try:
        import yaml as _yaml
        cfg = _yaml.safe_load(open(cfg_path)) or {}
        entry = cfg.get("diffuse") or {}
        if entry.get("model"):
            candidates.append(entry["model"])
        if entry.get("fallback"):
            candidates.append(entry["fallback"])
    except Exception:
        pass
    if model_info:
        candidates.append(model_info.get("enhance_model", ""))
    candidates.append("qwen3.5-4b")

    live: set = set()
    try:
        from diffuse.paths import LLAMA_SWAP_URL
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/v1/models", timeout=3) as r:
            live = {m["id"] for m in json.load(r).get("data", [])}
    except Exception:
        live = set()

    for cand in candidates:
        if cand and (not live or cand in live or (cand + ":think") in live):
            return cand
    return "qwen3.5-4b"


def main() -> None:
    args = parse_args()

    if args.lora:
        print_loras()
        sys.exit(0)

    if args.list:
        print_list()
        sys.exit(0)

    setup_environment()

    model_name = args.model
    model_info = MODELS[model_name]
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    backend_type = model_info.get("backend_type", "gemlite")

    # ── Defaults per backend ──
    # Steps: bonsai=4, qwen21-turbo=8, flux3=16, moss=50, qwen21=40, agate=50
    if args.steps is None:
        if backend_type == "qwen21_turbo_wan2gp":
            args.steps = 8           # official Alibaba turbo (distilled)
        elif backend_type == "flux3_wan2gp":
            args.steps = 16          # BFL distilled recipe
        elif backend_type == "moss_wan2gp":
            args.steps = 50          # OpenMOSS release recipe (CFG 3, SLG 22; sem distill oficial no WanGP)
        elif backend_type == "qwen21_wan2gp":
            args.steps = 40          # base 2.1 recipe (cfg 4.0; --turbo remaps para 8 no runner)
        elif backend_type == "agate":
            args.steps = 50
        elif backend_type == "fibo_sdcpp":
            args.steps = 4 if "turbo" in model_name else 20   # turbo: 4 passos (distill BRIA); dev: default do sd.cpp
        else:
            args.steps = 4

    # Size: model-specific defaults (512x512 for bonsai, model-specific for others)
    if args.size is None:
        default_size = model_info.get("default_size", (512, 512))
        width, height = default_size
    else:
        width, height = args.size

    # Agate: 001 is fixed 256×256; 003 is multi-res (256 or 512 native). Snap only for 001.
    if backend_type == "agate" and "agate-preview-001" in (args.model or model or ""):
        if (width, height) != (256, 256):
            print(f"  ⚠️  Agate 001 is a fixed 256×256 model (trained resolution) — ignoring --size {width}x{height}")
            print(f"     For larger output, upscale the result with Real-ESRGAN.")
            width, height = (256, 256)
    elif backend_type == "agate":
        # 003: multi-res (256/512). Other sizes → snap to nearest supported
        if (width, height) not in ((256, 256), (512, 512)):
            snapped = (256, 256) if max(width, height) < 384 else (512, 512)
            print(f"  ⚠️  Agate 003 supports 256×256 and 512×512 — snapping --size {width}x{height} to {snapped[0]}x{snapped[1]}")
            width, height = snapped

    prompt = args.prompt or get_prompt_interactive()

    original_prompt = prompt

    # Pre-flight
    if backend_type in ("flux3_wan2gp", "qwen21_wan2gp", "qwen21_turbo_wan2gp", "moss_wan2gp"):
        # Weights live in the WanGP ckpts/ tree; backends/wan2gp verifies per-file
        pass
    else:
        require_model_dir(model_name)

    # ── Validate --edit (qwen21, flux-3 e fibo suportam) ──
    ref_image_paths = None
    if args.edit:
        if backend_type not in ("flux3_wan2gp", "qwen21_wan2gp", "qwen21_turbo_wan2gp", "fibo_sdcpp"):
            print(f"  ⚠️  --edit is only supported with qwen-image-2.1, flux-3 or fibo (got {backend_type})")
            print(f"     Use: diffuse -m qwen-image-2.1 --edit {args.edit} -p 'instruction'")
            sys.exit(1)
        missing = [x for x in args.edit if not x.exists()]
        if missing:
            # If not found as-is, try resolving relative to original CWD
            # (the shell wrapper cds to SCRIPT_DIR before running generate.py)
            orig_cwd = os.environ.get("DIFFUSE_ORIG_CWD", "")
            if orig_cwd:
                resolved = [Path(orig_cwd) / x for x in args.edit]
                if all(x.exists() for x in resolved):
                    args.edit = resolved
                else:
                    print(f"  ✗ Edit image not found: {args.edit} (also tried {resolved})")
                    sys.exit(1)
            else:
                print(f"  ✗ Edit image not found: {args.edit}")
                sys.exit(1)
        ref_image_paths = [str(x.resolve()) for x in args.edit]
        print(f"  🖼️  Edit mode: {len(ref_image_paths)} ref image(s) → prompt as instruction")

    # ── FramePack I2V early path ──────────────────────────────────────────────
    if backend_type == "framepack":
        _run_framepack(args, model_name, model_info, prompt, original_prompt, seed, width, height)
        return

    # ── Bonsai subprocess early path ─────────────────────────────────────────
    if backend_type == "bonsai":
        _run_bonsai_image(args, model_name, model_info, prompt, original_prompt, seed, width, height)
        return

    # ── FLUX 3 Klein early path (T2I + editing via WanGP) ──
    if backend_type == "flux3_wan2gp":
        _run_flux3_wan2gp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
        return

    # ── Qwen-Image 2.1 UC early path (base uncensored via WanGP + flag --turbo) ──
    if backend_type == "qwen21_wan2gp":
        _run_qwen21_wan2gp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
        return
    # ── Qwen-Image 2.1 Turbo early path (T2I + editing via WanGP; Sakura TE) ──
    if backend_type == "qwen21_turbo_wan2gp":
        _run_qwen21_turbo_wan2gp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
        return

    # ── MOSS-Image early path (T2I via WanGP; layout-JSON nativo) ──
    if backend_type == "moss_wan2gp":
        _run_moss_wan2gp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
        return

    # ── Fibo 2.0 via sd.cpp (T2I + edição por referência) ──
    if backend_type == "fibo_sdcpp":
        _run_fibo_sdcpp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
        return

    # ── LLM eviction (free VRAM for diffusion) ──
    if args.evict_llm:
        running = llama_swap_running_models()
        if running:
            print(f"  🔄 Evicting LLM models: {', '.join(running)}")
            evicted = evict_llm()
            if evicted:
                print(f"     VRAM freed — diffusion pipeline can load")
            else:
                print(f"     Warning: eviction may not have fully completed")
        else:
            print(f"  ✅ No LLM models loaded — VRAM already free")
        print()

    # ── Fleet enhance config (~/.local/share/fleet/enhance-models.yaml) ─────
    # ── Prompt enhancement ──
    enhanced_prompt = None
    enhance_model: str = ""

    if args.enhance_with:
        # --enhance-with <model> — use any llama-swap model
        enhance_model = args.enhance_with
        args.enhance = True  # implied
    elif args.enhance:
        enhance_model = _load_enhance_config()
        args.enhance = True  # implied

    if args.enhance:
        enhance_type = model_info.get("enhance_type", "qwen21")

        # ── Normal enhancement (sem edit no main; edits vivos têm early path próprio) ──
        if not ref_image_paths:  # edit-analysis flow saiu com o hidream (F5); edits vivos têm early path próprio
            enhanced_result = prompt  # default: no change
            if enhance_type == "agate":
                print(f"  ✨ Enhancing prompt via {enhance_model} (agate mode)...")
                enhanced_result, raw_response = enhance_agate_prompt(prompt, enhance_model, nsfw=args.nsfw)
                if enhanced_result != prompt:
                    enhanced_prompt = enhanced_result
                    print(f"     Rewritten for Agate ({len(enhanced_result)} chars)")
                    print(f"     ─── Enhanced prompt ───")
                    import textwrap
                    for line in textwrap.wrap(enhanced_result, width=78):
                        print(f"     {line}")
                    print(f"     ────────────────────────")
                else:
                    print(f"     ⚠️ Enhancement failed — using raw prompt")
                    if raw_response and raw_response != prompt:
                        print(f"     ─── LLM response ───")
                        display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                        print(f"     {display}")
                        print(f"     ────────────────────")
            elif enhance_type == "vision":
                print(f"  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
                enhanced_result, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
                if enhanced_result != prompt:
                    enhanced_prompt = enhanced_result
                    print(f"     Expanded to English description ({len(enhanced_result)} chars)")
                    print(f"     ─── Enhanced prompt ───")
                    import textwrap
                    for line in textwrap.wrap(enhanced_result, width=78):
                        print(f"     {line}")
                    print(f"     ────────────────────────")
                else:
                    print(f"     ⚠️ Enhancement failed — using raw prompt")
                    if raw_response and raw_response != prompt:
                        print(f"     ─── LLM response ───")
                        display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                        print(f"     {display}")
                        print(f"     ────────────────────")
            else:
                print(f"  ✨ Enhancing prompt via {enhance_model} (legacy JSON mode)...")
                enhanced_result, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)
                if enhanced_result != prompt:
                    enhanced_prompt = enhanced_result
                    print(f"     Expanded to JSON ({len(enhanced_result)} chars)")
                    print(f"     ─── Enhanced prompt ───")
                    try:
                        parsed = json.loads(enhanced_result)
                        for key, val in parsed.items():
                            if isinstance(val, dict):
                                print(f"     {key}:")
                                for k, v in val.items():
                                    print(f"       {k}: {v}")
                            else:
                                print(f"     {key}: {val}")
                    except json.JSONDecodeError:
                        print(f"     {enhanced_result}")
                    print(f"     ────────────────────────")
                else:
                    print(f"     ⚠️ Enhancement failed — using raw prompt")
                    if raw_response and raw_response != prompt:
                        print(f"     ─── LLM response ───")
                        display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                        print(f"     {display}")
                        print(f"     ────────────────────")
            # Apply enhanced prompt
            if enhance_type in ("vision", "agate") and enhanced_result != prompt:
                prompt = enhanced_result

    # Show warm/cold estimate
    prior = get_previous_runs(model_name, width, height)
    print()
    if prior:
        mean_s = sum(prior) / len(prior)
        best_s = min(prior)
        print(f"  ⚡ {len(prior)} prior run(s) at {width}×{height} — warmed kernels available")
        print(f"     Historical wall: mean {mean_s:.1f}s, best {best_s:.1f}s")
    else:
        if backend_type == "agate":
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Expected: ~15s (imports + model load + CUDA graph warmup)")
            print(f"     Subsequent runs: ~2-4s")
        else:
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Cold start: ~30-60s (imports + model load + kernel JIT)")
            print(f"     Subsequent runs at this size will be faster (cached kernels)")
    print()

    # ── Phase 1.5: Evict LLMs after prompt enhancement ──
    # If we used an LLM for enhancement, evict it before loading the diffusion model
    if args.enhance:
        running = llama_swap_running_models()
        if running:
            print(f"  🔄 Evicting LLM models after enhancement: {', '.join(running)}")
            evict_llm()
            print(f"     VRAM freed for image generation")
            print()


    # ── Phase 1: Load pipeline ──
    print(f"  [1/3] Loading pipeline ({model_info['bits']})...")
    pipeline, load_time = load_pipeline(model_name, editing=bool(ref_image_paths))
    if backend_type == "gemlite":
        print(f"        Pipeline ready in {load_time:.1f}s")

    # ── Phase 2: Generate ──
    gen_desc = f"{width}×{height}, {args.steps} steps, seed={seed}"
    if ref_image_paths:
        gen_desc += ", editing"
    print(f"  [2/3] Generating ({gen_desc})...")
    wall_t0 = time.perf_counter()

    if backend_type == "gemlite":
        png_bytes, diffusion_time, peak_hbm = generate_image_gemlite(
            pipeline, prompt, seed, args.steps, width, height,
        )
        # Use the caller's cwd, not the script's directory
        orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
        output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)
        output_path.write_bytes(png_bytes)
    elif backend_type == "agate":
        # Default "sharper faces" do card 001: autoguide 1.0 (cfg continua em 3.0 a menos que --cfg passe 4.0 junto)
        agate_autoguide = args.agate_autoguide
        if agate_autoguide is None:
            agate_autoguide = 1.0 if (model_name or "").startswith("agate-preview-001") else 0.0
        png_bytes, diffusion_time, peak_hbm = generate_image_agate(
            pipeline, prompt, seed, args.steps, width, height,
            cfg=args.cfg, autoguide=agate_autoguide,
        )
        orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
        output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)
        output_path.write_bytes(png_bytes)
    wall_time = time.perf_counter() - wall_t0

    # ── Phase 3: Save + Unload ──
    print(f"  [3/3] Saving & unloading...")
    save_metadata(
        model_name, prompt, seed, width, height, args.steps,
        load_time, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced_prompt,
    )

    if backend_type in ("gemlite", "agate"):
        unload_pipeline()

    # ── Debrief ──
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, args.steps,
        load_time, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced_prompt,
        original_prompt=original_prompt,
    )

    # ── Open in viewer ──
    if args.open:
        open_image(output_path)


def _run_framepack(
    args: argparse.Namespace,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> None:
    """Handle FramePack I2V video generation — separate path from image generation."""
    from diffuse.backends.framepack import require_models, check_models

    # Validate input image
    input_image_path = args.input_image
    if input_image_path is None:
        print("  ✗ FramePack I2V requires --input-image")
        print("     Usage: diffuse -m framepack-i2v --input-image photo.png -p 'description'")
        sys.exit(1)
    if not input_image_path.exists():
        orig_cwd = os.environ.get("DIFFUSE_ORIG_CWD", "")
        if orig_cwd:
            resolved = Path(orig_cwd) / input_image_path
            if resolved.exists():
                input_image_path = resolved
            else:
                print(f"  ✗ Input image not found: {input_image_path} (also tried {resolved})")
                sys.exit(1)
        else:
            print(f"  ✗ Input image not found: {input_image_path}")
            sys.exit(1)

    # Video duration
    total_seconds = args.seconds if args.seconds is not None else model_info.get("default_seconds", 5.0)
    total_seconds = max(0.5, min(total_seconds, 120.0))
    use_teacache = not args.no_teacache
    steps = args.steps or model_info.get("default_steps", 25)

    print(f"  🎬 FramePack I2V: {width}×{height}, {total_seconds:.1f}s, {steps} steps, seed={seed}")
    print(f"     Input: {Path(input_image_path).name}")
    print(f"     TeaCache: {'ON' if use_teacache else 'OFF'}, CFG={args.cfg}, GS={args.gs}")

    # Evict LLMs before loading (FramePack needs ~4.5-6GB VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  🔄 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for video generation")

    print(f"  [1/3] Loading FramePack pipeline ({model_info['bits']})...")
    pipeline, load_time = load_pipeline(model_name)
    print(f"        Pipeline ready in {load_time:.1f}s")

    print(f"  [2/3] Generating video...")
    wall_t0 = time.perf_counter()

    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    out_dir = orig_cwd
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(":", "_").replace("/", "_")
    default_output = str(out_dir / f"diffuse_{safe_model}_{ts}_seed{seed}.mp4")

    output_file, diffusion_time = generate_video_framepack(
        pipeline,
        input_image_path=str(input_image_path),
        prompt=prompt,
        seed=seed,
        total_second_length=total_seconds,
        steps=steps,
        cfg=args.cfg if args.cfg is not None else 1.0,
        gs=args.gs,
        rs=0.0,
        gpu_memory_preservation=6.0,
        use_teacache=use_teacache,
        mp4_crf=16,
        output_path=args.output and str(args.output.with_suffix(".mp4")) or None,
    )
    wall_time = time.perf_counter() - wall_t0

    output_path = Path(output_file) if output_file else None
    peak_hbm = 0.0

    print(f"  [3/3] Unloading...")
    unload_pipeline()

    print()
    print("═══ diffuse — Video Generation Report ═══")
    print(f"  Model:       {model_name}")
    print(f"  Prompt:      \"{original_prompt}\"")
    print(f"  Input:       {Path(input_image_path).name}")
    print(f"  Duration:    {total_seconds:.1f}s")
    print(f"  Seed:        {seed}")
    print(f"  Resolution:  {width}×{height}")
    print(f"  Steps:       {steps}")
    print(f"  TeaCache:    {'ON' if use_teacache else 'OFF'}")
    print()
    print("  Timings:")
    print(f"    Setup:      {load_time:7.2f} s   (model load + DynamicSwap)")
    print(f"    Generation: {diffusion_time:7.2f} s   (video denoising + VAE decode)")
    print(f"    ─────────────────────")
    print(f"    Wall:       {wall_time:7.2f} s")
    print()
    print(f"  Output: {output_path}")
    print("══════════════════════════════════════")

    # Open video in player
    if args.open and output_path:
        import shutil
        import subprocess
        viewer = shutil.which("mpv") or shutil.which("vlc") or shutil.which("ffplay")
        if viewer:
            subprocess.Popen([viewer, str(output_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
def _show_enhanced_if_requested(args, enhanced: str | None) -> None:
    """Print the expanded prompt right after enhancement — ALWAYS (default since 09/10).

    Was opt-in via --show-enhanced (which now is kept as a no-op for script
    compatibility). Printing unconditionally lets the user abort a bad
    expansion (Ctrl+C) before paying for the render — mirrors the h3 wrapper.
    """
    if not enhanced:
        return
    print()
    print("  ── Enhanced prompt " + "─" * 52)
    for line in str(enhanced).split("\n"):
        print(f"  │ {line}")
    print("  " + "─" * 70)



def _run_bonsai_image(
    args: argparse.Namespace,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> None:
    """Handle Bonsai image generation via subprocess (separate venv)."""
    from diffuse.backends.bonsai import load_pipeline_bonsai, generate_image_bonsai

    steps = args.steps
    guidance = 1.0

    print(f"  🎨 Bonsai T2I: {width}×{height}, {steps} steps, seed={seed}")

    # Check model dir
    model_dir = Path(__file__).resolve().parent.parent / "models" / model_info["dir"]
    if not model_dir.exists():
        print(f"\n  ✗ Model not found: {model_dir}")
        print(f"    Run: diffuse download bonsai")
        sys.exit(1)

    # ── Prompt enhancement ──
    # fix V5: tags manuais extraídas ANTES do enhance (NameError pré-existente —
    # lora_tags era usada no reapply sem nunca ser definida neste runner)
    lora_tags = ""
    if "<lora:" in prompt:
        prompt, lora_tags = _extract_lora_tags(prompt)
    enhanced = None
    if args.enhance or args.enhance_with:
        # Resolve pela CONFIG DO FLEET (pode trazer :think). Antes esta linha
        # re-resolvia do models.py e o :think do fleet se perdia no payload.
        enhance_model = args.enhance_with or _load_enhance_config(model_info)
        enhance_type = model_info.get("enhance_type", "vision")

        if enhance_type == "vision":
            print(f"\n  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)

        if enhanced and enhanced != prompt:
            enhanced = _reapply_lora_tags(enhanced, lora_tags)
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced

    # Evict LLMs before loading
    running = llama_swap_running_models()
    if running:
        print(f"  🔄 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    print(f"  [1/3] Loading Bonsai pipeline ({model_info['bits']})...")
    config, load_time = load_pipeline_bonsai(model_name)
    print(f"        Pipeline ready in {load_time:.1f}s")

    # Output path
    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] Generating...")
    wall_t0 = time.perf_counter()
    output_path, diffusion_time, peak_hbm = generate_image_bonsai(
        config,
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        output_path=output_path,
    )
    wall_time = time.perf_counter() - wall_t0

    print(f"  [3/3] Done — unloading...")

    save_metadata(
        model_name, prompt, seed, width, height, steps,
        0.0, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
    )
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, steps, 0.0,
        diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
        original_prompt=original_prompt,
    )


def _run_fibo_sdcpp_image(
    args,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None,
) -> None:
    """Fibo 2.0 via sd.cpp: T2I no Dev/Turbo e edição por referência no Edit Turbo.

    Exceção contida à doutrina F5 (sd.cpp fora da frota): o Wan2GP não tem
    handler de Fibo e o sd.cpp upstream tem arch nativa (`#2180`).
    """
    from diffuse.backends.sdcpp import generate_image_sdcpp

    is_edit = bool(ref_image_paths)
    steps = args.steps if args.steps is not None else (4 if "turbo" in model_name else 20)
    cfg = model_info.get("cfg_scale", 1.0)

    print(f"  🎨 Fibo 2.0 {'edição' if is_edit else 'T2I'}: {width}×{height}, {steps} steps, cfg={cfg}, seed={seed}")

    require_model_dir(model_name)

    # ── Prompt enhancement ──
    lora_tags = ""
    if "<lora:" in prompt:
        prompt, lora_tags = _extract_lora_tags(prompt)
    enhanced = None
    if args.enhance or args.enhance_with:
        # Resolve pela CONFIG DO FLEET (pode trazer :think). Antes esta linha
        # re-resolvia do models.py e o :think do fleet se perdia no payload.
        enhance_model = args.enhance_with or _load_enhance_config(model_info)
        enhance_type = model_info.get("enhance_type", "qwen21")
        if enhance_type == "fibo":
            print(f"\n  ✨ Enhancing prompt via {enhance_model} (FIBO VGL JSON caption)...")
            from diffuse.enhance import enhance_fibo_prompt
            enhanced, _ = enhance_fibo_prompt(prompt, enhance_model, nsfw=args.nsfw)
        elif enhance_type == "vision":
            print(f"\n  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, _ = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, _ = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)
        if enhanced and enhanced != prompt:
            enhanced = _reapply_lora_tags(enhanced, lora_tags)
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced

    # Evict LLMs antes de carregar (o DiT do Fibo usa RAM pesada via --offload-to-cpu)
    running = llama_swap_running_models()
    if running:
        print(f"  🔄 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [1/3] Rodando sd-cli ({model_info['bits']})...")
    try:
        output_path, wall_time = generate_image_sdcpp(
            model_name=model_name,
            prompt=prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            output_path=output_path,
            ref_images=ref_image_paths,
            cfg_scale=cfg,
        )
    except RuntimeError as exc:
        print(f"\n  ✗ {exc}")
        sys.exit(1)

    print(f"  [3/3] Done.")

    save_metadata(
        model_name, prompt, seed, width, height, steps,
        cfg, 0.0, wall_time, 0.0, output_path,
        enhanced_prompt=enhanced,
    )
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, steps, cfg,
        0.0, wall_time, 0.0, output_path,
        enhanced_prompt=enhanced,
        original_prompt=original_prompt,
    )


def _run_flux3_wan2gp_image(
    args,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None,
) -> None:
    """Handle FLUX 3 Klein generation/editing via WanGP.

    Klein ships distilled: fixed 16 steps, cfg 1.0. Editing takes up to 10
    reference images. WanGP mmgp Profile 5 offloads blocks one-by-one —
    the same regime as Ming/H3 in the 6 GB fleet — so the denoise runs on
    GPU (sem fallback CPU — sd.cpp saiu da frota, F5).
    """
    from diffuse.backends.wan2gp import (
        load_pipeline_wan2gp,
        generate_image_flux3_wan2gp,
    )

    is_edit = bool(ref_image_paths)
    steps = args.steps if args.steps is not None else 16
    guidance = args.cfg if args.cfg is not None else 1.0

    if is_edit:
        print(f"  \U0001f3a8 FLUX 3 Klein editing: {Path(ref_image_paths[0]).name}")
        print(f"     Instruction: {prompt[:80] if len(prompt) > 80 else '...' if not prompt else prompt[:80]}")
    else:
        print(f"  \U0001f3a8 FLUX 3 Klein T2I")
    print(f"     {width}x{height}, {steps} steps, cfg={guidance}, seed={seed}")

    # -- Prompt enhancement --
    # Edit-aware: with refs, the VLM SEES the image and refines the INSTRUCTION
    # (analyze_and_enhance_edit, same one-shot flow as qwen21). The plain
    # T2I enhancer re-describes the scene from scratch ("young woman in her late
    # 20s...") and the diffusion loses the reference identity — the iany bug.
    # Without refs, standard T2I vision enhancement.
    enhanced = None
    if args.enhance or args.enhance_with:
        # Resolve pela CONFIG DO FLEET (pode trazer :think). Antes esta linha
        # re-resolvia do models.py e o :think do fleet se perdia no payload.
        enhance_model = args.enhance_with or _load_enhance_config(model_info)
        if is_edit and _check_model_vision(enhance_model):
            print(f"\n  \U0001f441\ufe0f\u2728 {enhance_model} analyzes the reference(s) and refines the edit instruction...")
            enhanced, raw_response = analyze_and_enhance_edit(
                ref_image_paths, prompt, enhance_model, nsfw=args.nsfw
            )
        else:
            print(f"\n  \u2728 Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        if enhanced and enhanced != prompt:
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced
        else:
            print(f"     \u26a0\ufe0f  Enhancement failed - using raw prompt")

    # Evict LLMs (TE runs on CPU, but nothing else should hold VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  \U0001f504 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    print(f"  [1/3] Loading FLUX 3 Klein pipeline ({model_info['bits']})...")
    config, load_time = load_pipeline_wan2gp(model_name)
    print(f"        WanGP config ready")

    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] {'Editing' if is_edit else 'Generating'} (WanGP profile 5, GPU offload)...")
    wall_t0 = time.perf_counter()
    output_path, diffusion_time, peak_hbm = generate_image_flux3_wan2gp(
        config, prompt, seed, width, height, output_path,
        ref_images=ref_image_paths, steps=steps, cfg_scale=guidance,
    )
    wall_time = time.perf_counter() - wall_t0

    print(f"  [3/3] Done - unloading...")

    save_metadata(
        model_name, prompt, seed, width, height, steps,
        0.0, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
    )
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, steps, 0.0,
        diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
        original_prompt=original_prompt,
    )


def _run_moss_wan2gp_image(
    args,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None,
) -> None:
    """Handle MOSS-Image SFT generation via WanGP (T2I only).

    OpenMOSS release recipe: 50 steps, cfg 3, shift 5, SLG layer-22 skip built
    into the pipeline. Uncensored by training doctrine — the official negative
    tag set (nsfw, worst quality...) is applied ONLY for SFW requests (--nsfw
    drops the negative entirely). No reference images: structured editing goes
    through the JSON layout prompt format.
    """
    from diffuse.backends.wan2gp import (
        load_pipeline_wan2gp,
        generate_image_wan2gp,
    )

    # model_type pro wgp = stem do defaults/*.json: moss_image (Base) e
    # moss_image_sft (def do DDM); kromoss = variant local (defaults criado pela casa).
    variant = ("moss_image" if model_name == "moss-image-base"
               else "moss_image_kromoss" if model_name == "moss-image-kromoss"
               else "moss_image_imagine" if model_name == "moss-imagine"
               else "moss_image_photoreal" if model_name == "moss-image-photoreal"
               else "moss_image_sft")
    steps = args.steps if args.steps is not None else 50
    guidance = args.cfg if args.cfg is not None else 3.0
    negative = None if args.nsfw else "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
    if args.nsfw and model_name == "moss-image-photoreal":
        # Default NSFW da casa p/ photoreal (ladder 27/out, seed fixa, medido com
        # user OK): clausula anti-quente = degrau FULL do A/B (saturacao de pele
        # -14%, R-ratio -3.5% vs controle; LIGHT sozinho = no-op, ORANGE = no-op
        # sobre FULL). Conceitos a suprimir APENAS — doutrina Round 3 SFT.
        negative = "skin redness, flushed cheeks, sunburn, tan, bronze skin, warm undertone"

    # --turbo: DMD 8-step distillation LoRA @1.0 + cfg 1.0, nativa por variante
    # (o arquivo que o card de cada variante embarca): SFT -> moss_dmd_8step
    # (destilada sobre o SFT); Base e Kromoss -> moss_dmd_8step_base (destilada
    # sobre o Base = a LoRA bundlada no proprio repo do Kromoss, sha-identica
    # f0e6cd07; PartiPrompts do card: 27.82 — o melhor score deles).
    # Cross SFT->kromoss = tag manual <lora:moss_dmd_8step:1.0> no prompt, que
    # vence o turbo (com tag de DMD explicita, a nativa NAO e injetada — evita
    # empilhar duas destilacoes). Sem tag, 8 steps / cfg 1.0 (lição do --pruna).
    # Non-selectable: injected via activated_loras directly, never via the
    # --enhance catalog. 8 steps cfg 1.0 validated E2E (poster A/B: 24s vs
    # 4m34s @512, zero visible loss).
    loras_extra: list[str] = []
    mults_extra: list[str] = []
    if getattr(args, "turbo", False):
        _dmd_native = {
            "moss_image_sft": ("moss_dmd_8step.safetensors", "DMD-SFT 8-step"),
            "moss_image": ("moss_dmd_8step_base.safetensors", "DMD-Base 8-step"),
            "moss_image_kromoss": ("moss_dmd_8step_base.safetensors",
                                   "DMD-Base 8-step (a LoRA que o proprio card do Kromoss embarca, sha f0e6cd07)"),
            "moss_image_imagine": ("moss_dmd_8step.safetensors",
                                   "DMD-SFT 8-step (Imagine e derivada do SFT; compat do merge rank-32 = probe)"),
            "moss_image_photoreal": ("moss_dmd_8step.safetensors",
                                     "DMD-SFT 8-step (Photoreal e merge sobre o SFT — mesmo DMD do SFT)"),
        }
        _dmd_file, _dmd_label = _dmd_native[variant]
        if "<lora:moss_dmd_8step:" in prompt or "<lora:moss_dmd_8step_base:" in prompt:
            print(f"     TURBO: tag manual de DMD vence — nativa ({_dmd_label}) não injetada")
        else:
            loras_extra = [_dmd_file]
            mults_extra = ["1.0"]
        # main() already materialized the per-backend default into args.steps
        # (50), so check against that, not None (the --pruna lesson).
        if args.steps in (None, 50):
            steps = 8
        if args.cfg is None:
            guidance = 1.0
            negative = None  # cfg 1.0: negative é no-op
        print(f"     TURBO: {_dmd_label} LoRA @1.0 (cfg {guidance}, {steps} steps)")

    label = {"moss_image": "Base", "moss_image_kromoss": "Kromoss",
             "moss_image_imagine": "Imagine"}.get(variant, "SFT")
    print(f"  \U0001f5bc\ufe0f MOSS-Image {label} T2I")
    print(f"     {width}x{height}, {steps} steps, cfg={guidance}, shift=5, seed={seed}")
    if args.nsfw:
        print(f"     NSFW: negative prompt desativado (modelo uncensored por doutrina de treino)")
    else:
        print(f"     SFW: negative '{negative}'")

    # -- Prompt enhancement + LoRA selection (moss, F4) --
    # seleção unificada via diffuse/lora_selector.py (rerank ColBERT no catálogo
    # loras_moss_image.json + LLM pela linha LORAS: OU modelo de decisão); o
    # backend wan2gp converte tags -> activated_loras. Fail-open garantido pelo
    # módulo: sem catálogo/rede/linha LORAS: = roda base.
    lora_tags = ""
    if "<lora:" in prompt:
        prompt, lora_tags = _extract_lora_tags(prompt)
    enhanced = None
    from diffuse.lora_selector import select_loras, load_selector_cfg
    _cfg = load_selector_cfg(args)
    # variant de catálogo: as 3 entries moss têm arquiteturas de DiT distintas
    # (base/sft: proj.q/k/v separados; kromoss: qkv fundido — keymap do wgp cobre,
    # provado no V-MOSS; o compat por variante fica pronto pra quando discriminar)
    _variant_moss = {"moss-image": "sft", "moss-image-base": "base",
                     "moss-image-kromoss": "kromoss", "moss-imagine": "imagine",
                     "moss-image-photoreal": "photoreal"}.get(model_name, "*")
    if args.enhance or args.enhance_with:
        # Resolve pela CONFIG DO FLEET (pode trazer :think). Antes esta linha
        # re-resolvia do models.py e o :think do fleet se perdia no payload.
        enhance_model = args.enhance_with or _load_enhance_config(model_info)
        print(f"\n  ✨ Enhancing prompt via {enhance_model} (moss doctrine)...")
        if _cfg.selector == "llm" and "<lora:" not in prompt:
            # uma call: enhance + seleção juntos (fluxo histórico do moss)
            prompt, _res, info = select_loras(
                prompt, root="moss_image", nsfw=bool(args.nsfw), variant=_variant_moss,
                cfg=_cfg, enhance_fn=enhance_moss_prompt, enhance_model=enhance_model)
            if info.get("body"):
                _show_enhanced_if_requested(args, info["body"])
        else:
            # decision (ou tag manual): enhance puro; seleção decision depois
            enhanced, raw_response = enhance_moss_prompt(prompt, enhance_model, nsfw=args.nsfw)
            if enhanced and enhanced != prompt:
                print(f"     Expanded to ({len(enhanced)} chars)")
                _show_enhanced_if_requested(args, enhanced)
                prompt = enhanced
            else:
                print(f"     ⚠️  Enhancement failed - using raw prompt")

    # tags manuais vencem SEMPRE — reapply FORA do bloco enhance (bug 20/out/2026:
    # sem --enhance, tags eram extraídas e nunca devolvidas; a LoRA era
    # silenciosamente ignorada — renders bit-idênticos com/sem tag, achado no V-MOSS)
    if lora_tags:
        prompt = _reapply_lora_tags(prompt, lora_tags)

    # seleção decision (sobre o texto final) — roda SEM --enhance e SEM --nsfw:
    # o decision não precisa de LLM enhance (julga o prompt cru) e o catálogo
    # tem LoRA estilo não-NSFW (ids 2/3/4/5/10) que merece seleção em prompt comum
    if _cfg.selector == "decision" and "<lora:" not in prompt:
        _pre_moss = prompt
        prompt, _res, _info = select_loras(
            prompt, root="moss_image", nsfw=bool(args.nsfw), variant=_variant_moss, cfg=_cfg)
        if prompt != _pre_moss and enhanced is None:
            enhanced = prompt

    # Evict LLMs (TE runs on CPU, but nothing else should hold VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  \U0001f504 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    print(f"  [1/3] Loading MOSS-Image pipeline ({model_info['bits']})...")
    config, load_time = load_pipeline_wan2gp(model_name)
    print(f"        WanGP config ready")

    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] Generating (WanGP profile 5, GPU offload)...")
    wall_t0 = time.perf_counter()
    output_path, diffusion_time, peak_hbm = generate_image_wan2gp(
        config, prompt, seed, width, height, output_path,
        ref_images=None, steps=steps, cfg_scale=guidance,
        model_type=variant, negative_prompt=negative or "",
        unit_name="diffuse-moss", lora_root_name="moss_image", flow_shift=5,
        activated_loras=loras_extra, loras_multipliers=mults_extra,
    )
    wall_time = time.perf_counter() - wall_t0

    print(f"  [3/3] Done - unloading...")

    save_metadata(
        model_name, prompt, seed, width, height, steps,
        0.0, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
    )
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, steps, 0.0,
        diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
        original_prompt=original_prompt,
    )


def _run_qwen21_turbo_wan2gp_image(
    args,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None,
) -> None:
    """Qwen-Image 2.1 Turbo via WanGP (official Alibaba distill).

    8 steps, cfg 1.0, qwen21_turbo scheduler (fixed released sigmas). TE = the
    local Sakura Heretic GGUF 8B (remapped natively by the handler). WanGP mmgp
    Profile 5 offloads blocks one-by-one — denoise on GPU, same regime as the
    flx3/Ming/H3 routes in the 6 GB fleet.
    """
    from diffuse.backends.wan2gp import (
        load_pipeline_wan2gp,
        generate_image_wan2gp,
    )

    is_edit = bool(ref_image_paths)
    steps = args.steps if args.steps is not None else 8
    guidance = args.cfg if args.cfg is not None else 1.0

    if is_edit:
        print(f"  \U0001f3a8 Qwen-Image 2.1 Turbo editing: {Path(ref_image_paths[0]).name}")
    else:
        print(f"  \U0001f3a8 Qwen-Image 2.1 Turbo T2I (official distill)")
    print(f"     {width}x{height}, {steps} steps, cfg={guidance}, seed={seed}")

    # -- Prompt enhancement + LoRA selection (qwen21 flow, ported from the
    # sd.cpp runner — catálogo ranked via ColBERT rerank, LLM picks 0-2 via the
    # LORAS: <ids> line, resolve -> <lora:tags> re-applied; the wan2gp backend
    # then converts tags -> activated_loras settings, see generate_image_wan2gp).
    lora_tags = ""
    if "<lora:" in prompt:
        prompt, lora_tags = _extract_lora_tags(prompt)
    enhanced = None
    if args.enhance or args.enhance_with or getattr(args, "enhance_edit_with", None):
        # Resolve pela CONFIG DO FLEET (pode trazer :think). Antes esta linha
        # re-resolvia do models.py e o :think do fleet se perdia no payload.
        enhance_model = args.enhance_with or _load_enhance_config(model_info)
        enhance_type = model_info.get("enhance_type", "qwen21")
        from diffuse.lora_selector import select_loras, load_selector_cfg
        cfg = load_selector_cfg(args)
        _manual_tag = bool(lora_tags)  # tags manuais vencem o seletor (decreto)

        if is_edit:
            # EDIT MULTI-REF: --enhance em edição SEMPRE usa o caminho vision (refs
            # no VLM, doutrina de refs numeradas). O enhancer text-only re-descreve
            # a cena do zero e MATA a identidade das refs (bug do iany, 2025) —
            # sem modelo vision = prompt CRU (refs fazem o trabalho), jamais re-descrição.
            edit_model = getattr(args, "enhance_edit_with", None) or enhance_model
            if not _check_model_vision(edit_model):
                print(f"  ⚠️  EDIT: {edit_model} sem visão — enhance SKIPADO: refs só existem pros olhos do diffusion; use --enhance-edit-with <modelo-vision> para o VLM ver as refs")
            else:
                print(f"\n  👁️✨ {edit_model} analisa a referência e refina a instrução (one-shot)...")
                if cfg.selector == "llm" and not _manual_tag:
                    def _edit_fn(p, m, nsfw, extra_system):
                        return analyze_and_enhance_edit(ref_image_paths, p, m,
                                                        nsfw=nsfw, extra_system=extra_system)
                    prompt, _res, info = select_loras(
                        prompt, root="qwen21", nsfw=bool(getattr(args, "nsfw", False)),
                        variant="*", cfg=cfg, enhance_fn=_edit_fn, enhance_model=edit_model)
                    if info.get("body"):
                        _show_enhanced_if_requested(args, info["body"])
                        import textwrap as _tw
                        print(f"     ─── Enhanced edit prompt ───")
                        for _line in _tw.wrap(info["body"], width=78):
                            print(f"     {_line}")
                        print(f"     ────────────────────────────")
                else:
                    enhanced, raw_response = analyze_and_enhance_edit(
                        ref_image_paths, prompt, edit_model, nsfw=args.nsfw)
        elif enhance_type == "qwen21":
            if getattr(args, "nsfw", False) and cfg.selector == "llm":
                pass  # --nsfw + llm: UMA call só no post-bloco (fluxo histórico)
            elif cfg.selector == "llm" and not _manual_tag:
                prompt, _res, info = select_loras(
                    prompt, root="qwen21", nsfw=False, variant="*", cfg=cfg,
                    enhance_fn=enhance_qwen21_prompt, enhance_model=enhance_model)
                if info.get("body"):
                    _show_enhanced_if_requested(args, info["body"])
            else:
                print(f"\n  ✨ Enhancing prompt via {enhance_model} (qwen21 mode)...")
                enhanced, raw_response = enhance_qwen21_prompt(prompt, enhance_model, nsfw=args.nsfw)
        elif enhance_type == "vision":
            print(f"\n  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)

        if enhanced and enhanced != prompt:
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced

    # tags manuais vencem SEMPRE — reapply FORA do bloco enhance (bug 20/out/2026:
    # sem --enhance, tags eram extraídas e nunca devolvidas; a LoRA era
    # silenciosamente ignorada — renders bit-idênticos com/sem tag, achado no V-MOSS)
    if lora_tags:
        prompt = _reapply_lora_tags(prompt, lora_tags)

    # ── seleção de LoRA (F3, lora_selector.py): rota --nsfw/llm (uma call) e
    # modo decision (pós-enhance, sobre o texto final). Fail-open garantido pelo
    # módulo; guarda '<lora:' not in prompt impede dupla seleção. ──────────────
    if "<lora:" not in prompt:
        from diffuse.lora_selector import select_loras, load_selector_cfg
        _cfg = load_selector_cfg(args)
        _nsfw = bool(getattr(args, "nsfw", False))
        _did_enhance = bool(args.enhance or args.enhance_with or getattr(args, "enhance_edit_with", None))
        _pre_sel = prompt
        if _cfg.selector == "decision":   # sempre: estilo não-NSFW também merece seleção
            prompt, _res, _info = select_loras(
                prompt, root="qwen21", nsfw=_nsfw, variant="*", cfg=_cfg)
        elif _nsfw and _cfg.selector == "llm" and _did_enhance:
            prompt, _res, _info = select_loras(
                prompt, root="qwen21", nsfw=True, variant="*", cfg=_cfg,
                enhance_fn=enhance_qwen21_prompt,
                enhance_model=args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b-heretic"))
            if _info.get("body"):
                _show_enhanced_if_requested(args, _info["body"])
        elif _nsfw and not _did_enhance and _cfg.selector == "llm":
            print("  ⚠️  --nsfw sem --enhance: sem seleção inteligente (modo llm); rodando base")
        if prompt != _pre_sel and enhanced is None:
            enhanced = prompt

    # Evict LLMs (TE runs on CPU with block offload; nothing else holds VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  \U0001f504 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    print(f"  [1/3] Loading Qwen-Image 2.1 Turbo pipeline (WanGP, {model_info['bits']})...")
    config, load_time = load_pipeline_wan2gp(model_name)
    print(f"        WanGP config ready")

    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] {'Editing' if is_edit else 'Generating'} (WanGP profile 5, GPU offload)...")
    wall_t0 = time.perf_counter()
    output_path, diffusion_time, peak_hbm = generate_image_wan2gp(
        config, prompt, seed, width, height, output_path,
        ref_images=ref_image_paths, steps=steps, cfg_scale=guidance,
        model_type="qwen_image_21_turbo_7B",
    )
    wall_time = time.perf_counter() - wall_t0

    print(f"  [3/3] Done - unloading...")

    save_metadata(
        model_name, prompt, seed, width, height, steps,
        0.0, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
    )
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, steps, 0.0,
        diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
        original_prompt=original_prompt,
    )


def _run_qwen21_wan2gp_image(
    args,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None,
) -> None:
    """Qwen-Image 2.1 UNCENSORED (abenzerps UC) via WanGP — the base route.

    40 steps, cfg 4.0 (base defaults), T2I + edit/outpaint/RGBA via the SAME
    WanGP qwen21 handler as the turbo sibling (json irmaos, mesmo handler;
    TE = qwen3vl_8b abliterated convrot, direcao de recusa fora). WanGP mmgp
    Profile 5 offloads blocks one-by-one — denoise on GPU, same regime as the
    flx3/Ming/H3 routes in the 6 GB fleet.
    """
    from diffuse.backends.wan2gp import (
        load_pipeline_wan2gp,
        generate_image_wan2gp,
    )

    is_edit = bool(ref_image_paths)
    steps = args.steps if args.steps is not None else 40
    guidance = args.cfg if args.cfg is not None else 4.0
    # --turbo: LoRA de destilacao aplicado SOBRE o base uncensored (a 2a passada
    # do plano: o turbo-CHECKPOINT da Alibaba e censurado de fabrica; o base
    # abenzerps nao tem refusals, o distill-lora acelera sem trocar o vocabulario).
    # Sem o lora em disco ou sem tag manual: roda base cru (avisado), nunca falha.
    loras_extra: list[str] = []
    mults_extra: list[str] = []
    turbo = getattr(args, "turbo", False)
    if turbo and "<lora:" in prompt:
        print("     TURBO: tag manual de LoRA no prompt vence — injecao nativa pulada (distill embate = monstro)")
        turbo = False
    if turbo:
        _wgp_loradir = Path.home() / "git" / "Wan2GP" / "loras" / "qwen21"
        candidates = sorted(_wgp_loradir.glob("*turbo*.safetensors")) if _wgp_loradir.exists() else []
        if candidates:
            loras_extra = [candidates[0].name]
            mults_extra = ["1.0"]
            if args.steps in (None, 40):
                steps = 8
            if args.cfg is None:
                guidance = 1.0
                negative = None
            print(f"     TURBO: distill LoRA {candidates[0].name} @1.0 (8 steps, cfg 1.0)")
        else:
            print("     TURBO: LoRA distill nao instalado ainda (passada 2) — rodando base cru")
    if is_edit:
        print(f"  \U0001f3a8 Qwen-Image 2.1 Uncensored editing: {Path(ref_image_paths[0]).name}")
    else:
        print(f"  \U0001f3a8 Qwen-Image 2.1 Uncensored T2I (base + TE abliterated)")
    print(f"     {width}x{height}, {steps} steps, cfg={guidance}, seed={seed}")

    # -- Prompt enhancement + LoRA selection (qwen21 flow, ported from the
    # sd.cpp runner — catálogo ranked via ColBERT rerank, LLM picks 0-2 via the
    # LORAS: <ids> line, resolve -> <lora:tags> re-applied; the wan2gp backend
    # then converts tags -> activated_loras settings, see generate_image_wan2gp).
    lora_tags = ""
    if "<lora:" in prompt:
        prompt, lora_tags = _extract_lora_tags(prompt)
    # Evict LLMs (TE runs on CPU with block offload; nothing else holds VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  \U0001f504 Evicting LLM models: {', '.join(running)}")
        evict_llm()
    enhanced = None
    if args.enhance or args.enhance_with or getattr(args, "enhance_edit_with", None):
        # Resolve pela CONFIG DO FLEET (pode trazer :think). Antes esta linha
        # re-resolvia do models.py e o :think do fleet se perdia no payload.
        enhance_model = args.enhance_with or _load_enhance_config(model_info)
        enhance_type = model_info.get("enhance_type", "qwen21")
        from diffuse.lora_selector import select_loras, load_selector_cfg
        cfg = load_selector_cfg(args)
        _manual_tag = bool(lora_tags)  # tags manuais vencem o seletor (decreto)

        if is_edit:
            # EDIT MULTI-REF: --enhance em edição SEMPRE usa o caminho vision (refs
            # no VLM, doutrina de refs numeradas). O enhancer text-only re-descreve
            # a cena do zero e MATA a identidade das refs (bug do iany, 2025) —
            # sem modelo vision = prompt CRU (refs fazem o trabalho), jamais re-descrição.
            edit_model = getattr(args, "enhance_edit_with", None) or enhance_model
            if not _check_model_vision(edit_model):
                print(f"  ⚠️  EDIT: {edit_model} sem visão — enhance SKIPADO: refs só existem pros olhos do diffusion; use --enhance-edit-with <modelo-vision> para o VLM ver as refs")
            else:
                print(f"\n  👁️✨ {edit_model} analisa a referência e refina a instrução (one-shot)...")
                if cfg.selector == "llm" and not _manual_tag:
                    def _edit_fn(p, m, nsfw, extra_system):
                        return analyze_and_enhance_edit(ref_image_paths, p, m,
                                                        nsfw=nsfw, extra_system=extra_system)
                    prompt, _res, info = select_loras(
                        prompt, root="qwen21", nsfw=bool(getattr(args, "nsfw", False)),
                        variant="*", cfg=cfg, enhance_fn=_edit_fn, enhance_model=edit_model)
                    if info.get("body"):
                        _show_enhanced_if_requested(args, info["body"])
                        import textwrap as _tw
                        print(f"     ─── Enhanced edit prompt ───")
                        for _line in _tw.wrap(info["body"], width=78):
                            print(f"     {_line}")
                        print(f"     ────────────────────────────")
                else:
                    # decision (ou tag manual): enhance de edição puro; a seleção
                    # decision roda depois, sobre o texto final (post-bloco abaixo)
                    enhanced, raw_response = analyze_and_enhance_edit(
                        ref_image_paths, prompt, edit_model, nsfw=args.nsfw)
        elif enhance_type == "qwen21":
            if getattr(args, "nsfw", False) and cfg.selector == "llm":
                pass  # --nsfw + llm: UMA call só no post-bloco (fluxo histórico)
            elif cfg.selector == "llm" and not _manual_tag:
                # gen (não-nsfw): enhance + seleção na mesma call
                prompt, _res, info = select_loras(
                    prompt, root="qwen21", nsfw=False, variant="*", cfg=cfg,
                    enhance_fn=enhance_qwen21_prompt, enhance_model=enhance_model)
                if info.get("body"):
                    _show_enhanced_if_requested(args, info["body"])
            else:
                # decision (ou tag manual): enhance puro; seleção decision depois
                print(f"\n  ✨ Enhancing prompt via {enhance_model} (qwen21 mode)...")
                enhanced, raw_response = enhance_qwen21_prompt(prompt, enhance_model, nsfw=args.nsfw)
        elif enhance_type == "vision":
            print(f"\n  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)

        if enhanced and enhanced != prompt:
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced

    # tags manuais vencem SEMPRE — reapply FORA do bloco enhance (bug 20/out/2026:
    # sem --enhance, tags eram extraídas e nunca devolvidas; a LoRA era
    # silenciosamente ignorada — renders bit-idênticos com/sem tag, achado no V-MOSS)
    if lora_tags:
        prompt = _reapply_lora_tags(prompt, lora_tags)

    # ── seleção de LoRA (F2, lora_selector.py): rota --nsfw/llm (uma call) e
    # modo decision (pós-enhance, sobre o texto final). O select_loras fail-open
    # sozinho (sem catálogo/sem elegíveis/rede fora → prompt intacto, nunca bloqueia).
    # Guarda "<lora:" not in prompt: tags manuais (acima) e seleções que já
    # escolheram pulam este bloco — nunca seleciona duas vezes. ──────────────
    if "<lora:" not in prompt:
        from diffuse.lora_selector import select_loras, load_selector_cfg
        _cfg = load_selector_cfg(args)
        _nsfw = bool(getattr(args, "nsfw", False))
        _did_enhance = bool(args.enhance or args.enhance_with or getattr(args, "enhance_edit_with", None))
        _pre_sel = prompt
        if _cfg.selector == "decision":   # sempre: estilo não-NSFW também merece seleção
            # decision: independe do enhance; julga o texto (cru ou expandido)
            prompt, _res, _info = select_loras(
                prompt, root="qwen21", nsfw=_nsfw, variant="*", cfg=_cfg)
        elif _nsfw and _cfg.selector == "llm" and _did_enhance:
            # --nsfw + llm: enhance + seleção na MESMA call (fluxo histórico)
            prompt, _res, _info = select_loras(
                prompt, root="qwen21", nsfw=True, variant="*", cfg=_cfg,
                enhance_fn=enhance_qwen21_prompt,
                enhance_model=args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b-heretic"))
            if _info.get("body"):
                _show_enhanced_if_requested(args, _info["body"])
        elif _nsfw and not _did_enhance and _cfg.selector == "llm":
            print("  ⚠️  --nsfw sem --enhance: sem seleção inteligente (modo llm); rodando base")
        if prompt != _pre_sel and enhanced is None:
            enhanced = prompt  # o registro (debrief) acompanha o prompt final
    print(f"  [1/3] Loading Qwen-Image 2.1 Uncensored pipeline (WanGP, {model_info['bits']})...")
    config, load_time = load_pipeline_wan2gp(model_name)
    print(f"        WanGP config ready")

    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] {'Editing' if is_edit else 'Generating'} (WanGP profile 5, GPU offload)...")
    wall_t0 = time.perf_counter()
    output_path, diffusion_time, peak_hbm = generate_image_wan2gp(
        config, prompt, seed, width, height, output_path,
        ref_images=ref_image_paths, steps=steps, cfg_scale=guidance,
        model_type="qwen_image_21_7B",
        activated_loras=loras_extra, loras_multipliers=mults_extra,
    )
    wall_time = time.perf_counter() - wall_t0

    print(f"  [3/3] Done - unloading...")

    save_metadata(
        model_name, prompt, seed, width, height, steps,
        0.0, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
    )
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, steps, 0.0,
        diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced,
        original_prompt=original_prompt,
    )

