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
from diffuse.backends.sd_cpp import generate_image_sd_cpp
from diffuse.backends.hidream import generate_image_hidream
from diffuse.backends.agate import generate_image_agate
from diffuse.backends.framepack import generate_video_framepack
from diffuse.llm import evict_llm, llama_swap_running_models
from diffuse.enhance import (
    enhance_prompt,
    enhance_vision_prompt,
    enhance_qwen21_prompt,
    enhance_agate_prompt,
    enhance_edit_prompt,
    analyze_image,
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
        if backend == "hidream":
            p = Path.home() / ".llama-models" / model_dir
        else:
            p = MODELS_DIR / model_dir
        return p.exists() and any(p.iterdir()) if p.exists() else False

    # Track total disk usage
    total_installed = 0.0
    total_pending = 0.0

    print()
    for cat_label, cat_key in [("IMAGE GENERATION", "image"), ("VIDEO GENERATION", "video")]:
        models = categories.get(cat_key, [])
        if not models:
            continue
        print(f"  {cat_label}")
        for name, info in sorted(models):
            installed = _check_installed(info)
            status = "" if installed else " [not installed]"
            print(f"    {name:20s} {info.get('description', ''):60s}{status}")

            total_gb = 0.0
            for comp in info.get("components", []):
                comp_name = comp["name"]
                comp_gb = comp["size_gb"]
                total_gb += comp_gb
                shared_tag = " [shared]" if "[shared]" in comp_name else ""
                print(f"      ├── {comp_name:55s} {_fmt_size_gb(comp_gb)}")

            if installed:
                total_installed += total_gb
            else:
                total_pending += total_gb
            print(f"      └── {'Total':55s} {_fmt_size_gb(total_gb)}")
            print()

    print(f"  TOTAL DISK: {_fmt_size_gb(total_installed + total_pending)} ({_fmt_size_gb(total_installed)} installed + {_fmt_size_gb(total_pending)} pending)")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffuse",
        description="diffuse — Local diffusion image generation CLI for NVIDIA RTX 3050 6GB",
        epilog=(
            "Examples:\n"
            "  diffuse -m ternary-gemlite -p 'a cat on the moon'\n"
            "  diffuse -m ternary-gemlite --enhance -p 'a rainy day at a coffee shop'\n"
            "  diffuse -m ternary-gemlite --enhance-with laguna-xs2 --evict-llm -p 'cyberpunk city'\n"
            "  diffuse -m hidream-sdnq -p 'a cat on a windowsill at golden hour'\n"
            "  diffuse -m hidream-sdnq -p 'add a red hat' --edit photo.png\n"
            "  diffuse --list                        # show model details\n"
            "\n"
            "Model capabilities:\n"
            "  ternary-gemlite  — 1.58-bit Bonsai, T2I only\n"
            "  binary-gemlite   — 1-bit Bonsai, T2I only\n"
            "  ideogram4-q4      — Ideogram 4, T2I (JSON prompts with --enhance)\n"
            "  hidream-sdnq      — HiDream-O1 SDNQ, T2I + image editing (--edit)\n"
            "  mageflow-edit-turbo — Mage-Flow-Edit-Turbo, 4B instruction-based image editing (--edit)\n"
            "\n"
            "Resolution guide (RTX 3050 6GB):\n"
            "  Bonsai:     512×512 max (OOM above this)\n"
            "  Ideogram 4: 1024×1024 native, up to 1920×1088 (~15 min)\n"
            "  HiDream:    snaps to 2048×2048 or 2560×1440 minimum\n"
            "              T2I: ~3 min | editing (--edit): ~8 min\n"
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
    p.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 4 for bonsai, 20 for ideogram4, 28 for hidream, 50 for agate).")
    p.add_argument(
        "--size", type=parse_size, default=None,
        help="Image size as WxH (default: 512x512 for bonsai, 480x480 for ideogram4, 1024x1024 for hidream; agate is fixed 256x256).",
    )
    p.add_argument("--output", type=Path, default=None, help="Output PNG path (auto-generated in cwd if not set).")
    p.add_argument("--open", action="store_true", help="Open the generated image with feh after saving.")
    p.add_argument(
        "--list", action="store_true",
        help="List available models with details and exit.",
    )
    p.add_argument(
        "--evict-llm", action="store_true",
        help="Evict all running LLM models (via llama-swap) to free VRAM before generating. "
             "Automatically done when --enhance is used (double-evict: before and after LLM call).",
    )
    p.add_argument(
        "--enhance", action="store_true",
        help="Expand prompt via LLM. "
             "For ideogram4: structured JSON with layout/colors/text. "
             "For hidream/bonsai: natural English description with character details. "
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
        help="Reference image for editing (hidream or mageflow-edit-turbo). Pass an image path to use instruction-based editing.",
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
        "--pruna", action="store_true",
        help="qwen-image-2.1 (base only): attach the Pruna 8-step distillation LoRA at strength 1.0 and default to 8 steps / cfg 1.0. Incompatible with -turbo (mixed distillation families).",
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
def main() -> None:
    args = parse_args()

    if args.list:
        print_list()
        sys.exit(0)

    setup_environment()

    model_name = args.model
    model_info = MODELS[model_name]
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    backend_type = model_info.get("backend_type", "gemlite")

    # ── Defaults per backend ──
    # Steps: bonsai=4, ideogram4=20, hidream=28, z-image=9 (8 NFE), qwen21=28, agate=50
    if args.steps is None:
        if backend_type == "hidream":
            args.steps = 28
        elif backend_type == "sd_cpp":
            args.steps = 20
        elif backend_type == "zimage_sd_cpp":
            args.steps = 9
        elif backend_type == "qwen21_sd_cpp":
            if model_name.endswith("-viggle-turbo"):
                args.steps = 6       # Viggle distill
            elif model_name.endswith("-turbo"):
                args.steps = 8       # official Alibaba turbo
            else:
                args.steps = 40
        elif backend_type == "agate":
            args.steps = 50
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
    if backend_type == "hidream":
        # HiDream models live in ~/.llama-models/, not diffuse/models/
        model_path = Path(os.path.expanduser(f"~/.llama-models/{model_info['dir']}"))
        if not model_path.exists():
            print(f"\n  ✗ Model not found: {model_path}")
            print(f"    Run: diffuse download {model_info['dir'].split('-')[0]}")
            sys.exit(1)
    else:
        require_model_dir(model_name)

    # ── Validate --edit (hidream, mageflow, and qwen21 support it) ──
    ref_image_paths = None
    if args.edit:
        if backend_type not in ("hidream", "mageflow_sd_cpp", "qwen21_sd_cpp"):
            print(f"  ⚠️  --edit is only supported with hidream, mageflow-edit-turbo or qwen-image-2.1 (got {backend_type})")
            print(f"     Use: diffuse -m hidream-sdnq --edit {args.edit} -p 'instruction'")
            print(f"      or: diffuse -m mageflow-edit-turbo --edit {args.edit} -p 'instruction'")
            print(f"      or: diffuse -m qwen-image-2.1 --edit {args.edit} -p 'instruction'")
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

    # ── Z-Image subprocess early path (uses sd-cli, same as Ideogram 4) ─────
    if backend_type == "zimage_sd_cpp":
        _run_zimage_sd_cpp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height)
        return

    # ── Mage-Flow-Edit-Turbo early path (uses sd-cli, instruction-based editing) ──
    if backend_type == "mageflow_sd_cpp":
        _run_mageflow_sd_cpp_edit(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
        return

    # ── Qwen-Image 2.1 early path (T2I + native editing via sd-cli) ──
    if backend_type == "qwen21_sd_cpp":
        _run_qwen21_sd_cpp_image(args, model_name, model_info, prompt, original_prompt, seed, width, height, ref_image_paths)
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
    def _load_enhance_config() -> str:
        """Resolve o modelo de enhance via config externa da frota.

        Ordem: chave 'diffuse' em enhance-models.yaml → fallback declarado →
        registry (enhance_model do models.py) → qwen3.5-4b. Valida os handles
        contra /v1/models do llama-swap (evita o 404 silencioso do episódio
        f210454). Fail-open se o swap estiver fora (aceita o primeiro).
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
        enhance_type = model_info.get("enhance_type", "ideogram")

        # ── Edit + Enhance: analyze image, then refine prompt ──
        if ref_image_paths and enhance_type == "vision":
            enhance_has_vision = _check_model_vision(enhance_model)

            if enhance_has_vision:
                # One-shot: enhance model has vision → single call for analysis + prompt
                print(f"  👁️✨ {enhance_model} has vision — one-shot image analysis + edit enhancement")
                print(f"  📷 Analyzing & enhancing edit prompt via {enhance_model}...")
                enhanced_result, raw_response = analyze_and_enhance_edit(
                    ref_image_paths[0], prompt, enhance_model, nsfw=args.nsfw
                )
                if enhanced_result != prompt:
                    enhanced_prompt = enhanced_result
                    print(f"     Expanded to edit instruction ({len(enhanced_result)} chars)")
                    print(f"     ─── Enhanced edit prompt ───")
                    import textwrap as _tw
                    for line in _tw.wrap(enhanced_result, width=78):
                        print(f"     {line}")
                    print(f"     ────────────────────────────")
                    prompt = enhanced_result
                else:
                    print(f"     ⚠️  One-shot vision+edit failed — falling back to raw prompt")
                    if raw_response and raw_response != prompt:
                        print(f"     ─── LLM response ───")
                        display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                        print(f"     {display}")
                        print(f"     ────────────────────")
            else:
                # Two-shot: separate vision model for analysis, then enhance model for prompt
                if args.vision_with:
                    vision_model = args.vision_with
                    print(f"  👁️  Using {vision_model} for image analysis (override)")
                else:
                    vision_model = DEFAULT_VISION_MODEL
                    print(f"  👁️  Using {vision_model} for image analysis (default)")

                # Step 1: Analyze the image with the vision model
                print(f"  📷 Analyzing reference image via {vision_model}...")
                image_description = analyze_image(ref_image_paths[0], vision_model, prompt, nsfw=args.nsfw)
                if not image_description:
                    print(f"     ⚠️  Image analysis failed — falling back to prompt-only enhancement")
                else:
                    print(f"     ─── Image description ({len(image_description)} chars) ───")
                    import textwrap
                    for line in textwrap.wrap(image_description, width=78):
                        print(f"     {line}")
                    print(f"     ────────────────────────────────────────")

                    # Evict vision model (it's different from enhance model)
                    running = llama_swap_running_models()
                    if running:
                        print(f"  🔄 Evicting {vision_model} after image analysis...")
                        evict_llm()
                        print(f"     VRAM freed for prompt enhancement")

                    # Step 2: Refine the edit prompt using the image description
                    print(f"  ✨ Enhancing edit prompt via {enhance_model} (vision + edit mode)...")
                    enhanced_result, raw_response = enhance_edit_prompt(image_description, prompt, enhance_model, nsfw=args.nsfw)
                    if enhanced_result != prompt:
                        enhanced_prompt = enhanced_result
                        print(f"     Expanded to edit instruction ({len(enhanced_result)} chars)")
                        print(f"     ─── Enhanced edit prompt ───")
                        import textwrap as _tw
                        for line in _tw.wrap(enhanced_result, width=78):
                            print(f"     {line}")
                        print(f"     ────────────────────────────")
                        prompt = enhanced_result
                    else:
                        print(f"     ⚠️  Edit-enhancement failed — using raw prompt")
                        if raw_response and raw_response != prompt:
                            print(f"     ─── LLM response ───")
                            display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                            print(f"     {display}")
                            print(f"     ────────────────────")

        # ── Normal enhancement (no edit, or ideogram type, or vision edit failed) ──
        if not ref_image_paths or enhance_type != "vision" or (ref_image_paths and enhance_type == "vision" and not enhanced_prompt):
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
                print(f"  ✨ Enhancing prompt via {enhance_model} (Ideogram JSON)...")
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
            if backend_type == "sd_cpp" and enhanced_result != prompt:
                prompt = enhanced_result
            elif enhance_type in ("vision", "agate") and enhanced_result != prompt:
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
        if backend_type == "hidream":
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Expected: ~3-4min (model load + CPU offload + 28 denoising steps)")
        elif backend_type == "agate":
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

    # ── Evict LLMs before HiDream (needs ~4.5GB VRAM) ──
    if backend_type == "hidream" and not args.enhance:
        running = llama_swap_running_models()
        if running:
            print(f"  🔄 Evicting LLM models for HiDream: {', '.join(running)}")
            evict_llm()
            print(f"     VRAM freed for HiDream (~4.5GB needed)")
            print()

    # ── Phase 1: Load pipeline ──
    print(f"  [1/3] Loading pipeline ({model_info['bits']})...")
    pipeline, load_time = load_pipeline(model_name, editing=bool(ref_image_paths))
    if backend_type == "gemlite":
        print(f"        Pipeline ready in {load_time:.1f}s")
    elif backend_type == "hidream":
        print(f"        Model dispatched (CPU offload) in {load_time:.1f}s")
    else:
        print(f"        sd-cli config ready")

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
    elif backend_type == "hidream":
        png_bytes, diffusion_time, peak_hbm = generate_image_hidream(
            pipeline, prompt, seed, args.steps, width, height,
            ref_image_paths=ref_image_paths,
        )
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
    elif backend_type == "sd_cpp":
        # For sd_cpp, we need an output path upfront
        orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
        output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)
        try:
            output_path, diffusion_time, peak_hbm = generate_image_sd_cpp(
                pipeline, prompt, seed, width, height, output_path,
                nsfw=args.nsfw,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) and args.cpu_fallback:
                print(f"  ⚠️  CUDA failed — retrying on CPU (this will be very slow)...")
                output_path, diffusion_time, peak_hbm = generate_image_sd_cpp(
                    pipeline, prompt, seed, width, height, output_path,
                    cpu_fallback=True,
                    nsfw=args.nsfw,
                )
            else:
                raise
        load_time = 0.0  # sd-cli handles its own loading

    wall_time = time.perf_counter() - wall_t0

    # ── Phase 3: Save + Unload ──
    print(f"  [3/3] Saving & unloading...")
    save_metadata(
        model_name, prompt, seed, width, height, args.steps,
        load_time, diffusion_time, wall_time, peak_hbm, output_path,
        enhanced_prompt=enhanced_prompt,
    )

    if backend_type in ("gemlite", "hidream", "agate"):
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
    enhanced = None
    if args.enhance or args.enhance_with:
        enhance_model = args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b")
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


def _run_zimage_sd_cpp_image(
    args,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> None:
    """Handle Z-Image-Turbo image generation via sd-cli (same path as Ideogram 4)."""
    from diffuse.backends.sd_cpp import load_pipeline_sd_cpp, generate_image_sd_cpp

    steps = args.steps if args.steps != 28 else 9  # Default 8 NFE for Turbo
    guidance = 1.0  # Turbo distilled: sd-cli warns "use cfg-scale=1 for distilled models"

    print(f"  🎨 Z-Image-Turbo T2I: {width}×{height}, {steps} steps (8 NFE), seed={seed}")

    # ── Prompt enhancement ──
    enhanced = None
    if args.enhance or args.enhance_with:
        enhance_model = args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b")
        enhance_type = model_info.get("enhance_type", "vision")

        if enhance_type == "vision":
            print(f"\n  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)

        if enhanced and enhanced != prompt:
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced

    # Evict LLMs before loading
    running = llama_swap_running_models()
    if running:
        print(f"  🔄 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    print(f"  [1/3] Loading Z-Image-Turbo pipeline ({model_info['bits']})...")
    config, load_time = load_pipeline_sd_cpp(model_name)
    config["steps"] = steps
    config["cfg"] = guidance
    print(f"        sd-cli config ready")

    # Output path
    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] Generating...")
    wall_t0 = time.perf_counter()
    try:
        output_path, diffusion_time, peak_hbm = generate_image_sd_cpp(
            config, prompt, seed, width, height, output_path,
            nsfw=args.nsfw,
        )
    except RuntimeError as e:
        if "CUDA" in str(e) and args.cpu_fallback:
            print(f"  ⚠️  CUDA failed — retrying on CPU (this will be very slow)...")
            output_path, diffusion_time, peak_hbm = generate_image_sd_cpp(
                config, prompt, seed, width, height, output_path, cpu_fallback=True,
                nsfw=args.nsfw,
            )
        else:
            raise
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


def _run_mageflow_sd_cpp_edit(
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
    """Handle Mage-Flow-Edit-Turbo image editing via sd-cli.

    Mage-Flow-Edit is instruction-based: pass a reference image with --edit
    and a text instruction as the prompt. No masks needed. Turbo = 4 steps.
    """
    from diffuse.backends.sd_cpp import (
        load_pipeline_sd_cpp,
        generate_image_mageflow_sd_cpp,
    )

    if not ref_image_paths:
        print("  ✗ Mage-Flow-Edit requires --edit IMAGE to specify a reference image")
        sys.exit(1)

    ref_image = ref_image_paths[0]
    steps = 4  # Turbo distilled
    guidance = 1.0  # Turbo: cfg=1.0

    print(f"  🎨 Mage-Flow-Edit-Turbo: editing {Path(ref_image).name}")
    print(f"     Instruction: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"     {width}x{height}, {steps} steps, seed={seed}")

    # ── Prompt enhancement ──
    enhanced = None
    if args.enhance or args.enhance_with:
        enhance_model = args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b")
        enhance_type = model_info.get("enhance_type", "vision")

        if enhance_type == "vision":
            print(f"\n  ✨ Enhancing edit instruction via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)

        if enhanced and enhanced != prompt:
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced

    # Evict LLMs before loading
    running = llama_swap_running_models()
    if running:
        print(f"  🔄 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image editing")

    print(f"  [1/3] Loading Mage-Flow-Edit-Turbo pipeline ({model_info['bits']})...")
    config, load_time = load_pipeline_sd_cpp(model_name)
    print(f"        sd-cli config ready")

    # Output path
    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] Editing...")
    wall_t0 = time.perf_counter()
    try:
        output_path, diffusion_time, peak_hbm = generate_image_mageflow_sd_cpp(
            config, prompt, seed, width, height, output_path, ref_image,
        )
    except RuntimeError as e:
        if "CUDA" in str(e) and args.cpu_fallback:
            print(f"  ⚠️  CUDA failed — retrying on CPU (this will be very slow)...")
            output_path, diffusion_time, peak_hbm = generate_image_mageflow_sd_cpp(
                config, prompt, seed, width, height, output_path, ref_image,
                cpu_fallback=True,
            )
        else:
            raise
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


def _run_qwen21_sd_cpp_image(
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
    """Handle Qwen-Image 2.1 generation/editing via sd-cli.

    Qwen-Image 2.1 is unified: the same model does T2I, native editing (up to 10
    reference images, no masks) and RGBA transparency. Unlike Z-Image-Turbo or
    Mage-Flow it is NOT a distilled model, but its own specification still calls
    for classifier-free guidance OFF (cfg 1.0) with 40 steps — see below.
    """
    from diffuse.backends.sd_cpp import (
        load_pipeline_sd_cpp,
        generate_image_qwen21_sd_cpp,
    )

    is_edit = bool(ref_image_paths)
    steps = args.steps if args.steps is not None else (
        6 if model_name.endswith("-viggle-turbo")
        else 8 if model_name.endswith("-turbo")
        else 40
    )
    # Qwen's own specification (vLLM Recipes): 40 steps with classifier-free
    # guidance OFF (cfg 1.0). Measured on this 3050: cfg 4.0 costs 11.03 s/it
    # because the DiT runs twice per step, while cfg 1.0 costs 5.5 s/it --
    # a 2.01x ratio. cfg 1.0 at 40 steps is both FASTER and better-looking
    # than cfg 4.0 at 28 steps. Do not raise this without an A/B.
    guidance = args.cfg if args.cfg is not None else 1.0

    # --pruna: Pruna 8-step distillation LoRA on the BASE model only. Mixing a
    # Pruna adapter with the Viggle-turbo DiT stacks two independent distillation
    # lines — the catalog explicitly says the families are not interchangeable.
    if getattr(args, "pruna", False):
        if model_name.endswith("-turbo"):
            raise SystemExit(
                "  ✗ --pruna is for the base model only (-m qwen-image-2.1). "
                "The turbo is already distilled (Viggle or official); stacking a Pruna LoRA on "
                "it mixes distillation families and corrupts the sampling recipe."
            )
        if "<lora:" not in prompt:
            prompt = f"<lora:p_qwen_image_2.1_8step_v0.1:1.0> {prompt}"
        # main() already materialized the per-backend default into args.steps
        # (40/6), so check against those, not None.
        if not args.steps or args.steps in (40, 6):
            steps = 8

    if is_edit:
        print(f"  \U0001f3a8 Qwen-Image 2.1 editing: {Path(ref_image_paths[0]).name}")
        print(f"     Instruction: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    else:
        print(f"  \U0001f3a8 Qwen-Image 2.1 T2I")
    print(f"     {width}x{height}, {steps} steps, cfg={guidance}, seed={seed}")

    # -- Prompt enhancement --
    enhanced = None
    if args.enhance or args.enhance_with or getattr(args, "enhance_edit_with", None):
        enhance_model = args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b")
        enhance_type = model_info.get("enhance_type", "vision")

        # Preserve <lora:...> tags: extract before enhance, re-apply after
        lora_tags = ""
        if "<lora:" in prompt:
            prompt, lora_tags = _extract_lora_tags(prompt)

        # --enhance-edit-with: VLM vê a reference image e refina a instrução (one-shot)
        if is_edit and getattr(args, "enhance_edit_with", None):
            edit_model = args.enhance_edit_with
            if not _check_model_vision(edit_model):
                print(f"  \u26a0\ufe0f  {edit_model} não tem visão — caindo pro enhance text-only")
                enhanced, raw_response = enhance_qwen21_prompt(prompt, enhance_model, nsfw=args.nsfw)
            else:
                print(f"\n  \U0001f441\ufe0f\u2728 {edit_model} analisa a referência e refina a instrução (one-shot)...")
                enhanced, raw_response = analyze_and_enhance_edit(
                    ref_image_paths[0], prompt, edit_model, nsfw=args.nsfw
                )
                if enhanced and enhanced != prompt:
                    print(f"     Expanded to ({len(enhanced)} chars)")
                    _show_enhanced_if_requested(args, enhanced)
                    print(f"     ─── Enhanced edit prompt ───")
                    import textwrap as _tw
                    for line in _tw.wrap(enhanced, width=78):
                        print(f"     {line}")
                    print(f"     ────────────────────────────")
                enhanced = enhanced if (enhanced and enhanced != prompt) else None
        elif enhance_type == "qwen21":
            print(f"\n  \u2728 Enhancing prompt via {enhance_model} (qwen21 mode)...")
            enhanced, raw_response = enhance_qwen21_prompt(prompt, enhance_model, nsfw=args.nsfw)
        elif enhance_type == "vision":
            print(f"\n  \u2728 Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced, raw_response = enhance_vision_prompt(prompt, enhance_model, nsfw=args.nsfw)
        else:
            enhanced, raw_response = enhance_prompt(prompt, enhance_model, nsfw=args.nsfw)

        if enhanced and enhanced != prompt:
            print(f"     Expanded to ({len(enhanced)} chars)")
            _show_enhanced_if_requested(args, enhanced)
            prompt = enhanced
        if lora_tags:
            prompt = _reapply_lora_tags(prompt, lora_tags)

    # --nsfw: seletor inteligente de LoRAs (mirror do H3, 03/out — decreto do
    # user: --nsfw PURO, o auto-attach ALL saiu). Fluxo:
    #   1. rerank ColBERT pré-ordena o catálogo (~/.local/share/diffuse/nsfw_catalog_qwen21.json)
    #   2. catálogo ranked + instrução "LORAS: <ids>" entram no system do enhance
    #   3. o output do enhance traz a linha LORAS: no fim → parse → resolve (paths
    #      relativos + strengths do catálogo + triggers-frase)
    #   4. tags <lora:...:str> re-aplicadas ao prompt enhanced
    # Fail-open na CADEIA inteira: sem enhance/sem resposta/sem linha LORAS:/IDs
    # inválidos → roda BASE sem tags (avisado), nunca bloqueia a geração.
    if getattr(args, "nsfw", False):
        import os as _os_i
        import re as _re_i
        import subprocess as _sp_i
        from diffuse.paths import MODELS_DIR
        # tags manuais do user vencem (o enhance preserva e re-aplica as delas depois)
        if "<lora:" not in prompt:
            if args.enhance or args.enhance_with:
                _cat_json = _os_i.environ.get("NSFW_CATALOG_JSON") or str(
                    Path.home() / ".local/share/diffuse/nsfw_catalog_qwen21.json")
                _rerank = str(Path.home() / ".local/share/diffuse/rerank_catalog_qwen21.py")
                _resolver = str(Path.home() / ".local/share/diffuse/nsfw_lora_resolve_qwen21.py")
                _catalog_block = None
                if Path(_rerank).exists() and Path(_cat_json).exists():
                    try:
                        _ranked = _sp_i.run(
                            ["python3", _rerank, "--prompt", prompt, "--n", "10", "--mode", "rank"],
                            capture_output=True, text=True, timeout=180)
                        if _ranked.returncode == 0 and _ranked.stdout.strip():
                            _catalog_block = _ranked.stdout.strip()
                    except Exception as e:  # fail-open: sem ranking, catálogo cru
                        print(f"  ⚠️  rerank indisponível ({e}); catálogo completo")
                elif not Path(_cat_json).exists():
                    print("  ⚠️  catálogo NSFW não encontrado; pulando seleção (roda base)")
                if _catalog_block:
                    _lora_rule = (
                        "LoRA SELECTION: at the very END of your output, on the LAST line, "
                        "output exactly: LORAS: <comma-separated numeric IDs>\n"
                        "- Choose the IDs best fit to the request (0 to 2 LoRAs, avoid stacking two "
                        "that target the same feature/body aspect).\n"
                        "- If genuinely nothing fits, output: LORAS: none\n"
                        "- Never mention catalog trigger phrases in the paragraph itself.\n\n"
                        + _catalog_block)
                    enhanced, raw_response = enhance_qwen21_prompt(
                        prompt, enhance_model, nsfw=True, extra_system=_lora_rule)
                    # (o bloco de exibição/logging abaixo, comum a todos os caminhos,
                    #  já mostra o prompt expandido — nada extra aqui)
                    _lora_ids = None
                    m = _re_i.search(r"LORAS:\s*([0-9,\s]+)", enhanced or "")
                    if m and m.group(1).strip():
                        _lora_ids = [int(x) for x in _re_i.findall(r"\d+", m.group(1))]
                    # strip da linha LORAS: do prompt (sintaxe interna, não é cena)
                    enhanced = _re_i.sub(r"^\s*LORAS:.*$", "", enhanced or "", flags=_re_i.M).strip()
                    if _lora_ids:
                        _res = _sp_i.run(["python3", _resolver, "--ids", ",".join(map(str, _lora_ids))],
                                         capture_output=True, text=True, timeout=30)
                        import json as _j
                        _sel = _j.loads(_res.stdout) if _res.returncode in (0, 3) and _res.stdout.strip() else {}
                        if _sel.get("names"):
                            _tags = " ".join(
                                f"<lora:{n}:{m_}>" for n, m_ in zip(_sel["names"], _sel["mults"]))
                            _trigs = [t for t in (_sel.get("triggers") or []) if t]
                            enhanced = f"{_tags} {enhanced}"
                            if _trigs:
                                enhanced = f"{', '.join(_trigs)} {enhanced}"
                            print(f"  🎯 LLM escolheu {len(_sel['names'])} LoRA(s): "
                                  f"{', '.join(Path(n).name for n in _sel['names'])}")
                        else:
                            print("  ⚠️  LLM respondeu LORAS: mas nada resolveu; rodando base (sem tags)")
                    elif enhanced:
                        print("  ℹ️  LLM não escolheu LoRA (LORAS: none ou linha ausente); rodando base")
                    prompt = enhanced
            else:
                print("  ⚠️  --nsfw sem --enhance: sem seleção inteligente; rodando base")

    # Evict LLMs before loading (the text encoder runs on CPU, but the DiT needs VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  \U0001f504 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for image generation")

    print(f"  [1/3] Loading Qwen-Image 2.1 pipeline ({model_info['bits']})...")
    config, load_time = load_pipeline_sd_cpp(model_name)
    print(f"        sd-cli config ready")

    # Output path
    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)

    print(f"  [2/3] {'Editing' if is_edit else 'Generating'}...")
    wall_t0 = time.perf_counter()
    try:
        output_path, diffusion_time, peak_hbm = generate_image_qwen21_sd_cpp(
            config, prompt, seed, width, height, output_path,
            ref_images=ref_image_paths, steps=steps, cfg_scale=guidance,
        )
    except RuntimeError as e:
        if "CUDA" in str(e) and args.cpu_fallback:
            print(f"  \u26a0\ufe0f  CUDA failed - retrying on CPU (this will be very slow)...")
            output_path, diffusion_time, peak_hbm = generate_image_qwen21_sd_cpp(
                config, prompt, seed, width, height, output_path,
                ref_images=ref_image_paths, cpu_fallback=True,
                steps=steps, cfg_scale=guidance,
            )
        else:
            raise
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
