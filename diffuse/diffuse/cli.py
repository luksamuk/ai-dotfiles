"""CLI — argument parsing, interactive prompt, and main() orchestration."""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

from diffuse.paths import setup_environment, DEFAULT_VISION_MODEL
from diffuse.models import MODELS
from diffuse.backends import load_pipeline, unload_pipeline, require_model_dir
from diffuse.backends.gemlite import generate_image_gemlite
from diffuse.backends.sd_cpp import generate_image_sd_cpp
from diffuse.backends.hidream import generate_image_hidream
from diffuse.backends.framepack import generate_video_framepack
from diffuse.llm import evict_llm, llama_swap_running_models
from diffuse.enhance import (
    enhance_prompt,
    enhance_vision_prompt,
    enhance_edit_prompt,
    analyze_image,
    analyze_and_enhance_edit,
    enhance_video_prompt,
    enhance_video_prompt_two_shot,
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
        if not 256 <= dim <= 2048:
            raise argparse.ArgumentTypeError(f"--size {name} {dim} out of range — must be 256–2048")
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
            "\n"
            "Resolution guide (RTX 3050 6GB):\n"
            "  Bonsai:     512×512 max (OOM above this)\n"
            "  Ideogram 4: 480×480 safe, up to 1920×1088 (~15 min)\n"
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
    p.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 4 for bonsai, 20 for ideogram4, 28 for hidream).")
    p.add_argument(
        "--size", type=parse_size, default=None,
        help="Image size as WxH (default: 512x512 for bonsai, 480x480 for ideogram4, 1024x1024 for hidream).",
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
        help="Enhance prompt using a specific llama-swap model. "
             "Any model available in llama-swap works (e.g. qwen3.5-4b, laguna-xs2). "
             "Implies --enhance.",
    )
    p.add_argument(
        "--show-enhanced", action="store_true",
        help="Print the full enhanced JSON prompt before generating.",
    )
    p.add_argument(
        "--cpu-fallback", action="store_true",
        help="If CUDA generation fails, automatically retry on CPU (very slow: ~30+ min).",
    )
    p.add_argument(
        "--edit", metavar="IMAGE", type=Path, default=None,
        help="Reference image for HiDream editing mode. Pass an image path to use instruction-based editing.",
    )
    p.add_argument(
        "--input-image", metavar="IMAGE", type=Path, default=None,
        help="Input image for FramePack I2V (image-to-video). Required for framepack-i2v model.",
    )
    p.add_argument(
        "--seconds", type=float, default=None,
        help="Video length in seconds for FramePack I2V (default: 5.0, max: 120).",
    )
    p.add_argument(
        "--cfg", type=float, default=None,
        help="CFG scale (default: 1.0 for FramePack/Wan2.2 Rapid, 7.0 for other image gen).",
    )
    p.add_argument(
        "--gs", type=float, default=4.5,
        help="Distilled guidance scale for FramePack I2V (default: 4.5).",
    )
    p.add_argument(
        "--no-teacache", action="store_true",
        help="Disable TeaCache for FramePack I2V (slower but potentially better quality).",
    )
    # ── Wan2.2 I2V video args ──
    p.add_argument(
        "--video-frames", type=int, default=None,
        help="Number of video frames for Wan2.2 I2V (default: 33, ~1.3s @ 24fps).",
    )
    p.add_argument(
        "--fps", type=int, default=None,
        help="Frames per second for Wan2.2 I2V output video (default: 24).",
    )
    p.add_argument(
        "--negative-prompt", type=str, default="",
        help="Negative prompt for Wan2.2 I2V.",
    )
    p.add_argument(
        "--flow-shift", type=float, default=None,
        help="Flow shift for Wan2.2 (default: 3.0).",
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
        print_models()
        sys.exit(0)

    setup_environment()

    model_name = args.model
    model_info = MODELS[model_name]
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    backend_type = model_info.get("backend_type", "gemlite")

    # ── Defaults per backend ──
    # Steps: bonsai=4, ideogram4=20, hidream=28
    if args.steps is None:
        if backend_type == "hidream":
            args.steps = 28
        elif backend_type == "sd_cpp":
            args.steps = 20
        else:
            args.steps = 4

    # Size: model-specific defaults (512x512 for bonsai, model-specific for others)
    if args.size is None:
        default_size = model_info.get("default_size", (512, 512))
        width, height = default_size
    else:
        width, height = args.size

    prompt = args.prompt or get_prompt_interactive()

    original_prompt = prompt

    # Pre-flight
    if backend_type == "hidream":
        # HiDream models live in ~/.llama-models/, not diffuse/models/
        hidream_model_path = Path(os.path.expanduser(f"~/.llama-models/{model_info['dir']}"))
        if not hidream_model_path.exists():
            print(f"\n  ✗ Model not found: {hidream_model_path}")
            print(f"    Download with: hf download WaveCut/HiDream-O1-Image-Dev-SDNQ-uint4-svd-r32-last8-odown-bf16 --local-dir {hidream_model_path}")
            sys.exit(1)
    else:
        require_model_dir(model_name)

    # ── Validate --edit (only for hidream backend) ──
    ref_image_paths = None
    if args.edit:
        if backend_type != "hidream":
            print(f"  ⚠️  --edit is only supported with hidream backend (got {backend_type})")
            print(f"     Use: diffuse -m hidream-sdnq --edit {args.edit} -p 'instruction'")
            sys.exit(1)
        if not args.edit.exists():
            # If not found as-is, try resolving relative to original CWD
            # (the shell wrapper cds to SCRIPT_DIR before running generate.py)
            orig_cwd = os.environ.get("DIFFUSE_ORIG_CWD", "")
            if orig_cwd:
                resolved = Path(orig_cwd) / args.edit
                if resolved.exists():
                    args.edit = resolved
                else:
                    print(f"  ✗ Edit image not found: {args.edit} (also tried {resolved})")
                    sys.exit(1)
            else:
                print(f"  ✗ Edit image not found: {args.edit}")
                sys.exit(1)
        ref_image_paths = [str(args.edit.resolve())]
        print(f"  🖼️  Edit mode: {args.edit.name} → prompt as instruction")

    # ── FramePack I2V early path ──────────────────────────────────────────────
    if backend_type == "framepack":
        _run_framepack(args, model_name, model_info, prompt, original_prompt, seed, width, height)
        return

    # ── Wan2.2 I2V video early path ───────────────────────────────────────────
    if backend_type == "sd_cpp_video":
        _run_sd_cpp_video(args, model_name, model_info, prompt, original_prompt, seed, width, height)
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

    # ── Prompt enhancement ──
    enhanced_prompt = None
    enhance_model = None

    if args.enhance_with:
        # --enhance-with <model> — use any llama-swap model
        enhance_model = args.enhance_with
        args.enhance = True  # implied
    elif args.enhance:
        enhance_model = model_info.get("enhance_model", "qwen3.5-4b")

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
                    ref_image_paths[0], prompt, enhance_model
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
                image_description = analyze_image(ref_image_paths[0], vision_model, prompt)
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
                    enhanced_result, raw_response = enhance_edit_prompt(image_description, prompt, enhance_model)
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
            if enhance_type == "vision":
                print(f"  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
                enhanced_result, raw_response = enhance_vision_prompt(prompt, enhance_model)
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
                enhanced_result, raw_response = enhance_prompt(prompt, enhance_model)
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
            elif enhance_type == "vision" and enhanced_result != prompt:
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
        else:
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Cold start: ~30-60s (imports + model load + kernel JIT)")
            print(f"     Subsequent runs at this size will be faster (cached kernels)")
    print()

    # ── Phase 1.5: Evict LLMs after prompt enhancement ──
    # If we used an LLM for enhancement, evict it before loading the diffusion model
    if args.enhance and backend_type == "hidream":
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
    elif backend_type == "sd_cpp":
        # For sd_cpp, we need an output path upfront
        orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
        output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)
        try:
            output_path, diffusion_time, peak_hbm = generate_image_sd_cpp(
                pipeline, prompt, seed, width, height, output_path,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) and args.cpu_fallback:
                print(f"  ⚠️  CUDA failed — retrying on CPU (this will be very slow)...")
                output_path, diffusion_time, peak_hbm = generate_image_sd_cpp(
                    pipeline, prompt, seed, width, height, output_path,
                    cpu_fallback=True,
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

    if backend_type in ("gemlite", "hidream"):
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


def _run_sd_cpp_video(
    args: argparse.Namespace,
    model_name: str,
    model_info: dict,
    prompt: str,
    original_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> None:
    """Handle Wan2.2 I2V video generation via sd-cli — separate path from image generation."""
    from diffuse.backends.sd_cpp import load_pipeline_sd_cpp_video, generate_video_sd_cpp

    # Validate input image
    input_image_path = args.input_image
    if input_image_path is None:
        print("  ✗ Wan2.2 I2V requires --input-image")
        print("     Usage: diffuse -m wan22-i2v --input-image photo.png -p 'description of motion'")
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

    # Video parameters with model defaults
    video_frames = args.video_frames or model_info.get("default_video_frames", 33)
    fps = args.fps or model_info.get("default_fps", 24)
    steps = args.steps or model_info.get("default_steps", 4)
    cfg_scale = args.cfg if args.cfg is not None else model_info.get("default_cfg", 1.0)
    flow_shift = args.flow_shift if args.flow_shift is not None else model_info.get("default_flow_shift", 3.0)
    negative_prompt = args.negative_prompt or (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    duration_s = video_frames / fps

    print(f"  🎬 Wan2.2 I2V: {width}×{height}, {video_frames} frames @ {fps} fps ({duration_s:.1f}s)")
    print(f"     Steps: {steps}, CFG: {cfg_scale}, Flow shift: {flow_shift}")
    print(f"     Input: {Path(input_image_path).name}")
    print(f"     Seed: {seed}")

    # ── Prompt enhancement (vision-based for I2V) ──────────────────────────
    if args.enhance or args.enhance_with:
        enhance_model = args.enhance_with or model_info.get("enhance_model", "qwen3.6-35b-a3b")
        enhance_has_vision = _check_model_vision(enhance_model)

        if enhance_has_vision:
            # One-shot: enhance model has vision → single call with image
            print(f"\n  ✨ Video-enhancing prompt via {enhance_model} (vision + I2V mode)...")
            print(f"  📷 Analyzing input image & writing motion prompt...")
            enhanced_result, raw_response = enhance_video_prompt(
                str(input_image_path), prompt, enhance_model
            )
        else:
            # Two-shot: separate vision model analyzes image, then enhance model refines
            if args.vision_with:
                vision_model = args.vision_with
            else:
                vision_model = DEFAULT_VISION_MODEL
            print(f"\n  👁️  Using {vision_model} for image analysis (default)")
            print(f"  📷 Analyzing input image via {vision_model}...")
            image_description = analyze_image(str(input_image_path), vision_model, prompt)
            if not image_description:
                print(f"     ⚠️  Image analysis failed — falling back to raw prompt")
                enhanced_result = prompt
                raw_response = ""
            else:
                print(f"     ─── Image description ({len(image_description)} chars) ───")
                import textwrap as _tw
                for line in _tw.wrap(image_description, width=78):
                    print(f"     {line}")
                print(f"     ────────────────────────────────────────")

                # Evict vision model before loading enhance model
                running = llama_swap_running_models()
                if running:
                    print(f"  🔄 Evicting {vision_model} after image analysis...")
                    evict_llm()
                    print(f"     VRAM freed for prompt enhancement")

                print(f"  ✨ Video-enhancing prompt via {enhance_model} (two-shot, text-only)...")
                enhanced_result, raw_response = enhance_video_prompt_two_shot(
                    image_description, prompt, enhance_model
                )
        if enhanced_result and enhanced_result != prompt:
            print(f"     Expanded to video prompt ({len(enhanced_result)} chars)")
            print(f"     ─── Enhanced video prompt ───")
            import textwrap as _tw
            for line in _tw.wrap(enhanced_result, width=78):
                print(f"     {line}")
            print(f"     ──────────────────────────────")
            prompt = enhanced_result
        else:
            print(f"     ⚠️  Video-enhancement failed — using raw prompt")
            if raw_response and raw_response != prompt:
                display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                print(f"     {display}")

    # Evict LLMs before loading (Wan2.2 needs all available VRAM)
    running = llama_swap_running_models()
    if running:
        print(f"  🔄 Evicting LLM models: {', '.join(running)}")
        evict_llm()
        print(f"     VRAM freed for video generation")

    print(f"  [1/3] Loading sd-cli config ({model_info['bits']})...")
    config, load_time = load_pipeline_sd_cpp_video(model_name)
    print(f"        Config ready (sd-cli handles model loading at runtime)")

    # Output path
    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    out_dir = orig_cwd
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(":", "_").replace("/", "_")
    default_output = str(out_dir / f"diffuse_{safe_model}_{ts}_seed{seed}.mp4")
    output_path = Path(args.output and str(args.output.with_suffix(".mp4")) or default_output)

    print(f"  [2/3] Generating video...")
    wall_t0 = time.perf_counter()

    output_path, diffusion_time, peak_hbm = generate_video_sd_cpp(
        config,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        video_frames=video_frames,
        fps=fps,
        steps=steps,
        cfg_scale=cfg_scale,
        flow_shift=flow_shift,
        input_image=str(input_image_path),
        output_path=output_path,
        max_vram=5.1,
    )
    wall_time = time.perf_counter() - wall_t0

    print(f"  [3/3] Done — unloading (sd-cli exits automatically)")

    print()
    print("═══ diffuse — Wan2.2 I2V Video Report ═══")
    print(f"  Model:       {model_name}")
    print(f"  Prompt:      \"{original_prompt}\"")
    print(f"  Input:       {Path(input_image_path).name}")
    print(f"  Frames:      {video_frames} @ {fps} fps ({duration_s:.1f}s)")
    print(f"  Seed:        {seed}")
    print(f"  Resolution:  {width}×{height}")
    print(f"  Steps:       {steps}")
    print(f"  CFG:         {cfg_scale}")
    print(f"  Flow shift:  {flow_shift}")
    print()
    print("  Timings:")
    print(f"    Generation: {diffusion_time:7.2f} s   (sd-cli wall time)")
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