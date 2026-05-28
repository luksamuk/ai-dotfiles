#!/usr/bin/env python3
"""diffuse — Local Diffusion Image Generation CLI.

Load pipeline → prompt → generate → stats debrief → unload.
Designed for NVIDIA RTX 3050 6GB. Model-agnostic — add new backends via MODELS registry.

Currently supports: Bonsai Image 4B (gemlite + HQQ kernels on CUDA).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("diffuse")

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
TRITON_CACHE_DIR = OUTPUTS_DIR / ".triton_cache"
GEMLITE_PERSIST_PATH = OUTPUTS_DIR / ".gemlite_cache" / "autotune.json"

# ── Model registry ─────────────────────────────────────────────────────────
# Add new diffusion models here. Each entry needs:
#   backend_id:     identifier for the pipeline backend
#   hf_repo:        HuggingFace repo for download
#   dir:            local directory name under models/
#   backend_type:   "gemlite" (CUDA) — extend with "mlx", "diffusers", etc.
#   bits:           human-readable weight format
#   description:    short label for --help
MODELS = {
    "binary-gemlite": {
        "backend_id": "bonsai-binary-gemlite",
        "hf_repo": "prism-ml/bonsai-image-binary-4B-gemlite-1bit",
        "dir": "bonsai-image-4B-binary-gemlite",
        "backend_type": "gemlite",
        "bits": "1-bit",
        "transformer_kwarg": "binary_transformer_path",
        "description": "1-bit {−1, +1} — 0.93 GB transformer, 88% of FP16 quality",
    },
    "ternary-gemlite": {
        "backend_id": "bonsai-ternary-gemlite",
        "hf_repo": "prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
        "dir": "bonsai-image-4B-ternary-gemlite",
        "backend_type": "gemlite",
        "bits": "1.58-bit",
        "transformer_kwarg": "ternary_transformer_path",
        "description": "1.58-bit {−1, 0, +1} — 1.21 GB transformer, 95% of FP16 quality",
    },
}


# ── Environment setup (must happen before torch/triton imports) ─────────────
def setup_environment() -> None:
    """Set cache directories before any torch/triton imports."""
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GEMLITE_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))


# ── Pipeline helpers ────────────────────────────────────────────────────────
def _find_subdir(root: Path, *hints: str) -> Path:
    """Find a child directory whose name contains any of the hints."""
    matches = [p for p in root.iterdir() if p.is_dir() and any(h in p.name for h in hints)]
    if not matches:
        present = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "(empty)"
        raise FileNotFoundError(f"No subdir matching {hints!r} under {root}. Present: {present}")
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def require_model_dir(model_name: str) -> Path:
    """Ensure model weights are present and return path."""
    model_info = MODELS[model_name]
    model_root = MODELS_DIR / model_info["dir"]
    if not model_root.exists():
        print(f"\n  ✗ Model not found: {model_root}")
        print(f"    Run: diffuse download {model_name.split('-')[0]}")
        sys.exit(1)
    return model_root


def load_pipeline_gemlite(model_name: str) -> tuple:
    """Load a gemlite-backed pipeline. Returns (pipeline, load_time_seconds)."""
    from backend_gpu.pipeline_gpu import GpuPipeline
    from gemlite.core import GemLiteLinearTriton

    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)
    text_encoder_dir = _find_subdir(model_root, "text_encoder")
    transformer_dir = _find_subdir(model_root, "transformer")

    t0 = time.perf_counter()

    # GpuPipeline requires ALL transformer paths to be set, even for
    # backends you don't use. Pass the same transformer dir for both
    # binary and ternary — the unused backend slot is simply ignored.
    pipeline = GpuPipeline(
        backend=model_info["backend_id"],
        binary_transformer_path=str(transformer_dir),
        ternary_transformer_path=str(transformer_dir),
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(_find_subdir(model_root, "vae")),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )

    # Load persisted autotune cache (speeds up subsequent runs)
    if GEMLITE_PERSIST_PATH.exists():
        GemLiteLinearTriton.load_config(str(GEMLITE_PERSIST_PATH), print_error=False)

    pipeline.prewarm()
    load_time = time.perf_counter() - t0

    # Persist any new autotune configs
    GemLiteLinearTriton.cache_config(str(GEMLITE_PERSIST_PATH))

    return pipeline, load_time


def load_pipeline(model_name: str) -> tuple:
    """Load a pipeline based on the model's backend_type. Returns (pipeline, load_time_seconds)."""
    model_info = MODELS[model_name]
    backend_type = model_info.get("backend_type", "gemlite")

    if backend_type == "gemlite":
        return load_pipeline_gemlite(model_name)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def unload_pipeline() -> None:
    """Force-unload pipeline from VRAM and system memory."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    log.info("Pipeline unloaded from VRAM")


# ── LLM eviction (llama-swap coordination) ──────────────────────────────────
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")
LLAMA_SWAP_CLI = os.environ.get("LLAMA_SWAP_CLI", os.path.expanduser("~/git/ai-dotfiles/llama-swap/llama-swap-cli"))


def _llama_swap_running_models() -> list[str]:
    """Query llama-swap /running endpoint. Returns list of model IDs or empty list."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/running", timeout=3) as resp:
            data = json.loads(resp.read())
            return [m.get("model", m.get("id", "?")) for m in data.get("running", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def evict_llm() -> bool:
    """Evict all loaded LLM models from llama-swap to free VRAM for diffusion.

    Uses llama-swap-cli unload which gracefully stops all running models.
    Returns True if eviction happened (models were running), False if already empty.
    """
    running = _llama_swap_running_models()
    if not running:
        log.info("No LLM models loaded — VRAM already free")
        return False

    log.info("Evicting LLM models from llama-swap: %s", ", ".join(running))

    # Try llama-swap-cli unload first (graceful, lets llama-swap manage shutdown)
    if os.path.isfile(LLAMA_SWAP_CLI) and os.access(LLAMA_SWAP_CLI, os.X_OK):
        result = subprocess.run(
            [LLAMA_SWAP_CLI, "unload"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            log.info("llama-swap-cli unload succeeded: %s", result.stdout.strip())
        else:
            log.warning("llama-swap-cli unload failed (rc=%d): %s", result.returncode, result.stderr.strip())
    else:
        # Fallback: POST to llama-swap /v1/unload (may not exist in all versions)
        import urllib.request
        import urllib.error
        log.info("llama-swap-cli not found — trying direct API eviction")
        try:
            req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/unload", method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                log.info("API eviction response: %s", resp.read().decode())
        except (urllib.error.URLError, OSError) as e:
            log.warning("Could not evict LLM models via API: %s", e)
            log.warning("Models %s may still be loaded — generation may fail or be slow", ", ".join(running))

    # Verify eviction
    import time as _time
    for _ in range(10):
        _time.sleep(0.5)
        if not _llama_swap_running_models():
            log.info("LLM models evicted — VRAM free for diffusion")
            return True

    # Even if verification fails, proceed — llama-swap may report stale state
    log.warning("Could not confirm LLM eviction — proceeding anyway")
    return True


# ── Generation ─────────────────────────────────────────────────────────────
def generate_image(
    pipeline,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
) -> tuple[bytes, float, float]:
    """Generate a PNG image. Returns (png_bytes, diffusion_time, peak_hbm_mb)."""
    log.info("Generating: prompt=%r seed=%d steps=%d size=%dx%d", prompt, seed, steps, width, height)

    t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt=prompt,
        seed=seed,
        steps=steps,
        height=height,
        width=width,
    )
    diffusion_time = time.perf_counter() - t0

    peak_hbm = pipeline.last_peak_memory_mb or 0.0
    log.info("Diffusion done in %.2fs (peak HBM %.1f MiB)", diffusion_time, peak_hbm)

    return png_bytes, diffusion_time, peak_hbm


# ── Output helpers ─────────────────────────────────────────────────────────
def resolve_output_path(model_name: str, seed: int, output_arg: str | None, cwd: Path | None = None) -> Path:
    """Determine output PNG path.

    If --output is given, use it directly.
    Otherwise, save to the current working directory (where the user ran diffuse),
    not to the script's own outputs/ dir.
    """
    if output_arg is not None:
        return Path(output_arg).expanduser().resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = cwd if cwd else Path.cwd()
    out = base_dir / f"diffuse_{model_name}_{ts}_seed{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Also write to internal log dir for metadata
    internal_dir = OUTPUTS_DIR / model_name
    internal_dir.mkdir(parents=True, exist_ok=True)

    return out


def save_metadata(
    model_name: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    load_time: float,
    diffusion_time: float,
    wall_time: float,
    peak_hbm: float,
    output_path: Path,
) -> Path:
    """Append generation record to JSON log."""
    meta_dir = OUTPUTS_DIR / model_name
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "generations.json"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "load_seconds": round(load_time, 3),
        "diffusion_seconds": round(diffusion_time, 3),
        "wall_seconds": round(wall_time, 3),
        "peak_hbm_mib": round(peak_hbm, 1),
        "output": str(output_path),
    }

    existing = []
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []

    existing.append(record)
    meta_path.write_text(json.dumps(existing, indent=2) + "\n")
    return meta_path


# ── Debrief display ────────────────────────────────────────────────────────
def print_debrief(
    model_name: str,
    model_info: dict,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    load_time: float,
    diffusion_time: float,
    wall_time: float,
    peak_hbm: float,
    output_path: Path,
) -> None:
    """Print generation stats report."""
    print()
    print("═══ diffuse — Generation Report ═══")
    print(f"  Model:       {model_name} ({model_info['bits']})")
    print(f"  Prompt:      \"{prompt}\"")
    print(f"  Seed:        {seed}")
    print(f"  Resolution:  {width} × {height}")
    print(f"  Steps:       {steps}")
    print()
    print("  Timings:")
    print(f"    Setup:      {load_time:7.2f} s   (imports + model load + kernel JIT)")
    print(f"    Diffusion: {diffusion_time:7.2f} s   ({steps} denoising steps + VAE decode)")
    print(f"    ─────────────────────")
    print(f"    Wall:       {wall_time:7.2f} s")
    print()
    if peak_hbm > 0:
        print("  Memory:")
        print(f"    Peak HBM:  {peak_hbm:,.0f} MiB")
        print()
    print(f"  Output: {output_path}")
    print("══════════════════════════════════════")


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffuse",
        description="diffuse — Local diffusion image generation CLI",
        epilog=(
            "Recommended sizes:\n"
            "  Fast (512x512):   512x512, 624x416, 416x624\n"
            "  Quality (1024x1024): 1024x1024, 1248x832, 832x1248\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-m", "--model",
        choices=sorted(MODELS),
        default="ternary-gemlite",
        help="Model variant (default: ternary-gemlite)",
    )
    p.add_argument("-p", "--prompt", help="Text prompt. If omitted, prompted interactively.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (random if not set).")
    p.add_argument("--steps", type=int, default=4, help="Denoising steps (default: 4, recommended: 4).")
    p.add_argument(
        "--size", type=parse_size, default=(512, 512),
        help="Image size as WxH (default: 512x512).",
    )
    p.add_argument("--output", type=Path, default=None, help="Output PNG path (auto-generated in cwd if not set).")
    p.add_argument("--open", action="store_true", help="Open the generated image with feh after saving.")
    p.add_argument(
        "--evict-llm", action="store_true",
        help="Evict all running LLM models (via llama-swap) to free VRAM before generating.",
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


# ── Previous runs ───────────────────────────────────────────────────────────
def get_previous_runs(model_name: str, width: int, height: int) -> list[float]:
    """Check historical generation times at this resolution."""
    meta_path = OUTPUTS_DIR / model_name / "generations.json"
    if not meta_path.exists():
        return []
    try:
        entries = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [
        float(e["wall_seconds"])
        for e in entries
        if isinstance(e, dict)
        and e.get("width") == width
        and e.get("height") == height
        and isinstance(e.get("wall_seconds"), (int, float))
    ]


# ── Open image in viewer ────────────────────────────────────────────────────
def open_image(path: Path) -> None:
    """Open the generated image with feh in a background process."""
    import shutil
    import subprocess

    viewer = shutil.which("feh")
    if not viewer:
        log.warning("feh not found — skipping image viewer")
        return

    subprocess.Popen(
        [viewer, "--scale-down", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    log.info("Opened in feh: %s", path)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    setup_environment()

    model_name = args.model
    model_info = MODELS[model_name]
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    width, height = args.size
    prompt = args.prompt or get_prompt_interactive()

    # Pre-flight
    require_model_dir(model_name)

    # ── LLM eviction (free VRAM for diffusion) ──
    if args.evict_llm:
        running = _llama_swap_running_models()
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

    # Show warm/cold estimate
    prior = get_previous_runs(model_name, width, height)
    print()
    if prior:
        mean_s = sum(prior) / len(prior)
        best_s = min(prior)
        print(f"  ⚡ {len(prior)} prior run(s) at {width}×{height} — warmed kernels available")
        print(f"     Historical wall: mean {mean_s:.1f}s, best {best_s:.1f}s")
    else:
        print(f"  ⏳ First run at {width}×{height}")
        print(f"     Cold start: ~30-60s (imports + model load + kernel JIT)")
        print(f"     Subsequent runs at this size will be faster (cached kernels)")
    print()

    # ── Phase 1: Load pipeline ──
    print(f"  [1/3] Loading pipeline ({model_info['bits']})...")
    pipeline, load_time = load_pipeline(model_name)
    print(f"        Pipeline ready in {load_time:.1f}s")

    # ── Phase 2: Generate ──
    print(f"  [2/3] Generating ({width}×{height}, {args.steps} steps, seed={seed})...")
    wall_t0 = time.perf_counter()

    png_bytes, diffusion_time, peak_hbm = generate_image(
        pipeline, prompt, seed, args.steps, width, height,
    )
    wall_time = time.perf_counter() - wall_t0

    # ── Phase 3: Save + Unload ──
    print(f"  [3/3] Saving & unloading...")
    # Use the caller's cwd, not the script's directory
    orig_cwd = Path(os.environ.get("DIFFUSE_ORIG_CWD", str(Path.cwd())))
    output_path = resolve_output_path(model_name, seed, args.output, cwd=orig_cwd)
    output_path.write_bytes(png_bytes)
    save_metadata(
        model_name, prompt, seed, width, height, args.steps,
        load_time, diffusion_time, wall_time, peak_hbm, output_path,
    )
    unload_pipeline()

    # ── Debrief ──
    print_debrief(
        model_name, model_info, prompt, seed,
        width, height, args.steps,
        load_time, diffusion_time, wall_time, peak_hbm, output_path,
    )

    # ── Open in viewer ──
    if args.open:
        open_image(output_path)


if __name__ == "__main__":
    main()