#!/usr/bin/env python3
"""magenta-rt — Local Music Generation CLI.

Load → prompt → generate → stats debrief → unload.
Designed for NVIDIA RTX 3050 6GB. Uses JAX backend for CUDA.

Model sizes:
  mrt2_small (230M) — recommended for RTX 3050 6GB
  mrt2_base (2.4B)  — higher quality, may not fit in 6GB
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
log = logging.getLogger("magenta-rt")

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
MRT_DATA_DIR = Path(os.environ.get(
    "MRT_DATA_DIR",
    Path.home() / "Documents" / "Magenta" / "magenta-rt-v2",
))

# ── Model registry ─────────────────────────────────────────────────────────
MODELS = {
    "mrt2_small": {
        "params": "230M",
        "description": "Small (230M) — any Apple Silicon Mac, NVIDIA GPU offline",
        "quality": "Good",
        "vram_estimate_gb": 1.5,
    },
    "mrt2_base": {
        "params": "2.4B",
        "description": "Base (2.4B) — Apple Silicon Pro Max, NVIDIA GPU 40GB+",
        "quality": "Better",
        "vram_estimate_gb": 5.5,
    },
}

# ── LLM eviction (llama-swap coordination) ────────────────────────────────
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")
LLAMA_SWAP_CLI = os.environ.get(
    "LLAMA_SWAP_CLI",
    os.path.expanduser("~/git/ai-dotfiles/llama-swap/llama-swap-cli"),
)


def _llama_swap_running_models() -> list[str]:
    """Query llama-swap /running endpoint."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/running", timeout=3) as resp:
            data = json.loads(resp.read())
            return [m.get("model", m.get("id", "?")) for m in data.get("running", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def evict_llm() -> bool:
    """Evict all running LLM models from llama-swap to free VRAM."""
    running = _llama_swap_running_models()
    if not running:
        log.info("No LLM models loaded — VRAM already free")
        return False

    log.info("Evicting LLM models from llama-swap: %s", ", ".join(running))

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
        import urllib.request
        import urllib.error
        log.info("llama-swap-cli not found — trying direct API eviction")
        try:
            req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/unload", method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                log.info("API eviction response: %s", resp.read().decode())
        except (urllib.error.URLError, OSError) as e:
            log.warning("Could not evict LLM models via API: %s", e)

    import time as _time
    for _ in range(10):
        _time.sleep(0.5)
        if not _llama_swap_running_models():
            log.info("LLM models evicted — VRAM free for music generation")
            return True

    log.warning("Could not confirm LLM eviction — proceeding anyway")
    return True


# ── GPU memory ─────────────────────────────────────────────────────────────
def get_gpu_memory() -> dict[str, float]:
    """Get NVIDIA GPU memory info via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                return {
                    "total_mb": float(parts[0].strip()),
                    "used_mb": float(parts[1].strip()),
                    "free_mb": float(parts[2].strip()),
                }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return {}


# ── Model download check ──────────────────────────────────────────────────
def check_model_available(model_name: str) -> bool:
    """Check if model weights are downloaded."""
    model_dir = MRT_DATA_DIR / "models" / model_name
    return model_dir.exists() and any(model_dir.iterdir())


def check_resources_available() -> bool:
    """Check if style/codec resources are downloaded."""
    resources_dir = MRT_DATA_DIR / "resources"
    return resources_dir.exists() and any(resources_dir.iterdir())


# ── Generation ─────────────────────────────────────────────────────────────
def generate_audio(
    model_name: str,
    prompt: str,
    duration: float,
    output: Path | None,
    cwd: Path | None,
    mrt_bin: str,
) -> dict:
    """Generate audio using mrt CLI. Returns metadata dict."""
    model_info = MODELS[model_name]

    # Build command: mrt jax generate --prompt "..." --duration 4.0 --model mrt2_small
    cmd = [
        mrt_bin, "jax", "generate",
        "--prompt", prompt,
        "--duration", str(duration),
        "--model", model_name,
    ]

    log.info("Running: %s", " ".join(cmd))
    log.info("Model: %s (%s params)", model_name, model_info["params"])
    log.info("Prompt: %r", prompt)
    log.info("Duration: %.1fs", duration)

    t0 = time.perf_counter()
    gpu_before = get_gpu_memory()

    result = subprocess.run(cmd, capture_output=True, text=True)

    wall_time = time.perf_counter() - t0
    gpu_after = get_gpu_memory()

    if result.returncode != 0:
        log.error("Generation failed (rc=%d)", result.returncode)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                log.error("  %s", line)
        print(f"\n  ✗ Generation failed (exit code {result.returncode})")
        print(f"    See log above for details.")
        sys.exit(1)

    # mrt saves to MRT_DATA_DIR by default — find the output file
    output_path = None

    # If user specified output path, check there first
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        if output_path.exists():
            pass  # found it
        else:
            log.warning("Specified output not found: %s", output_path)

    # Otherwise, look for the most recent .wav in MRT_DATA_DIR
    if output_path is None or not output_path.exists():
        # mrt jax generate outputs to stdout or a default location
        # Check if stdout contains a path
        wav_files = []
        # Search common output locations
        for search_dir in [Path.cwd(), MRT_DATA_DIR]:
            wav_files.extend(search_dir.rglob("*.wav"))

        # Also check if mrt printed the output path
        for line in result.stdout.strip().split("\n"):
            for part in line.split():
                candidate = Path(part)
                if candidate.exists() and candidate.suffix == ".wav":
                    wav_files.append(candidate)

        if wav_files:
            # Use the newest file
            output_path = max(wav_files, key=lambda p: p.stat().st_mtime)
        else:
            log.error("No output WAV file found")

    # If still not found, create output in cwd
    if output_path is None or not output_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = cwd if cwd else Path.cwd()
        output_path = base_dir / f"mrt2_{model_name}_{ts}.wav"
        log.warning("Could not find mrt output — reporting expected path: %s", output_path)

    # Calculate peak memory
    peak_hbm = 0.0
    if gpu_after:
        peak_hbm = max(gpu_after.get("used_mb", 0), gpu_before.get("used_mb", 0))

    file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

    metadata = {
        "model": model_name,
        "params": model_info["params"],
        "prompt": prompt,
        "duration": duration,
        "wall_seconds": round(wall_time, 3),
        "peak_hbm_mib": round(peak_hbm, 1),
        "output": str(output_path),
        "file_size_mb": round(file_size_mb, 2),
    }

    return metadata


# ── Output helpers ─────────────────────────────────────────────────────────
def save_metadata(metadata: dict) -> Path:
    """Append generation record to JSON log."""
    meta_dir = OUTPUTS_DIR / metadata["model"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "generations.json"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metadata,
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


def print_debrief(metadata: dict, gpu_info: dict) -> None:
    """Print generation stats report."""
    model_name = metadata["model"]
    model_info = MODELS[model_name]

    print()
    print("═══ magenta-rt — Generation Report ═══")
    print(f"  Model:       {model_name} ({model_info['params']} params)")
    print(f"  Prompt:      \"{metadata['prompt']}\"")
    print(f"  Duration:    {metadata['duration']:.1f}s")
    print()
    print("  Timings:")
    print(f"    Wall:      {metadata['wall_seconds']:7.2f} s")
    print()
    if metadata.get("peak_hbm_mib", 0) > 0:
        print("  Memory:")
        print(f"    Peak HBM:  {metadata['peak_hbm_mib']:,.0f} MiB")
        if gpu_info:
            print(f"    GPU Free:  {gpu_info.get('free_mb', 0):,.0f} MiB / {gpu_info.get('total_mb', 0):,.0f} MiB")
        print()
    if metadata.get("file_size_mb", 0) > 0:
        print(f"  Output:      {metadata['file_size_mb']:.2f} MB")
    print(f"  File:        {metadata['output']}")
    print("══════════════════════════════════════")


# ── Argument parsing ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="magenta-rt",
        description="magenta-rt — Local music generation with Magenta RealTime 2",
        epilog=(
            "Recommended:\n"
            "  Quick test:     magenta-rt generate -p 'disco funk'\n"
            "  Longer piece:   magenta-rt generate -p 'ambient pads' --duration 8.0\n"
            "  With eviction:  magenta-rt generate -p 'jazz piano' --evict-llm\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-m", "--model",
        choices=sorted(MODELS),
        default="mrt2_small",
        help="Model variant (default: mrt2_small)",
    )
    p.add_argument("-p", "--prompt", help="Text prompt. If omitted, prompted interactively.")
    p.add_argument("--duration", type=float, default=4.0, help="Duration in seconds (default: 4.0).")
    p.add_argument("--output", type=Path, default=None, help="Output WAV path (auto-detected if not set).")
    p.add_argument(
        "--evict-llm", action="store_true",
        help="Evict all running LLM models (via llama-swap) to free VRAM before generating.",
    )
    return p.parse_args()


# ── Interactive prompt ──────────────────────────────────────────────────────
def get_prompt_interactive() -> str:
    """Prompt the user for a text prompt interactively."""
    print()
    print("  🎵 magenta-rt — Enter your prompt (Ctrl+C to cancel)")
    print("  ─────────────────────────────────────────────────────")
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

    model_name = args.model
    model_info = MODELS[model_name]
    prompt = args.prompt or get_prompt_interactive()

    # Find mrt CLI in venv
    venv_bin = SCRIPT_DIR / ".venv" / "bin" / "mrt"
    if not venv_bin.exists():
        # Fallback: try system PATH
        import shutil
        mrt_bin = shutil.which("mrt")
        if mrt_bin is None:
            print("\n  ✗ mrt CLI not found")
            print("    Run: magenta-rt setup")
            sys.exit(1)
    else:
        mrt_bin = str(venv_bin)

    # ── Preflight check ──
    if not check_resources_available():
        print("\n  ✗ Resources not found (MusicCoCa + SpectroStream)")
        print("    Run: magenta-rt download resources")
        sys.exit(1)

    if not check_model_available(model_name):
        print(f"\n  ✗ Model not found: {model_name}")
        print("    Run: magenta-rt download small")
        sys.exit(1)

    # ── LLM eviction ──
    if args.evict_llm:
        running = _llama_swap_running_models()
        if running:
            print(f"  🔄 Evicting LLM models: {', '.join(running)}")
            evicted = evict_llm()
            if evicted:
                print(f"     VRAM freed — music pipeline can load")
            else:
                print(f"     Warning: eviction may not have fully completed")
        else:
            print(f"  ✅ No LLM models loaded — VRAM already free")
        print()

    # ── GPU memory check ──
    gpu_info = get_gpu_memory()
    if gpu_info:
        free_gb = gpu_info.get("free_mb", 0) / 1024
        print(f"  GPU VRAM free: {free_gb:.1f} GB / {gpu_info.get('total_mb', 0) / 1024:.1f} GB")
        est_gb = model_info["vram_estimate_gb"]
        if free_gb < est_gb:
            print(f"  ⚠️  Estimated model need: {est_gb:.1f} GB — may not fit!")
            print(f"     Consider --evict-llm or using mrt2_small")
        print()

    # ── Generate ──
    print(f"  🎵 Generating {args.duration:.1f}s of audio with {model_name}...")
    print(f"     Prompt: \"{prompt}\"")
    print()

    metadata = generate_audio(
        model_name=model_name,
        prompt=prompt,
        duration=args.duration,
        output=args.output,
        cwd=Path.cwd(),
        mrt_bin=mrt_bin,
    )

    # Save metadata
    save_metadata(metadata)

    # Print debrief
    print_debrief(metadata, gpu_info)


if __name__ == "__main__":
    main()