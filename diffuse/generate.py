#!/usr/bin/env python3
"""diffuse — Local Diffusion Image Generation CLI.

Load pipeline → prompt → generate → stats debrief → unload.
Designed for NVIDIA RTX 3050 6GB. Model-agnostic — add new backends via MODELS registry.

Currently supports:
  - Bonsai Image 4B (gemlite + HQQ kernels on CUDA)
  - Ideogram 4 Q4 (sd-cli/stable-diffusion.cpp with CUDA offload)

Usage:
  diffuse -m ternary-gemlite -p 'a cat on the moon'
  diffuse -m ideogram4-q4 --enhance -p 'a rainy coffee shop' --show-enhanced
  diffuse -m ideogram4-q4:think --enhance --evict-llm -p 'cyberpunk city'
  diffuse -m ideogram4-q4 --cpu-fallback -p 'sunset'  # fallback if CUDA OOM
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
SD_CLI_PATH = SCRIPT_DIR / "bin" / "sd-cli"

# ── LLM prompt enhancement ─────────────────────────────────────────────────
ENHANCE_SYSTEM_PROMPT = """You are a prompt engineer for Ideogram 4, a text-to-image model.
Convert the user's simple text prompt into a structured JSON object that Ideogram 4 uses for image generation.

REQUIRED format — respond ONLY with valid JSON, no markdown, no explanation:
{
  "high_level_description": "<detailed scene description in 2-3 sentences>",
  "style_description": {
    "aesthetics": "<art style, medium, visual qualities>",
    "lighting": "<lighting description>",
    "color_palette": ["<hex color 1>", "<hex color 2>", "<hex color 3>", "<hex color 4>", "<hex color 5>"]
  }
}

GUIDELINES:
- high_level_description: Expand the prompt into a rich, specific scene. Add details about position, pose, expression, setting.
- aesthetics: Describe the art style (e.g. "vibrant digital illustration, clean lines, detailed", "photorealistic photography", "oil painting style"). Be specific.
- lighting: Describe the lighting (e.g. "warm golden hour sunlight", "dramatic side lighting", "soft diffused studio lighting").
- color_palette: 5 hex colors that define the mood. Use a cohesive palette. Prefer saturated, distinctive colors over generic ones.
- The JSON must be parseable. No trailing commas, no comments.
- Keep the description focused on what should APPEAR in the image, not abstract concepts.
- If the prompt requests text in the image, include it in compositional_deconstruction elements."""

THINK_ENHANCE_SYSTEM_PROMPT = """You are a prompt engineer for Ideogram 4, a text-to-image model.
Convert the user's simple text prompt into a detailed structured JSON object for maximum quality image generation.

REQUIRED format — respond ONLY with valid JSON, no markdown, no explanation:
{
  "high_level_description": "<detailed scene description in 2-4 sentences, very specific>",
  "style_description": {
    "aesthetics": "<detailed art style, medium, visual qualities>",
    "lighting": "<detailed lighting description with direction and mood>",
    "color_palette": ["<hex 1>", "<hex 2>", "<hex 3>", "<hex 4>", "<hex 5>"]
  },
  "compositional_deconstruction": {
    "canvas": "<canvas description: size, orientation, layout style>",
    "background": "<detailed background description>",
    "layout": "<layout description: where elements are placed>",
    "elements": [
      {"type": "obj", "desc": "<detailed description of element 1>"},
      {"type": "obj", "desc": "<detailed description of element 2>"},
      {"type": "text", "desc": "<any text that should appear in the image, if requested>"}
    ]
  }
}

Be extremely detailed and specific. Think about the composition, colors, lighting, and every element carefully before writing the JSON."""

# ── Model registry ─────────────────────────────────────────────────────────
MODELS = {
    # Bonsai Image 4B (gemlite CUDA)
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
    # Ideogram 4 (sd-cli / stable-diffusion.cpp)
    "ideogram4-q4": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
        "enhance_model": "qwen3.5-4b",
        "default_size": (480, 480),
    },
    "ideogram4-q4:think": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — enhanced prompts via Qwen 3.5 4B (thinking mode)",
        "enhance_model": "qwen3.5-4b",
        "enhance_think": True,
        "default_size": (480, 480),
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

    pipeline = GpuPipeline(
        backend=model_info["backend_id"],
        binary_transformer_path=str(transformer_dir),
        ternary_transformer_path=str(transformer_dir),
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(_find_subdir(model_root, "vae")),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )

    if GEMLITE_PERSIST_PATH.exists():
        GemLiteLinearTriton.load_config(str(GEMLITE_PERSIST_PATH), print_error=False)

    pipeline.prewarm()
    load_time = time.perf_counter() - t0

    GemLiteLinearTriton.cache_config(str(GEMLITE_PERSIST_PATH))

    return pipeline, load_time

def load_pipeline(model_name: str) -> tuple:
    """Load a pipeline based on the model's backend_type. Returns (pipeline_or_config, load_time_seconds)."""
    model_info = MODELS[model_name]
    backend_type = model_info.get("backend_type", "gemlite")

    if backend_type == "gemlite":
        return load_pipeline_gemlite(model_name)
    elif backend_type == "sd_cpp":
        return load_pipeline_sd_cpp(model_name)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

def load_pipeline_sd_cpp(model_name: str) -> tuple:
    """Prepare sd-cli configuration. Returns (config_dict, 0.0)."""
    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)

    sd_cli = str(SD_CLI_PATH)
    if not Path(sd_cli).exists():
        # Try stable-diffusion.cpp build location
        alt = Path.home() / "git" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
        if alt.exists():
            sd_cli = str(alt)
        else:
            raise FileNotFoundError(f"sd-cli not found at {SD_CLI_PATH}. Run: diffuse build-sd-cpp")

    config = {
        "sd_cli": sd_cli,
        "diffusion_model": str(model_root / "ideogram4-Q4_0.gguf"),
        "uncond_diffusion_model": str(model_root / "ideogram4_uncond-Q4_0.gguf"),
        "llm": str(model_root / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "vae": str(model_root / "vae" / "flux2-vae.safetensors"),
    }

    # Verify all files exist
    for key, path in config.items():
        if key == "sd_cli" and not Path(path).exists():
            raise FileNotFoundError(f"{key} not found: {path}")

    return config, 0.0

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
    """Evict all loaded LLM models from llama-swap to free VRAM for diffusion."""
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

    log.warning("Could not confirm LLM eviction — proceeding anyway")
    return True

# ── LLM prompt enhancement ─────────────────────────────────────────────────
def enhance_prompt(prompt: str, model: str, think: bool = False) -> str:
    """Use an LLM via llama-swap to expand a simple prompt into Ideogram 4 JSON."""
    system = THINK_ENHANCE_SYSTEM_PROMPT if think else ENHANCE_SYSTEM_PROMPT

    log.info("Enhancing prompt via %s (think=%s)", model, think)
    t0 = time.perf_counter()

    # Call llama-swap OpenAI-compatible API
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7 if not think else 0.4,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLAMA_SWAP_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            enhanced = data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
        log.error("Prompt enhancement failed: %s — using raw prompt", e)
        return prompt

    elapsed = time.perf_counter() - t0
    log.info("Prompt enhanced in %.1fs", elapsed)

    # Try to extract JSON from the response
    # The LLM might wrap it in ```json``` blocks
    enhanced = enhanced.strip()
    if enhanced.startswith("```json"):
        enhanced = enhanced[7:]
    if enhanced.startswith("```"):
        enhanced = enhanced[3:]
    if enhanced.endswith("```"):
        enhanced = enhanced[:-3]
    enhanced = enhanced.strip()

    # Validate it's parseable JSON
    try:
        parsed = json.loads(enhanced)
        if "high_level_description" not in parsed:
            log.warning("Enhanced prompt missing 'high_level_description' — using raw prompt")
            return prompt
        # Return the JSON string as-is — sd-cli accepts JSON prompts
        return enhanced
    except json.JSONDecodeError:
        log.warning("Enhanced prompt is not valid JSON — using raw prompt")
        return prompt

# ── Generation ─────────────────────────────────────────────────────────────
def generate_image_gemlite(pipeline, prompt: str, seed: int, steps: int, width: int, height: int) -> tuple:
    """Generate a PNG image using gemlite pipeline. Returns (png_bytes, diffusion_time, peak_hbm_mb)."""
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

def generate_image_sd_cpp(config: dict, prompt: str, seed: int, width: int, height: int, output_path: Path, cpu_fallback: bool = False) -> tuple:
    """Generate image using sd-cli. Returns (output_path, wall_time_seconds, 0.0)."""
    log.info("Generating via sd-cli: seed=%d size=%dx%d cpu_fallback=%s", seed, width, height, cpu_fallback)

    cmd = [
        config["sd_cli"],
        "--diffusion-model", config["diffusion_model"],
        "--uncond-diffusion-model", config["uncond_diffusion_model"],
        "--llm", config["llm"],
        "--vae", config["vae"],
        "-p", prompt,
        "--diffusion-fa",
        "--offload-to-cpu",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "--max-vram", "4.8",
        "--stream-layers",
        "-H", str(height),
        "-W", str(width),
        "--seed", str(seed),
        "-o", str(output_path),
    ]

    # CPU fallback: remove VRAM limits and force everything on CPU
    if cpu_fallback:
        log.warning("Retrying with CPU-only backend — this will be very slow (~30+ minutes)")
        cmd = [
            config["sd_cli"],
            "--diffusion-model", config["diffusion_model"],
            "--uncond-diffusion-model", config["uncond_diffusion_model"],
            "--llm", config["llm"],
            "--vae", config["vae"],
            "-p", prompt,
            "--backend", "cpu",
            "-H", str(height),
            "-W", str(width),
            "--seed", str(seed),
            "-o", str(output_path),
        ]

    timeout_secs = 2400 if cpu_fallback else 600  # 40 min for CPU fallback, 10 min for CUDA
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_secs)
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        # Print last 20 lines of stderr for debugging
        stderr_lines = result.stderr.strip().split("\n")[-20:]
        for line in stderr_lines:
            log.error("sd-cli: %s", line)
        raise RuntimeError(f"sd-cli failed (rc={result.returncode}). Last error: {stderr_lines[-1] if stderr_lines else 'unknown'}")

    if not output_path.exists():
        raise FileNotFoundError(f"sd-cli did not produce output: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("sd-cli completed in %.1fs, output %.2f MiB", wall_time, file_size_mb)

    return output_path, wall_time, 0.0

# ── Output helpers ─────────────────────────────────────────────────────────
def resolve_output_path(model_name: str, seed: int, output_arg: str | None, cwd: Path | None = None) -> Path:
    """Determine output PNG path."""
    if output_arg is not None:
        return Path(output_arg).expanduser().resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filesystem (replace : and other problematic chars)
    safe_model = model_name.replace(":", "_").replace("/", "_")
    base_dir = cwd if cwd else Path.cwd()
    out = base_dir / f"diffuse_{safe_model}_{ts}_seed{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Also write to internal log dir for metadata
    internal_dir = OUTPUTS_DIR / model_name
    internal_dir.mkdir(parents=True, exist_ok=True)

    return out

def save_metadata(
    model_name: str, prompt: str, seed: int, width: int, height: int,
    steps: int, load_time: float, diffusion_time: float, wall_time: float,
    peak_hbm: float, output_path: Path, enhanced_prompt: str | None = None,
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
    if enhanced_prompt:
        record["enhanced_prompt"] = enhanced_prompt
        record["prompt_enhanced"] = True

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

# ── Debrief display ───────────────────────────────────────────────────────
def print_debrief(
    model_name: str, model_info: dict, prompt: str, seed: int,
    width: int, height: int, steps: int, load_time: float,
    diffusion_time: float, wall_time: float, peak_hbm: float,
    output_path: Path, enhanced_prompt: str | None = None,
    original_prompt: str | None = None,
) -> None:
    """Print generation stats report."""
    print()
    print("═══ diffuse — Generation Report ═══")
    print(f"  Model:       {model_name} ({model_info['bits']})")
    if original_prompt and original_prompt != prompt:
        print(f"  Prompt:      \"{original_prompt}\"")
        print(f"  Enhanced:    Yes → Ideogram 4 JSON")
    else:
        print(f"  Prompt:      \"{prompt}\"")
    print(f"  Seed:        {seed}")
    print(f"  Resolution:  {width} × {height}")
    print(f"  Steps:       {steps}")
    if original_prompt and original_prompt != prompt and enhanced_prompt:
        print()
        print("  Enhanced prompt (JSON):")
        try:
            parsed = json.loads(enhanced_prompt)
            for key, val in parsed.items():
                if isinstance(val, dict):
                    print(f"    {key}:")
                    for k, v in val.items():
                        if isinstance(v, list):
                            print(f"      {k}:")
                            for item in v:
                                print(f"        {item}")
                        else:
                            print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {val}")
        except json.JSONDecodeError:
            for line in enhanced_prompt.split("\n"):
                print(f"    {line[:120]}")
    print()
    print("  Timings:")
    print(f"    Setup:      {load_time:7.2f} s   (imports + model load)")
    print(f"    Diffusion: {diffusion_time:7.2f} s   (denoising + VAE decode)")
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
        print(f"  {name}")
        print(f"    {bits}  |  {backend}  |  default {size_str}")
        if enhance:
            think = " (thinking mode)" if info.get("enhance_think") else ""
            print(f"    enhance: {enhance}{think}")
        print(f"    {desc}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffuse",
        description="diffuse — Local diffusion image generation CLI for NVIDIA RTX 3050 6GB",
        epilog=(
            "Examples:\n"
            "  diffuse -m ternary-gemlite -p 'a cat on the moon'\n"
            "  diffuse -m ideogram4-q4 --enhance -p 'a rainy day at a coffee shop'\n"
            "  diffuse -m ideogram4-q4:think --evict-llm --enhance -p 'cyberpunk city'\n"
            "  diffuse -m ideogram4-q4 -p '{\"high_level_description\": \"...\"}' --size 480x480\n"
            "  diffuse --list                        # show model details\n"
            "\n"
            "Ideogram 4 requires structured JSON prompts for best results.\n"
            "Use --enhance to auto-expand simple text into JSON via LLM.\n"
            "The ':think' suffix uses thinking mode for richer prompts.\n"
            "\n"
            "Recommended sizes for RTX 3050 6GB:\n"
            "  Safe:  480x480, 624x416, 416x624 (fits VRAM comfortably)\n"
            "  Max:   512x512, 624x448, 448x624 (may OOM with other GPU load)\n"
            "  Bonsai: 512x512 is the default and safe for all models\n"
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
    p.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 4 for bonsai, 20 for ideogram4).")
    p.add_argument(
        "--size", type=parse_size, default=None,
        help="Image size as WxH (default: 512x512 for bonsai, 480x480 for ideogram4).",
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
        help="Expand simple text prompt into Ideogram 4 JSON via LLM (qwen3.5-4b). "
             "Essential for ideogram4 — plain text produces poor results. "
             "Uses model's 'enhance_model' or qwen3.5-4b by default. "
             "The ':think' suffix activates thinking mode for more detailed prompts.",
    )
    p.add_argument(
        "--show-enhanced", action="store_true",
        help="Print the full enhanced JSON prompt before generating.",
    )
    p.add_argument(
        "--cpu-fallback", action="store_true",
        help="If CUDA generation fails, automatically retry on CPU (very slow: ~30+ min).",
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

    if args.list:
        print_models()
        sys.exit(0)

    setup_environment()

    model_name = args.model
    model_info = MODELS[model_name]
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    backend_type = model_info.get("backend_type", "gemlite")

    # ── Defaults per backend ──
    # Steps: bonsai=4, ideogram4=20
    if args.steps is None:
        args.steps = 20 if backend_type == "sd_cpp" else 4

    # Size: model-specific defaults (480x480 for ideogram4 on 6GB VRAM, 512x512 for bonsai)
    if args.size is None:
        default_size = model_info.get("default_size", (512, 512))
        width, height = default_size
    else:
        width, height = args.size

    prompt = args.prompt or get_prompt_interactive()

    original_prompt = prompt

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

    # ── Prompt enhancement ──
    enhanced_prompt = None
    enhance_think = model_info.get("enhance_think", False)
    enhance_model = model_info.get("enhance_model")

    if args.enhance:
        if not enhance_model:
            enhance_model = "qwen3.5-4b"
        print(f"  ✨ Enhancing prompt via {enhance_model}{' (thinking mode)' if enhance_think else ''}...")
        enhanced_prompt = enhance_prompt(prompt, enhance_model, think=enhance_think)
        if enhanced_prompt != prompt:
            print(f"     Expanded to JSON ({len(enhanced_prompt)} chars)")
            if args.show_enhanced:
                print(f"     ─── Enhanced prompt ───")
                try:
                    parsed = json.loads(enhanced_prompt)
                    for key, val in parsed.items():
                        if isinstance(val, dict):
                            print(f"     {key}:")
                            for k, v in val.items():
                                print(f"       {k}: {v}")
                        else:
                            print(f"     {key}: {val}")
                except json.JSONDecodeError:
                    print(f"     {enhanced_prompt}")
                print(f"     ────────────────────────")
        else:
            print(f"     Enhancement failed — using raw prompt")
        # For sd_cpp, use the enhanced JSON prompt
        if backend_type == "sd_cpp" and enhanced_prompt != prompt:
            prompt = enhanced_prompt

    # Show warm/cold estimate
    prior = get_previous_runs(model_name, width, height)
    print()
    if prior:
        mean_s = sum(prior) / len(prior)
        best_s = min(prior)
        print(f"  ⚡ {len(prior)} prior run(s) at {width}×{height} — warmed kernels available")
        print(f"     Historical wall: mean {mean_s:.1f}s, best {best_s:.1f}s")
    else:
        if backend_type == "sd_cpp":
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Expected: ~80-100s (model load + offload + 20 denoising steps)")
        else:
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Cold start: ~30-60s (imports + model load + kernel JIT)")
            print(f"     Subsequent runs at this size will be faster (cached kernels)")
    print()

    # ── Phase 1.5: Evict LLMs after prompt enhancement ──
    # If we used an LLM for enhancement, evict it before loading sd-cli
    # (sd-cli also needs VRAM for the diffusion model)
    if args.enhance and backend_type == "sd_cpp":
        running = _llama_swap_running_models()
        if running:
            print(f"  🔄 Evicting LLM models after enhancement: {', '.join(running)}")
            evict_llm()
            print(f"     VRAM freed for image generation")
            print()

    # ── Phase 1: Load pipeline ──
    print(f"  [1/3] Loading pipeline ({model_info['bits']})...")
    pipeline, load_time = load_pipeline(model_name)
    if backend_type == "gemlite":
        print(f"        Pipeline ready in {load_time:.1f}s")
    else:
        print(f"        sd-cli config ready")

    # ── Phase 2: Generate ──
    print(f"  [2/3] Generating ({width}×{height}, {args.steps} steps, seed={seed})...")
    wall_t0 = time.perf_counter()

    if backend_type == "gemlite":
        png_bytes, diffusion_time, peak_hbm = generate_image_gemlite(
            pipeline, prompt, seed, args.steps, width, height,
        )
        # Use the caller's cwd, not the script's directory
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

    if backend_type == "gemlite":
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


if __name__ == "__main__":
    main()