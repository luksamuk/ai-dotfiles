#!/usr/bin/env python3
"""diffuse — Local Diffusion Image Generation CLI.

Load pipeline → prompt → generate → stats debrief → unload.
Designed for NVIDIA RTX 3050 6GB. Model-agnostic — add new backends via MODELS registry.

Currently supports:
  - Bonsai Image 4B (gemlite + HQQ kernels on CUDA)
  - Ideogram 4 Q4 (sd-cli/stable-diffusion.cpp with CUDA offload)
  - HiDream-O1-Image-Dev SDNQ (transformers + accelerate CPU offload)

Usage:
  diffuse -m ternary-gemlite -p 'a cat on the moon'
  diffuse -m ideogram4-q4 --enhance -p 'a rainy coffee shop'
  diffuse -m ideogram4-q4 --enhance-with laguna-xs2 --evict-llm -p 'cyberpunk city'
  diffuse -m hidream-sdnq -p 'a cat on the moon'              # text-to-image
  diffuse -m hidream-sdnq -p 'add a hat' --edit photo.png     # image editing
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
Convert the user's simple text prompt into a detailed structured JSON object for maximum quality image generation.

You MUST respond with ONLY the raw JSON object below — no markdown code fences, no explanation, no commentary, no text before or after the JSON.

REQUIRED format:
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

Be extremely detailed and specific. Think about the composition, colors, lighting, and every element carefully before writing the JSON.

IMPORTANT: Output ONLY the JSON object. Do NOT wrap it in ```json``` code blocks. Do NOT add any text before or after the JSON."""

# ── Vision enhancement prompt (for HiDream, Bonsai) ────────────────────────
ENHANCE_VISION_SYSTEM_PROMPT = """You are a prompt engineer for text-to-image models (HiDream, Bonsai).
Your job is to expand simple prompts into rich, detailed English descriptions that produce stunning images.

RULES:
1. If the prompt is not in English, translate it to English first, then expand.
2. For named characters from games, anime, books, or mythology (e.g. Alucard, Goku, Gandalf, Sonic the Hedgehog) — ALWAYS include the character's FULL NAME in the prompt so the image model can anchor on known training data. THEN describe their PHYSICAL APPEARANCE in detail as a complement: body type, skin tone, hair color/style, facial features, clothing/armor details, weapons, posture, aura. Do NOT assume the model knows who they are from description alone — give both the name AND the visual description.
3. Describe the art style explicitly: medium (oil painting, watercolor, digital art, etc.), lighting (golden hour, dramatic chiaroscuro, soft ambient, etc.), color palette, mood.
4. Add compositional details: camera angle, depth of field, background atmosphere, environmental storytelling.
5. Keep the output as a SINGLE natural-language paragraph. No JSON, no bullet points, no headers.
6. Be specific and vivid — vague descriptions produce vague images.

EXAMPLE INPUT: "Alucard from Castlevania SOTN, estilo Ayami Kojima, óleo sobre tela"
EXAMPLE OUTPUT: "Alucard from Castlevania: Symphony of the Night — a pale androgynous male vampire with long flowing silver-white hair and crimson eyes, wearing an ornate black Victorian-era coat with gold filigree embroidery over a white ruffled cravat, a dark cape draped over one shoulder, black leather gloves, standing in a gothic cathedral interior with towering stained glass windows casting purple and crimson light. He holds a glowing sword. Oil on canvas style with visible brushstrokes, heavy chiaroscuro lighting, muted earthy tones of burgundy, deep purple, aged gold, and cold blue, in the manner of Ayami Kojima's Castlevania character art — melancholic, romantic, with an ethereal painterly quality and soft edges blending into the dark background."

OUTPUT ONLY the expanded prompt. No commentary, no labels, no markdown."""

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
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
    },
    "ternary-gemlite": {
        "backend_id": "bonsai-ternary-gemlite",
        "hf_repo": "prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
        "dir": "bonsai-image-4B-ternary-gemlite",
        "backend_type": "gemlite",
        "bits": "1.58-bit",
        "transformer_kwarg": "ternary_transformer_path",
        "description": "1.58-bit {−1, 0, +1} — 1.21 GB transformer, 95% of FP16 quality",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
    },
    # Ideogram 4 (sd-cli / stable-diffusion.cpp)
    "ideogram4-q4": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "ideogram",
        "default_size": (480, 480),
    },
    # HiDream-O1-Image-Dev SDNQ (transformers + accelerate CPU offload)
    "hidream-sdnq": {
        "dir": "HiDream-O1-Image-Dev-SDNQ-last8",
        "backend_type": "hidream",
        "bits": "4-bit SDNQ (uint4+svd-r32 last8-odown-bf16)",
        "description": "HiDream-O1-Image-Dev SDNQ — 8B unified (T2I + editing + IP), ~3min/2048² on 6GB VRAM",
        "default_size": (1024, 1024),
        "hidream_repo": "~/git/HiDream-O1-Image",
        "enhance_model": "qwen3.5-4b",
        "enhance_type": "vision",
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

def load_pipeline(model_name: str, editing: bool = False) -> tuple:
    """Load a pipeline based on the model's backend_type. Returns (pipeline_or_config, load_time_seconds)."""
    model_info = MODELS[model_name]
    backend_type = model_info.get("backend_type", "gemlite")

    if backend_type == "gemlite":
        return load_pipeline_gemlite(model_name)
    elif backend_type == "sd_cpp":
        return load_pipeline_sd_cpp(model_name)
    elif backend_type == "hidream":
        return load_pipeline_hidream(model_name, editing=editing)
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


def _wait_for_model_ready(model: str, timeout: int = 120) -> bool:
    """Poll llama-swap /running until the requested model is in 'ready' state.

    llama-swap starts the model on first request but the health check takes
    several seconds. Without waiting, the first request hits the backend
    before it's ready and gets a 400.

    Returns True if model is ready, False if timeout exceeded.
    """
    import urllib.request
    import urllib.error
    import time

    # Extract base model name (strip :think, :code, etc. suffixes)
    base_model = model.split(":")[0]
    t0 = time.perf_counter()

    while True:
        try:
            with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/running", timeout=3) as resp:
                data = json.loads(resp.read())
                for m in data.get("running", []):
                    if m.get("model") == base_model and m.get("state") == "ready":
                        return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass

        if time.perf_counter() - t0 > timeout:
            return False

        time.sleep(1)

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
def _extract_json(text: str) -> str | None:
    """Extract the first valid JSON object from text.

    Handles: raw JSON, ```json``` code blocks, ``` code blocks,
    text before/after JSON, and nested braces correctly.
    """
    import re

    if not text:
        return None

    text = text.strip()

    # 1. Try the whole text as JSON (most common case)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Strip ```json ... ``` or ``` ... ``` code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 3. Find the first top-level JSON object using brace matching
    #    This handles text before/after the JSON reliably
    first_brace = text.find("{")
    if first_brace >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(first_brace, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        # Keep searching for next '{' after this point
                        next_brace = text.find("{", i + 1)
                        if next_brace < 0:
                            break
                        # Continue from outer loop won't work, so restart
                        first_brace = next_brace
                        depth = 0
                        in_string = False
                        escape = False
                        continue
        # If we exhausted brace matching without a valid object, try last candidate
        # as a fallback (might be truncated but still useful for debugging)

    return None


def enhance_prompt(prompt: str, model: str) -> tuple:
    """Use an LLM via llama-swap to expand a simple prompt into Ideogram 4 JSON.

    Returns (enhanced_prompt, raw_response).
    On failure, enhanced_prompt is the original prompt and raw_response contains the LLM output.
    """
    system = ENHANCE_SYSTEM_PROMPT

    log.info("Enhancing prompt via %s", model)
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
        "temperature": 0.7,
        "max_tokens": 8192,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLAMA_SWAP_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Retry loop: llama-swap may return 400 while model is loading
    # (health check not passed yet). Wait and retry.
    max_retries = 12  # up to ~2 minutes
    enhanced = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                # Thinking models (e.g. qwen3.6:think) put reasoning in
                # reasoning_content and the actual answer in content.
                # If content is empty but reasoning_content exists, the
                # model likely ran out of max_tokens during thinking.
                enhanced = msg.get("content", "").strip()
                reasoning = msg.get("reasoning_content", "")
                if not enhanced and reasoning:
                    log.warning(
                        "Enhancement model returned empty content with %d chars of reasoning — "
                        "likely hit max_tokens during thinking. Increase max_tokens or use non-think variant.",
                        len(reasoning),
                    )
                    return prompt, reasoning
                break  # success
        except urllib.error.HTTPError as e:
            if e.code == 400 and attempt < max_retries - 1:
                log.info("Model loading (attempt %d/%d), retrying in 10s...", attempt + 1, max_retries)
                time.sleep(10)
                continue
            log.error("Prompt enhancement failed: %s — using raw prompt", e)
            return prompt, str(e)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            log.error("Prompt enhancement failed: %s — using raw prompt", e)
            return prompt, str(e)

    elapsed = time.perf_counter() - t0
    log.info("Prompt enhanced in %.1fs", elapsed)

    # Extract JSON from the response — handles code fences, text before/after, etc.
    extracted = _extract_json(enhanced)
    if extracted is None:
        log.warning("Could not extract valid JSON from enhanced prompt — using raw prompt")
        return prompt, enhanced

    # Validate it has the expected structure
    try:
        parsed = json.loads(extracted)
        if "high_level_description" not in parsed:
            log.warning("Enhanced prompt missing 'high_level_description' — using raw prompt")
            return prompt, extracted
        # Return the JSON string as-is — sd-cli accepts JSON prompts
        return extracted, enhanced
    except json.JSONDecodeError:
        log.warning("Enhanced prompt is not valid JSON — using raw prompt")
        return prompt, enhanced

def enhance_vision_prompt(prompt: str, model: str) -> tuple:
    """Use an LLM via llama-swap to expand a prompt for vision models (HiDream, Bonsai).

    Unlike enhance_prompt (which returns Ideogram JSON), this returns a natural-language
    English paragraph that describes characters physically, translates non-English prompts,
    and specifies art style, lighting, and composition.

    Returns (enhanced_prompt, raw_response).
    On failure, enhanced_prompt is the original prompt.
    """
    system = ENHANCE_VISION_SYSTEM_PROMPT

    log.info("Vision-enhancing prompt via %s", model)
    t0 = time.perf_counter()

    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 8192,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLAMA_SWAP_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Retry loop: llama-swap may return 400 while model is loading
    # (health check not passed yet). Wait and retry.
    max_retries = 12  # up to ~2 minutes
    enhanced = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                enhanced = msg.get("content", "").strip()
                reasoning = msg.get("reasoning_content", "")
                if not enhanced and reasoning:
                    log.warning("Vision-enhancement model returned empty content with %d chars of reasoning", len(reasoning))
                    return prompt, reasoning
                break
        except urllib.error.HTTPError as e:
            if e.code == 400 and attempt < max_retries - 1:
                log.info("Model loading (attempt %d/%d), retrying in 10s...", attempt + 1, max_retries)
                time.sleep(10)
                continue
            log.error("Vision-enhancement failed: %s — using raw prompt", e)
            return prompt, str(e)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            log.error("Vision-enhancement failed: %s — using raw prompt", e)
            return prompt, str(e)

    elapsed = time.perf_counter() - t0
    log.info("Vision-enhancement completed in %.1fs", elapsed)

    if not enhanced:
        log.warning("Empty vision-enhancement response — using raw prompt")
        return prompt, enhanced

    return enhanced, enhanced

# ── Generation ─────────────────────────────────────────────────────────────
def generate_image_gemlite(pipeline, prompt: str, seed: int, steps: int, width: int, height: int) -> tuple:
    """Generate a PNG image using gemlite pipeline with text-encoder offload.

    Text encoder (2.84 GB) is moved to CPU after encoding the prompt, freeing
    VRAM for the diffusion loop and VAE decode. This allows larger resolutions
    without OOMing on 6 GB cards.

    Returns (png_bytes, diffusion_time, peak_hbm_mb).
    """
    import torch
    from backend_gpu.diffusion_klein import _encode_klein_qwen3_prompt

    log.info("Generating: prompt=%r seed=%d steps=%d size=%dx%d", prompt, seed, steps, width, height)

    text_encoder = pipeline._text_encoder
    tokenizer = pipeline._tokenizer

    # 1. Encode prompt (text encoder on GPU)
    log.info("Encoding prompt (text encoder on GPU)...")
    t_enc = time.perf_counter()
    prompt_embeds = _encode_klein_qwen3_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
        max_sequence_length=512,
    )
    log.info("Prompt encoded in %.2fs", time.perf_counter() - t_enc)

    # 2. Offload text encoder to CPU — frees ~2.8 GB VRAM
    if torch.cuda.is_available():
        log.info("Offloading text encoder to CPU...")
        text_encoder.to("cpu")
        torch.cuda.empty_cache()
        log.info("Text encoder offloaded to CPU (~2.8 GB VRAM freed)")

    # 3. Diffusion + VAE decode (only transformer + VAE on GPU)
    t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt="",  # Ignored — we pass precomputed embeds
        seed=seed,
        steps=steps,
        height=height,
        width=width,
        precomputed_prompt_embeds=prompt_embeds,
    )
    diffusion_time = time.perf_counter() - t0

    # 4. Move text encoder back to GPU for next call
    if torch.cuda.is_available():
        text_encoder.to(pipeline.device)
        log.info("Text encoder restored to GPU")

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
        "--max-vram", "5.1",
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

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)  # no timeout — let sd-cli finish naturally
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


# ── HiDream backend (transformers + accelerate CPU offload) ─────────────────
HIDREAM_REPO = Path(os.path.expanduser("~/git/HiDream-O1-Image"))

def load_pipeline_hidream(model_name: str, editing: bool = False) -> tuple:
    """Load HiDream-O1-Image-Dev SDNQ model with accelerate CPU offload.

    Returns (pipeline_dict, load_time_seconds).
    pipeline_dict contains model, processor, tokenizer — not a callable pipeline.

    If editing=True, reserves more VRAM headroom for reference image encoding.
    """
    import torch
    from accelerate import infer_auto_device_map, dispatch_model

    # Disable cuDNN to avoid CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH crash
    # with Conv3d + accelerate CPU offload on RTX 3050
    torch.backends.cudnn.enabled = False

    model_info = MODELS[model_name]
    model_root = Path(os.path.expanduser(f"~/.llama-models/{model_info['dir']}"))
    hidream_repo = Path(os.path.expanduser(model_info["hidream_repo"]))

    # Model path already validated in main() — skip duplicate check here
    if not hidream_repo.exists():
        print(f"\n  ✗ HiDream repo not found: {hidream_repo}")
        print(f"    Clone from: https://github.com/HiDream-ai/HiDream-O1-Image")
        sys.exit(1)

    # Add repo to sys.path so we can import sdnq, models, inference
    if str(hidream_repo) not in sys.path:
        sys.path.insert(0, str(hidream_repo))

    import sdnq
    from transformers import AutoProcessor
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    from inference import add_special_tokens, get_tokenizer

    torch.cuda.empty_cache()
    gc.collect()

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    t0 = time.perf_counter()
    print("  Loading SDNQ model on CPU...")

    processor = AutoProcessor.from_pretrained(str(model_root))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_root),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()

    load_time = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded on CPU in {load_time:.1f}s ({param_count/1e9:.1f}B params)")

    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    # Dispatch model across GPU/CPU with memory limit.
    # Editing mode (ref_image) needs ~1 GB more VRAM for reference image encoding.
    # T2I works fine with 1.8 GB headroom; editing needs 3.0+ GB.
    headroom_gb = 3.0 if editing else 1.8
    max_vram_mb = int((total / 1e9 - headroom_gb) * 1024)
    print(f"  Mode: {'editing' if editing else 'T2I'} — reserving {headroom_gb:.1f} GB for activations")
    print(f"  Max GPU for model layers: {max_vram_mb} MiB")

    device_map = infer_auto_device_map(
        model,
        max_memory={0: f"{max_vram_mb}MiB", "cpu": "24GiB"},
        no_split_module_classes=["Qwen3VLDecoderLayer"],
    )

    gpu_layers = len([v for v in device_map.values() if v == 0 or v == torch.device(0)])
    cpu_layers = len([v for v in device_map.values() if v == "cpu" or v == torch.device("cpu")])
    print(f"  Device map: {gpu_layers} layers on GPU, {cpu_layers} on CPU")

    model = dispatch_model(model, device_map=device_map)

    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM after dispatch: {free/1e9:.1f} GB free")

    return {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "hidream_repo": hidream_repo,
    }, load_time


def generate_image_hidream(
    pipeline_dict: dict,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    ref_image_paths: list[str] | None = None,
) -> tuple:
    """Generate an image using HiDream pipeline. Returns (output_path, diffusion_time, peak_hbm).

    If ref_image_paths is provided, performs image editing (instruction-based).
    """
    import torch
    from models.pipeline import generate_image

    model = pipeline_dict["model"]
    processor = pipeline_dict["processor"]
    is_editing = ref_image_paths is not None and len(ref_image_paths) > 0

    # Dev model defaults: 28 steps, guidance_scale=0, shift=1.0
    # Editing uses flash scheduler (flow_match causes cuDNN Conv3d crash with CPU offload on RTX 3050)
    num_inference_steps = steps or 28
    guidance_scale = 0.0
    shift = 1.0
    scheduler_name = "flash"
    noise_scale_start = 7.5
    noise_scale_end = 7.5
    noise_clip_std = 2.5
    extra_kwargs = {
        "noise_scale_start": noise_scale_start,
        "noise_scale_end": noise_scale_end,
        "noise_clip_std": noise_clip_std,
    }

    mode_str = "editing" if is_editing else "T2I"
    log.info("HiDream %s: prompt=%r seed=%d steps=%d size=%dx%d",
             mode_str, prompt[:80], seed, num_inference_steps, width, height)

    t0 = time.perf_counter()

    image = generate_image(
        model=model,
        processor=processor,
        prompt=prompt,
        ref_image_paths=ref_image_paths,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        scheduler_name=scheduler_name,
        seed=seed,
        **extra_kwargs,
    )

    diffusion_time = time.perf_counter() - t0
    peak_hbm = 0.0  # HiDream doesn't expose peak HBM easily

    # Save to temp PNG bytes
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    return png_bytes, diffusion_time, peak_hbm


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
            "  diffuse -m ideogram4-q4 --enhance -p 'a rainy day at a coffee shop'\n"
            "  diffuse -m ideogram4-q4 --enhance-with laguna-xs2 --evict-llm -p 'cyberpunk city'\n"
            "  diffuse -m hidream-sdnq -p 'a cat on a windowsill at golden hour'\n"
            "  diffuse -m hidream-sdnq -p 'add a red hat' --edit photo.png\n"
            "  diffuse -m ideogram4-q4 -p '{\"high_level_description\": \"...\"}' --size 480x480\n"
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
        help="Expand prompt via LLM. For ideogram4: structured JSON. "
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
    # Steps: bonsai=4, ideogram4=20, hidream=28
    if args.steps is None:
        if backend_type == "sd_cpp":
            args.steps = 20
        elif backend_type == "hidream":
            args.steps = 28
        else:
            args.steps = 4

    # Size: model-specific defaults (480x480 for ideogram4 on 6GB VRAM, 512x512 for bonsai)
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
            print(f"    Download with: huggingface-cli download WaveCut/HiDream-O1-Image-Dev-SDNQ-uint4-svd-r32-last8-odown-bf16 --local-dir {hidream_model_path}")
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
            print(f"  ✗ Edit image not found: {args.edit}")
            sys.exit(1)
        ref_image_paths = [str(args.edit.resolve())]
        print(f"  🖼️  Edit mode: {args.edit.name} → prompt as instruction")

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
    enhance_model = None

    if args.enhance_with:
        # --enhance-with <model> — use any llama-swap model
        enhance_model = args.enhance_with
        args.enhance = True  # implied
    elif args.enhance:
        enhance_model = model_info.get("enhance_model", "qwen3.5-4b")

    if args.enhance:
        enhance_type = model_info.get("enhance_type", "ideogram")
        if enhance_type == "vision":
            print(f"  ✨ Enhancing prompt via {enhance_model} (vision mode)...")
            enhanced_result, raw_response = enhance_vision_prompt(prompt, enhance_model)
        else:
            print(f"  ✨ Enhancing prompt via {enhance_model} (Ideogram JSON)...")
            enhanced_result, raw_response = enhance_prompt(prompt, enhance_model)
        if enhanced_result != prompt:
            enhanced_prompt = enhanced_result  # for metadata/debrief
            if enhance_type == "vision":
                print(f"     Expanded to English description ({len(enhanced_result)} chars)")
                print(f"     ─── Enhanced prompt ───")
                # Print wrapped text at 80 chars
                import textwrap
                for line in textwrap.wrap(enhanced_result, width=78):
                    print(f"     {line}")
                print(f"     ────────────────────────")
            else:
                print(f"     Expanded to JSON ({len(enhanced_result)} chars)")
                # ALWAYS print enhanced JSON — not just with --show-enhanced
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
                # Truncate very long responses
                display = raw_response[:500] + ("..." if len(raw_response) > 500 else "")
                print(f"     {display}")
                print(f"     ────────────────────")
        # For sd_cpp, use the enhanced JSON prompt; for vision models, use the expanded text
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
        if backend_type == "sd_cpp":
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Expected: ~80-100s (model load + offload + 20 denoising steps)")
        elif backend_type == "hidream":
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Expected: ~3-4min (model load + CPU offload + 28 denoising steps)")
        else:
            print(f"  ⏳ First run at {width}×{height}")
            print(f"     Cold start: ~30-60s (imports + model load + kernel JIT)")
            print(f"     Subsequent runs at this size will be faster (cached kernels)")
    print()

    # ── Phase 1.5: Evict LLMs after prompt enhancement ──
    # If we used an LLM for enhancement, evict it before loading the diffusion model
    if args.enhance and backend_type in ("sd_cpp", "hidream"):
        running = _llama_swap_running_models()
        if running:
            print(f"  🔄 Evicting LLM models after enhancement: {', '.join(running)}")
            evict_llm()
            print(f"     VRAM freed for image generation")
            print()

    # ── Evict LLMs before HiDream (needs ~4.5GB VRAM) ──
    if backend_type == "hidream" and not args.enhance:
        running = _llama_swap_running_models()
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


if __name__ == "__main__":
    main()