"""LLM integration — llama-swap coordination for prompt enhancement and VRAM eviction."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

from diffuse.paths import LLAMA_SWAP_URL

log = logging.getLogger("diffuse.llm")

# ── llama-swap CLI path ─────────────────────────────────────────────────────
LLAMA_SWAP_CLI = os.environ.get(
    "LLAMA_SWAP_CLI",
    os.path.expanduser("~/git/ai-dotfiles/llama-swap/llama-swap-cli"),
)


# ── LLM eviction (llama-swap coordination) ─────────────────────────────────
def llama_swap_running_models() -> list[str]:
    """Query llama-swap /running endpoint. Returns list of model IDs or empty list."""
    try:
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/running", timeout=3) as resp:
            data = json.loads(resp.read())
            return [m.get("model", m.get("id", "?")) for m in data.get("running", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def wait_for_model_ready(model: str, timeout: int = 120) -> bool:
    """Poll llama-swap /running until the requested model is in 'ready' state.

    llama-swap starts the model on first request but the health check takes
    several seconds. Without waiting, the first request hits the backend
    before it's ready and gets a 400.

    Returns True if model is ready, False if timeout exceeded.
    """
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
    running = llama_swap_running_models()
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
    for _ in range(10):
        time.sleep(0.5)
        if not llama_swap_running_models():
            log.info("LLM models evicted — VRAM free for diffusion")
            return True

    log.warning("Could not confirm LLM eviction — proceeding anyway")
    return True