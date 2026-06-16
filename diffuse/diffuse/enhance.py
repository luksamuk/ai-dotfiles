"""LLM-based prompt enhancement — Ideogram JSON, vision, and edit modes."""
from __future__ import annotations

import json
import logging
import re
import time

from diffuse.paths import LLAMA_SWAP_URL
from diffuse.prompts import (
    get_ideogram_enhance_prompt,
    get_vision_enhance_prompt,
    get_vision_analysis_prompt,
    get_edit_enhance_prompt,
    get_edit_vision_prompt,
)

log = logging.getLogger("diffuse")


# ── JSON extraction ────────────────────────────────────────────────────────
def _extract_json(text: str) -> str | None:
    """Extract the first valid JSON object from text.

    Handles: raw JSON, ```json``` code blocks, ``` code blocks,
    text before/after JSON, and nested braces correctly.
    """
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


# ── Ideogram JSON enhancement ───────────────────────────────────────────────
def enhance_prompt(prompt: str, model: str) -> tuple:
    """Use an LLM via llama-swap to expand a simple prompt into Ideogram 4 JSON.

    Returns (enhanced_prompt, raw_response).
    On failure, enhanced_prompt is the original prompt and raw_response contains the LLM output.
    """
    import urllib.request
    import urllib.error

    system = get_ideogram_enhance_prompt()

    log.info("Enhancing prompt via %s", model)
    t0 = time.perf_counter()

    # max_tokens must be generous for thinking models — reasoning tokens
    # consume most of the budget before the JSON answer is emitted.
    # qwen3.6-35b-a3b:think uses ~15-25k tokens on reasoning alone.
    # 16384 is the minimum that works reliably; 8192 causes empty content
    # because the model exhausts the budget during the thinking phase.
    max_tokens_val = 16384

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens_val,
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


# ── Vision (natural-language) enhancement ──────────────────────────────────
def enhance_vision_prompt(prompt: str, model: str) -> tuple:
    """Use an LLM via llama-swap to expand a prompt for vision models (HiDream, Bonsai).

    Unlike enhance_prompt (which returns Ideogram JSON), this returns a natural-language
    English paragraph that describes characters physically, translates non-English prompts,
    and specifies art style, lighting, and composition.

    Returns (enhanced_prompt, raw_response).
    On failure, enhanced_prompt is the original prompt.
    """
    import urllib.request
    import urllib.error

    system = get_vision_enhance_prompt()

    log.info("Vision-enhancing prompt via %s", model)
    t0 = time.perf_counter()

    # Same reasoning as ideogram enhance: thinking models need more tokens
    max_tokens_val = 16384

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens_val,
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


# ── Vision capability check ────────────────────────────────────────────────
def _check_model_vision(model: str) -> bool:
    """Check if a llama-swap model supports vision (image input) via the API.

    Strips :think/:code suffixes before matching, since the /v1/models endpoint
    lists base model names (e.g. 'gemma4-12b') without suffixes.
    """
    import urllib.request
    import urllib.error
    base_model = model.split(":")[0]
    try:
        req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=None) as resp:
            data = json.loads(resp.read())
            for m in data.get("data", []):
                if m.get("id") == base_model:
                    features = m.get("meta", {}).get("llamaswap", {}).get("features", {})
                    return features.get("image", False) or features.get("vision", False)
    except Exception as e:
        log.warning("Could not check model vision capability: %s", e)
    return False


# ── Image analysis ─────────────────────────────────────────────────────────
def analyze_image(image_path: str, model: str, user_prompt: str) -> str:
    """Use a vision model to analyze a reference image for editing.

    Sends the image plus the user's edit instruction to a vision model,
    which returns a detailed visual description of what's in the image
    and what elements matter for the edit.

    Returns the visual description string, or empty string on failure.
    """
    import base64
    import urllib.request
    import urllib.error
    from pathlib import Path as _Path

    log.info("Analyzing reference image via %s", model)

    # Read and base64-encode the image
    img_data = _Path(image_path).read_bytes()
    b64 = base64.b64encode(img_data).decode("ascii")

    # Detect mime type from extension
    ext = _Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}
    mime = mime_map.get(ext, "image/jpeg")

    # Build the user message with image + edit instruction context
    user_content = "I want to edit this image. My edit instruction: " + user_prompt + "\n\nDescribe this image in detail, focusing on elements relevant to the edit."
    system = get_vision_analysis_prompt()

    def _build_payload():
        return json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + b64}},
                    {"type": "text", "text": user_content},
                ]},
            ],
            "temperature": 0.4,
            "max_tokens": 2048,
        }).encode("utf-8")

    t0 = time.perf_counter()
    max_retries = 12
    analysis = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{LLAMA_SWAP_URL}/v1/chat/completions",
                data=_build_payload(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=None) as resp:
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                analysis = msg.get("content", "").strip()
                reasoning = msg.get("reasoning_content", "")
                if not analysis and reasoning:
                    log.warning("Vision model returned empty content with %d chars of reasoning", len(reasoning))
                    return ""
                break
        except urllib.error.HTTPError as e:
            if e.code == 400 and attempt < max_retries - 1:
                log.info("Model loading (attempt %d/%d), retrying in 10s...", attempt + 1, max_retries)
                time.sleep(10)
                continue
            log.error("Image analysis failed: %s", e)
            return ""
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            log.error("Image analysis failed: %s", e)
            return ""

    elapsed = time.perf_counter() - t0
    log.info("Image analysis completed in %.1fs", elapsed)
    return analysis


# ── Edit enhancement (text-only) ──────────────────────────────────────────
def enhance_edit_prompt(image_description: str, user_prompt: str, model: str) -> tuple:
    """Combine a visual description with the user's edit instruction into a
    refined edit prompt for HiDream.

    Returns (enhanced_prompt, raw_response).
    On failure, enhanced_prompt is the original prompt and raw_response contains the LLM output.
    """
    import urllib.request
    import urllib.error

    system = get_edit_enhance_prompt()
    user_msg = "REFERENCE IMAGE DESCRIPTION:\n" + image_description + "\n\nUSER'S EDIT INSTRUCTION:\n" + user_prompt + "\n\nWrite the refined edit prompt:"

    log.info("Enhancing edit prompt via %s", model)
    t0 = time.perf_counter()

    # Same reasoning as ideogram enhance: thinking models need more tokens
    max_tokens_val = 16384

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens_val,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLAMA_SWAP_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    max_retries = 12
    enhanced = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                enhanced = msg.get("content", "").strip()
                reasoning = msg.get("reasoning_content", "")
                if not enhanced and reasoning:
                    log.warning("Edit-enhancement model returned empty content with %d chars of reasoning", len(reasoning))
                    return user_prompt, reasoning
                break
        except urllib.error.HTTPError as e:
            if e.code == 400 and attempt < max_retries - 1:
                log.info("Model loading (attempt %d/%d), retrying in 10s...", attempt + 1, max_retries)
                time.sleep(10)
                continue
            log.error("Edit-enhancement failed: %s — using raw prompt", e)
            return user_prompt, str(e)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            log.error("Edit-enhancement failed: %s — using raw prompt", e)
            return user_prompt, str(e)

    elapsed = time.perf_counter() - t0
    log.info("Edit-enhancement completed in %.1fs", elapsed)

    if not enhanced:
        log.warning("Empty edit-enhancement response — using raw prompt")
        return user_prompt, enhanced

    return enhanced, enhanced


# ── One-shot vision + edit ──────────────────────────────────────────────────
def analyze_and_enhance_edit(image_path: str, user_prompt: str, model: str) -> tuple:
    """One-shot vision + edit enhancement: send image + instruction to a
    vision-capable model and get a refined edit prompt back.

    Used when the enhance model already has vision capability (e.g. gemma4-12b),
    avoiding two separate LLM calls (analyze then enhance).

    Returns (enhanced_prompt, raw_response).
    On failure, enhanced_prompt is the original prompt and raw_response contains
    the LLM output or error string.
    """
    import base64
    import urllib.request
    import urllib.error
    from pathlib import Path as _Path

    log.info("One-shot vision+edit enhancement via %s", model)

    # Read and base64-encode the image
    img_data = _Path(image_path).read_bytes()
    b64 = base64.b64encode(img_data).decode("ascii")

    ext = _Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}
    mime = mime_map.get(ext, "image/jpeg")

    system = get_edit_vision_prompt()
    user_text = (
        "I want to edit this image. My edit instruction: " + user_prompt
        + "\n\nLook at the image carefully, describe the relevant elements you see, "
        "and write a refined edit instruction that combines what you see with what I want changed."
    )

    def _build_payload():
        return json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + b64}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            "temperature": 0.5,
            "max_tokens": 16384,
        }).encode("utf-8")

    t0 = time.perf_counter()
    max_retries = 12

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{LLAMA_SWAP_URL}/v1/chat/completions",
                data=_build_payload(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=None) as resp:
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                enhanced = msg.get("content", "").strip()
                reasoning = msg.get("reasoning_content", "")
                if not enhanced and reasoning:
                    log.warning(
                        "One-shot vision+edit returned empty content with %d chars of reasoning",
                        len(reasoning),
                    )
                    return user_prompt, reasoning
                break
        except urllib.request.HTTPError as e:
            if e.code == 400 and attempt < max_retries - 1:
                log.info("Model loading (attempt %d/%d), retrying in 10s...", attempt + 1, max_retries)
                time.sleep(10)
                continue
            log.error("One-shot vision+edit failed: %s", e)
            return user_prompt, str(e)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            log.error("One-shot vision+edit failed: %s", e)
            return user_prompt, str(e)

    elapsed = time.perf_counter() - t0
    log.info("One-shot vision+edit completed in %.1fs", elapsed)

    if not enhanced:
        log.warning("Empty one-shot vision+edit response — using raw prompt")
        return user_prompt, enhanced

    return enhanced, enhanced