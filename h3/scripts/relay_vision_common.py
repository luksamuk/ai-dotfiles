#!/usr/bin/env python3
"""relay_vision_common.py — biblioteca de visão/auto-despacho compartilhada entre ltx e h3.

Funções extraídas/inspiradas no ltx_prompt_relay.py (motor LTX) para uso geral:
  - resolve_think_variant(model): preferir :think c/ validação no fleet
  - check_vision(model): features.vision / input_modalities / heurística de nome
  - image_to_data_uri(path)
  - describe_image_vlm(path, model, nsfw): fallback lfm2.5-vl-3b descreve
  - enhance_call(model, system, prompt, images=None, nsfw=False, temperature=0.7, max_tokens=2048, timeout=300):
      chamada one-shot: imagens inline se o modelo VÊ; senão VLM descreve e injeta como media context
      retorna (texto, via) onde via ∈ {'inline-vision','vlm-describing','text-only','passthrough-error'}

Uso nos scripts:
  sys.path.insert(0, '/home/alchemist/.local/bin')
  from relay_vision_common import enhance_call
"""
import json
import base64
import os
import sys
import urllib.request

LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")
VLM_FALLBACK = os.environ.get("RELAY_VLM_FALLBACK", "lfm2.5-vl-3b")


def resolve_think_variant(model: str) -> str:
    try:
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
        ids = {m.get("id", "") for m in data.get("data", [])}
        if f"{model}:think" in ids:
            return f"{model}:think"
    except Exception:
        pass
    return model


def check_vision(model: str):
    base_model = model.split(":")[0]
    try:
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
        for m in data.get("data", []):
            if m.get("id") == base_model:
                feat = ((m.get("meta") or {}).get("llamaswap") or {}).get("features") or {}
                if feat.get("vision") or feat.get("image"):
                    return True, "features.vision"
                arch = m.get("architecture") or {}
                if "image" in (arch.get("input_modalities") or []):
                    return True, "input_modalities"
        name = base_model.lower()
        return any(t in name for t in ("vl", "vision", "llava", "gemma4", "minicpm-v")), "name-heuristic"
    except Exception as e:
        print(f"[WARN] capability check falhou: {e}", file=sys.stderr)
        return False, "check-failed"


def image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


def describe_image_vlm(path: str, model: str, nsfw: bool) -> str:
    content = [
        {"type": "text", "text": "Describe this frame factually for video prompt use: subjects, bodies, poses, clothing, environment, lighting, camera angle. Complete precise anatomical language if requested context is explicit." if nsfw else "Describe this frame factually: subjects, poses, environment, lighting, camera angle."},
        {"type": "image_url", "image_url": {"url": image_to_data_uri(path)}},
    ]
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as resp:
        d = json.loads(resp.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def chat_once(model: str, system: str, user_content, temperature: float, max_tokens: int, timeout: int, think_ok: bool = True):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content, reasoning


def enhance_call(model: str, system: str, prompt: str, images=None, nsfw: bool = False,
                 temperature: float = 0.7, max_tokens: int = 2048, timeout: int = 300,
                 clean_fn=None):
    """Chamada enhance completa com despacho visual automático.
    clean_fn: função opcional de pós-processamento do texto (ex.: clean_output do relay ltx)."""
    model = resolve_think_variant(model)
    images = [p for p in (images or []) if p and os.path.exists(p)]
    has_vision, via = check_vision(model)

    if images and has_vision:
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": image_to_data_uri(p)}} for p in images
        ]
        via_mode = "inline-vision"
    elif images:
        descs = []
        for p in images:
            try:
                d = describe_image_vlm(p, VLM_FALLBACK, nsfw)
                descs.append(f"<Picture {len(descs)+1}>: {d}")
            except Exception as e:
                print(f"[WARN] VLM falhou em {p}: {e}", file=sys.stderr)
        media_ctx = ("\n\nImage descriptions (from vision model):\n" + "\n".join(descs)) if descs else ""
        system = system + media_ctx
        content = prompt
        via_mode = "vlm-described" if descs else "no-image-describe"
    else:
        content = prompt
        via_mode = "text-only"

    content_text, reasoning = chat_once(model, system, content, temperature, max_tokens, timeout)
    # LLMs :think às vezes entregam TODO o output no reasoning_content e deixam content vazio/válido-esperando
    if not content_text.strip() and reasoning.strip():
        content_text = reasoning  # o reasoning contém o texto (e a linha LORAS: se pedida)
    if clean_fn:
        content_text = clean_fn(content_text)
    return content_text, reasoning, via_mode, model


def evict_unload():
    import subprocess
    if os.environ.get("RELAY_EVICT", "1") != "0":
        try:
            import subprocess as sp
            sp.run(["llama-swap-cli", "unload"], timeout=30, capture_output=True)
        except FileNotFoundError:
            pass