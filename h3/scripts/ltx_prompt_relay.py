#!/usr/bin/env python3
# ltx_prompt_relay.py — relay de prompt pro LTX com VISION automática
# Copia o pattern do h3 (models/minimax_h3/prompt_enhancer.py vars) mas com
# capability detection nativa do llama-swap:
#   1. O modelo de enhance é checado em /v1/models (features.vision ou arch input_modalities)
#   2. SE tem vision → imagens (--image-start / --image-end / --ref-img) vão INLINE
#      (content multimodal, base64 data-uri) no MESMO pedido do enhance — one-shot
#   3. SE NÃO tem vision → LFM2.5-VL descreve cada imagem (vision describe), e a
#      descrição entra como media context no pedido (o enhance lê como notas internas)
# Nunca passa cegamente: se o modelo não suporta, o fallback é AUTOMÁTICO.
import json
import base64
import os
import sys
import subprocess
import urllib.request
import urllib.error

LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")
VLM_FALLBACK = os.environ.get("LTX_VLM_FALLBACK", "lfm2.5-vl-3b")


def resolve_think_variant(model: str) -> str:
    """Preferir variante :think quando existir no fleet (enable_thinking true + top_p 0.95).
    O base do heretic roda com thinking DESLIGADO via setParamsByID (config llama-swap) —
    o sufixo :think é o que ativa o COT pra valer, não o contrário."""
    try:
        with urllib.request.urlopen(f"{LLAMA_SWAP_URL}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
        ids = {m.get("id", "") for m in data.get("data", [])}
        if f"{model}:think" in ids:
            return f"{model}:think"
    except Exception:
        pass
    return model


def check_vision(model: str) -> tuple[bool, str]:
    """True se o modelo aceita image input. Testa features.vision e input_modalities."""
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
        # modelo não listado no llama-swap (ex.: ollama:) → heurística de nome
        name = base_model.lower()
        return any(t in name for t in ("vl", "vision", "llava", "gemma4", "minicpm-v", "ornith")), "name-heuristic"
    except Exception as e:
        print(f"[WARN] capability check falhou: {e}", file=sys.stderr)
        # conservador: assume sem vision, cai no fallback VLM
        return False, "check-failed"


def image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


def describe_image_vlm(path: str, model: str, nsfw: bool) -> str:
    """Fase 2 do fallback: VLM descreve a imagem em texto pro enhance ler."""
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


def clean_output(text: str) -> str:
    """Proibir multi-frame e fences: parágrafo único contínuo (doutrina LTX distilled)."""
    import re
    # Remover code fences
    text = re.sub(r"```[a-z]*\n?", "", text)
    # Remover linhas "Frame N:" reescrevendo como texto contínuo
    if re.search(r"Frame\s*\d+\s*:", text, re.I):
        parts = re.split(r"Frame\s*\d+\s*:", text)
        body = " ".join(p.strip() for p in parts if p.strip() and "Here is" not in p[:30] and "prompt for" not in p[:60].lower())
        text = body
    # Remover headers markdown e comentários do LLM
    text = re.sub(r"\*\*Prompt:\*\*", "", text)
    text = re.sub(r"^This prompt maintains.*$", "", text, flags=re.M)
    text = re.sub(r"^Here is.*$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True, help="system prompt (ex.: LTX_I2AV_SYSTEM_PROMPT)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", action="append", default=[], help="imagens de contexto (repeatable)")
    ap.add_argument("--nsfw", action="store_true")
    ap.add_argument("--show-enhanced", action="store_true")
    args = ap.parse_args()
    args.model = resolve_think_variant(args.model)

    has_vision, via = check_vision(args.model)
    images = [p for p in args.image if os.path.exists(p)]

    if images and has_vision:
        # ONE-SHOT: imagens inline no próprio pedido do enhance
        print(f"[vision] {args.model} tem imagem ({via}) — envio INLINE ({len(images)} img)", file=sys.stderr)
        content = [{"type": "text", "text": args.prompt}]
        for p in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(p)}})
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        raw = True
    elif images:
        # FALLBACK AUTOMÁTICO: VLM descreve → descrição vira media context
        print(f"[vision] {args.model} SEM imagem ({via}) — fallback VLM ({VLM_FALLBACK})", file=sys.stderr)
        descs = []
        for p in images:
            try:
                d = describe_image_vlm(p, VLM_FALLBACK, args.nsfw)
                descs.append(f"<Picture {len(descs)+1}>: {d}")
            except Exception as e:
                print(f"[WARN] VLM falhou em {p}: {e}", file=sys.stderr)
        media_ctx = ""
        if descs:
            media_ctx = "\n\nImage descriptions (from vision model):\n" + "\n".join(descs)
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system + media_ctx},
                {"role": "user", "content": args.prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        raw = False
    else:
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        raw = False

    req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read())
        content_text = clean_output((d["choices"][0]["message"].get("content") or "").strip())
        if args.show_enhanced:
            print("── Enhanced prompt ──", file=sys.stderr)
            print(content_text, file=sys.stderr)
        print(content_text)
    except Exception as e:
        print(f"[ERROR] enhance falhou: {e} — passthrough do prompt original", file=sys.stderr)
        print(args.prompt)
    finally:
        # Evict automático: libera a VRAM do LLM antes da geração (mesma mecânica do diffuse)
        if os.environ.get("LTX_EVICT", "1") != "0":
            try:
                subprocess.run(["llama-swap-cli", "unload"], timeout=30,
                               capture_output=True)
            except FileNotFoundError:
                pass  # CLI não instalado — o swap manual do llama-swap resolve


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()