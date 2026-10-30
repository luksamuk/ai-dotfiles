#!/usr/bin/env python3
"""musica — enhance de legenda via frota llama-swap + params de console via d1.

Padrão idêntico ao enhance do h3/ltx/diffuse:
  - config: ~/.local/share/fleet/enhance-models.yaml, seção `musica:`
  - rewrite:  POST localhost:12434/v1/chat/completions (LLM reescreve SÓ a legenda)
  - decisão:  POST localhost:12434/v1/systemone (d1, N julgamentos noul binários)

O LLM recebe a letra como CONTEXTO SOMENTE-LEITURA — o output é só o caption.
BPM (só ace_step aceita): candidatos extraídos do caption enhanced, cada um
julgado num noul independente; score >= threshold vence, abaixo descarta
(mantém o default do motor). Param explícito na CLI nem chega aqui.

Saída: JSON único em stdout. Erros: stderr, exit != 0.
Rodar com PYTHONPATH limpo (o gateway vaza numpy 3.14 em venvs de projeto).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

FLEET_CFG = os.environ.get(
    "MUSICA_FLEET_CFG", os.path.expanduser("~/.local/share/fleet/enhance-models.yaml")
)
BASE = os.environ.get("MUSICA_SWAP_URL", "http://localhost:12434")
MAX_DECISION_CHARS = 600  # batch 512 do d1: payload tem que ser curto

SYSTEM_PROMPT = """\
You are a music-generation caption engineer for a local ACE-Step model.
Rewrite the user's rough music description into ONE dense structured-caption
paragraph in ENGLISH suitable as a text-to-music prompt.

Rules (non-negotiable):
- ONE paragraph, 3-6 sentences. No bullet lists, no markdown, no preamble.
- Cover: genre/subgenre and cultural origin, instrumentation (be concrete:
  instruments, drum pattern names, bass character), vocal details if vocals
  are implied, production/sonics (lo-fi, cassette grit, sidechain, mastering
  style), mood and groove evolution.
- Preserve the user's genre intent EXACTLY — never substitute a cousin genre.
  If they ask for funk carioca, it stays funk carioca (tamborzão), not phonk,
  not reggaeton, not US funk.
- If the description implies a tempo, you MAY state it as "NNN BPM" once.
- If lyrics are provided as reference, they are READ-ONLY context to calibrate
  genre, tempo and mood. You must NOT rewrite, translate, summarize or echo
  them in the output.
- Output ONLY the caption paragraph. No quotes around it."""


def read_fleet(tool: str) -> dict:
    try:
        import yaml
    except ImportError:  # yaml é dependência da casa; sem ele, config morta
        print("aviso: pyyaml ausente — usando defaults embutidos", file=sys.stderr)
        return {}
    try:
        with open(FLEET_CFG) as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        print(f"aviso: config de frota ausente: {FLEET_CFG}", file=sys.stderr)
        return {}
    return cfg.get(tool) or {}


def post(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def enhance(prompt: str, lyrics: str, model: str) -> str:
    user = f"Rough description:\n{prompt}"
    if lyrics:
        user += (
            f"\n\nLyrics (READ-ONLY reference — never rewrite or echo these):\n"
            f"{lyrics[:4000]}"
        )
    resp = post(
        "http://localhost:12434/v1/chat/completions",
        {
            "model": model,
            "temperature": 0.4,
            # sem max_tokens: sem teto de tokens (convenção da casa — handles
            # :think queimam tokens de reasoning antes do caption; diffuse não
            # limita e o h3 usa 65536). O guard de finish_reason fica por garantia.
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        },
        timeout=600,
    )
    choice = resp["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    if not text and choice.get("finish_reason") == "length":
        raise RuntimeError(
            "enhance cortado em max_tokens (reasoning comeu o orçamento) — tente um handle sem :think"
        )
    # higiene: LLMs teimosos às vezes cercam com ``` ou "caption:" — cortar
    text = re.sub(r"^```[a-z]*\n?|```\n?$", "", text, flags=re.M).strip()
    text = text.strip('"')
    # frase incompleta no fim (corte de tokens): recorta na última sentença
    if not text.endswith((".", "!", "?")):
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > 0:
            text = text[: last + 1]
    if not text:
        raise RuntimeError("enhance devolveu vazio")
    return text


def bpm_candidates(caption: str) -> list[int]:
    """BPMs citados no caption (números com 'bpm' por perto). Sem inventar."""
    found, seen = [], set()
    for m in re.finditer(
        r"(\d{2,3})\s*(?:bpm|beats per minute)", caption, re.I
    ):
        v = int(m.group(1))
        if 40 <= v <= 220 and v not in seen:
            seen.add(v)
            found.append(v)
    return found[:6]


def decide_bpm(caption: str, model: str, threshold: float) -> dict:
    cands = bpm_candidates(caption)
    if not cands:
        return {"bpm": None, "reason": "sem candidatos de bpm no caption"}
    short = caption[:MAX_DECISION_CHARS]
    questions = {
        f"bpm{v}": {
            "type": "noul",
            "instructions": (
                f"USE quando a descrição musical pedir exatamente um tempo de "
                f"{v} BPM. A descrição abaixo claramente pede esse tempo "
                f"({v} BPM)? Descrição: {short}"
            ),
        }
        for v in cands
    }
    try:
        resp = post(
            "http://localhost:12434/v1/systemone",
            {"model": model, "state": short, "questions": questions},
            timeout=120,
        )
    except (urllib.error.URLError, TimeoutError, KeyError) as e:
        return {"bpm": None, "reason": f"systemone indisponível: {e}"}

    answers = resp.get("answers", {})
    scored = []
    for qid, ans in answers.items():
        p = ans.get("noul")
        if p is None:
            continue
        try:
            scored.append((float(p), int(qid[3:])))
        except (ValueError, TypeError):
            continue
    if not scored:
        return {"bpm": None, "reason": "sem respostas tipadas"}
    best_p, best_v = max(scored)
    if best_p < threshold:
        return {
            "bpm": None,
            "reason": f"abaixo do corte (top {best_v} BPM p={best_p:.2f} < {threshold})",
            "scores": sorted([(v, round(p, 3)) for p, v in scored], reverse=True),
        }
    return {
        "bpm": best_v,
        "confidence": round(best_p, 3),
        "scores": sorted([(v, round(p, 3)) for p, v in scored], reverse=True),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="legenda crua")
    ap.add_argument("--lyrics", default="", help="contexto somente-leitura")
    ap.add_argument("--model", default="", help="override do handle de enhance")
    ap.add_argument("--no-params", action="store_true", help="pula decisão d1")
    a = ap.parse_args()

    cfg = read_fleet("musica")
    model = a.model or cfg.get("model") or ""
    d_model = cfg.get("decision_model", "d1-2b")
    threshold = float(cfg.get("decision_threshold", 0.5))
    if not model:
        print("erro: sem handle de enhance (fleet cfg sem secao musica?)", file=sys.stderr)
        return 1

    out: dict = {"model": model, "decision_model": d_model}
    out["caption"] = enhance(a.prompt, a.lyrics, model)
    if not a.no_params:
        out["params"] = decide_bpm(out["caption"], d_model, threshold)

    # decisão é best-effort: se ela morreu mas o enhance saiu, entrega o caption
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())