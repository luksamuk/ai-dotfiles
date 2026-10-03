#!/usr/bin/env python3
"""nsfw_lora_resolve.py — resolutor central de LoRAs NSFW H3 (usado pelo h3).

Modos:
  --ids "1,2,3"    → resolve IDs do catálogo (force; ignora LLM)
  --out <arquivo>  → lê output do enhance, extrai linha LORAS: <ids|none>, valida

Exit codes: 0 ok; 2 = linha LORAS ausente; 3 = LoRA pedida sem arquivo local.
stdout: JSON único:
  {"source": "force|llm|llm-none", "names": [...], "mults": [...], "triggers": [...],
   "unknown": [...], "fallback_ids": [...], "fallback_names": [...]}
Regar:
  - IDs desconhecidos/sem arquivo são DESCARTADOS (com lista em unknown)
  - "LORAS: none" no LLM = escolha vazia consciente (não ativa fallback)
  - trigger só entra se for hard-token (hmotion, grindtime, bl0w_job, cmst...)
"""
import json, os, re, sys, argparse

CATALOG_LORA = os.environ.get("NSFW_CATALOG_JSON") or os.path.expanduser("~/.local/share/h3/nsfw_catalog_h3.json")
LORA_DIR = os.environ.get("N3_DIR", os.path.expanduser("~/git/Wan2GP/loras/minimax_h3"))

cat = json.load(open(CATALOG_LORA))
by_id = {e["id"]: e for e in cat["loras"]}

ap = argparse.ArgumentParser()
ap.add_argument("--ids", help='IDs sep. por vírgula (force): "1,2,9"')
ap.add_argument("--out", help="arquivo contendo o output do enhance")
args = ap.parse_args()

if args.ids:
    source = "force"
    tokens = [t.strip() for t in re.split(r"[,\s]+", args.ids) if t.strip()]
    ids = [int(t) for t in tokens if t.isdigit() and int(t) in by_id]
else:
    text = open(args.out).read() if args.out else ""
    m = re.search(r'LORAS:\s*([^\n]+)', text, re.IGNORECASE)
    if not m:
        print(json.dumps({"error": "no_loras_line", "fallback_ids": cat.get("fallback_pilha", [])}))
        sys.exit(2)
    raw = m.group(1).strip()
    if raw.lower() == "none":
        ids, source = [], "llm-none"
    else:
        tokens = [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]
        ids = [int(t) for t in tokens if t.isdigit() and int(t) in by_id]
        source = "llm"

names, mults, triggers, unknown = [], [], [], []
for i in ids:
    e = by_id[i]
    st = e.get("safetensors")
    if not st:
        unknown.append(i); continue
    stem = st[:-len(".safetensors")]
    if not os.path.exists(os.path.join(LORA_DIR, st)):
        unknown.append(i); continue
    names.append(stem)
    mults.append(round(float(e.get("strength_autor") or 1.0), 2))
    trig = (e.get("trigger_instr_arquivo") or "").strip()
    if trig and "," in trig:
        trig = trig.split(",")[0].strip()
    hard = bool(re.search(r'\b(hmotion|grindtime|hmpussy|inniepussy|hmcumshot|penislora|bl0w_j0b|cmst|cumsh0t|sensual_fingering|rjvideo|squirting|hmasturbation|oralmove1|oraldetail1|deepthroat|blowjob_wearing_thong|0\.5-1str|use10erosmax)\b', (trig or "").lower()))
    triggers.append((trig[:64] if hard else ""))

fb_names = []
for fid in cat.get("fallback_pilha", []):
    fe = by_id.get(fid)
    if fe and fe.get("safetensors") and os.path.exists(os.path.join(LORA_DIR, fe["safetensors"])):
        fb_names.append(fe["safetensors"][:-len(".safetensors")])

import sys as _s
if os.environ.get("NSFW_RESOLVE_RAW") == "1":
    # modo raw p/ bash: names por linha, depois mults por linha, depois triggers por linha
    for _n in names: _s.stdout.write(_n + "\n")
    _s.stdout.write("---\n")
    for _m in mults: _s.stdout.write(str(_m) + "\n")
    _s.stdout.write("---\n")
    for _t in triggers: _s.stdout.write(("" if not _t else _t) + "\n")
else:
    _s.stdout.write(json.dumps({
        "source": source, "names": names, "mults": mults, "triggers": triggers,
        "unknown": unknown, "fallback_ids": cat.get("fallback_pilha", []),
        "fallback_names": fb_names,
    }))