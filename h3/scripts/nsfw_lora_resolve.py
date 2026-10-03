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

import sys as _s

names, mults, triggers, unknown = [], [], [], []
# TRAVA DE MODALIDADE (03/out): variantes FL2VA/REF2VA são excludentes — o run é de
# UM modality só (REF2VA_MODE=true no wrapper quando há refs). Env NSFW_RUN_MODALITY
# = "fl2va"|"ref2va" (exportado pelo h3); LoRA de outra modalidade é DROPPADA (não
# erro: degrade silenciosa com aviso no stderr) — cobre LLM e --force-ids igualmente.
RUN_MODALITY = (os.environ.get("NSFW_RUN_MODALITY") or "").strip().lower()
# Mapa de família: i2v/t2v são ramos do FL2VA (um run fl2va aceita os três tags;
# um run ref2va aceita só ref2va) — MESMO mapa do rerank_catalog.py
FAMILY = {"fl2va": {"fl2va", "t2v", "i2v"}, "ref2va": {"ref2va"}}
def _modality_ok(e):
    if RUN_MODALITY not in FAMILY:
        return True  # env ausente = rota fora do wrapper (testes) — sem filtro
    mods = e.get("modalidade") or []
    if not mods:
        return True   # LoRA universal (anatomia/cena, sem variantes)
    return bool(set(mods) & FAMILY[RUN_MODALITY])
dropped_modality = []
for i in ids:
    e = by_id[i]
    if not _modality_ok(e):
        dropped_modality.append(i)
        _s.stderr.write(f"[resolve] LoRA {i} ({e.get('safetensors')}) é de outra modalidade "
                        f"(run={RUN_MODALITY}, lora={e.get('modalidade')}) — dropada\n")
        continue
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
    hard = bool(re.search(r'\b(hmotion|grindtime|hmpussy|inniepussy|hmcumshot|penislora|bl0w_j0b|cmst|cumsh0t|sensual_fingering|rjvideo|squirting|hmasturbation|oralmove1|oraldetail1|deepthroat|blowjob_wearing_thong|0\.5-1str|use10erosmax)\b|she (?:is|has) (?:a )?fl\w+ chestr?', (trig or "").lower()))
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