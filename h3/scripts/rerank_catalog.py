#!/usr/bin/env python3
"""Pré-filtro ColBERT do catálogo NSFW H3.

Ranking: embeddings mean-pooled do LFM2.5-ColBERT-350M (CPU-only no llama-swap)
cos(prompt, desc_llm de cada LoRA) → top-N.

Ranks:
  default          → imprime CATÁLOGO REORDENADO (formato do build_catalog_prompt)
  --mode ids       → imprime apenas os IDs top-N, vírgula-separados (pra pipeline)
  --mode shadow    → imprime no stderr "[COLBERT-SHADOW] top ids=..." e no stdout o catálogo
                     ORIGINAL (sem ordem mexida) — para comparar antes de ativar.
  --mode both      → stdout = catálogo reordenado; stderr = linha shadow p/ log paralelo
"""
import argparse
import json
import re
import math
import os
import sys
import urllib.request

CATALOG = os.environ.get("NSFW_CATALOG_JSON") or os.path.expanduser("~/.local/share/h3/nsfw_catalog_h3.json")
DEFAULT_N = 10
COLBERT_MODEL = "lfm2.5-colbert-350m"
COLBERT_URL = "http://localhost:12434/v1/embeddings"


def load_catalog():
    cat = json.load(open(CATALOG))
    return cat["loras"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_all(texts):
    body = json.dumps({"model": COLBERT_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(COLBERT_URL, body, {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=90).read())
    out = [e["embedding"] for e in d["data"]]
    if len(out) != len(texts):
        raise RuntimeError(f"esperava {len(texts)} embeddings, veio {len(out)}")
    return out


SYN = {
    "punheta": ["handjob", "punheta", "hand job", "wank", "stroking", "bate"],
    "gozada": ["cumshot", "gozada", "gozar", "ejacula", "cum", "sêmen", "semen", "esguich", "facial", "esporra", "goza"],
    "boquete": ["blowjob", "boquete", "chupada", "mamada", "oral", "suck", "deepthroat"],
    "de_quatro": ["doggy", "quatro", "behind", "por trás"],
    "mission": ["missionary", "missionário"],
    "cunni": ["cunnilingus", "chupa ela", "lamb", "linga", "grega"],
    "peito": ["seios", "mamas", "tetas", "jiggle", "bounce", "peito"],
    "anal": ["anal", "rimjob", "ass licking"],
    "femorg": ["squirt", "masturba"],
    "bondage": ["amarr", "atad", "bondage", "tied"],
    "buceta": ["buceta", "pussy", "vulva", "vagina"],
    "peluda": ["hairy", "peluda", "pelo"],
    "dupla": ["duas", "2 girls", "dupla"],
    "pov": ["pov", "primeira pessoa", "fpov"],
    "pegging": ["pegging", "strap"],
    "tanga": ["thong", "tanga"],
    "cowgirl": ["cowgirl", "monta", "cavalg", "straddl", "riding"],
    "sentar": ["monta", "sitting on"],
}


def _kw_hits(prompt_lower, it):
    """quantas keywords (desc+trigger da lora) batem no prompt, com sinônimos pt"""
    text = (it.get("desc_llm") or "") + " " + (it.get("trigger_instr_arquivo") or "") + " " + it["safetensors"].lower()
    hits = 0
    for group in SYN.values():
        # o grupo conta se ALGUM sinônimo do grupo aparece na desc E (algo do grupo OU a keyword bruta no prompt)
        in_desc = any(w in text.lower() for w in group)
        if not in_desc:
            continue
        if any(w in prompt_lower for w in group):
            hits += 1
    return hits


def rank(prompt, loras, n):
    prompt_lower = prompt.lower()
    embs = embed_all([prompt] + [it.get("desc_llm") or it["safetensors"] for it in loras])
    q = embs[0]
    scored = []
    for it, e in zip(loras, embs[1:]):
        cos = cosine(q, e)
        kw = _kw_hits(prompt_lower, it)
        score = 0.4 * cos / 0.99 + 0.6 * min(kw, 3) / 3.0  # cos ~0.98→1; kw 0-3 normalizado
        scored.append((it["id"], score, cos, kw))
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = [(i, s) for i, s, _c, _k in scored[:n]]
    return top, scored


def fmt_block(ids, desc_by_id, cat_by_id, cat_by_orig, ordered_ids=None, scores=None):
    """Catálogo p/ o system do enhance.

    ordered_ids presente (modo ranked): seção top-N EM ORDEM DE RELEVÂNCIA com score
    + seção "demais disponíveis" com o RESTO do catálogo (nunca some do radar do LLM —
    o pré-filtro é priorização, não exclusão).
    Sem ordered_ids (modo completo/fail-open): lista única em id ascendente.
    """
    lines = ["Catalogo de LoRAs H3 disponiveis (IDs numericos) - escolha os adequados ao pedido:"]
    if ordered_ids is not None:
        lines.append("")
        lines.append("## Mais relevantes ao pedido (ordenados por relevância; score entre parenteses):")
        by_id = {it["id"]: it for it in cat_by_orig}
        for lid in ordered_ids:
            it = by_id.get(lid)
            if not it:
                continue
            cat = it.get("categoria", "?")
            mod = "[REF2VA] " if "ref2va" in (it.get("modalidade") or []) else ""
            desc = mod + (it.get("desc_llm") or it["safetensors"])
            sc = scores.get(lid)
            sc_str = f" ({sc:.3f})" if sc is not None else ""
            lines.append(f'{it["id"]}: [{cat}]{sc_str} {desc}')
        rest = [it for it in cat_by_orig if it["id"] not in set(ordered_ids)]
        if rest:
            lines.append("")
            lines.append("## Demais LoRAs disponiveis (menos relevantes ao pedido atual, mas use se fizer sentido):")
            for it in rest:
                cat = it.get("categoria", "?")
                mod = "[REF2VA] " if "ref2va" in (it.get("modalidade") or []) else ""
                desc = mod + (it.get("desc_llm") or it["safetensors"])
                lines.append(f'{it["id"]}: [{cat}] {desc}')
    else:
        ids_set = set(ids)
        for it in cat_by_orig:
            if it["id"] in ids_set:
                cat = it.get("categoria", "?")
                desc = it.get("desc_llm") or it["safetensors"]
                lines.append(f'{it["id"]}: [{cat}] {desc}')
    return chr(10).join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--mode", choices=["rank", "ids", "shadow", "both"], default="rank")
    args = ap.parse_args()

    loras = load_catalog()
    desc_by_id = {it["id"]: (it.get("desc_llm") or it["safetensors"]) for it in loras}
    cat_by_orig = sorted(loras, key=lambda x: x["id"])

    try:
        top, all_s = rank(args.prompt, loras, args.n)
    except Exception as e:
        # FALHA (llama-swap fora, colbert carregando...) = fail-open: imprime catálogo inteiro
        print(f"[rerank] colbert indisponível ({e}); imprimindo catálogo completo", file=sys.stderr)
        if args.mode == "ids":
            print(",".join(str(it["id"]) for it in cat_by_orig))
        else:
            print(fmt_block([it["id"] for it in cat_by_orig], desc_by_id, None, cat_by_orig))
        return 0

    top_ids = [i for i, _ in top]

    # shadow/both: linha de auditoria no stderr (o h3 redireciona stderr pro log)
    shadow_line = "[COLBERT-SHADOW] prompt=%r top=%s all_scores=%s" % (
        args.prompt[:60],
        ",".join(map(str, top_ids)),
        " ".join(f"{i}:{s:.3f}" for i, s, _c, _k in all_s),
    )
    if args.mode in ("shadow", "both"):
        print(shadow_line, file=sys.stderr)

    if args.mode == "ids":
        print(",".join(map(str, top_ids)))
    elif args.mode == "shadow":
        print(fmt_block([it["id"] for it in cat_by_orig], desc_by_id, None, cat_by_orig))
    else:  # rank | both — ranked: top-N em ordem de relevância + os demais no fim (priorização, não exclusão)
        scores_map = {i: s for i, s, _c, _k in all_s}
        ordered = [i for i, _s in top]
        print(fmt_block(top_ids, desc_by_id, None, cat_by_orig, ordered_ids=ordered, scores=scores_map))
    return 0


if __name__ == "__main__":
    sys.exit(main())
