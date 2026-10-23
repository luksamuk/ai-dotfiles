"""Seleção unificada de LoRAs para o MiniMax H3 — porte do módulo do diffuse.

Diferenças em relação ao original (deliberadas, todas marcadas com [H3]):
  [H3] SHARE = ~/.local/share/h3   (catálogo loras_minimax_h3.json)
  [H3] env prefix H3_ em vez de DIFFUSE_
  [H3] seção `h3` do enhance-models.yaml
  [H3] o rerank da casa aqui se chama rerank_catalog.py (sem sufixo de root)
  [H3] `modality` (fl2va|ref2va): LoRA de modalidade roda só na sua modalidade

Modos (SelectorCfg.selector):
  - "llm"      (alias "reranker"): rerank ColBERT pré-ordena e o LLM escolhe via "LORAS:".
  - "decision": N perguntas `noul` independentes no /v1/systemone (d1-3b), corte + teto.

Fail-open por doutrina: qualquer falha devolve ([], []) e a geração roda base.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SHARE = Path.home() / ".local" / "share" / "h3"                      # [H3]
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")

_DISTILL = re.compile(r"dmd|turbo_distill|lightx2v|_acc_|acc_8step", re.I)


@dataclass
class SelectorCfg:
    selector: str = "llm"
    decision_model: str = "d1-3b"
    decision_threshold: float = 0.5
    decision_max: int = 4


# ── config: CLI > env > yaml > default ──────────────────────────────────────

def load_selector_cfg(args) -> SelectorCfg:
    def cli(k):
        return getattr(args, k, None)

    def env(k, cast):
        v = os.environ.get(f"H3_{k.upper()}")                          # [H3]
        if v in (None, ""):
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    def yml(k, cast):
        try:
            import yaml
            cfg = yaml.safe_load(open(Path.home() / ".local" / "share" / "fleet"
                                      / "enhance-models.yaml")) or {}
            v = (cfg.get("h3") or {}).get(k)                           # [H3]
            if v in (None, ""):
                return None
            try:
                return cast(v)
            except (TypeError, ValueError):
                return None
        except Exception:
            return None

    sel = cli("lora_selector") or env("lora_selector", str) or yml("lora_selector", str) or "llm"
    if sel == "reranker":
        sel = "llm"
    if sel not in ("llm", "decision"):
        print(f"  [warn] lora_selector invalido {sel!r} — usando 'llm'")
        sel = "llm"

    model = cli("decision_model") or env("decision_model", str) or yml("decision_model", str) or "d1-3b"
    th = cli("decision_threshold")
    if th is None:
        th = env("decision_threshold", float)
    if th is None:
        th = yml("decision_threshold", float)
    th = 0.5 if th is None else float(th)

    mx = cli("decision_max")
    if mx is None:
        mx = env("decision_max", int)
    if mx is None:
        mx = yml("decision_max", int)
    mx = 4 if mx is None else int(mx)

    return SelectorCfg(selector=sel, decision_model=model,
                       decision_threshold=th, decision_max=max(1, min(mx, 8)))


# ── catálogo ───────────────────────────────────────────────────────────────

_CACHE: dict = {}


def load_catalog(root: str):
    """Entradas de loras_<root>.json (memoizado). None se não existir."""
    if root in _CACHE:
        return _CACHE[root]
    for alias in (root, root.split("_")[0]):
        p = SHARE / f"loras_{alias}.json"
        if p.exists():
            try:
                _CACHE[root] = json.loads(p.read_text()).get("loras", [])
            except Exception as e:
                print(f"  [warn] {p.name} ilegível ({e}) — sem seleção")
                _CACHE[root] = None
            return _CACHE[root]
    _CACHE[root] = None
    return None


def eligible(entries, nsfw: bool, variant: str = "*", modality: str = ""):
    """Filtra por conteúdo, variante, modalidade e destilados."""
    out = []
    for e in entries or []:
        if _DISTILL.search(e.get("safetensors", "")):
            continue
        if e.get("nsfw") and not nsfw:
            continue
        comp = e.get("compat", ["*"])
        if not comp:
            continue
        if "*" not in comp and variant not in comp:
            continue
        # [H3] modalidade: campo opcional; ausente = serve nas duas
        mod = e.get("modality") or ""
        if mod and modality and mod != modality:
            continue
        out.append(e)
    return out


# ── decision: N perguntas noul no /v1/systemone ─────────────────────────────

def _decision_select(prompt: str, entries, cfg: SelectorCfg):
    questions = {}
    for e in entries:
        nome = e["safetensors"].replace(".safetensors", "")
        desc = (e.get("desc_llm") or "").replace("\n", " ")[:400]  # 400: clausula de escopo no fim tem que caber (era 160 e cortava)
        questions[f"lora_{e['id']}"] = {
            "type": "noul",
            "instructions": f"This LoRA applies and should be used for the request: "
                            f"[{e.get('categoria','')}] {nome}: {desc}",
        }
    payload = json.dumps({
        "model": cfg.decision_model,
        "state": {"request": prompt},
        "questions": questions,
    }).encode("utf-8")
    req = urllib.request.Request(f"{LLAMA_SWAP_URL}/v1/systemone", data=payload,
                                 headers={"Content-Type": "application/json"})
    # retry: o d1 pode estar sendo carregado ou evicted durante um render na GPU.
    # Sem isso, um 500 transitorio derrubava a selecao e o run seguia SEM LoRA nenhuma
    # (silencioso: num NSFW perde a identidade inteira).
    answers, ultimo_erro = None, None
    for tentativa, espera in enumerate((0, 8, 25, 60), 1):
        if espera:
            time.sleep(espera)
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                answers = (json.loads(r.read()) or {}).get("answers") or {}
            if tentativa > 1:
                print(f"  [seletor] decision respondeu na tentativa {tentativa}")
            break
        except Exception as e:
            ultimo_erro = e
            print(f"  [seletor] tentativa {tentativa} falhou: {e}")
    if answers is None:
        print(f"  [warn] seletor decision indisponivel apos 4 tentativas ({ultimo_erro}); "
              f"rodando base SEM LoRA — selecione manualmente com --lora se for NSFW")
        return [], None
    scores = {}
    for k, v in answers.items():
        p = v.get("noul") if isinstance(v, dict) else None
        scores[str(k).replace("lora_", "")] = float(p) if isinstance(p, (int, float)) else 0.0
    rank = sorted(scores.items(), key=lambda x: -x[1])
    # [H3] no maximo UM LoRA por conceito: versoes da mesma familia (VBVR Pro/v1,
    # better_human_motion/motion_combat) competem entre si, nao ocupam duas vagas
    # [H3] EIXO DE PAPEL (22/out): cada LoRA ocupa no maximo UM papel. Antes o dedupe era
    # por "conceito", e como hmmotion/grindtime/AIO tinham conceitos distintos, dois priors
    # de movimento (ou de ato) entravam juntos e brigavam no denoise.
    # GATE DE ATO: LoRA de ato so entra se o pedido descreve um ato. Sem gate, cena solo
    # de nudez carregava LoRA de ato e o modelo inventava parceiro (bunda masculina, 22/out).
    papel_por_id = {str(e["id"]): _papel(e) for e in entries}
    conceito_por_id = {str(e["id"]): str(e.get("conceito") or "") for e in entries}
    tem_ato = bool(ACT_RE.search(prompt))
    sel, ocupados = [], set()
    # PIN: em cena com ato, o AIO entra SEMPRE como o ato principal (decisao do usuario,
    # 22/out), independente da nota — ele e a base do ato e pontua baixo sozinho.
    if tem_ato:
        _pin = [i for i, _ in rank if conceito_por_id.get(str(i)) == "ato-aio"]
        if _pin:
            sel.append(_pin[0]); ocupados.add("ato")
    for i, p in rank:
        if p < cfg.decision_threshold or len(sel) >= cfg.decision_max:
            break
        papel = papel_por_id.get(str(i), "outro")
        if papel == "ato" and not tem_ato:
            continue
        if papel in ocupados:
            continue
        ocupados.add(papel)
        sel.append(i)
    return sel, rank


# ── papel de cada LoRA e gate de ato ────────────────────────────────────────

ACT_RE = re.compile(
    r"\b(fuck|fucked|fucking|penetrat\w*|thrust\w*|sex|blowjob|handjob|hj|bj|"
    r"riding|rides|mount\w*|cum\w*|ejaculat\w*|oral|masturbat\w*|fingering|"
    r"doggystyle|cowgirl|missionary|gangbang|org\w*|grind\w*|girando|gyrat\w*|"
    r"transa\w*|fod\w*|penetr\w*|chup\w*|punhet\w*|sentando|goz\w*)\b",
    re.I)


def _papel(e):
    """Papel funcional: identidade | ato | movimento | estilo | camera | corpo | fala | outro."""
    txt = (str(e.get("categoria") or "") + " " + str(e.get("conceito") or "")).lower()
    for chave, papel in (
        ("identidade", "identidade"),
        # movimento ANTES de ato: "movimento-ato" contem "ato" e cairia no papel errado
        ("movimento", "movimento"), ("motion", "movimento"), ("grind", "movimento"),
        ("ato", "ato"), ("cumshot", "ato"),
        ("camera", "camera"), ("zoom", "camera"),
        ("corpo", "corpo"), ("conceito", "corpo"),
        ("fala", "fala"), ("speech", "fala"),
        ("estilo", "estilo"), ("style", "estilo"), ("anime", "estilo"),
        ("olhos", "estilo"), ("skin", "estilo"),
        ("ref2v", "referencia"), ("vbvr", "referencia"),
    ):
        if chave in txt:
            return papel
    return "outro"


# ── resolve ids -> tags ─────────────────────────────────────────────────────

def resolve_ids(entries, ids):
    by_id = {str(e["id"]): e for e in entries or []}
    out = []
    for i in ids:
        e = by_id.get(str(i))
        if not e:
            continue
        out.append((e["safetensors"], float(e.get("strength_autor") or 1.0),
                    [t.strip() for t in (e.get("trigger_instr_arquivo") or "").split(",") if t.strip()]))
    return out


def build_prompt(prompt: str, resolved):
    if not resolved:
        return prompt
    tags = " ".join(f"<lora:{n[:-len('.safetensors')] if n.endswith('.safetensors') else n}:{m}>"
                    for n, m, _ in resolved)
    trigs = ", ".join(t for _, _, ts in resolved for t in ts)
    parts = [p for p in (trigs, tags, prompt) if p]
    return " ".join(parts)


# ── orquestrador ────────────────────────────────────────────────────────────

def select_loras(prompt: str, *, root: str = "minimax_h3", nsfw: bool = False,
                 variant: str = "*", modality: str = "", cfg: SelectorCfg = None,
                 enhance_fn=None, enhance_model: str = ""):
    """Seleciona LoRAs. Retorna (prompt_final, resolved, info)."""
    cfg = cfg or SelectorCfg()
    entries = eligible(load_catalog(root), nsfw, variant, modality)
    if not entries:
        return prompt, [], {"mode": cfg.selector, "ids": [], "reason": "sem elegíveis"}

    if cfg.selector == "decision":
        t0 = time.time()
        ids, rank = _decision_select(prompt, entries, cfg)
        info = {"mode": "decision", "ids": ids, "rank": rank,
                "secs": round(time.time() - t0, 1), "candidates": len(entries)}
        if not ids:
            info["reason"] = ("seletor decision indisponível" if rank is None
                              else "nenhuma acima do corte")
            return prompt, [], info
        resolved = resolve_ids(entries, ids)
        return build_prompt(prompt, resolved), resolved, info

    if enhance_fn is None:
        return prompt, [], {"mode": "llm", "ids": [], "reason": "sem enhance_fn"}
    block = _rerank_block(prompt, entries, root, modality)
    if not block:
        return prompt, [], {"mode": "llm", "ids": [], "reason": "rerank indisponível"}
    rule = (
        "LoRA SELECTION: at the very END of your output, on the LAST line, "
        "output exactly: LORAS: <comma-separated numeric IDs>\n"
        "- Choose the IDs best fit to the request (0 to 2 LoRAs, avoid stacking two "
        "that target the same feature/body aspect).\n"
        "- If genuinely nothing fits, output: LORAS: none\n"
        "- Never mention catalog trigger phrases in the paragraph itself.\n\n"
        + block)
    enhanced, raw = enhance_fn(prompt, enhance_model, nsfw=nsfw, extra_system=rule)
    m = re.search(r"LORAS:\s*([0-9,\s]+)", enhanced or "")
    ids = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    body = re.sub(r"^\s*LORAS:.*$", "", enhanced or "", flags=re.M)
    body = re.sub(r"\bLORAS:\s*[0-9,\s]*", "", body).strip()
    info = {"mode": "llm", "ids": ids, "body": body, "candidates": len(entries)}
    if not body:
        body = prompt
    if not ids:
        return body, [], info
    resolved = resolve_ids(entries, ids)
    return build_prompt(body, resolved), resolved, info


def _rerank_block(prompt: str, entries, root: str, modality: str = ""):
    """Roda o rerank ColBERT da casa sobre os elegíveis; devolve o bloco rankeado."""
    script = None
    for cand in (f"rerank_catalog_{root}.py", f"rerank_catalog_{root.split('_')[0]}.py",
                 "rerank_catalog.py"):                                  # [H3]
        p = SHARE / cand
        if p.exists():
            script = p
            break
    if script is None:
        print("  [warn] rerank não encontrado; sem seleção llm")
        return None
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"loras": entries}, tf, ensure_ascii=False)
        tmp = tf.name
    env = dict(os.environ)
    env["CATALOG_JSON"] = tmp
    env["NSFW_CATALOG_JSON"] = tmp
    try:
        cmd = ["python3", str(script), "--prompt", prompt, "--n", "10", "--mode", "rank"]
        if modality:                                                   # [H3]
            cmd += ["--modality", modality]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
        if r.returncode == 0 and r.stdout.strip():
            print(f"  🔎 Rerank ColBERT: "
                  f"{len(re.findall(r'^\d+:', r.stdout, flags=re.M))} LoRAs pré-ordenados "
                  f"({time.time() - t0:.1f}s)")
            return r.stdout.strip()
        print(f"  [warn] rerank falhou (rc={r.returncode}); catálogo completo")
        return "\n".join(f"{e['id']}. [{e.get('categoria','')}] "
                         f"{e['safetensors'].replace('.safetensors','')}: "
                         f"{(e.get('desc_llm') or '')[:400]}" for e in entries)
    except Exception as e:
        print(f"  [warn] rerank indisponível ({e}); catálogo completo")
        return "\n".join(f"{e['id']}. [{e.get('categoria','')}] "
                         f"{e['safetensors'].replace('.safetensors','')}: "
                         f"{(e.get('desc_llm') or '')[:400]}" for e in entries)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
