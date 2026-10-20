"""Seleção unificada de LoRAs — dois seletores, uma fonte de verdade.

Modos (SelectorCfg.selector):
  - "llm"      (alias "reranker"): rerank ColBERT pré-ordena o catálogo elegível e o
               LLM escolhe 0-2 IDs via linha "LORAS:" na própria resposta do enhance
               (comportamento histórico da casa).
  - "decision": o modelo de decisão (fleet /v1/systemone) recebe o prompt (normalmente
               o já-expandido) e responde N perguntas `noul` independentes — uma por
               LoRA elegível — com probabilidade por candidata. Corte cru no threshold,
               teto em max. Validado no harness 20/out: d1-3b recall@3 93%, recall@4
               100%, 0 falsos no controle negativo (laya: 21% — descartado).

Catálogo: ~/.local/share/diffuse/loras_<root>.json (formato F0: nsfw:bool + compat por
variante; IDs herdados do ex-nsfw, NUNCA renumerar). Fallback legado catalog_/nsfw_catalog_
não existe aqui — os runners antigos continuam nos arquivos legados até a migração.

Fail-open por doutrina: qualquer falha (swap fora, catálogo ausente, zero elegíveis)
retorna ([], []) e a geração roda base — nunca bloqueia.
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

SHARE = Path.home() / ".local" / "share" / "diffuse"
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://localhost:12434")

# destilados nunca são selecionáveis — entram só por --turbo / tag manual
_DISTILL = re.compile(r"dmd|turbo_distill", re.I)


@dataclass
class SelectorCfg:
    selector: str = "llm"          # "llm" | "decision"
    decision_model: str = "d1-3b"
    decision_threshold: float = 0.5
    decision_max: int = 4


# ── config: CLI > env > yaml > default ──────────────────────────────────────

def load_selector_cfg(args) -> SelectorCfg:
    """Resolve a configuração do seletor. `args` = namespace do argparse (flags CLI)."""
    def cli(k):
        return getattr(args, k, None)

    def env(k, cast):
        v = os.environ.get(f"DIFFUSE_{k.upper()}")
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
            v = (cfg.get("diffuse") or {}).get(k)
            if v in (None, ""):
                return None
            try:
                return cast(v)
            except (TypeError, ValueError):
                return None
        except Exception:
            return None

    sel = cli("lora_selector") or env("lora_selector", str) or yml("lora_selector", str) or "llm"
    if sel == "reranker":          # alias aceito (o rerank é a 1ª etapa; quem decide é o LLM)
        sel = "llm"
    if sel not in ("llm", "decision"):
        print(f"  [warn] lora_selector invalido {sel!r} — usando 'llm'")
        sel = "llm"

    model = cli("decision_model") or env("decision_model", str) or yml("decision_model", str) or "d1-3b"
    th = cli("decision_threshold")
    if th is None:
        th = env("decision_threshold", float) if env("decision_threshold", float) is not None else None
    if th is None:
        th = yml("decision_threshold", float)
    th = 0.5 if th is None else float(th)

    mx = cli("decision_max")
    if mx is None:
        mx = env("decision_max", int) if env("decision_max", int) is not None else None
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


def eligible(entries, nsfw: bool, variant: str):
    """Filtra por conteúdo, compatibilidade de variante e destilados."""
    out = []
    for e in entries or []:
        if _DISTILL.search(e.get("safetensors", "")):
            continue
        if e.get("nsfw") and not nsfw:
            continue
        comp = e.get("compat", ["*"])
        if not comp:
            continue  # explicitamente fora (ex.: femboy LyCORIS ate o V-MOSS validar)
        if "*" not in comp and variant not in comp:
            continue
        out.append(e)
    return out


# ── decision: N perguntas noul no /v1/systemone ─────────────────────────────

def _decision_select(prompt: str, entries, cfg: SelectorCfg):
    """Retorna (ids_selecionados, ranking_completo). Falha → ([], None) com aviso."""
    questions = {}
    for e in entries:
        nome = e["safetensors"].replace(".safetensors", "")
        desc = (e.get("desc_llm") or "").replace("\n", " ")[:160]
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
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            answers = (json.loads(r.read()) or {}).get("answers") or {}
    except Exception as e:
        print(f"  [warn] seletor decision indisponível ({e}); rodando base sem LoRA")
        return [], None
    scores = {}
    for k, v in answers.items():
        p = v.get("noul") if isinstance(v, dict) else None
        scores[str(k).replace("lora_", "")] = float(p) if isinstance(p, (int, float)) else 0.0
    rank = sorted(scores.items(), key=lambda x: -x[1])
    sel = [i for i, p in rank if p >= cfg.decision_threshold][:cfg.decision_max]
    return sel, rank


# ── resolve ids -> tags ─────────────────────────────────────────────────────

def resolve_ids(entries, ids):
    """IDs -> [(safetensors, strength, triggers)] resolvido em memória."""
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
    """Prepende tags <lora:...> e triggers ao prompt (mesma ordem histórica)."""
    if not resolved:
        return prompt
    # contrato historico: tag <lora:NOME:mult> SEM extensão (nsfw_lora_resolve:41)
    tags = " ".join(f"<lora:{n[:-len('.safetensors')] if n.endswith('.safetensors') else n}:{m}>"
                    for n, m, _ in resolved)
    trigs = ", ".join(t for _, _, ts in resolved for t in ts)
    parts = [p for p in (tags, trigs, prompt) if p]
    return " ".join(parts)


# ── orquestrador ────────────────────────────────────────────────────────────

def select_loras(prompt: str, *, root: str, nsfw: bool, variant: str,
                 cfg: SelectorCfg, enhance_fn=None, enhance_model: str = ""):
    """Seleciona LoRAs para o pedido. Retorna (prompt_final, resolved, info).

    - llm:      uma chamada de enhance faz expandir + selecionar (regra LORAS:).
                enhance_fn(prompt, model, nsfw, extra_system) -> (enhanced, raw).
                Devolve o ENHANCED como prompt_final (fluxo histórico).
    - decision: NÃO chama LLM; o `prompt` recebido (normalmente o já-expandido
                pelo enhance comum) vai ao systemone; prompt_final = o mesmo
                prompt com tags prependadas.
    info: dict com modo, ids, ranking (decision) — para display/telemetria.
    """
    entries = eligible(load_catalog(root), nsfw, variant)
    if not entries:
        return prompt, [], {"mode": cfg.selector, "ids": [], "reason": "sem elegíveis"}

    if cfg.selector == "decision":
        t0 = time.time()
        ids, rank = _decision_select(prompt, entries, cfg)
        info = {"mode": "decision", "ids": ids, "rank": rank,
                "secs": round(time.time() - t0, 1), "candidates": len(entries)}
        if not ids:
            # rank=None = seletor indisponível (rede/API); rank=list = corte vazio honesto
            info["reason"] = ("seletor decision indisponível" if rank is None
                              else "nenhuma acima do corte")
            return prompt, [], info
        resolved = resolve_ids(entries, ids)
        return build_prompt(prompt, resolved), resolved, info

    # llm
    if enhance_fn is None:
        return prompt, [], {"mode": "llm", "ids": [], "reason": "sem enhance_fn"}
    block = _rerank_block(prompt, entries, root)
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
    body = re.sub(r"^\s*LORAS:.*$", "", enhanced or "", flags=re.M).strip()
    info = {"mode": "llm", "ids": ids, "body": body}
    if not ids:
        return (body or prompt), [], info
    resolved = resolve_ids(entries, ids)
    return build_prompt(body or prompt, resolved), resolved, info


def _rerank_block(prompt: str, entries, root: str):
    """Rode o rerank ColBERT da casa sobre os elegíveis; devolve o bloco rankeado."""
    script = None
    for cand in (f"rerank_catalog_{root}.py", f"rerank_catalog_{root.split('_')[0]}.py"):
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
        r = subprocess.run(["python3", str(script), "--prompt", prompt,
                            "--n", "10", "--mode", "rank"],
                           capture_output=True, text=True, timeout=180, env=env)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        print(f"  [warn] rerank falhou (rc={r.returncode}); catálogo completo")
        # fallback: bloco NÃO rankeado (o LLM escolhe entre todos)
        lines = []
        for e in entries:
            nome = e["safetensors"].replace(".safetensors", "")
            desc = (e.get("desc_llm") or "").replace("\n", " ")[:150]
            lines.append(f"{e['id']}. [{e.get('categoria','')}] {nome}: {desc}")
        return "\n".join(lines)
    except Exception as e:
        print(f"  [warn] rerank indisponível ({e}); catálogo completo")
        return "\n".join(
            f"{e['id']}. [{e.get('categoria','')}] "
            f"{e['safetensors'].replace('.sft','').replace('.safetensors','')}: "
            f"{(e.get('desc_llm') or '')[:150]}"
            for e in entries)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass