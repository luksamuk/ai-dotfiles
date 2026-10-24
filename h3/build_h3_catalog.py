#!/usr/bin/env python3
"""Monta ~/.local/share/h3/loras_minimax_h3.json a partir do DISCO.
Schema = union: campos legados (id, categoria, desc_llm, trigger_instr_arquivo,
safetensors, strength_autor) + novos (nsfw, compat, fonte).
IDs: reaproveita os do nsfw_catalog_h3.json quando o arquivo sobreviveu."""
import json, os, glob, struct

H3 = os.path.expanduser("~/.local/share/h3")
LORAS = os.path.expanduser("~/git/Wan2GP/loras/minimax_h3")

# Curadoria vem de catalog_cur.py (local, fora do git). Ausente = catalogo sem descricoes.
try:
    from catalog_cur import CUR, CONCEITO_MANUAL, DESATIVADOS, DOWNLOAD_MAP, LEGACY_ID
except ImportError:
    CUR, CONCEITO_MANUAL, DESATIVADOS, DOWNLOAD_MAP, LEGACY_ID = {}, {}, {}, {}, {}
    print("aviso: catalog_cur.py ausente — catalogo sem curadoria")
LEGACY = os.path.join(H3, "nsfw_catalog_h3.json")
OUT = os.path.join(H3, "loras_minimax_h3.json")

# curadoria: chave = nome do arquivo. campos que o disco nao da.

legacy = {}
# prioridade de ID: catalogo NOVO (ja publicado) > legado > novo sequencial.
# Sem isso, adicionar um LoRA cujo nome ordena antes renumeraria todos os outros.
if os.path.exists(OUT):
    for e in json.load(open(OUT)).get("loras", []):
        legacy[e.get("safetensors", "")] = e.get("id")
p = LEGACY
if os.path.exists(p):
    for e in json.load(open(p)).get("loras", []):
        legacy.setdefault(e.get("safetensors", ""), e.get("id"))


# LoRAs DESATIVADAS (23/out): pausa no NSFW por decreto do user. Ficam registradas com
# link e sha256 para volta; nao exigem arquivo em disco e o seletor as ignora.


# Referencia de download das LoRAs ATIVAS (23/out): prova por sha256 de que o arquivo
# local e o mesmo do Civitai, apesar do rename. Sem isso, apagar a LoRA = perde-la.

disk = sorted(os.path.basename(x) for x in glob.glob(os.path.join(LORAS, "*.safetensors")))
legacy.update({k: v for k, v in LEGACY_ID.items()})   # LEGACY_ID e autoritativo
prox = max([i for i in legacy.values() if isinstance(i, int)] or [0]) + 1
out, faltando, usados = [], [], set()

# Fora do catalogo (nao sao LoRA de estilo; o seletor nunca deve escolher):
#  - facial-realism: empurra textura de poro e derrete rosto a poucos steps (22/out)
#  - *_turbo_*: aceleradoras de step, carregadas explicitamente com --turbo-file/--lora
NAO_CATALOGAR = {"minimax-h3-facial-realism-closeup-cp2000.safetensors"}

for nome in disk:
    if nome in NAO_CATALOGAR or "_turbo_" in nome:
        continue
    if nome.endswith("_conv.safetensors") and nome.replace("_conv.safetensors", ".safetensors") in disk:
        continue  # duplicata tecnica do mesmo LoRA em outro formato
    c = CUR.get(nome)
    if c is None:
        faltando.append(nome); c = dict(categoria="?", nsfw=True, strength_autor=0.8,
                                        trigger_instr_arquivo="", fonte="?", desc_llm="")
    i = legacy.get(nome)
    if i is None or i in usados:
        i = prox; prox += 1
    usados.add(i)
    # conceito: familia do LoRA. O seletor escolhe no maximo UM por conceito,
    # senao ele gasta vagas com versoes da mesma coisa (VBVR Pro + v1,
    # better_human_motion + motion_combat). Derivado do nome; overrides abaixo.
    def _conceito(nm):
        if nm in CONCEITO_MANUAL:
            return CONCEITO_MANUAL[nm]
        import re as _r
        b = nm.replace(".safetensors", "")
        b = _r.sub(r'[-_](fl2va|ref2va)$', '', b, flags=_r.I)   # variante arquitetural
        b = _r.sub(r'[-_ ](pro|v\d+)\b', '', b, flags=_r.I)     # VBVR-H3-Pro-attn vs -v1-attn
        b = _r.sub(r'[-_]attn$', '', b, flags=_r.I)
        b = _r.sub(r'[\s_-]*\d{6,}$', '', b)                    # id civitai colado
        return _r.sub(r'[-_]+$', '', b).strip().lower()

    # modalidade: variantes -fl2va/-ref2va do MESMO conceito nunca devem
    # concorrer na mesma run (o seletor ja filtra pelo campo "modality")
    import re as _re
    _m = _re.search(r'[-_](fl2va|ref2va)\b', nome, _re.I)
    out.append({"id": i, "safetensors": nome, "conceito": _conceito(nome),
                "modality": (_m.group(1).lower() if _m else ""),
                "categoria": c["categoria"],
                "desc_llm": c["desc_llm"], "trigger_instr_arquivo": c["trigger_instr_arquivo"],
                "strength_autor": c["strength_autor"], "fonte": c["fonte"],
                "nsfw": c["nsfw"], "compat": ["*"]})

doc = {"motor": "minimax_h3", "versao": 1,
       "nota": "Catalogo unificado do H3 (schema do diffuse lora_selector: nsfw + compat). "
               "IDs legados preservados; nunca renumerar. Gerado a partir do disco.",
       "fonte": "disco ~/git/Wan2GP/loras/minimax_h3 + curadoria Civitai (21/out/2026)",
       "formato_selecao": "LORAS: <int-id,int-id,...> (ultima linha do enhance)",
       "loras": out}
# download das ativas: anexa o bloco de recuperacao a cada entrada com arquivo em disco
for e in out:
    d = DOWNLOAD_MAP.get(e["safetensors"])
    if d:
        e["download"] = dict(provider="civitai",
                             url=f"https://civitai.com/api/download/models/{d['version_id']}",
                             version_id=d["version_id"], remote_file=d["remote"],
                             size_bytes=d["size"], sha256=d["sha256"])

# desativadas: entram no catalogo como registro, nunca como candidata
for nome, d in DESATIVADOS.items():
    if any(e["safetensors"] == nome for e in out):
        continue
    c = CUR.get(nome) or {}
    out.append(dict(
        id=legacy.get(nome, prox), safetensors=nome,
        categoria=c.get("categoria", ""), conceito=c.get("conceito", ""),
        desc_llm=c.get("desc_llm", ""), strength_autor=c.get("strength_autor", 1.0),
        trigger_instr_arquivo=c.get("trigger_instr_arquivo", ""), fonte=c.get("fonte", ""),
        nsfw=True, compat="", modality="",
        disabled=True, baixado=False,
        download=dict(provider="civitai", model_id=d["model_id"], version_id=d["version_id"],
                      url=f"https://civitai.com/api/download/models/{d['version_id']}",
                      size_bytes=d["size"], sha256=d["sha256"])))

json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)

# 22/out: MESMA lista no catalogo que o enhancer le (rerank_catalog.py le
# nsfw_catalog_h3.json). Antes eram dois arquivos com IDs diferentes: o enhance
# escolhia IDs de um catalogo de 43 entradas (40 fantasmas) que nao existiam mais,
# e a resolucao falhava em silencio ("0 LoRAs selecionadas").
ALT = os.path.join(H3, "nsfw_catalog_h3.json")
alt = [dict(e, baixado=True, na_pilha_doutrina=False) for e in out]
json.dump({"nota": "espelho do loras_minimax_h3.json para o enhancer (mesmos ids)",
           "formato_selecao": "LORAS: <int-id,int-id,...> (ultima linha do enhance)",
           "loras": alt}, open(ALT, "w"), indent=1, ensure_ascii=False)

print(f"=== {OUT}")
print(f"    {len(out)} entradas | nsfw: {sum(1 for e in out if e['nsfw'])} | "
      f"sfw: {sum(1 for e in out if not e['nsfw'])}")
print(f"    ids: {[e['id'] for e in out]}")
if faltando:
    print(f"    SEM CURADORIA ({len(faltando)}): {faltando}")
print(f"    ignorados por duplicata tecnica: _conv")
