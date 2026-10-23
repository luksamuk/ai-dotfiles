#!/usr/bin/env python3
"""Monta ~/.local/share/h3/loras_minimax_h3.json a partir do DISCO.
Schema = union: campos legados (id, categoria, desc_llm, trigger_instr_arquivo,
safetensors, strength_autor) + novos (nsfw, compat, fonte).
IDs: reaproveita os do nsfw_catalog_h3.json quando o arquivo sobreviveu."""
import json, os, glob, struct

H3 = os.path.expanduser("~/.local/share/h3")
LORAS = os.path.expanduser("~/git/Wan2GP/loras/minimax_h3")
LEGACY = os.path.join(H3, "nsfw_catalog_h3.json")
OUT = os.path.join(H3, "loras_minimax_h3.json")

# curadoria: chave = nome do arquivo. campos que o disco nao da.
CUR = {
 "h3_motion_combat_reasoning_v3377709.safetensors": dict(
   categoria="acao-combate", nsfw=False, strength_autor=0.8, trigger_instr_arquivo="",
   fonte="civitai:3377709",
   desc_llm="Enhances combat, action and motion in highly dynamic scenes; follows fight "
            "choreography, impact and staging. Melhora combate, acao e movimento em cenas "
            "dinamicas. Consensus merge (FourBunny + MisticRain69). FL2VA/REF2VA/Hybrid."),
 "h3_better_human_motion_v3256084.safetensors": dict(
   categoria="movimento", nsfw=False, strength_autor=0.6, trigger_instr_arquivo="",
   fonte="civitai:3256084",
   desc_llm="Natural and consistent body movement, less limb warping, better weight and "
            "follow-through. Movimento corporal natural e consistente. Treinado em T2V "
            "720x1280, 15-30 steps, peso 0.4-0.8."),
 "h3_anime_shinkai_style_v3302128.safetensors": dict(
   categoria="estilo-anime-cenario", nsfw=False, strength_autor=0.9,
   trigger_instr_arquivo="2d anime style, art by makoto shinkai",
   fonte="civitai:3302128",
   desc_llm="Makoto Shinkai environmental anime look: 2D anime style for backgrounds, skies, "
            "landscapes, light blooms. Estilo anime de CENARIO (nao estiliza personagem). "
            "Treinado em 228 frames de 'Your Name', 1536x864."),
 "h3_2d_anime_slider_v3378274.safetensors": dict(
   categoria="estilo-slider", nsfw=False, strength_autor=3.0, trigger_instr_arquivo="",
   fonte="civitai:3378274",
   desc_llm="Style slider between 2D anime and photorealism: positive leans 2D anime, "
            "negative leans photoreal, range -5..5 (3 ou -3 ja define). Dial de estilo "
            "anime x realista."),
 "h3_camera_motion_v3297480.safetensors": dict(
   categoria="camera", nsfw=False, strength_autor=1.0,
   trigger_instr_arquivo="camera motion",
   fonte="civitai:3297480",
   desc_llm="Camera movement control: pan, tilt, dolly, tracking, zoom and speed. Controle "
            "de camera. Trigger 'camera motion' OBRIGATORIO no INICIO do prompt, seguido do "
            "movimento e da velocidade."),
 "h3_crash_zoom_v3294020.safetensors": dict(
   categoria="camera", nsfw=False, strength_autor=0.9, trigger_instr_arquivo="crash_zm",
   fonte="civitai:3294020",
   desc_llm="Fast aggressive punch-in crash zoom. Zoom rapido de impacto. Forca 0.8-1.0."),
 "h3_polyhedron_eyes_skin_v3278036.safetensors": dict(
   categoria="detalhe", nsfw=False, strength_autor=0.5,
   trigger_instr_arquivo="perfe8ct, perfect skin, perfect eyes, perfect hands",
   fonte="civitai:3278036",
   desc_llm="Skin texture, eyes and hands detail fixer. Corrige pele, olhos e maos. Autor "
            "avisa que pode estar overtrained: usar peso baixo."),
 # ── os que ja estavam no disco ──
 "MiniMaxH3_Ref2V_JoinsTheBattle_v1.0.safetensors": dict(
   categoria="intro-card-meme", nsfw=False, strength_autor=0.9, trigger_instr_arquivo="",
   fonte="local (descrito pelo dono, 21/out)",
   desc_llm="Meme intro card: the subject enters frame striking a cool pose under a "
            "'NAME JOINS THE BATTLE!' style title card. Card de entrada de meme, pose "
            "de apresentacao. NAO serve para cena de luta/combate em andamento."),
 "VBVR-H3-Pro-attn.safetensors": dict(
   categoria="controle-timeline", nsfw=False, strength_autor=1.0, trigger_instr_arquivo="",
   fonte="local",
   desc_llm="Prompt-following and video reasoning: follows scheduled shot transitions and "
            "timelines, keeps multi-step instructions. Segue roteiro e transicoes agendadas."),
 "VBVR-H3-v1-attn.safetensors": dict(
   categoria="controle-timeline", nsfw=False, strength_autor=1.0, trigger_instr_arquivo="",
   fonte="local",
   desc_llm="Versao leve do VBVR: aderencia ao roteiro e a transicoes agendadas."),
 "anirl_mmh3_t2va.safetensors": dict(
   categoria="estilo-anime-rosto", nsfw=False, strength_autor=1.0,
   trigger_instr_arquivo="ANIRL, anime-rendered face and hair",
   fonte="local (Anime IRL, civitai)",
   desc_llm="Anime-rendered FACE and hair on an otherwise photoreal subject (cel-shaded "
            "rosto/cabelo, corpo continua real). Anime IRL. Forca 1.0+."),
 "minimax-h3-facial-realism-closeup-cp2000.safetensors": dict(
   categoria="detalhe", nsfw=False, strength_autor=1.0,
   trigger_instr_arquivo="Facial Realism",
   fonte="local (prithivMLmods)",
   desc_llm="Facial realism for close-ups: pores, skin texture, micro-expression. Realismo "
            "facial em close. Trigger 'Facial Realism' no inicio."),
 "natural_face_speech_h3_lora_v1_500.safetensors": dict(
   categoria="fala", nsfw=False, strength_autor=0.6, trigger_instr_arquivo="",
   fonte="local (vpakarinen)",
   desc_llm="Detailed facial acting: micro-expressions, glare, gritted teeth, tears in "
            "close-ups and emotional beats; natural mouth dynamics on dialogue. Card 0.4-0.8."),
 "hmmotion_minimax-h3_epoch12.safetensors": dict(
   categoria="movimento-ato", nsfw=True, strength_autor=1.0,
   trigger_instr_arquivo="hmotion",
   fonte="civitai 2834417 v1.0",
   conceito="penetracao",
   desc_llm="Penetration motion for explicit scenes: drives the physical rhythm of intercourse — thrusting, bodies meeting, the mechanical motion of penetration. Select when the prompt describes intercourse or penetration and the motion itself must read clearly. Not for solo nudity or static poses."),
 "grindtime_v1_h3_motion_r16.safetensors": dict(
   categoria="movimento-ato", nsfw=True, strength_autor=1.0,
   trigger_instr_arquivo="grindtime, a woman performing continuous rhythmic gyrating hip motion while having sex",
   fonte="civitai 2862363 v1.0",
   conceito="grinding",
   desc_llm="Grinding and gyrating hip motion during sex: slow, continuous, rhythmic circling of the hips rather than thrusting. Select when the prompt asks for grinding, gyrating, riding with circular motion, or slow rhythmic hip movement. Angle keywords front view, rear view or side view can be added to the prompt. Not for plain penetration or solo scenes."),
 "HMNSFW-AIO-V2.5.safetensors": dict(
   categoria="ato-geral", nsfw=True, strength_autor=1.0, trigger_instr_arquivo="",
   fonte="civitai 2834417 V2.5",
   conceito="ato-aio",
   desc_llm="General explicit-act realism LoRA trained on a broad range of sexual acts: penetration, handjobs, blowjobs, fingering, riding, cumshots, with natural skin and realistic bodies. Select ONLY when the prompt describes a sexual act happening between two or more participants. Do NOT select for a person alone, for nudity, posing, or a solo scene with no act. No trigger token. Author settings: 12 steps, shift 6, turbo strength 0.5. Works far better with an image reference (I2V/Ref2VA) than text-only."),
 "mm-h3-flat-chested.safetensors": dict(
   categoria="conceito", nsfw=True, strength_autor=1.0,
   trigger_instr_arquivo="She is flat chested.",
   fonte="local",
   desc_llm="Body concept: flat chested. Select ONLY when the prompt explicitly describes a flat-chested woman — it injects the phrase 'She is flat chested.' into the prompt. Do NOT select for a generic nude or body shot."),
 "MM-H3 - Sydney Sweeney.safetensors": dict(
   categoria="identidade", nsfw=False, strength_autor=1.0, trigger_instr_arquivo="",
   fonte="local",
   desc_llm="Likeness of the actress SIDNEY SWEENEY: blonde waves, wide-set blue eyes, "
            "rounded face, full lips. ALWAYS select whenever the prompt names her or asks "
            "for her face/body in a realistic shot. Identidade de pessoa real."),
 "h3-cumshot-final-step16600-r16-k3nk.safetensors": dict(
   categoria="cumshot", nsfw=True, strength_autor=0.85, trigger_instr_arquivo="",
   fonte="local (F4C3SPL4SH k3nk)",
   desc_llm="Ejaculation onto the subject at the END of a shot, with audio. Select ONLY "
            "when the prompt explicitly asks for semen/ejaculation/cumshot. Do NOT select "
            "for a merely nude or sexual scene."),
}

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
DESATIVADOS = {
    'hmmotion_minimax-h3_epoch12.safetensors': dict(model_id=2834417, version_id=3198686, size=309964680, sha256="aa31d84116b689e840cd4e218c305a2995de448d84d48d35217efa70f6bb29bf"),
    'grindtime_v1_h3_motion_r16.safetensors': dict(model_id=2862363, version_id=3233508, size=489776860, sha256="ce270fd83ec994a4a52265ba237d511d0c7aa047e1873923adc070c91ea43326"),
    'HMNSFW-AIO-V2.5.safetensors': dict(model_id=2834417, version_id=3268303, size=86040232, sha256="a07732a84fd733085eb5d910f602f918fa7a3658117116927e4329f5951a9d2d"),
    'mm-h3-flat-chested.safetensors': dict(model_id=2761774, version_id=3203865, size=155110304, sha256="702f1ef4645dc822f8446bea3c35b45906d05a9634b6df0ab0ecb4daa6f4d8d4"),
    'h3-cumshot-final-step16600-r16-k3nk.safetensors': dict(model_id=2857340, version_id=3329529, size=310206704, sha256="8fb36c6e70cdb25adafb72530e7fbebb95c9dfd7de4fc5e23f6542502d84d2ef"),
}

disk = sorted(os.path.basename(x) for x in glob.glob(os.path.join(LORAS, "*.safetensors")))
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
    CONCEITO_MANUAL = {
        "h3_better_human_motion_v3256084.safetensors": "motion",
        "h3_motion_combat_reasoning_v3377709.safetensors": "motion",
        # atos: familias distintas, nao competem entre si (penetracao != grinding != ato geral)
        "hmmotion_minimax-h3_epoch12.safetensors": "penetracao",
        "grindtime_v1_h3_motion_r16.safetensors": "grinding",
        "HMNSFW-AIO-V2.5.safetensors": "ato-aio",
    }
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
