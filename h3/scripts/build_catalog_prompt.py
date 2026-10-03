#!/usr/bin/env python3
"""build_catalog_prompt.py — gera o bloco de catálogo compacto injetado no system do enhance.
Lê $NSFW_CATALOG_JSON (default: ~/.local/share/h3/nsfw_catalog_h3.json) e produz texto de 1 linha por LoRA."""
import json, os, sys

CAT = os.environ.get('NSFW_CATALOG_JSON') or os.path.expanduser('~/.local/share/h3/nsfw_catalog_h3.json')
try:
    d = json.load(open(CAT))
except Exception:
    print("(catalogo indisponível)", file=sys.stderr); sys.exit(0)

lines = ["Catalogo de LoRAs H3 disponiveis (IDs numericos) - escolha os adequados ao pedido:"]
for e in sorted(d['loras'], key=lambda x: x['id']):
    trig = e.get('trigger_instr_arquivo') or ''
    # desc curta já é 1 linha
    lines.append(f"{e['id']}: [{e['categoria']}] {e['desc_llm']}")
print('\n'.join(lines))