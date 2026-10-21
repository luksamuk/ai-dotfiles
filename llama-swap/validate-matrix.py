#!/usr/bin/env python3
"""Valida a matrix do config buildado ANTES de sincronizar no swap."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("bc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-config.py"))
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
import yaml

text, active, disabled = bc.build_config()
cfg = yaml.safe_load(text)

models = set(cfg["models"])
vars_ = cfg["matrix"]["vars"]
sets = cfg["matrix"]["sets"]
costs = cfg["matrix"]["evict_costs"]

errs = []
aposentados = ["laguna-xs-2.1", "lfm2.5-2.6b", "minicpm-v-4.7-1b", "minicpm5-2b",
               "nanbeige4.2-3b", "nemotron-3-embed-1b", "neohorse-1-4b", "bonsai2-2b", "bonsai2-4b"]

# 1. var -> model exists
for v, m in vars_.items():
    if m not in models:
        errs.append(f"var '{v}' aponta p/ modelo inexistente: {m}")

# 2. var values nao podem ser aposentados (nem handle do aposentado)
for v, m in vars_.items():
    for a in aposentados:
        if m == a or m.split(":")[0] == a:
            errs.append(f"var '{v}' ainda aponta p/ aposentado {m}")

# 3. tokens dos sets existem como var
TOK = re.compile(r"[A-Za-z0-9_.\-]+")
for sname, expr in sets.items():
    for tok in TOK.findall(expr):
        if tok not in vars_:
            errs.append(f"set '{sname}' usa token desconhecido: '{tok}'")

# 4. evict_costs keys existem como var
for k in costs:
    if k not in vars_:
        errs.append(f"evict_costs '{k}' nao e var da matrix")

# 5. modelos aposentados fora do config
for a in aposentados:
    if a in models:
        errs.append(f"modelo aposentado ainda no config: {a}")

print(f"models no config: {len(models)}")
print(f"vars: {len(vars_)} | sets: {len(sets)} | evict_costs: {len(costs)}")
print()
if errs:
    print("ERROS:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("MATRIX OK — nenhum erro")
# dump pra inspecao
open("/tmp/config-built.yaml", "w").write(text)
print("config buildado em /tmp/config-built.yaml")
print("\nheavy :", sets.get("heavy"))
print("bs2_embed:", sets.get("bs2_embed"))
print("medium:", "presente" if "medium" in sets else "REMOVIDO")
print("\nnovas vars:", {k: vars_[k] for k in ("orn35", "bs2", "lfm24", "mcv35", "q36") if k in vars_})
