#!/usr/bin/env bash
# Remove os PESOS do H3 listados no manifesto, preservando codigo, catalogo e LoRAs.
# Nao usa rm -rf: apaga caminho por caminho, do manifesto, e mostra o que fez.
# Uso: h3-uninstall-pesos.sh [--exec] [manifesto]
set -u
EXEC=false
COM_LORAS=false
MAN="$HOME/.local/share/h3/h3_manifest_2026-10-23.json"
for a in "$@"; do
  case "$a" in
    --exec) EXEC=true ;;
    --loras) COM_LORAS=true ;;
    *) MAN="$a" ;;
  esac
done
W="$HOME/git/Wan2GP"
[ -f "$MAN" ] || { echo "manifesto nao encontrado: $MAN"; exit 1; }

LIB=0
echo "=== manifesto: $MAN"
python3 - "$MAN" <<'PY' > /tmp/h3_rm_lista.txt
import json, sys
m = json.load(open(sys.argv[1]))
for b in m["bases"]:
    print(f"{b['dest']}\t{b['size']}")
t = m["turbo"]
print(f"{t['dest']}\t{t['size']}")
PY

while IFS=$'\t' read -r dest size; do
  p="$W/$dest"
  if [ -f "$p" ]; then
    echo "  ${size} B  $dest"
    LIB=$((LIB + size))
    if [ "$EXEC" = true ]; then rm -f "$p"; fi
  fi
done < /tmp/h3_rm_lista.txt

echo ""
echo "=== symlinks que apontariam para alvo ausente:"
python3 - "$MAN" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
W = os.path.expanduser("~/git/Wan2GP")
for link, alvo in m["symlinks"]:
    lp = os.path.join(W, link)
    if os.path.lexists(lp):
        print("  " + link)
PY

# LoRAs recuperaveis (as orfas ficam: nao ha de onde baixar)
LM="$HOME/.local/share/h3/h3_loras_manifest_2026-10-23.json"
if [ "$COM_LORAS" = true ] && [ -f "$LM" ]; then
  echo ""
  echo "=== LoRAs recuperaveis (orfas ficam):"
  python3 - "$LM" <<'PYL' > /tmp/h3_lora_rm.txt
import json, sys
for x in json.load(open(sys.argv[1]))["loras"]:
    print(f"{x['dest']}\t{x['size'] or 0}")
PYL
  while IFS=$'\t' read -r dest size; do
    p="$W/$dest"
    [ -f "$p" ] || continue
    echo "  ${size} B  $dest"
    LIB=$((LIB + size))
    [ "$EXEC" = true ] && rm -f "$p"
  done < /tmp/h3_lora_rm.txt
fi

if [ "$EXEC" != true ]; then
  echo ""
  echo "DRY RUN — nada foi apagado. Total que sairia: $(echo "scale=1; $LIB/1073741824" | bc) GiB"
  echo "Rode com --exec para apagar."
fi
