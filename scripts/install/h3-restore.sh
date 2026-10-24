#!/usr/bin/env bash
# Reinstala os PESOS do MiniMax H3 a partir do manifesto.
# Baixa do HF pelo nome exato, confere tamanho e sha256, recria symlinks e rebuild do catalogo.
# Uso: h3-restore.sh [manifesto]
set -u
MAN="${1:-$HOME/.local/share/h3/h3_manifest_2026-10-23.json}"
W="$HOME/git/Wan2GP"
[ -f "$MAN" ] || { echo "manifesto nao encontrado: $MAN"; exit 1; }
cd "$W" || exit 1

echo "=== manifesto: $MAN"
python3 - "$MAN" <<'PY' > /tmp/h3_restore_lista.txt
import json, sys
m = json.load(open(sys.argv[1]))
for b in m["bases"]:
    print(f"{b['repo']}\t{b['file']}\t{b['dest']}\t{b['size']}\t{b['sha256']}")
t = m["turbo"]
print(f"{t['repo']}\t{t['file']}\t{t['dest']}\t{t['size']}\t{t['sha256']}")
PY

FALHOU=0
while IFS=$'\t' read -r repo file dest size sha; do
  echo ""
  echo "--- $file"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then
    echo "    ja presente com o tamanho certo; pulando"
  else
    hf download "$repo" "$file" --local-dir ckpts 2>/dev/null || \
      hf download "$repo" "$file" --local-dir "$(dirname "$dest")" || { echo "    FALHOU download"; FALHOU=1; continue; }
  fi
  # turbo vai para loras/, nao ckpts/
  if [[ "$dest" == loras/* ]]; then
    [ -f "$dest" ] || cp -f "ckpts/$file" "$dest" 2>/dev/null || true
  fi
  if [ -f "$dest" ]; then
    TS=$(stat -c %s "$dest")
    if [ "$TS" = "$size" ]; then echo "    tamanho OK ($TS)"; else echo "    TAMANHO ERRADO: $TS != $size"; FALHOU=1; fi
  fi
done < /tmp/h3_restore_lista.txt

echo ""
LM="$HOME/.local/share/h3/h3_loras_manifest_2026-10-23.json"
if [ -f "$LM" ]; then
  echo "=== LoRAs (Civitai, precisa de ~/.civitai_key)"
  TOK=$(cat "$HOME/.civitai_key" 2>/dev/null)
  python3 - "$LM" <<'PYL' > /tmp/h3_lora_lista.txt
import json, sys
for x in json.load(open(sys.argv[1]))["loras"]:
    print(f"{x['url']}\t{x['dest']}\t{x['size']}\t{x['sha256']}")
PYL
  while IFS=$'\t' read -r url dest size sha; do
    [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] && { echo "  ok  $(basename "$dest")"; continue; }
    echo "  baixando $(basename "$dest")"
    curl -sSL -A "Mozilla/5.0" -H "Authorization: Bearer $TOK" -o "$dest" "$url?token=$TOK" || { echo "    FALHOU"; continue; }
    A=$(sha256sum "$dest" | cut -d' ' -f1)
    [ "$A" = "$sha" ] && echo "    sha256 OK" || echo "    SHA256 DIFERENTE"
  done < /tmp/h3_lora_lista.txt
else
  echo "=== sem manifesto de LoRAs (pulando)"
fi

echo ""
echo "=== symlinks"
python3 - "$MAN" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
W = os.path.expanduser("~/git/Wan2GP")
for link, alvo in m["symlinks"]:
    lp = os.path.join(W, link)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    if os.path.lexists(lp):
        os.remove(lp)
    os.symlink(alvo, lp)
    print(f"  {link} -> {alvo}  ({'ok' if os.path.exists(lp) else 'ALVO AUSENTE'})")
PY

echo ""
echo "=== catalogo"
python3 "$HOME/.local/share/h3/build_h3_catalog.py" | head -3

echo ""
[ "$FALHOU" = "0" ] && echo "RESTORE OK" || echo "RESTORE COM FALHAS — confira acima"
