#!/usr/bin/env bash
# Escada de offload do main FAT (8.2 GiB) com vision na 3050 6GB
NGL=${1:?}
CTX=${2:-8192}
MM=${3:-no}
MODEL=/home/alchemist/.llama-models/sensenova-loop-9b/SenseNova-Loop-9B-Preview-Q4_K_M-Vision.gguf
MMPROJ=/home/alchemist/.llama-models/sensenova-loop-9b/mmproj-SenseNova-Loop-F16.gguf
cd /home/alchemist/git/llama.cpp-sn-vision || exit 1

pkill -f 'llama-server.*SenseNova' 2>/dev/null; sleep 3

ARGS=(-m "$MODEL" --port 10123 -ngl "$NGL" -c "$CTX" -fa on -ctk q4_0 -ctv q4_0 \
  --jinja -t 10 -b 512 -ub 256 --no-webui --no-mmproj-offload)
[ "$MM" = "yes" ] && ARGS+=(--mmproj "$MMPROJ")

env GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ./build/bin/llama-server "${ARGS[@]}" > /tmp/sn_vis.log 2>&1 &
SRV=$!

HEALTH=000
for i in $(seq 1 150); do
  BODY=$(curl -s http://127.0.0.1:10123/health 2>/dev/null)
  case "$BODY" in *ok*) HEALTH=ok; break ;; esac
  kill -0 $SRV 2>/dev/null || break
  sleep 1
done

if [ "$HEALTH" = "ok" ]; then
  PICO=$(timeout 6 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 2>/dev/null | sort -n | tail -1)
  echo "NGL=$NGL CTX=$CTX MM=$MM HEALTH=ok PICO=$PICO"
  if [ "$MM" = "yes" ]; then
    python3 - <<'EOF'
import base64, io, json, time, urllib.request
from PIL import Image, ImageDraw
img = Image.new('RGB', (256, 256), 'white')
d = ImageDraw.Draw(img)
d.ellipse([64, 64, 192, 192], fill='red')
buf = io.BytesIO(); img.save(buf, 'PNG')
b64 = base64.b64encode(buf.getvalue()).decode()
req = json.dumps({"model":"sn9","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}},
  {"type":"text","text":"What color is the circle? One short sentence."}]}],
  "max_tokens":300,"temperature":0}).encode()
t0 = __import__('time').time()
try:
    r = urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:10123/v1/chat/completions',
        data=req, headers={'Content-Type':'application/json'}), timeout=280)
    d = json.load(r)
    m = d['choices'][0]['message']
    print(f"VISION PROBE ({time.time()-t0:.0f}s): finish={d['choices'][0].get('finish_reason')}")
    print("reasoning len:", len(m.get('reasoning_content') or ''))
    print("content:", (m.get('content') or '').strip()[:250])
except Exception as e:
    print("VISION PROBE FALHOU:", repr(e)[:300])
EOF
  else
    curl -s -m 200 http://127.0.0.1:10123/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"sn9","messages":[{"role":"user","content":"One short sentence about the sea."}],"max_tokens":96,"temperature":0}' \
      | python3 -c "import json,sys; d=json.load(sys.stdin); ct=d['usage'].get('completion_time',{}); print('decode t/s:', round(ct['predicted_n']/ct['predicted_ms']*1000,1) if ct else 'n/a'); print('content ok:', bool(d['choices'][0]['message'].get('content')))" 2>&1
  fi
else
  echo "NGL=$NGL CTX=$CTX MM=$MM HEALTH=000 (load falhou)"
  grep -E 'failed to allocate|failed to initialize|error' /tmp/sn_vis.log | tail -2
fi
pkill -f 'llama-server.*SenseNova' 2>/dev/null
sleep 3