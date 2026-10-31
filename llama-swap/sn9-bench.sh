#!/usr/bin/env bash
# Bench de teto de contexto do SenseNova-Loop-9B Q4_0 na 3050 (Vulkan)
# uso: bench_sn9.sh <CTX> <ctk> <ctv> [gen_tokens]
CTX=${1:?}
CTK=${2:?}
CTV=${3:?}
GEN=${4:-40}
MODEL=/home/alchemist/.llama-models/sensenova-loop-9b/SenseNova-Loop-9B-Preview-Q4_0.gguf
cd /home/alchemist/git/llama.cpp-sensenova-loop || exit 1

pkill -f 'llama-server.*SenseNova-Loop' 2>/dev/null; sleep 3

env GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ./build/bin/llama-server \
  -m "$MODEL" --port 10123 -ngl 99 -c "$CTX" -fa on -ctk "$CTK" -ctv "$CTV" \
  --jinja -b 512 -ub 128 --no-webui > /tmp/sn_server.log 2>&1 &
SRV=$!

HEALTH=000
for i in $(seq 1 120); do
  BODY=$(curl -s http://127.0.0.1:10123/health 2>/dev/null)
  case "$BODY" in
    *ok*) HEALTH=ok; break ;;
  esac
  killed=$(kill -0 $SRV 2>/dev/null; echo $?)
  [ "$killed" != "0" ] && break
  sleep 1
done
echo "HEALTH=$HEALTH"

if [ "$HEALTH" = "ok" ]; then
  PICO=$(timeout 6 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 2>/dev/null | sort -n | tail -1)
  echo "PICO_VRAM_MIB=$PICO"
  R=$(curl -s -m 180 http://127.0.0.1:10123/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"sn9\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one short sentence about the moon.\"}],\"max_tokens\":$GEN,\"temperature\":0}")
  echo "$R" | python3 -c "import json,sys
try:
  d=json.load(sys.stdin)
  u=d['choices'][0]
  print(u['message'].get('content','')[:200])
  t=d['usage']
  print('gen t/s:', round(t['completion_time']['predicted_n']/t['completion_time']['predicted_ms']*1000,1) if 'completion_time' in t else 'n/a', t)
except Exception as e:
  print('ERR', e)" 2>&1 | head -8
fi

pkill -f 'llama-server.*SenseNova-Loop' 2>/dev/null
sleep 3
grep -E 'KV buffer|compute buffer|failed to allocate|load_time' /tmp/sn_server.log | tail -6