#!/usr/bin/env bash
# Ladder ngl x ctx para SenseNova-Loop-9B Q4_0 (Vulkan, 3050 6GB)
# uso: sn9-ladder.sh <NGl> <CTX> [gen]
NGL=${1:?}
CTX=${2:?}
GEN=${3:-96}
MODEL=/home/alchemist/.llama-models/sensenova-loop-9b/SenseNova-Loop-9B-Preview-Q4_0.gguf
cd /home/alchemist/git/llama.cpp-sensenova-loop || exit 1

pkill -f 'llama-server.*SenseNova-Loop' 2>/dev/null; sleep 3

env GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ./build/bin/llama-server \
  -m "$MODEL" --port 10123 -ngl "$NGL" -c "$CTX" -fa on -ctk q4_0 -ctv q4_0 \
  --jinja -t 10 -b 512 -ub 128 --no-webui > /tmp/sn_ladder.log 2>&1 &
SRV=$!

HEALTH=000
for i in $(seq 1 140); do
  BODY=$(curl -s http://127.0.0.1:10123/health 2>/dev/null)
  case "$BODY" in *ok*) HEALTH=ok; break ;; esac
  kill -0 $SRV 2>/dev/null || break
  sleep 1
done

if [ "$HEALTH" = "ok" ]; then
  PICO=$(timeout 6 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 2>/dev/null | sort -n | tail -1)
  T0=$(date +%s)
  curl -s -m 240 http://127.0.0.1:10123/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"sn9\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in one short paragraph why memory-efficient attention helps small GPUs.\"}],\"max_tokens\":$GEN,\"temperature\":0}" > /tmp/sn_gen.json
  T1=$(date +%s)
  echo "NGL=$NGL CTX=$CTX HEALTH=ok PICO=$PICO WALL=$((T1-T0))s"
  grep -E 'print_timing' /tmp/sn_ladder.log | tail -3 | grep -oE '(eval time|prompt eval).*' | head -3
else
  echo "NGL=$NGL CTX=$CTX HEALTH=000 (load falhou)"
  grep -E 'failed to allocate|failed to initialize' /tmp/sn_ladder.log | tail -2
fi
pkill -f 'llama-server.*SenseNova-Loop' 2>/dev/null
sleep 3