#!/usr/bin/env bash
# turboquant-bench.sh — Benchmark comparativo: ik (q4_0+hadamard) vs bee (turbo3_tcq)
# Roda UM modelo por vez (6GB VRAM!). Nunca dois ao mesmo tempo.
# Mede: velocidade (tok/s), qualidade (Snake benchmark), footprint (VRAM/RAM)
#
# Uso: ./turboquant-bench.sh [model]
#   model: e2b | e4b | 4b | 9b | 12b | all (default: all)
#
# Resultados em: ~/turboquant-bench-results/

set -euo pipefail

RESULTS_DIR="$HOME/turboquant-bench-results"
mkdir -p "$RESULTS_DIR"

IK_SERVER="$HOME/git/ik_llama.cpp/build/bin/llama-server"
BEE_SERVER="$HOME/git/beellama.cpp/build/bin/llama-server"
MODELS_DIR="$HOME/.llama-models"
SNAKE_DIR="$HOME/projects/ai/benchmarks/snake"
PORT=19999
LLAMASWAP_PORT=12434

# Snake prompt (standardized)
SNAKE_PROMPT="Write one complete Python 3 file using only the standard library. Return only Python code. Do not use markdown, comments, tests, examples, or explanatory text. Implement a deterministic Task store module with a compact, repetitive structure that is easy to predict. Required shape: - imports: dataclasses, datetime, typing - dataclass Task with fields id: int, title: str, status: str, created_at: str - class TaskStore with an internal dict[int, Task] - methods: add, get, rename, mark_done, reopen, delete, clear, list_all, list_open, list_done, count_open, count_done, titles, to_dicts, __len__, __contains__ - add assigns increasing integer ids starting at 1 - valid statuses are \"open\" and \"done\" - all list methods return tasks sorted by id - count_open and count_done use explicit loops - titles returns task titles sorted by task id - to_dicts returns deterministic dictionaries sorted by id - to_dicts includes id, title, status, and created_at keys for every task - raise ValueError for empty title or missing task id - use straightforward if statements and explicit loops - keep method bodies short and similar in style - no argparse, no JSON, no file IO, no unittest, no pytest - target about 110 to 132 lines of code - define __all__ = [\"Task\", \"TaskStore\"] - stop immediately after defining __all__"

# Model definitions
declare -A MODEL_FILES
MODEL_FILES[e2b]="gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
MODEL_FILES[e4b]="gemma-4-E4B-it-Q4_K_M.gguf"
MODEL_FILES[4b]="MoQ-3.75.gguf"
MODEL_FILES[9b]="Qwen3.5-9B-MoQ-3.6.gguf"
MODEL_FILES[12b]="gemma-4-12b-it-qat-q4_0.gguf"

declare -A MODEL_NAMES
MODEL_NAMES[e2b]="Gemma4-E2B"
MODEL_NAMES[e4b]="Gemma4-E4B"
MODEL_NAMES[4b]="Qwen3.5-4B"
MODEL_NAMES[9b]="Qwen3.5-9B"
MODEL_NAMES[12b]="Gemma4-12B"

declare -A MODEL_PARAMS
MODEL_PARAMS[e2b]="4650000000"
MODEL_PARAMS[e4b]="7520000000"
MODEL_PARAMS[4b]="4210000000"
MODEL_PARAMS[9b]="8950000000"
MODEL_PARAMS[12b]="12000000000"

declare -A MODEL_MMPROJ
MODEL_MMPROJ[e2b]="--no-mmproj"
MODEL_MMPROJ[e4b]="--no-mmproj"
MODEL_MMPROJ[4b]="--jinja --parallel-tool-calls"
MODEL_MMPROJ[9b]="--jinja --parallel-tool-calls"
MODEL_MMPROJ[12b]="--mmproj ${MODELS_DIR}/mmproj-gemma-4-12b-it-qat-q4_0.gguf"

declare -A MODEL_EXTRA
MODEL_EXTRA[e2b]="--no-mmproj"
MODEL_EXTRA[e4b]="--no-mmproj"
MODEL_EXTRA[4b]="--jinja --parallel-tool-calls"
MODEL_EXTRA[9b]="--jinja --parallel-tool-calls"
MODEL_EXTRA[12b]="--mmproj ${MODELS_DIR}/mmproj-gemma-4-12b-it-qat-q4_0.gguf --jinja --reasoning on --min-p 0.01 --repeat-penalty 1.05"

stop_server() {
    echo "  Stopping server on port $PORT..."
    pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
    sleep 3
    # Verify it's really gone
    if curl -s --max-time 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "  WARNING: Server still alive, force killing..."
        pkill -9 -f "llama-server.*--port ${PORT}" 2>/dev/null || true
        sleep 2
    fi
}

get_vram() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

get_ram() {
    free -m | awk '/Mem:/{printf "%.0f", $3}'
}

get_metrics() {
    # Get VRAM and RAM usage
    local vram=$(get_vram)
    local ram=$(get_ram)
    echo "vram_mb=${vram} ram_mb=${ram}"
}

run_benchmark() {
    local model_key=$1
    local backend=$2  # "ik" or "bee"
    local server_bin=$3
    local cache_k=$4
    local cache_v=$5
    local extra_cache_args=$6

    local model_file="${MODEL_FILES[$model_key]}"
    local model_name="${MODEL_NAMES[$model_key]}"
    local model_params="${MODEL_PARAMS[$model_key]}"
    local model_extra="${MODEL_EXTRA[$model_key]}"

    local label="${model_key}_${backend}"
    local result_file="${RESULTS_DIR}/${label}.json"

    echo ""
    echo "============================================"
    echo "  BENCHMARK: ${model_name} (${backend})"
    echo "  Cache: K=${cache_k} V=${cache_v}"
    echo "============================================"

    # Stop any existing server
    stop_server

    # Record baseline VRAM/RAM before loading
    local vram_before=$(get_vram)
    local ram_before=$(get_ram)
    echo "  VRAM before: ${vram_before} MB, RAM before: ${ram_before} MB"

    # Build server command
    local cmd="${server_bin} --port ${PORT}"
    cmd+=" --model ${MODELS_DIR}/${model_file}"
    cmd+=" --n-gpu-layers 99"
    cmd+=" --ctx-size 65536"  # 64K for benchmark (faster startup)
    cmd+=" --timeout 0"
    cmd+=" --temp 0.7"
    cmd+=" --top-p 0.9"
    cmd+=" --top-k 40"
    cmd+=" --min-p 0.01"
    cmd+=" --repeat-penalty 1.0"
    cmd+=" --reasoning on"
    cmd+=" --cache-type-k ${cache_k}"
    cmd+=" --cache-type-v ${cache_v}"
    cmd+=" ${extra_cache_args}"
    cmd+=" --flash-attn on"
    cmd+=" --metrics"
    cmd+=" ${model_extra}"
    echo "  Starting server..."
    echo "  CMD: ${cmd}" | head -1
    eval "$cmd" &>/dev/null &
    local server_pid=$!

    # Wait for server to be ready
    echo "  Waiting for model to load..."
    local max_wait=180
    local waited=0
    while ! curl -s --max-time 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: Server failed to start within ${max_wait}s"
            kill $server_pid 2>/dev/null || true
            echo '{"error": "server_timeout"}' > "$result_file"
            return 1
        fi
    done
    echo "  Server ready after ${waited}s (PID: ${server_pid})"

    # Record VRAM/RAM after loading
    sleep 2  # Let VRAM settle
    local vram_after_load=$(get_vram)
    local ram_after_load=$(get_ram)
    echo "  VRAM after load: ${vram_after_load} MB, RAM after load: ${ram_after_load} MB"

    # Run Snake benchmark
    echo "  Running Snake benchmark..."
    local start_time=$(date +%s%N)
    local response=$(curl -s --max-time 300 "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model_key}\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$SNAKE_PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}],
            \"temperature\": 0.0,
            \"max_tokens\": 4096
        }")
    local end_time=$(date +%s%N)

    # Parse response
    local completion_tokens=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

    local prompt_tokens=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('usage', {}).get('prompt_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

    local content=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except:
    print('')
" 2>/dev/null || echo "")

    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))
    local tok_per_sec=0
    if [ "$completion_tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ]; then
        tok_per_sec=$(python3 -c "print(f'{$completion_tokens / ($elapsed_ms / 1000):.1f}')")
    fi

    # Run Snake scoring
    echo "  Scoring Snake benchmark..."
    local snake_score="N/A"
    if [ -f "${SNAKE_DIR}/ENUNCIADO.md" ] && [ -d "${SNAKE_DIR}" ]; then
        echo "$content" > "${RESULTS_DIR}/${label}_output.py"
        # Try to run snake scoring
        if [ -f "${SNAKE_DIR}/score.py" ]; then
            snake_score=$(cd "${SNAKE_DIR}" && python3 score.py "${RESULTS_DIR}/${label}_output.py" 2>/dev/null || echo "N/A")
        else
            snake_score=$(python3 -c "
# Quick line count as proxy
lines = open('${RESULTS_DIR}/${label}_output.py').read().strip().split('\n')
non_empty = [l for l in lines if l.strip() and not l.strip().startswith('#')]
print(f'{len(non_empty)} lines')
" 2>/dev/null || echo "N/A")
        fi
    fi

    # Record final VRAM (after generation)
    local vram_final=$(get_vram)
    local ram_final=$(get_ram)

    # Write results
    python3 -c "
import json
results = {
    'model': '${model_name}',
    'model_key': '${model_key}',
    'backend': '${backend}',
    'cache_k': '${cache_k}',
    'cache_v': '${cache_v}',
    'params': '${model_params}',
    'vram_before_mb': ${vram_before},
    'vram_after_load_mb': ${vram_after_load},
    'vram_after_gen_mb': ${vram_final},
    'ram_before_mb': ${ram_before},
    'ram_after_load_mb': ${ram_after_load},
    'ram_after_gen_mb': ${ram_final},
    'vram_delta_load_mb': ${vram_after_load} - ${vram_before},
    'vram_delta_gen_mb': ${vram_final} - ${vram_after_load},
    'prompt_tokens': ${prompt_tokens},
    'completion_tokens': ${completion_tokens},
    'elapsed_ms': ${elapsed_ms},
    'tok_per_sec': '${tok_per_sec}',
    'snake_score': '${snake_score}',
    'ctx_size': 65536,
}
with open('${result_file}', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
"

    echo "  Done: ${tok_per_sec} tok/s, Snake: ${snake_score}"
    echo "  VRAM: ${vram_before} -> ${vram_after_load} (load) -> ${vram_final} (after gen)"

    # Stop server
    stop_server

    # Wait for VRAM to free
    sleep 5
}

# Main
TARGETS=()
if [ -z "${1:-}" ] || [ "$1" = "all" ]; then
    TARGETS=(e2b e4b 4b 9b 12b)
else
    TARGETS=("$1")
fi

echo "TurboQuant Benchmark — $(date)"
echo "Models: ${TARGETS[*]}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Make sure llama-swap is stopped so we have full VRAM
echo "Pausing llama-swap..."
curl -s http://localhost:${LLAMASWAP_PORT}/v1/models 2>/dev/null | python3 -c "import sys,json; [print(f'  Active: {m[\"id\"]}') for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null || echo "  llama-swap not running"
echo ""

for model_key in "${TARGETS[@]}"; do
    echo ""
    echo "========================================"
    echo "  MODEL: ${MODEL_NAMES[$model_key]}"
    echo "========================================"

    # Run ik benchmark (q4_0/q8_0 + hadamard)
    run_benchmark "$model_key" "ik" "$IK_SERVER" "q8_0" "q4_0" "--k-cache-hadamard --v-cache-hadamard"

    # Cool down
    echo "  Cooling down (10s)..."
    sleep 10

    # Run bee benchmark (turbo3_tcq)
    run_benchmark "$model_key" "bee" "$BEE_SERVER" "turbo3_tcq" "turbo3_tcq" ""

    # Cool down between models
    echo "  Cooling down (15s)..."
    sleep 15
done

echo ""
echo "============================================"
echo "  BENCHMARKS COMPLETE"
echo "============================================"
echo "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"*.json 2>/dev/null

echo ""
echo "Summary:"
for model_key in "${TARGETS[@]}"; do
    for backend in ik bee; do
        f="${RESULTS_DIR}/${model_key}_${backend}.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
print(f\"  {d['model']:20s} {d['backend']:5s}: {d.get('tok_per_sec','?'):>6s} tok/s, VRAM load={d.get('vram_delta_load_mb','?'):>5s} MB, Snake={d.get('snake_score','?')}\")
" 2>/dev/null || echo "  $f: ERROR"
        fi
    done
done