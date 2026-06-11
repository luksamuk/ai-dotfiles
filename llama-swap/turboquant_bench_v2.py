#!/usr/bin/env python3
"""TurboQuant Benchmark v2 — ik/q4_0+hadamard vs bee/turbo3_tcq for dense models.
Key constraint: turbo3_tcq requires ALL KV cache on GPU (no CPU offload).
Models that don't fit entirely in VRAM cannot use turbo3_tcq."""
import json, subprocess, time, os, sys, requests

RESULTS_DIR = os.path.expanduser("~/turboquant-bench-results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS_DIR = os.path.expanduser("~/.llama-models")
IK_SERVER = os.path.expanduser("~/git/ik_llama.cpp/build/bin/llama-server")
UPSTREAM_SERVER = os.path.expanduser("~/git/llama.cpp/build/bin/llama-server")
BEE_SERVER = os.path.expanduser("~/git/beellama.cpp/build/bin/llama-server")
PORT = 19999
CTX = 32768  # 32K context for faster benchmarks

SNAKE_PROMPT = """Write one complete Python 3 file using only the standard library. Return only Python code. Do not use markdown, comments, tests, examples, or explanatory text. Implement a deterministic Task store module with a compact, repetitive structure that is easy to predict. Required shape: - imports: dataclasses, datetime, typing - dataclass Task with fields id: int, title: str, status: str, created_at: str - class TaskStore with an internal dict[int, Task] - methods: add, get, rename, mark_done, reopen, delete, clear, list_all, list_open, list_done, count_open, count_done, titles, to_dicts, __len__, __contains__ - add assigns increasing integer ids starting at 1 - valid statuses are "open" and "done" - all list methods return tasks sorted by id - count_open and count_done use explicit loops - titles returns task titles sorted by task id - to_dicts returns deterministic dictionaries sorted by id - to_dicts includes id, title, status, and created_at keys for every task - raise ValueError for empty title or missing task id - use straightforward if statements and explicit loops - keep method bodies short and similar in style - no argparse, no JSON, no file IO, no unittest, no pytest - target about 110 to 132 lines of code - define __all__ = ["Task", "TaskStore"] - stop immediately after defining __all__"""

BENCHES = [
    # E2B — smallest, fastest to bench
    {
        "key": "gemma4-e2b", "name": "Gemma 4 E2B",
        "model_file": f"{MODELS_DIR}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "configs": [
            ("upstream_q4_0", UPSTREAM_SERVER, "q4_0", "q4_0",
             ["--no-mmproj", "--fit", "on", "--fit-ctx", "4096",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "40",
              "--min-p", "0.01", "--repeat-penalty", "1.0"]),
            ("bee_turbo3_tcq", BEE_SERVER, "turbo3_tcq", "turbo3_tcq",
             ["--no-mmproj", "--fit", "on", "--fit-ctx", "4096",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "40",
              "--min-p", "0.01", "--repeat-penalty", "1.0"]),
        ],
    },
    # 4B — MoQ quant, very fast
    {
        "key": "qwen3.5-4b", "name": "Qwen 3.5 4B",
        "model_file": f"{MODELS_DIR}/MoQ-3.75.gguf",
        "configs": [
            ("ik_q8_q4_hadamard", IK_SERVER, "q8_0", "q4_0",
             ["--jinja", "--parallel-tool-calls", "--fit", "--fit-margin", "512",
              "--reasoning", "on",
              "--k-cache-hadamard", "--v-cache-hadamard",
              "--flash-attn", "auto",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "20",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
            ("bee_turbo3_tcq", BEE_SERVER, "turbo3_tcq", "turbo3_tcq",
             ["--jinja", "--fit", "on", "--fit-ctx", "4096",
              "--reasoning", "on", "--flash-attn", "on",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "20",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
        ],
    },
    # E4B — medium size
    {
        "key": "gemma4-e4b", "name": "Gemma 4 E4B",
        "model_file": f"{MODELS_DIR}/gemma-4-E4B-it-Q4_K_M.gguf",
        "configs": [
            ("ik_q8_q4_hadamard", IK_SERVER, "q8_0", "q4_0",
             ["--fit", "--fit-margin", "512",
              "--jinja", "--parallel-tool-calls", "--reasoning", "on",
              "--k-cache-hadamard", "--v-cache-hadamard",
              "--flash-attn", "auto",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "40",
              "--min-p", "0.01", "--repeat-penalty", "1.0"]),
            ("bee_turbo3_tcq", BEE_SERVER, "turbo3_tcq", "turbo3_tcq",
             ["--no-mmproj", "--fit", "on", "--fit-ctx", "4096",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "40",
              "--min-p", "0.01", "--repeat-penalty", "1.0"]),
        ],
    },
    # 9B — larger dense
    {
        "key": "qwen3.5-9b", "name": "Qwen 3.5 9B",
        "model_file": f"{MODELS_DIR}/Qwen3.5-9B-MoQ-3.6.gguf",
        "configs": [
            ("ik_q8_q4_hadamard", IK_SERVER, "q8_0", "q4_0",
             ["--jinja", "--parallel-tool-calls", "--fit", "--fit-margin", "512",
              "--reasoning", "on",
              "--k-cache-hadamard", "--v-cache-hadamard",
              "--flash-attn", "auto",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "20",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
            ("bee_turbo3_tcq", BEE_SERVER, "turbo3_tcq", "turbo3_tcq",
             ["--jinja", "--fit", "on", "--fit-ctx", "4096",
              "--reasoning", "on", "--flash-attn", "on",
              "--temp", "0.7", "--top-p", "0.9", "--top-k", "20",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
        ],
    },
    # 12B — NO mmproj (text-only), turbo3_tcq might be tight
    {
        "key": "gemma4-12b", "name": "Gemma 4 12B (text-only)",
        "model_file": f"{MODELS_DIR}/gemma-4-12b-it-qat-q4_0.gguf",
        "configs": [
            ("upstream_q4_0_mmproj", UPSTREAM_SERVER, "q4_0", "q4_0",
             ["--mmproj", f"{MODELS_DIR}/mmproj-gemma-4-12b-it-qat-q4_0.gguf",
              "--fit", "on", "--fit-target", "768", "--fit-ctx", "4096",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
            ("bee_turbo3_tcq_nomproj", BEE_SERVER, "turbo3_tcq", "turbo3_tcq",
             ["--no-mmproj", "--fit", "on", "--fit-ctx", "4096",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
        ],
    },
]

def get_vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True)
    return int(r.stdout.strip().split('\n')[0])

def kill_servers():
    subprocess.run(["pkill", "-9", "-f", "llama-server.*--port"], capture_output=True)
    time.sleep(5)

def start_server(server_bin, port, model_file, cache_k, cache_v, extra_args, ctx=32768):
    cmd = [server_bin, "--port", str(port), "--model", model_file,
            "--ctx-size", str(ctx), "--timeout", "0",
            "--cache-type-k", cache_k, "--cache-type-v", cache_v,
            "--metrics"] + extra_args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc

def wait_for_server(port, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200 and "ok" in r.text.lower():
                return True
        except:
            pass
        time.sleep(2)
    return False

def run_benchmark(port, model_key, max_tokens=4096, timeout=600):
    payload = {
        "model": model_key,
        "messages": [{"role": "user", "content": SNAKE_PROMPT}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    start = time.time()
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions",
                         json=payload, timeout=timeout)
        elapsed = time.time() - start
        d = r.json()
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.time() - start}

    usage = d.get("usage", {})
    msg = d.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    if not content.strip() and reasoning.strip():
        content = reasoning

    prompt_tok = usage.get("prompt_tokens", 0)
    comp_tok = usage.get("completion_tokens", 0)
    lines = len([l for l in content.split('\n') if l.strip()])
    tok_sec = round(comp_tok / elapsed, 1) if comp_tok > 0 and elapsed > 0 else 0

    return {
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "lines": lines,
        "elapsed_s": round(elapsed, 1),
        "tok_per_sec": tok_sec,
        "content": content,
    }

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    for bench in BENCHES:
        if target != "all" and bench["key"] != target:
            continue

        print(f"\n{'='*60}")
        print(f"  MODEL: {bench['name']} ({bench['key']})")
        print(f"{'='*60}")

        for label, server_bin, cache_k, cache_v, extra_args in bench["configs"]:
            print(f"\n--- {label}: cache_k={cache_k}, cache_v={cache_v} ---")

            kill_servers()
            vram_before = get_vram()
            print(f"  VRAM before: {vram_before} MiB")

            proc = start_server(server_bin, PORT, bench["model_file"], cache_k, cache_v, extra_args, CTX)

            if not wait_for_server(PORT, timeout=180):
                print(f"  ERROR: Server failed to start!")
                proc.kill()
                # Print error log
                err = proc.stderr.read().decode() if proc.stderr else ""
                print(f"  Last lines: {err[-500:]}")
                continue

            time.sleep(2)
            vram_loaded = get_vram()
            print(f"  VRAM after load: {vram_loaded} MiB (delta: {vram_loaded - vram_before} MiB)")

            print(f"  Running Snake benchmark (max_tokens=4096, ctx={CTX})...")
            result = run_benchmark(PORT, bench["key"], max_tokens=4096, timeout=600)
            vram_after = get_vram()

            result.update({
                "model": bench["name"], "model_key": bench["key"],
                "backend": label, "cache_k": cache_k, "cache_v": cache_v,
                "vram_before_mb": vram_before, "vram_loaded_mb": vram_loaded,
                "vram_after_gen_mb": vram_after,
                "vram_delta_load_mb": vram_loaded - vram_before,
                "vram_delta_gen_mb": vram_after - vram_loaded,
                "ctx_size": CTX,
            })

            # Remove content from JSON (too large)
            result_save = {k:v for k,v in result.items() if k != "content"}

            out_file = os.path.join(RESULTS_DIR, f"{bench['key']}_{label}.json")
            with open(out_file, 'w') as f:
                json.dump(result_save, f, indent=2)

            code_file = os.path.join(RESULTS_DIR, f"{bench['key']}_{label}_output.py")
            content = result.get("content", "")
            if content:
                with open(code_file, 'w') as f:
                    f.write(content)

            print(f"  Result: {result_save.get('tok_per_sec', 'ERR')} tok/s, "
                  f"{result_save.get('lines', '?')} lines, "
                  f"{result_save.get('completion_tokens', '?')} tokens, "
                  f"{result_save.get('elapsed_s', '?')}s")
            if "error" in result_save:
                print(f"  ERROR: {result_save['error'][:200]}")

            proc.kill()
            proc.wait(timeout=10)

            print("  Cooling down (10s)...")
            time.sleep(10)

    print(f"\n{'='*60}")
    print("  BENCHMARKS COMPLETE")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*60}")

    # Print summary
    print(f"\n{'Model':<22} {'Backend':<22} {'tok/s':>7} {'Lines':>6} {'Tokens':>7} {'Time':>6} {'VRAM':>6} {'dGen':>6}")
    print("-" * 90)
    for bench in BENCHES:
        if target != "all" and bench["key"] != target:
            continue
        for label, _, _, _, _ in bench["configs"]:
            f = os.path.join(RESULTS_DIR, f"{bench['key']}_{label}.json")
            if os.path.exists(f):
                d = json.load(open(f))
                print(f"{d.get('model','?'):<22} {label:<22} {d.get('tok_per_sec','ERR'):>7} {d.get('lines','?'):>6} {d.get('completion_tokens','?'):>7} {d.get('elapsed_s','?'):>6} {d.get('vram_loaded_mb','?'):>6} {d.get('vram_delta_gen_mb','?'):>6}")
            else:
                print(f"{bench['name']:<22} {label:<22} {'MISSING':>7}")

if __name__ == "__main__":
    main()