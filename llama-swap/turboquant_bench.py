#!/usr/bin/env python3
"""TurboQuant Benchmark: ik (q4_0/q8_0+hadamard) vs bee (turbo3_tcq) for dense models."""
import json, subprocess, time, os, sys, requests

RESULTS_DIR = os.path.expanduser("~/turboquant-bench-results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS_DIR = os.path.expanduser("~/.llama-models")
IK_SERVER = os.path.expanduser("~/git/ik_llama.cpp/build/bin/llama-server")
UPSTREAM_SERVER = os.path.expanduser("~/git/llama.cpp/build/bin/llama-server")
BEE_SERVER = os.path.expanduser("~/git/beellama.cpp/build/bin/llama-server")
PORT = 19999

SNAKE_PROMPT = """Write one complete Python 3 file using only the standard library. Return only Python code. Do not use markdown, comments, tests, examples, or explanatory text. Implement a deterministic Task store module with a compact, repetitive structure that is easy to predict. Required shape: - imports: dataclasses, datetime, typing - dataclass Task with fields id: int, title: str, status: str, created_at: str - class TaskStore with an internal dict[int, Task] - methods: add, get, rename, mark_done, reopen, delete, clear, list_all, list_open, list_done, count_open, count_done, titles, to_dicts, __len__, __contains__ - add assigns increasing integer ids starting at 1 - valid statuses are "open" and "done" - all list methods return tasks sorted by id - count_open and count_done use explicit loops - titles returns task titles sorted by task id - to_dicts returns deterministic dictionaries sorted by id - to_dicts includes id, title, status, and created_at keys for every task - raise ValueError for empty title or missing task id - use straightforward if statements and explicit loops - keep method bodies short and similar in style - no argparse, no JSON, no file IO, no unittest, no pytest - target about 110 to 132 lines of code - define __all__ = ["Task", "TaskStore"] - stop immediately after defining __all__"""

# Bench configs: (key, name, model_file, backend_configs)
# backend_configs: list of (label, server_bin, cache_k, cache_v, extra_args_list)
BENCHES = [
    {
        "key": "gemma4-12b",
        "name": "Gemma 4 12B",
        "model_file": f"{MODELS_DIR}/gemma-4-12b-it-qat-q4_0.gguf",
        "mmproj": f"{MODELS_DIR}/mmproj-gemma-4-12b-it-qat-q4_0.gguf",
        "configs": [
            ("upstream", UPSTREAM_SERVER, "q4_0", "q4_0",
             ["--mmproj", f"{MODELS_DIR}/mmproj-gemma-4-12b-it-qat-q4_0.gguf",
              "--fit", "on", "--fit-target", "768", "--fit-ctx", "8192",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
            ("bee_turbo3_tcq", BEE_SERVER, "turbo3_tcq", "turbo3_tcq",
             ["--mmproj", f"{MODELS_DIR}/mmproj-gemma-4-12b-it-qat-q4_0.gguf",
              "--fit", "on", "--fit-target", "768", "--fit-ctx", "8192",
              "--jinja", "--reasoning", "on", "--flash-attn", "on",
              "--min-p", "0.01", "--repeat-penalty", "1.05"]),
        ],
    },
    {
        "key": "gemma4-e2b",
        "name": "Gemma 4 E2B",
        "model_file": f"{MODELS_DIR}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "mmproj": None,
        "configs": [
            ("upstream", UPSTREAM_SERVER, "q4_0", "q4_0",
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
    {
        "key": "gemma4-e4b",
        "name": "Gemma 4 E4B",
        "model_file": f"{MODELS_DIR}/gemma-4-E4B-it-Q4_K_M.gguf",
        "mmproj": None,
        "configs": [
            ("ik_hadamard", IK_SERVER, "q8_0", "q4_0",
             ["--no-mmproj", "--fit", "--fit-margin", "512",
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
    {
        "key": "qwen3.5-4b",
        "name": "Qwen 3.5 4B",
        "model_file": f"{MODELS_DIR}/MoQ-3.75.gguf",
        "mmproj": None,
        "configs": [
            ("ik_hadamard", IK_SERVER, "q8_0", "q4_0",
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
    {
        "key": "qwen3.5-9b",
        "name": "Qwen 3.5 9B",
        "model_file": f"{MODELS_DIR}/Qwen3.5-9B-MoQ-3.6.gguf",
        "mmproj": None,
        "configs": [
            ("ik_hadamard", IK_SERVER, "q8_0", "q4_0",
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
]

def get_vram():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    return int(result.stdout.strip().split('\n')[0])

def kill_servers():
    subprocess.run(["pkill", "-9", "-f", "llama-server.*--port"], capture_output=True)
    time.sleep(3)

def start_server(server_bin, port, model_file, cache_k, cache_v, extra_args):
    cmd = [server_bin, "--port", str(port), "--model", model_file,
            "--ctx-size", "131072", "--timeout", "0",
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

def run_benchmark(port, model_key, max_tokens=8192):
    """Run the Snake benchmark and return results."""
    payload = {
        "model": model_key,
        "messages": [{"role": "user", "content": SNAKE_PROMPT}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    start = time.time()
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions",
                         json=payload, timeout=900)
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
        "content_preview": content[:500],
    }

def main():
    # If a specific model key is given, only bench that one
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

            print(f"  Starting {server_bin.split('/')[-2]}/{server_bin.split('/')[-1]}...")
            proc = start_server(server_bin, PORT, bench["model_file"], cache_k, cache_v, extra_args)

            if not wait_for_server(PORT, timeout=120):
                print(f"  ERROR: Server failed to start!")
                proc.kill()
                continue

            time.sleep(2)
            vram_loaded = get_vram()
            print(f"  VRAM after load: {vram_loaded} MiB (delta: {vram_loaded - vram_before} MiB)")

            print(f"  Running Snake benchmark...")
            result = run_benchmark(PORT, bench["key"])
            vram_after = get_vram()

            result.update({
                "model": bench["name"],
                "model_key": bench["key"],
                "backend": label,
                "cache_k": cache_k,
                "cache_v": cache_v,
                "vram_before_mb": vram_before,
                "vram_loaded_mb": vram_loaded,
                "vram_after_gen_mb": vram_after,
                "vram_delta_load_mb": vram_loaded - vram_before,
                "vram_delta_gen_mb": vram_after - vram_loaded,
                "ctx_size": 131072,
            })

            # Save
            out_file = os.path.join(RESULTS_DIR, f"{bench['key']}_{label}.json")
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Save output code
            code_file = os.path.join(RESULTS_DIR, f"{bench['key']}_{label}_output.py")
            content = result.get("content_preview", "")
            if content:
                with open(code_file, 'w') as f:
                    f.write(content)

            print(f"  Result: {result.get('tok_per_sec', '?')} tok/s, "
                  f"{result.get('lines', '?')} lines, "
                  f"{result.get('completion_tokens', '?')} tokens, "
                  f"{result.get('elapsed_s', '?')}s")
            print(f"  VRAM: loaded={vram_loaded}MiB, after_gen={vram_after}MiB, "
                  f"delta_gen={vram_after - vram_loaded}MiB")

            proc.kill()
            proc.wait(timeout=10)

            # Cool down
            print("  Cooling down (10s)...")
            time.sleep(10)

    print(f"\n{'='*60}")
    print("  BENCHMARKS COMPLETE")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Model':<22} {'Backend':<18} {'tok/s':>7} {'Lines':>6} {'Tokens':>7} {'Time':>6} {'VRAM_load':>9} {'VRAM_genΔ':>10}")
    print("-" * 100)
    for bench in BENCHES:
        if target != "all" and bench["key"] != target:
            continue
        for label, _, cache_k, _, _ in bench["configs"]:
            f = os.path.join(RESULTS_DIR, f"{bench['key']}_{label}.json")
            if os.path.exists(f):
                d = json.load(open(f))
                print(f"{d.get('model','?'):<22} {label:<18} {d.get('tok_per_sec','?'):>7} {d.get('lines','?'):>6} {d.get('completion_tokens','?'):>7} {d.get('elapsed_s','?'):>6} {d.get('vram_loaded_mb','?'):>9} {d.get('vram_delta_gen_mb','?'):>10}")
            else:
                print(f"{bench['name']:<22} {label:<18} {'N/A':>7}")

if __name__ == "__main__":
    main()