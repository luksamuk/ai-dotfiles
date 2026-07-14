#!/usr/bin/env python3
"""Bonsai 27B Code Benchmark — Snake prompt via llama-swap API.
Runs the Snake benchmark (structured Python code generation) against Bonsai 27B Ternary.
Also runs against comparison models if specified."""
import json, time, os, sys, requests

SWAP_URL = "http://localhost:12434/v1/chat/completions"
RESULTS_DIR = os.path.expanduser("~/bonsai-bench-results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SNAKE_PROMPT = """Write one complete Python 3 file using only the standard library. Return only Python code. Do not use markdown, comments, tests, examples, or explanatory text. Implement a deterministic Task store module with a compact, repetitive structure that is easy to predict. Required shape: - imports: dataclasses, datetime, typing - dataclass Task with fields id: int, title: str, status: str, created_at: str - class TaskStore with an internal dict[int, Task] - methods: add, get, rename, mark_done, reopen, delete, clear, list_all, list_open, list_done, count_open, count_done, titles, to_dicts, __len__, __contains__ - add assigns increasing integer ids starting at 1 - valid statuses are "open" and "done" - all list methods return tasks sorted by id - count_open and count_done use explicit loops - titles returns task titles sorted by task id - to_dicts returns deterministic dictionaries sorted by id - to_dicts includes id, title, status, and created_at keys for every task - raise ValueError for empty title or missing task id - use straightforward if statements and explicit loops - keep method bodies short and similar in style - no argparse, no JSON, no file IO, no unittest, no pytest - target about 110 to 132 lines of code - define __all__ = ["Task", "TaskStore"] - stop immediately after defining __all__"""

def run_benchmark(model_key, max_tokens=4096, timeout=600, think=False):
    model_id = f"{model_key}:think" if think else model_key
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": SNAKE_PROMPT}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    print(f"  Sending request to {model_id}...")
    start = time.time()
    try:
        r = requests.post(SWAP_URL, json=payload, timeout=timeout)
        elapsed = time.time() - start
        d = r.json()
    except Exception as e:
        return {"error": str(e), "elapsed_s": round(time.time() - start, 1)}

    usage = d.get("usage", {})
    msg = d.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")

    # If thinking mode and content is empty, use reasoning
    if not content.strip() and reasoning.strip():
        content = reasoning

    prompt_tok = usage.get("prompt_tokens", 0)
    comp_tok = usage.get("completion_tokens", 0)

    # Count lines of code (non-empty)
    lines = len([l for l in content.split('\n') if l.strip()])
    tok_sec = round(comp_tok / elapsed, 1) if comp_tok > 0 and elapsed > 0 else 0

    # Basic quality checks
    has_dataclass = "dataclass" in content
    has_task_store = "TaskStore" in content
    has_all = "__all__" in content
    has_add = "def add" in content
    has_get = "def get" in content
    has_delete = "def delete" in content
    has_mark_done = "def mark_done" in content
    has_list_all = "def list_all" in content
    has_count_open = "def count_open" in content
    has_to_dicts = "def to_dicts" in content
    has_contains = "__contains__" in content
    has_len = "__len__" in content

    # Count methods implemented
    methods = sum([
        has_add, has_get, "def rename" in content, has_mark_done,
        "def reopen" in content, has_delete, "def clear" in content,
        has_list_all, "def list_open" in content, "def list_done" in content,
        has_count_open, "def count_done" in content, "def titles" in content,
        has_to_dicts, has_len, has_contains
    ])

    # Check for markdown (should have none)
    has_markdown = "```" in content

    result = {
        "model": model_id,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "lines": lines,
        "elapsed_s": round(elapsed, 1),
        "tok_per_sec": tok_sec,
        "methods_implemented": methods,
        "methods_total": 16,
        "has_dataclass": has_dataclass,
        "has_task_store": has_task_store,
        "has_all": has_all,
        "has_markdown": has_markdown,
        "content": content,
        "reasoning_tokens": len(reasoning.split()) if reasoning else 0,
    }
    return result

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "bonsai-27b-ternary"
    think = "--think" in sys.argv
    max_tokens = 4096
    if "--max-tokens" in sys.argv:
        idx = sys.argv.index("--max-tokens")
        max_tokens = int(sys.argv[idx + 1])

    models_to_test = [target]
    if "--all" in sys.argv:
        models_to_test = ["bonsai-27b-ternary", "qwen3.6-35b-a3b", "agents-a1-35b", "glm-4.7-flash"]

    print(f"\n{'='*70}")
    print(f"  SNAKE BENCHMARK — Structured Python Code Generation")
    print(f"  Models: {', '.join(models_to_test)}{' (thinking mode)' if think else ''}")
    print(f"  Max tokens: {max_tokens}")
    print(f"{'='*70}")

    all_results = []

    for model_key in models_to_test:
        print(f"\n--- {model_key}{' (think)' if think else ''} ---")
        result = run_benchmark(model_key, max_tokens=max_tokens, think=think)
        all_results.append(result)

        # Save individual result
        save_result = {k: v for k, v in result.items() if k not in ("content",)}
        out_file = os.path.join(RESULTS_DIR, f"{model_key}{'_think' if think else ''}_snake.json")
        with open(out_file, 'w') as f:
            json.dump(save_result, f, indent=2)

        # Save code output
        code_file = os.path.join(RESULTS_DIR, f"{model_key}{'_think' if think else ''}_snake.py")
        content = result.get("content", "")
        if content:
            with open(code_file, 'w') as f:
                f.write(content)

        if "error" in result:
            print(f"  ERROR: {result['error'][:200]}")
        else:
            print(f"  Speed: {result['tok_per_sec']} tok/s")
            print(f"  Tokens: {result['completion_tokens']} ({result['prompt_tokens']} prompt)")
            print(f"  Lines: {result['lines']}")
            print(f"  Time: {result['elapsed_s']}s")
            print(f"  Methods: {result['methods_implemented']}/{result['methods_total']}")
            print(f"  Has markdown: {result['has_markdown']}")
            print(f"  Reasoning tokens: {result.get('reasoning_tokens', 0)}")

        # Cooldown between models
        if model_key != models_to_test[-1]:
            print("  Cooling down 10s...")
            time.sleep(10)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'tok/s':>7} {'Lines':>6} {'Tokens':>7} {'Time':>6} {'Methods':>8} {'Markdown':>10}")
    print("-" * 85)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<30} {'ERROR':>7}")
        else:
            print(f"{r['model']:<30} {r['tok_per_sec']:>7} {r['lines']:>6} {r['completion_tokens']:>7} {r['elapsed_s']:>6} {r['methods_implemented']:>4}/{r['methods_total']:<3} {'YES' if r['has_markdown'] else 'NO':>10}")

    print(f"\n  Results saved to: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()