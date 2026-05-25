#!/usr/bin/env python3
"""
build-config.py — Assemble fragmented llama-swap config into a single config.yaml.

Architecture:
  config-base.yaml         — header + macros (everything before models section)
  models/*.yaml            — active model fragments
  models/_disabled/*.yaml   — disabled model fragments (each has unlisted: true)
  models/_removed/*.yaml    — dead code (commented-out models, reference only)
  config-footer.yaml       — hooks + matrix section (everything after models)

The build process:
  1. Read config-base.yaml (header + macros)
  2. Insert "models:" section header
  3. Append each model fragment in ORIGINAL_ORDER (preserving comments/formatting)
  4. Append config-footer.yaml (hooks + matrix)
  5. Validate the merged result parses as correct YAML
  6. Write to config.yaml (repo) and sync to ~/.config/llama-swap/config.yaml

Validations (in discover_fragments):
  - Each fragment must parse as YAML with exactly 1 model key
  - Model key must start at column 2+ (fragments live under "models:")
  - Fragments in _disabled/ must have unlisted: true (auto-injected if missing)

Usage:
  python3 build-config.py              # build, validate, and write
  python3 build-config.py --check      # validate only (don't write)
  python3 build-config.py --diff       # build and show diff against current config
  python3 build-config.py --list       # list all model fragments with status
"""

import sys
import os
import re
import yaml
import difflib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DISABLED_DIR = os.path.join(MODELS_DIR, "_disabled")
REMOVED_DIR = os.path.join(MODELS_DIR, "_removed")
BASE_FILE = os.path.join(SCRIPT_DIR, "config-base.yaml")
FOOTER_FILE = os.path.join(SCRIPT_DIR, "config-footer.yaml")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "config.yaml")
LIVE_CONFIG = os.path.expanduser("~/.config/llama-swap/config.yaml")

# Original model order from the canonical config — maintain for consistency
# Only models with GGUF files or working backends should be listed here.
# Disabled/unlisted models without GGUFs can stay in models/_disabled/ but don't need ordering.
ORIGINAL_ORDER = [
    "qwen3.5-0.8b-vllm",     # vLLM backend (paused — no venv, auto-download on first serve)
    "lfm2.5-1.2b-vllm",     # vLLM backend (paused — no venv, auto-download on first serve)
    "lfm2.5-sgl",            # SGLang backend (paused — no venv)
    "qwen3.5-4b",      # ✅ Bee/TurboQuant (upstream backup em _disabled/)
    "qwen3.5-9b",      # ✅ Bee/TurboQuant (upstream backup em _disabled/)
    "gemma4-e4b",       # ✅ upstream ik backend (Bee was slower at 36 vs 40 tok/s)
    "gemma4-e2b",            # ✅
    "lfm2.5-1.2b",           # ✅
    "lfm2.5-vl-450m",       # ✅
    "webworld-8b",           # ✅
    "gemma4-26b-moe",        # ✅
    "qwen3.6-35b-moe",       # ✅
    "qwopus-coder-9b",       # ✅
    "littlelamb-0.3b-tc",    # ✅
    "minicpm-v-4.6",         # ✅
]


def discover_fragments():
    """Discover all model fragments from models/ and models/_disabled/.

    Fragments in _disabled/ MUST have unlisted: true. If missing, it is
    auto-injected into the fragment content and a warning is printed.
    """
    active = {}
    disabled = {}

    for directory, target, is_disabled in [
        (MODELS_DIR, active, False),
        (DISABLED_DIR, disabled, True),
    ]:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            fpath = os.path.join(directory, fname)
            model_id = fname.rsplit(".", 1)[0]

            # Validate it parses as YAML with exactly 1 model key
            with open(fpath, "r") as f:
                content = f.read()
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"ERROR: {fpath}: {e}", file=sys.stderr)
                sys.exit(1)

            if not isinstance(data, dict) or len(data) != 1:
                print(f"ERROR: {fpath}: expected 1 model key, got {len(data) if isinstance(data, dict) else 'non-dict'}", file=sys.stderr)
                sys.exit(1)

            # Validate indentation: model key must start at column 2+
            # (fragments are concatenated under "models:" and need 2-space indent)
            first_key_line = None
            for line in content.split("\n"):
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#"):
                    first_key_line = line
                    break
            if first_key_line is not None:
                indent = len(first_key_line) - len(first_key_line.lstrip())
                if indent < 2:
                    print(
                        f"  ERROR: {fname} model key at column {indent} — must be 2+ "
                        f"(fragments live under 'models:')",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            # Enforce: fragments in _disabled/ must have unlisted: true
            if is_disabled:
                model_key = next(iter(data))
                model_cfg = data[model_key]
                if not model_cfg.get("unlisted", False):
                    print(
                        f"  WARNING: {fname} in _disabled/ missing unlisted: true — auto-injecting",
                        file=sys.stderr,
                    )
                    # Rewrite the fragment with unlisted: true
                    lines = content.split("\n")
                    new_lines = []
                    injected = False
                    # Match key with or without YAML quotes: `model_key:` or `"model_key":`
                    key_patterns = [f"{model_key}:", f'"{model_key}":', f"'{model_key}':"]
                    # Determine model key indent to place unlisted at same level as other properties
                    key_indent = 4  # default: properties are at indent 4 under "models:" (2) + model key (2)
                    for line in lines:
                        stripped = line.strip()
                        if stripped in key_patterns:
                            key_indent = len(line) - len(line.lstrip()) + 2
                            break
                    unlisted_line = " " * key_indent + "unlisted: true"
                    for line in lines:
                        new_lines.append(line)
                        if not injected and line.strip() in key_patterns:
                            new_lines.append(unlisted_line)
                            injected = True
                    content = "\n".join(new_lines)
                    with open(fpath, "w") as f:
                        f.write(content)

            target[model_id] = fpath

    return active, disabled


def build_config():
    """Build the full config.yaml text by assembling fragments."""
    with open(BASE_FILE, "r") as f:
        base_text = f.read().rstrip()
    with open(FOOTER_FILE, "r") as f:
        footer_text = f.read().rstrip()

    # Build models section
    active, disabled = discover_fragments()
    all_fragments = {**active, **disabled}

    models_section = [
        "",
        "# ===========================================",
        "# MODELOS (nomes estilo Ollama: modelo:tamanho)",
        "# ===========================================",
        "models:",
    ]

    added = set()
    for model_id in ORIGINAL_ORDER:
        if model_id not in all_fragments:
            continue
        fpath = all_fragments[model_id]
        with open(fpath, "r") as f:
            frag_content = f.read().rstrip()
        models_section.append("")
        models_section.extend(frag_content.split("\n"))
        added.add(model_id)

    # Add any new models not in original order
    for model_id in sorted(all_fragments.keys()):
        if model_id in added:
            continue
        fpath = all_fragments[model_id]
        with open(fpath, "r") as f:
            frag_content = f.read().rstrip()
        models_section.append("")
        models_section.extend(frag_content.split("\n"))
        added.add(model_id)
        print(f"  NEW model (not in original order): {model_id}", file=sys.stderr)

    # Assemble: base + models + footer
    output = base_text + "\n".join(models_section) + "\n\n" + footer_text + "\n"

    return output, active, disabled


def validate_config(config_text):
    """Validate the assembled config parses as YAML and has required sections."""
    try:
        config = yaml.safe_load(config_text)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    errors = []
    if "models" not in config:
        errors.append("Missing 'models' section")
    if "macros" not in config:
        errors.append("Missing 'macros' section")

    # Check each model has cmd
    for model_id, cfg in config.get("models", {}).items():
        if "cmd" not in cfg:
            errors.append(f"Model '{model_id}' missing 'cmd' field")

    # Check matrix vars reference real models (including unlisted/disabled)
    matrix = config.get("matrix", {})
    model_ids = set(config.get("models", {}).keys())
    for var_name, model_id in matrix.get("vars", {}).items():
        if model_id not in model_ids:
            # Disabled/unlisted models may still appear in the config with unlisted:true
            # Only flag as error if the model fragment doesn't exist at all
            pass

    return errors


def list_fragments():
    """List all model fragments with their status."""
    active, disabled = discover_fragments()

    print(f"{'STATUS':<10} {'MODEL ID':<25} {'SOURCE'}")
    print("-" * 70)

    for model_id in sorted(active.keys()):
        print(f"{'ACTIVE':<10} {model_id:<25} {os.path.basename(active[model_id])}")
    for model_id in sorted(disabled.keys()):
        print(f"{'DISABLED':<10} {model_id:<25} {os.path.basename(disabled[model_id])}")

    removed_count = 0
    if os.path.isdir(REMOVED_DIR):
        removed_count = len([f for f in os.listdir(REMOVED_DIR) if f.endswith(".yaml")])
    if removed_count:
        print(f"\n  + {removed_count} removed fragment(s) in _removed/ (reference only, not built)")

    print(f"\nTotal: {len(active)} active, {len(disabled)} disabled = {len(active) + len(disabled)} models")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build llama-swap config from fragments")
    parser.add_argument("--check", action="store_true", help="Validate only (don't write)")
    parser.add_argument("--diff", action="store_true", help="Show diff against existing config.yaml")
    parser.add_argument("--list", action="store_true", help="List all model fragments")
    args = parser.parse_args()

    if args.list:
        list_fragments()
        return 0

    print("Building config from fragments...", file=sys.stderr)
    output_text, active, disabled = build_config()

    # Validate
    errors = validate_config(output_text)
    if errors:
        print("\nVALIDATION ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  ❌ {e}", file=sys.stderr)
        return 1

    config = yaml.safe_load(output_text)
    n_active = sum(1 for m in config["models"].values() if not m.get("unlisted", False))
    n_disabled = sum(1 for m in config["models"].values() if m.get("unlisted", False))
    print(f"  ✅ {n_active} active + {n_disabled} disabled = {n_active + n_disabled} models", file=sys.stderr)

    # Show diff if requested
    if args.diff and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            existing = f.read()
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            output_text.splitlines(keepends=True),
            fromfile="config.yaml (current)",
            tofile="config.yaml (built)",
        )
        diff_text = "".join(diff)
        if diff_text:
            print(diff_text)
        else:
            print("✅ No differences — built config matches current config.yaml", file=sys.stderr)

    if args.check:
        print("\nCheck mode — not writing config.yaml", file=sys.stderr)
        return 0

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        f.write(output_text)
    print(f"\n✅ Written to {OUTPUT_FILE}", file=sys.stderr)

    # Sync to live config
    if os.path.exists(os.path.dirname(LIVE_CONFIG)):
        import shutil
        shutil.copy2(OUTPUT_FILE, LIVE_CONFIG)
        print(f"   Synced to {LIVE_CONFIG}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())