#!/usr/bin/env python3
"""
build-config.py — Assemble fragmented llama-swap config into a single config.yaml.

Architecture:
  config-base.yaml         — header + macros (everything before models section)
  models/*.yaml            — active model fragments
  models/_disabled/*.yaml   — disabled model fragments (excluded from config.yaml)
  models/_removed/*.yaml    — dead code (commented-out models, reference only)
  config-footer.yaml       — hooks + matrix section (everything after models)

The build process (YAML merge):
  1. Parse config-base.yaml as YAML dict (preserving comments via ruamel.yaml)
  2. Parse each model fragment as YAML (preserving scalar styles via ruamel.yaml)
  3. Collect models into ordered dict under "models" key
  4. Parse config-footer.yaml as YAML dict (preserving comments via ruamel.yaml)
  5. Merge: base + models + footer → single document
  6. Validate the merged result parses as correct YAML
  7. Write to config.yaml and sync to live config path

Uses ruamel.yaml round-trip mode to preserve comments, key ordering,
multiline block scalars (|), and quoted string styles. Falls back to
pyyaml when ruamel.yaml is not available (with reduced formatting).

Validations (in discover_fragments):
  - Each fragment must parse as YAML with exactly 1 model key
  - Fragments in _disabled/ are excluded from the generated config.yaml

Usage:
  python3 build-config.py              # build, validate, and write
  python3 build-config.py --check      # validate only (don't write)
  python3 build-config.py --diff       # build and show diff against current config
  python3 build-config.py --list       # list all model fragments with status
"""

import sys
import os
import difflib
import yaml

try:
    from ruamel.yaml import YAML as RuamelYAML
    from ruamel.yaml.comments import CommentedMap
    HAS_RUAMEL = True
except ImportError:
    HAS_RUAMEL = False
    CommentedMap = dict  # type: ignore[misc,assignment]

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
# Disabled models without GGUFs can stay in models/_disabled/ but don't need ordering.
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
    "qwen3.6-35b-a3b",       # ✅
    "ornith-1.0-35b",        # ✅ Post-trained Qwen 3.5 35B MoE (agentic coding RL)
    "nex-n2-mini",           # ✅ Nex-AGI agentic model (Agentic Thinking, Qwen3.5-35B-A3B base)
    "agentworld-35b",         # ✅ World model (7 domains)
    "qwopus-coder-9b",       # ✅
    # "littlelamb-0.3b-tc",   # REMOVED Jun 2026 — tool-calling broken, too small
    "minicpm-v-4.6",         # ✅
    "minicpm5-1b",           # ✅ tool calling fixed (llama.cpp b9833+), dual think/no-think
    "nanbeige4.1-3b",         # ⚠️ verbose reasoning, always thinks, multi-turn tool calls BROKEN (#22684)
    # "mellum2-12b-thinking",   # DISABLED: tool calling broken (no chat_template in GGUF, --tool-call-parser hermes is vLLM-only). Awaiting community GGUF.
    # "ornstein-36-35b",        # DISABLED: Q4_K_M too large (21.7GB) — needs APEX I-Compact conversion (~16.1GB)
]


def _make_ruamel_yaml():
    """Create a ruamel.yaml instance for round-trip preservation."""
    ry = RuamelYAML()
    ry.preserve_quotes = True
    ry.default_flow_style = False
    ry.width = 4096  # Avoid line wrapping
    ry.indent(mapping=2, sequence=2, offset=2)
    return ry


def _deep_copy_commented(obj):
    """Deep copy a ruamel.yaml CommentedMap/CommentedSeq, preserving
    scalar string types (LiteralScalarString, etc.) and comments.
    Plain dicts/lists are returned as-is (caller should not mutate originals).
    """
    if isinstance(obj, CommentedMap):
        new = CommentedMap()
        for k, v in obj.items():
            new[k] = _deep_copy_commented(v)
        # Copy comment info if present
        if hasattr(obj, "ca") and obj.ca.comment:
            new.ca.comment = obj.ca.comment
        return new
    elif isinstance(obj, list):
        # Could be CommentedSeq — just copy items
        return [_deep_copy_commented(item) for item in obj]
    else:
        # Scalars (including LiteralScalarString etc.) are immutable;
        # return as-is so scalar style is preserved in output.
        return obj


def discover_fragments():
    """Discover all model fragments from models/ and models/_disabled/.

    Each fragment is validated as YAML with exactly 1 model key.
    Fragments in _disabled/ are excluded from the generated config.yaml.

    Returns dict of {model_id: fpath} for active and disabled.
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

            # Validate: fragment must parse as YAML with exactly 1 model key
            with open(fpath, "r") as f:
                content = f.read()
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"ERROR: {fpath}: {e}", file=sys.stderr)
                sys.exit(1)

            if not isinstance(data, dict) or len(data) != 1:
                n_keys = len(data) if isinstance(data, dict) else "non-dict"
                print(
                    f"ERROR: {fpath}: expected 1 model key, got {n_keys}",
                    file=sys.stderr,
                )
                sys.exit(1)

            target[model_id] = fpath

    return active, disabled




def load_fragment_roundtrip(fpath):
    """Load a model fragment using ruamel.yaml round-trip mode.

    Preserves scalar styles (quoted strings, literal blocks |) and
    fragment-internal comments. Returns (model_key, model_config)
    where model_config is a CommentedMap.
    """
    ry = _make_ruamel_yaml()
    with open(fpath, "r") as f:
        data = ry.load(f)
    model_key = next(iter(data))
    return model_key, data[model_key]


def load_fragment_plain(fpath):
    """Load a model fragment using pyyaml (fallback).

    Returns (model_key, model_config) as plain dicts.
    """
    with open(fpath, "r") as f:
        content = f.read()
    data = yaml.safe_load(content)
    model_key = next(iter(data))
    return model_key, data[model_key]


def _extract_fragment_comment(fpath):
    """Extract pre-key comment text from a model fragment file.

    When ruamel.yaml loads a fragment, comments before the model key
    are stored as document-level comments (data.ca.comment). This function
    extracts those comments and returns them as plain text (without '#'
    prefix, since ruamel adds that during serialization).

    Returns the comment text string, or None if no comments.
    """
    if not HAS_RUAMEL:
        return None

    ry = _make_ruamel_yaml()
    with open(fpath, "r") as f:
        data = ry.load(f.read())

    if data is None or not hasattr(data, 'ca') or data.ca.comment is None:
        return None

    # Document comments are stored as [None, [CommentToken, ...]]
    doc_comment = data.ca.comment
    if not isinstance(doc_comment, list) or len(doc_comment) < 2 or doc_comment[1] is None:
        return None

    lines = []
    for ct in doc_comment[1]:
        text = ct.value.rstrip('\n')
        # Strip '#' prefix — ruamel adds it during serialization
        if text.startswith('# '):
            text = text[2:]
        elif text.startswith('#'):
            text = text[1:].lstrip()
        else:
            text = text  # blank line or non-comment text
        lines.append(text)

    comment_text = '\n'.join(lines)
    return comment_text if comment_text.strip() else None


def build_config():
    """Build the full config.yaml by parsing and merging YAML dicts.

    Instead of text concatenation, we parse each component as YAML,
    merge the dicts programmatically, and dump the result. This ensures
    structural correctness and eliminates format errors from concatenation.

    When ruamel.yaml is available:
      - Base and footer are parsed in round-trip mode (preserves comments)
      - Fragment model data preserves scalar styles (|, quoted strings)
      - Comments in base/footer survive; comments in fragment files are
        lost (they're ephemeral build artifacts)
    """
    if HAS_RUAMEL:
        # Load base and footer with round-trip to preserve comments
        ry = _make_ruamel_yaml()
        with open(BASE_FILE, "r") as f:
            base_data = ry.load(f)
        with open(FOOTER_FILE, "r") as f:
            footer_data = ry.load(f)

        if base_data is None:
            base_data = CommentedMap()
        if footer_data is None:
            footer_data = CommentedMap()
    else:
        with open(BASE_FILE, "r") as f:
            base_data = yaml.safe_load(f)
        with open(FOOTER_FILE, "r") as f:
            footer_data = yaml.safe_load(f)

        if base_data is None:
            base_data = {}
        if footer_data is None:
            footer_data = {}

    # Build models section by loading each fragment
    active, disabled = discover_fragments()
    # Only include active fragments in config — disabled models stay in _disabled/ YAMLs
    # They're not listed in the API and shouldn't pollute the config.
    all_fragments = active

    # Collect model configs in ORIGINAL_ORDER, then alphabetically for new ones
    models_ordered = []
    added = set()

    load_fn = load_fragment_roundtrip if HAS_RUAMEL else load_fragment_plain

    # Collect fragment comments (pre-key comment headers from each fragment file)
    fragment_comments = {}

    for model_id in ORIGINAL_ORDER:
        if model_id not in all_fragments:
            continue
        model_key, model_cfg = load_fn(all_fragments[model_id])
        models_ordered.append((model_key, model_cfg))
        added.add(model_id)
        # Collect fragment pre-key comments if using ruamel
        if HAS_RUAMEL:
            fragment_comments[model_key] = _extract_fragment_comment(all_fragments[model_id])

    # Add any new models not in original order
    for model_id in sorted(all_fragments.keys()):
        if model_id in added:
            continue
        model_key, model_cfg = load_fn(all_fragments[model_id])
        models_ordered.append((model_key, model_cfg))
        added.add(model_id)
        print(f"  NEW model (not in original order): {model_id}", file=sys.stderr)
        # Collect fragment pre-key comments if using ruamel
        if HAS_RUAMEL:
            fragment_comments[model_key] = _extract_fragment_comment(all_fragments[model_id])

    # Merge into final config
    if HAS_RUAMEL:
        config_data = _merge_ruamel(base_data, models_ordered, footer_data, fragment_comments)
    else:
        config_data = _merge_plain(base_data, models_ordered, footer_data)

    # Serialize
    output = _serialize(config_data)

    return output, active, disabled


def _merge_ruamel(base_data, models_ordered, footer_data, fragment_comments):
    """Merge config parts using ruamel.yaml CommentedMaps.

    Preserves comments from base_data, footer_data, and fragment files.
    Models are inserted in order into a CommentedMap under 'models'.
    Each model config is deep-copied to preserve scalar styles.
    Fragment pre-key comments are attached as before-key comments.
    Footer key-level comments are transferred to the base map.
    """
    import copy

    # Build models section as CommentedMap with ordered keys
    models_map = CommentedMap()
    for model_key, model_cfg in models_ordered:
        if isinstance(model_cfg, CommentedMap):
            models_map[model_key] = _deep_copy_commented(model_cfg)
        elif isinstance(model_cfg, dict):
            models_map[model_key] = CommentedMap(model_cfg)
        else:
            models_map[model_key] = model_cfg

        # Attach fragment pre-key comments as before-key comments on the model
        if model_key in fragment_comments and fragment_comments[model_key]:
            _set_before_key_comment(models_map, model_key, fragment_comments[model_key])

    # Insert models into base
    base_data["models"] = models_map

    # Merge footer keys into base (hooks, matrix, etc.)
    # Transfer footer key-level comments to base
    if isinstance(footer_data, CommentedMap):
        for key in footer_data:
            base_data[key] = footer_data[key]
            # Transfer before-key comments from footer to base
            if hasattr(footer_data, 'ca') and footer_data.ca.items and key in footer_data.ca.items:
                if not hasattr(base_data, 'ca') or base_data.ca.items is None:
                    base_data.ca.items = {}
                base_data.ca.items[key] = copy.deepcopy(footer_data.ca.items[key])

        # Transfer footer document comment to the first footer key in base
        if hasattr(footer_data, 'ca') and footer_data.ca.comment:
            first_key = next(iter(footer_data))
            _merge_document_comment(base_data, first_key, footer_data.ca.comment)
    elif isinstance(footer_data, dict):
        for key in footer_data:
            base_data[key] = footer_data[key]

    return base_data


def _set_before_key_comment(commented_map, key, comment_text):
    """Set a before-key comment on a key in a CommentedMap.

    comment_text should be plain text without '#' prefix —
    ruamel.yaml will add the '#' prefix when serializing.
    """
    commented_map.yaml_set_comment_before_after_key(key, before=comment_text)


def _merge_document_comment(commented_map, key, doc_comment):
    """Merge a document-level comment (from ruamel.yaml parsing) as a
    before-key comment on the specified key in the map.

    Handles the [None, [CommentToken...]] format from ruamel.yaml.
    """
    if doc_comment is None:
        return

    # Extract text from CommentTokens if present
    if isinstance(doc_comment, list) and len(doc_comment) >= 2 and doc_comment[1]:
        comment_tokens = doc_comment[1]
        lines = []
        for ct in comment_tokens:
            text = ct.value.rstrip('\n')
            # Strip '#' prefix since yaml_set_comment_before_after_key adds it
            if text.startswith('# '):
                text = text[2:]
            elif text.startswith('#'):
                text = text[1:].lstrip()
            lines.append(text)
        comment_text = '\n'.join(lines)
        if comment_text:
            commented_map.yaml_set_comment_before_after_key(key, before=comment_text)


def _merge_plain(base_data, models_ordered, footer_data):
    """Merge config parts using plain dicts (fallback)."""
    models_dict = {}
    for model_key, model_cfg in models_ordered:
        models_dict[model_key] = model_cfg

    merged = dict(base_data)
    merged["models"] = models_dict
    if footer_data:
        for key, value in footer_data.items():
            merged[key] = value
    return merged


def _serialize(config_data):
    """Serialize config data to YAML string.

    Uses ruamel.yaml for round-trip preservation if available,
    otherwise falls back to pyyaml.
    """
    if HAS_RUAMEL:
        import io
        ry = _make_ruamel_yaml()
        stream = io.StringIO()
        ry.dump(config_data, stream)
        yaml_output = stream.getvalue()
    else:
        yaml_output = yaml.dump(
            config_data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=4096,
        )

    # Ensure file ends with single newline
    yaml_output = yaml_output.rstrip("\n") + "\n"
    return yaml_output


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

    # Check matrix vars reference real models (including disabled)
    matrix = config.get("matrix", {})
    model_ids = set(config.get("models", {}).keys())
    for var_name, model_id in matrix.get("vars", {}).items():
        if model_id not in model_ids:
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

    engine = "ruamel.yaml round-trip" if HAS_RUAMEL else "pyyaml"
    print(f"Building config from fragments (YAML merge, {engine})...", file=sys.stderr)
    output_text, active, disabled = build_config()

    # Validate
    errors = validate_config(output_text)
    if errors:
        print("\nVALIDATION ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  ❌ {e}", file=sys.stderr)
        return 1

    config = yaml.safe_load(output_text)
    n_active = len(config["models"])
    n_disabled = len(disabled)
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