# llama-swap Config Architecture

This directory uses a **fragmented config system** — instead of one monolithic `config.yaml`, the configuration is split into modular files that are assembled by `build-config.py`.

## Why Fragments?

The original `config.yaml` grew to 1100+ lines with 16 model definitions, 9 of which are disabled (`unlisted: true`) with comment-heavy tombstones. This made it hard to:
- Find and edit a specific model's parameters
- Track which models are active vs disabled
- Maintain per-model documentation without scroll fatigue
- Avoid merge conflicts when editing multiple models

## Structure

```
llama-swap/
├── config-base.yaml          # Header + macros (everything before models)
├── config-footer.yaml        # Hooks + matrix (everything after models)
├── config.yaml               # ⚡ GENERATED — do not edit directly!
├── build-config.py           # Assembler script
├── models/
│   ├── gemma4-26b-moe.yaml   # Active models
│   ├── gemma4-e4b.yaml
│   ├── gemma4-e2b.yaml
│   ├── lfm2.5-1.2b.yaml
│   ├── qwen3.5-4b.yaml
│   ├── qwen3.6-35b-moe.yaml
│   ├── _disabled/             # Disabled models (each has unlisted: true)
│   │   ├── gpt-oss-20b.yaml
│   │   ├── lfm2-24b.yaml
│   │   ├── lfm2.5-1.2b-think.yaml
│   │   ├── lfm2.5-1.2b-vllm.yaml
│   │   ├── lfm2.5-sgl.yaml
│   │   ├── lfm2.5-vl-450m.yaml
│   │   ├── qwopus-35b.yaml
│   │   ├── qwen3.5-0.8b.yaml
│   │   ├── qwen3.5-0.8b-vllm.yaml
│   │   └── qwen3.5-9b.yaml
│   └── _removed/              # Dead code (commented-out models, reference only)
│       └── ds-r1-distill-14b-32b.yaml
├── download-models.sh
├── llama-swap-cli
├── run.sh
├── llama-swap.service.template
├── MTP-NOTES.md
├── README.md
└── ARCHITECTURE.md            # ← You are here
```

## Workflow

### Editing a Model

1. Edit the model's fragment file in `models/` or `models/_disabled/`
2. Run `python3 build-config.py` to regenerate `config.yaml`
3. The script auto-syncs to `~/.config/llama-swap/config.yaml`
4. Reload llama-swap: `systemctl --user restart llama-swap`

### Activating a Disabled Model

1. Move `models/_disabled/<model>.yaml` → `models/<model>.yaml`
2. Remove `unlisted: true` from the fragment (it's only needed in `_disabled/`)
3. Add the model ID to the `ORIGINAL_ORDER` list in `build-config.py`
4. Run `python3 build-config.py`

### Disabling an Active Model

1. Move `models/<model>.yaml` → `models/_disabled/<model>.yaml`
2. Add `unlisted: true` after the `description:` field in the fragment
3. Remove the model ID from the `ORIGINAL_ORDER` list in `build-config.py`
4. Run `python3 build-config.py`

### Adding a New Model

1. Create `models/<model-id>.yaml` with the model definition (see existing fragments for format)
2. Add the model ID to `ORIGINAL_ORDER` in `build-config.py` (in the desired position)
3. Add matrix vars/sets/evict_costs in `config-footer.yaml`
4. Run `python3 build-config.py`

### Permanently Removing a Model

1. Delete `models/_disabled/<model>.yaml`
2. Remove the model ID from `ORIGINAL_ORDER` in `build-config.py`
3. Remove matrix vars/sets/evict_costs in `config-footer.yaml`
4. Run `python3 build-config.py`

If you want to keep the config for future reference, move it to `models/_removed/` instead of deleting.

## Build Script Reference

```
python3 build-config.py              # Build, validate, and write config.yaml
python3 build-config.py --check      # Validate only (don't write)
python3 build-config.py --diff       # Show diff against current config.yaml
python3 build-config.py --list       # List all model fragments with status
```

Validation checks:
- Each fragment must parse as valid YAML with exactly 1 model key
- The assembled config must have `models` and `macros` sections
- Each model must have `cmd` and `name` fields
- Matrix `vars` must reference existing model IDs

## Fragment Format

Each model fragment is a YAML file containing a single model definition with its comments. Example:

```yaml
  # -------------------------------------------
  # QWEN3.5-4B - MULTIMODAL (VISION + TEXT)
  # Cabe inteiro na VRAM com mmproj
  # -------------------------------------------
  "qwen3.5-4b":
    name: "Qwen3.5 4B"
    description: "Multimodal reasoning and agentic chat with 201-language support"
    cmd: |
      ${llama_server} --port ${PORT}
      --model ${models_dir}/Qwen3.5-4B-UD-Q3_K_XL.gguf
      ...
```

Note: The 2-space indent is required — model keys and properties are at the same indent level as they appear in the final `config.yaml` under the `models:` key.

## Matrix Design Principles

### Embedding/Reranker Coexistence (CRITICAL)

Embedding and reranker models (`nomic`, `lfmemb`, `lfmcol`) are CPU-only (~0.5GB total, zero VRAM). They **must** coexist with any LLM in the swap matrix.

**Why**: The solver uses set membership to determine coexistence. If an embedding var is missing from an LLM's set, the solver sees them as mutually exclusive. This causes **ping-pong eviction** — the embedding evicts the LLM (high cost), then the LLM evicts the embedding (low cost), repeating forever.

**Rule**: Every LLM set (`small_only`, `small_vision`, `medium`, `heavy`, `vision_only`, etc.) must AND-in the embedding/reranker vars:

```yaml
# CORRECT
heavy: "(qw36 | orn35 | a1 | aw | glm) & (nomic | lfmemb)"

# WRONG — lfmemb missing = ping-pong eviction
heavy: "(qw36 | orn35 | a1 | aw | glm) & nomic"
```

When adding a new embedding/reranker model, add it to the `&` clause of **every** LLM set. When adding a new LLM set, include all existing embedding/reranker vars in the `&` clause.

See `references/new-model-workflow.md` §6 for the full workflow rule.

## Key Files

| File | Purpose |
|------|---------|
| `config-base.yaml` | Header comments + macros (no models, no hooks, no matrix) |
| `config-footer.yaml` | Hooks + matrix section (vars, sets, evict_costs) |
| `config.yaml` | ⚡ Generated output — DO NOT EDIT |
| `build-config.py` | Assembler: base + models + footer → config.yaml |
| `models/*.yaml` | Active model definitions |
| `models/_disabled/*.yaml` | Disabled models (unlisted: true) |
| `models/_removed/*.yaml` | Dead code for reference (not built) |

## Original Order Preservation

Models are assembled in a specific order defined by `ORIGINAL_ORDER` in `build-config.py`. This matches the original monolithic config to minimize diffs. New models not in this list are appended alphabetically at the end.