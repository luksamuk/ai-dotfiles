# New Model Addition Workflow

## Steps (in order)

### 0. Pre-flight: Backend Decision Matrix

Before anything, determine which backend(s) can run the model:

| Backend | Strengths | Weaknesses |
|---------|-----------|------------|
| **ik_llama.cpp** | `--fit`, `--cpu-moe`, `--defer-experts`, `--parallel-tool-calls`, `--jinja`, row-interleaved quants, hadamard KV | No auto attn_rot, no TurboQuant, mmproj limited |
| **llama.cpp (upstream)** | Auto attn_rot, `--jinja`, `ngram-cache`, `--reasoning on`, full mmproj support | No `--fit`, no `--cpu-moe`, no hadamard, no `--parallel-tool-calls` |
| **BeeLlama.cpp** | TurboQuant KV, DFlash speculative, `ngram-mod`, `--reasoning-loop-guard` | No `--fit` (uses `--n-gpu-layers 99`), no `--cpu-moe`, no `--parallel-tool-calls`, mmproj may crash on CUDA |
| **vLLM** | `--tool-call-parser hermes`, `--trust-remote-code`, custom model support | No `--fit`, no `--cpu-moe`, Python venv required, no real-time streaming |

**Priority rule**: ik_llama.cpp for MoE models (CPU-MoE offload). Upstream or Bee for dense models. vLLM only for models that need `--trust-remote-code`.

### 1. Architecture Check (GATE)

Before investing research time, verify the model can run on llama-swap:

- Does a GGUF conversion exist? (Search HuggingFace for `<model>-GGUF` repos)
- Is the architecture in llama.cpp? (Search for `case LLM_ARCH_<NAME>:` in `llama-model.cpp`)
- Is the architecture in ik_llama.cpp? (Same check in ik's copy)
- **⚠️ Same-arch duplicate check**: If the candidate is a fine-tune of an architecture already in the fleet (e.g., Jan-Code-4B is Qwen3), it inherits parent bugs. Evaluate value-add.
- **⚠️ Architecture name in GGUF may differ from marketing name**: Phi-4-mini uses `phi3` in GGUF, NOT `phi4`. Verify by running `strings <file>.gguf | grep 'general.architecture'`.
- **head_dim check**: `head_dim = embedding_length / head_count`. If `head_dim % 64 != 0`, attn_rot is NOT supported → use `q8_0 + q8_0` KV cache.
- If architecture is unsupported, note the blocker and skip. Do NOT download or create config fragments for unsupported architectures.

### 2. Research

Check benchmark scores, VRAM requirements, attn_rot compatibility, and chat template format.

### 2.5. Chat Template / Tool-Calling GATE

Before downloading, verify tool calling capability:

- **Check `chat_template` in `tokenizer_config.json`**. If empty/missing → no embedded Jinja template. `--jinja` cannot format tool calls → garbled output.
- **Check `--tool-call-parser` availability**. ik_llama and upstream do NOT have this flag. If model HF page says "serve with `--tool-call-parser hermes`" → vLLM-only, won't work in llama-swap.
- **If both missing** → model **cannot do tool calling** on llama-swap. Mark `tools: false` in fragment.
- **Test early**: if `finish_reason: "stop"` (not `"tool_calls"`) and content is garbled → tool calling is broken. Don't waste time on multiple templates.
- **Parallel tool calls**: Check if model supports calling multiple tools in one response. Test with 4+ tool calls. If only 1 call per response, add `--parallel-tool-calls` (ik-only) and re-test.
- **Always-thinking models**: If the model ALWAYS generates reasoning (no `enable_thinking` toggle, e.g. Nanbeige4.1, gpt-oss-20b), deploy without `:think` variant. Add `--n-predict 32768` (or higher) so reasoning tokens don't consume output budget.

### 3. Download

Add entry to `download-models.sh` (5 locations!) and run:

```bash
cd ~/git/ai-dotfiles/llama-swap
./download-models.sh <model>
```

⚠️ **`huggingface-cli` is DEPRECATED** — use `hf download <repo> <file> --local-dir <path>` instead.

If model has mmproj, also add to `MMPROJ` dict.

### 4. Create Fragment YAML

Create `models/<model-name>.yaml` with **canonical 0-indent YAML** (the build system now uses `ruamel.yaml` merge, not text concatenation):

```yaml
# -------------------------------------------
# MODEL_NAME - BACKEND (why this backend)
# Architecture notes, VRAM fit, quant optimization notes
# Tool calling notes, template requirements, known issues
# -------------------------------------------
model-name:
  name: "Model Display Name"
  description: "Short description from Ollama/HF model card — what the model is for, not backend/quant details"
  cmd: |
    ${llama_server|ik_llama_server|bee_server} --port ${PORT}
    --model ${models_dir}/Model-File.Q4_K_M.gguf
    [--mmproj ${models_dir}/mmproj-Model-F16.gguf]
    [--embeddings --pooling mean]   # embedding models only
    [--fit on --fit-target ${vram_margin} --fit-ctx ${small_model_min_ctx}]
    [--no-mmproj]                  # vision models with broken mmproj on CUDA
    [--n-gpu-layers 99]            # BeeLlama only (no --fit)
    [--moe on --moe-experts-cpu on] # ik only, MoE models
    [--parallel-tool-calls]        # ik only, models with tool calling
    --ctx-size ${small_model_max_ctx}
    --temp ${default_temp} --top-p ${default_top_p} --top-k ${default_top_k}
    --min-p ${default_min_p} --repeat-penalty ${default_repeat_penalty}
    [--reasoning on --reasoning-loop-guard force-close]  # thinking models
    [--reasoning off]               # vision/embedding models
    --cache-type-k q8_0 --cache-type-v q4_0  # or turbo3_tcq for Bee
    --flash-attn on --metrics
  filters:
    stripParams: "temperature, top_p"
    setParamsByID:
      "${MODEL_ID}":
        chat_template_kwargs:
          enable_thinking: false
        temperature: 0.7
        top_p: 0.9
      "${MODEL_ID}:think":
        chat_template_kwargs:
          enable_thinking: true
        temperature: 0.6
        top_p: 0.9
  ttl: 60
  metadata:
    architecture: <arch>           # from GGUF: general.architecture
    parameters: <int>              # from GGUF: general.parameter_count
    context_length: <int>          # from GGUF: <arch>.context_length
    embedding_length: <int>        # from GGUF: <arch>.embedding_length
    quantization: <string>         # from GGUF quant level
    size: "~2.71 GB (i1-Q4_K_M)"
    context: "64K-128K (dynamic)"
    vram_usage: "~3.5GB"
    source: "mradermacher/Repo-Name-GGUF"
    kv_cache: "q8_0/q4_0 or turbo3_tcq etc."
    features:
      completion: true|false      # true for chat models, false for embedding
      tools: true|false
      thinking: true|false
      vision: true|false           # true if mmproj present and working
      embedding: true|false        # true for embedding-only models
      insert: false                # rarely true
      audio: false                 # rarely true
      image: false                 # rarely true
```

**Description style**: Use the model card's short description, not backend/quant details. E.g., for Gemma 4: "Gemma 4 models are designed to deliver frontier-level performance at each size. Well-suited for reasoning, agentic workflows, coding, and multimodal understanding." If no Ollama/HF model card description, write a didactic one.

**Extract metadata from GGUF**: Use `python3 -c "from gguf import GGUFReader; ..."` or check `tokenizer_config.json` for `context_length`, `embedding_length`, `parameter_count`.

### 5. Register in Build Script

Add model ID to `ORIGINAL_ORDER` in `build-config.py`.

### 6. Matrix

Edit `config-footer.yaml`:
- **vars**: add short alias (e.g., `abl4: qwen3.5-4b-abliterated`)
- **sets**: add to appropriate swap groups (`small_vision`, `small_only`, `medium`, `heavy`, `vision_only`)
- **evict_costs**: add cost (1=ultra-light, 2=small, 3=medium, 4=heavy)

#### ⚠️ CRITICAL: Embedding/Reranker Coexistence Rule

**All embedding and reranker models (e.g., `nomic`, `lfmemb`, `lfmcol`) MUST be included in EVERY swap set that defines an LLM group.** These models are CPU-only (~0.5GB total) and do not consume VRAM, so they can always coexist with any LLM — including heavy offload models.

If an embedding/reranker model is NOT in the same set as an LLM, the solver treats them as mutually exclusive. This causes **ping-pong eviction**: the embedding evicts the LLM (high cost), then the LLM evicts the embedding (low cost), repeat forever.

**Rule**: When adding a new LLM set or modifying an existing one, always AND-in the embedding/reranker vars:

```yaml
# CORRECT — embedding coexists with heavy LLMs
heavy: "(qw36 | orn35 | a1 | aw | glm) & (nomic | lfmemb)"

# WRONG — embedding excluded, causes ping-pong eviction
heavy: "(qw36 | orn35 | a1 | aw | glm) & nomic"
# lfmemb is missing — lfm2.5-embedding-350m will evict Ornith and vice versa
```

The same applies to `small_only`, `medium`, `vision_only`, and any other set that groups LLMs. Embedding/reranker vars (`nomic`, `lfmemb`, `lfmcol`) should appear in the `&` clause of every LLM set.

### 7. Download Script (5 LOCATIONS)

Edit `download-models.sh`. Must update ALL 5 locations:
1. **Header comment** (lines 1-29): model name + size + description
2. **`MODELS` dict**: `["<model-id>"]="repo remote_file [local_file]"`
3. **Error message** in `download_model()`: add to "Available:" list
4. **`case` statement** (bottom): add `"<model-id>"` pattern
5. **`show_sizes()`**: add size/description line
- If model has mmproj, also add to `MMPROJ` dict.
- ⚠️ **Never skip this step** — it has been forgotten multiple times.

### 8. Build & Sync

```bash
cd ~/git/ai-dotfiles/llama-swap
python3 build-config.py --diff    # verify changes
python3 build-config.py           # build + sync
```

### 9. SIGHUP / Restart

```bash
systemctl --user restart llama-swap
# or: kill -HUP $(pgrep llama-swap)
```

### 10. Quick Test

```bash
curl -s http://127.0.0.1:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-name>","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}'
```

### 11. Tool-Calling Test

Send request with `tools` parameter. Verify `finish_reason: "tool_calls"`.

### 12. Agent Integration (for agentic models)

If the model has `tools: true` and `thinking: true` and tolerates agentic use:
- **OpenCode**: Add to `~/git/ai-dotfiles/configs/opencode/opencode.json` under the appropriate provider
- **Pi Coding Agent**: Add to `~/git/ai-dotfiles/configs/pi/` config
- **Crush**: Add to `~/git/ai-dotfiles/configs/crush/crush.json`
- **Other agents**: Check `~/git/ai-dotfiles/configs/` for available agents
- **:think variants**: When deploying a `:think` variant with a custom display name, prefix the name with 🧠 emoji

### 13. Commit

```bash
cd ~/git/ai-dotfiles
# Check for PII first
git diff -p | grep -iE '(password|secret|api.key|token|cpf|rg )' | head
git add -A
git commit -m "llama-swap: add <model-name>"
```

## Naming Convention

- **Model IDs**: Use `name-parameters` format (e.g., `qwen3.5-4b`, `gemma4-e4b`)
- **MoE models**: Include active parameters: `qwen3.6-35b-a3b` (not `qwen3.6-35b-moe`)
- **:think variants**: Deploy as separate entries with `:think` suffix (e.g., `qwen3.5-4b:think`)
- **Always-thinking models**: Deploy WITHOUT `:think` variant — the base model is already always-thinking

## Fragment System

```
~/git/ai-dotfiles/llama-swap/
├── config-base.yaml          # Header + macros
├── config-footer.yaml        # Hooks + matrix (vars, sets, evict_costs)
├── config.yaml               # ⚡ GENERATED — do not edit directly!
├── build-config.py           # Assembler (ruamel.yaml merge)
├── download-models.sh         # Download script (5 locations per model)
├── llama-swap-cli             # CLI tool (list, ps, testchat, etc.)
├── models/*.yaml              # Active model fragments (0-indent YAML)
├── models/_disabled/*.yaml    # Unlisted models (each has unlisted: true)
└── models/_removed/*.yaml    # Dead code (reference only, not built)
```

**Key rules:**
- **DO NOT edit `config.yaml` directly** — it's generated by `build-config.py`
- **Fragments use canonical 0-indent YAML** — `yaml.safe_load()` parseable, merged by `build-config.py`
- **`_disabled/` fragments MUST have `unlisted: true`** — auto-injected if missing
- **Model order** is controlled by `ORIGINAL_ORDER` in `build-config.py`
- **Build auto-syncs** to `~/.config/llama-swap/config.yaml`
- **After rebuild**: `systemctl --user restart llama-swap`

## Pitfalls

### Chat Template / Tool Calling
- No `chat_template` in GGUF = NO tool calling on llama-swap (vLLM-only `--tool-call-parser` doesn't exist)
- Always test with `curl` before adding `tools: true`
- CopySpec BREAKS tool calling on Qwen3.5-4B
- ngram-mod BREAKS parallel tool calling on Gemma4-E4B
- `--parallel-tool-calls` is ik-only; upstream handles via Jinja template

### Always-Thinking Models
- Always-thinking models (Nanbeige4.1, gpt-oss-20b) need `--n-predict 32768` or higher
- Without it, reasoning tokens consume the entire output budget → empty `content`

### VRAM and Quantization
- Every model MUST fit in 6GB VRAM. Use `--fit` on ik/upstream or `--n-gpu-layers 99` on Bee
- TurboQuant (Bee only): `turbo3_tcq` for KV cache ~5x compression
- IK MoE: `--moe on --moe-experts-cpu on --defer-experts`
- mmproj on CUDA: Gemma 4 mmproj crashes with SIGABRT (issue #21402). Use `--no-mmproj`.

### Unicode in Bash Comments
- **Never paste non-ASCII** (Chinese characters, emoji, thinking tokens) into bash-script comments
- Use plain-English descriptions instead

### Fragment YAML
- Each fragment must be valid YAML with exactly 1 model key
- Use 0-indent (canonical YAML) — the build system uses ruamel.yaml merge
- Run `python3 build-config.py --check` to validate

### Matrix — Embedding/Reranker Coexistence
- **Embedding and reranker models are CPU-only (~0.5GB) and must coexist with any LLM.**
- If an embedding var is missing from an LLM set, the solver ping-pongs: embedding evicts LLM (cost 9+), LLM evicts embedding (cost 1), repeat.
- Always AND-in `nomic | lfmemb | lfmcol` in every LLM set's coexistence clause.
- See "Step 6: Matrix" above for the full rule and examples.
