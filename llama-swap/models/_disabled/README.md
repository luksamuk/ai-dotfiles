# Disabled Models — Reativation Guide

This directory holds model fragments that are **disabled** (excluded from
the generated `config.yaml`). The corresponding GGUF files have been moved to
`~/.llama-models/_disabled/` to free disk space.

Disabled on: 2026-07-05

## Disabled models

| Model ID          | GGUF (in `~/.llama-models/_disabled/`)               | Size  | Fragment (in `models/_removed/`)    |
|-------------------|------------------------------------------------------|-------|--------------------------------------|
| `north-mini-code` | `North-Mini-Code-1.0-Q4_K_M.gguf` (18.6 GB)          | 18 GB | `north-mini-code.yaml`               |
| `agents-a1-35b`   | `Agents-A1-APEX-I-Compact.gguf` (16.5 GB)            | 16 GB | `agents-a1-35b.yaml`                 |

> **Note:** `mv` within the same filesystem does not free space on its own —
> it only relocates inodes. To actually reclaim the ~34 GB, delete the GGUFs
> from `~/.llama-models/_disabled/` once you are confident you will not
> reactivate. The directory exists precisely to keep the operation reversible
> until that decision is made.

## How to re-enable a model

For each model you want back (e.g. `north-mini-code`):

### 1. Restore the GGUF

```sh
mv ~/.llama-models/_disabled/<Model>.gguf ~/.llama-models/<model-dir>/
# Also restore any mmproj if it was moved (none for these two models)
```

Specifically:

```sh
# north-mini-code
mv ~/.llama-models/_disabled/North-Mini-Code-1.0-Q4_K_M.gguf \
   ~/.llama-models/north-mini-code-q4km/

# agents-a1-35b
mv ~/.llama-models/_disabled/Agents-A1-APEX-I-Compact.gguf \
   ~/.llama-models/agents-a1-35b/
```

### 2. Restore the llama-swap fragment

```sh
mv ~/git/ai-dotfiles/llama-swap/models/_removed/<model>.yaml \
   ~/git/ai-dotfiles/llama-swap/models/
```

Specifically:

```sh
mv ~/git/ai-dotfiles/llama-swap/models/_removed/north-mini-code.yaml \
   ~/git/ai-dotfiles/llama-swap/models/
mv ~/git/ai-dotfiles/llama-swap/models/_removed/agents-a1-35b.yaml \
   ~/git/ai-dotfiles/llama-swap/models/
```

Then rebuild the llama-swap config:

```sh
python3 ~/git/ai-dotfiles/llama-swap/build-config.py
```

### 3. Uncomment harness references

Search for the marker `DISABLED 2026-07-05` in each harness and remove the
comment prefix. The harnesses touched were:

- `~/git/ai-dotfiles/configs/opencode/config.json`
- `~/git/ai-dotfiles/configs/pi/models.json`
- `~/git/ai-dotfiles/configs/droid/settings.json`
- `~/git/ai-dotfiles/configs/crush/crush.json`
- `~/git/ai-dotfiles/configs/codex/config.toml`
- `~/git/ai-dotfiles/configs/sprachspiel/models.toml`
- `~/git/ai-dotfiles/configs/nanocoder/agents.config.json`
- `~/git/ai-dotfiles/configs/vscode/chatLanguageModels.json`
- `~/git/ai-dotfiles/configs/zed/settings.json`
- `~/git/ai-dotfiles/configs/omp/config.yml`
- `~/git/ai-dotfiles/configs/omp/models.yml`
- `~/git/ai-dotfiles/configs/copilot-cli/copilot-local`
- `~/.emacs.d/init.el`
- `~/.emacs.d/init.org`

Each commented block is prefixed with a header line containing
`DISABLED 2026-07-05: GGUF moved to ~/.llama-models/_disabled/` followed by
the original lines commented out. Strip the comment marker (`//`, `#`, or `;`
depending on the file type) from each line and remove the two header lines.

### 4. Restart llama-swap

```sh
# Restart the llama-swap service to pick up the new config.yaml
systemctl --user restart llama-swap
# Or whichever supervisor you use
```

## Convention

- `models/*.yaml`           — active fragments (included in `config.yaml`)
- `models/_disabled/*.yaml` — disabled but kept locally (excluded; reserved
  for fragments whose GGUF still lives in `~/.llama-models/`)
- `models/_removed/*.yaml`  — dead code, reference only (excluded; used when
  the GGUF has been moved away or the model is unlikely to come back)