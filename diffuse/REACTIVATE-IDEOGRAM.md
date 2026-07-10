# Ideogram 4 in diffuse — Auto-Download Reference

The Ideogram 4 (sd_cpp) backend has been **reactivated** with auto-download support.
Models are no longer stored locally — they download from HuggingFace on first use.

## How it works

When a user runs `diffuse -m ideogram4-q4`, the CLI checks if model weights are present
locally. If not, and the model has `hf_files` metadata in `diffuse/models.py`, the CLI
automatically downloads the required files using `hf download`.

No pre-download is needed for the GGUF files. The only prerequisite is:

- `hf` CLI installed (`pip install huggingface-hub`)
- For the VAE file: the user must accept the license at
  https://huggingface.co/black-forest-labs/FLUX.2-dev and run `hf auth login`
  once (it's a gated repo)

## Model files (auto-downloaded)

| File | Size | HF Repo | Notes |
|------|------|---------|-------|
| `ideogram4-Q4_0.gguf` | 5.64 GB | `leejet/ideogram-4-GGUF` | Main diffusion model |
| `ideogram4_uncond-Q4_0.gguf` | 5.64 GB | `leejet/ideogram-4-GGUF` | Unconditional model |
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 5.03 GB | `unsloth/Qwen3-VL-8B-Instruct-GGUF` | Renamed from `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` |
| `vae/flux2-vae.safetensors` | 336 MB | `black-forest-labs/FLUX.2-dev` | Gated repo, renamed from `ae.safetensors` |

Total: ~16 GB downloaded on first use.

## Steps to re-enable

**Already re-enabled — models auto-download on first use.**

If you need to manually re-download (e.g. corrupted files):

```bash
cd ~/git/ai-dotfiles/diffuse
rm -rf models/ideogram-4-Q4_0
diffuse -m ideogram4-q4 -p "test"  # will auto-download
```

## Files involved

| File | Role |
|------|------|
| `diffuse/backends/sd_cpp.py` | Backend module (sd-cli subprocess wrapper) |
| `diffuse/models.py` | `ideogram4-q4` entry with `hf_files` download metadata |
| `diffuse/backends/__init__.py` | `require_model_dir()` with auto-download, `load_pipeline()` sd_cpp dispatch |
| `diffuse/cli.py` | CLI integration: imports, generation block, help text, defaults |
| `diffuse/enhance.py` | `enhance_prompt()` for Ideogram JSON format |
| `diffuse/prompts.py` | `get_ideogram_enhance_prompt()` accessor |
| `diffuse/output.py` | "Enhanced: Yes → Ideogram 4 JSON" label in debrief |

## History

This backend was previously disabled to free ~16GB of disk space. It was reactivated
with auto-download support so models are fetched on demand rather than stored locally.