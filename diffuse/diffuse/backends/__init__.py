"""Backend registry — load_pipeline and unload_pipeline dispatch to the right backend."""
from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from diffuse.paths import MODELS_DIR
from diffuse.models import MODELS

log = logging.getLogger("diffuse")


# ── Shared helpers ──────────────────────────────────────────────────────────
def _find_subdir(root: Path, *hints: str) -> Path:
    """Find a child directory whose name contains any of the hints."""
    matches = [p for p in root.iterdir() if p.is_dir() and any(h in p.name for h in hints)]
    if not matches:
        present = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "(empty)"
        raise FileNotFoundError(f"No subdir matching {hints!r} under {root}. Present: {present}")
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def require_model_dir(model_name: str) -> Path:
    """Ensure model weights are present and return path.

    If the model directory doesn't exist and the model has 'hf_files' metadata,
    automatically download the weights from HuggingFace using the hf CLI.
    """
    model_info = MODELS[model_name]
    model_root = MODELS_DIR / model_info["dir"]
    if not model_root.exists():
        hf_files = model_info.get("hf_files")
        if hf_files:
            _auto_download_model(model_name, model_info, model_root)
            return model_root
        print(f"\n  ✗ Model not found: {model_root}")
        print(f"    Run: diffuse download {model_name.split('-')[0]}")
        sys.exit(1)
    return model_root


def _auto_download_model(model_name: str, model_info: dict, model_root: Path) -> None:
    """Download model weights from HuggingFace using the hf CLI.

    Reads the 'hf_files' metadata from the model entry and fetches each file
    to the correct local path. Handles subdirs and renames.
    """
    # Check hf CLI is available (huggingface-cli is deprecated, use hf)
    hf_cli = shutil.which("hf")
    if not hf_cli:
        print("\n  ✗ hf CLI is not installed.")
        print("    Install it with:  pip install huggingface-hub")
        print(f"    Then re-run:  diffuse -m {model_name}")
        sys.exit(1)

    hf_files = model_info["hf_files"]
    total_files = sum(len(entry["files"]) for entry in hf_files)

    print(f"\n  📦 Downloading model weights from HuggingFace ({total_files} files, ~16 GB)...")
    print(f"     Model: {model_name}")
    print(f"     Target: {model_root}")
    print()

    model_root.mkdir(parents=True, exist_ok=True)

    for entry in hf_files:
        repo = entry["repo"]
        files = entry["files"]
        rename_map = entry.get("rename", {})
        subdir = entry.get("subdir")

        for filename in files:
            # Determine destination directory
            if subdir:
                dest_dir = model_root / subdir
            else:
                dest_dir = model_root
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Download the file
            print(f"  ⬇️  {repo} → {filename}")
            cmd = [hf_cli, "download", repo, filename, "--local-dir", str(dest_dir)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n  ✗ Download failed: {repo}/{filename}")
                if "gated" in str(result.stderr or "").lower() or "401" in str(result.stderr or ""):
                    print(f"    This is a gated repo. You need to:")
                    print(f"    1. Visit https://huggingface.co/{repo} and accept the license")
                    print(f"    2. Run: hf auth login")
                    print(f"    3. Re-run: diffuse -m {model_name}")
                sys.exit(1)

            # Rename if needed
            local_name = rename_map.get(filename)
            if local_name and local_name != filename:
                downloaded_path = dest_dir / filename
                target_path = dest_dir / local_name
                if downloaded_path.exists():
                    downloaded_path.rename(target_path)
                    print(f"     Renamed: {filename} → {local_name}")

            print(f"     ✓ Done")

    print(f"\n  ✅ All model weights downloaded to {model_root}")
    print()


# ── Backend dispatch ─────────────────────────────────────────────────────────
def load_pipeline(model_name: str, editing: bool = False) -> tuple:
    """Load a pipeline based on the model's backend_type.

    Returns (pipeline_or_config, load_time_seconds).
    """
    model_info = MODELS[model_name]
    backend_type = model_info.get("backend_type", "gemlite")

    if backend_type == "gemlite":
        from diffuse.backends.gemlite import load_pipeline_gemlite
        return load_pipeline_gemlite(model_name)
    elif backend_type == "sd_cpp":
        from diffuse.backends.sd_cpp import load_pipeline_sd_cpp
        return load_pipeline_sd_cpp(model_name)
    elif backend_type == "hidream":
        from diffuse.backends.hidream import load_pipeline_hidream
        return load_pipeline_hidream(model_name, editing=editing)
    elif backend_type == "framepack":
        from diffuse.backends.framepack import load_pipeline as load_pipeline_framepack
        return load_pipeline_framepack(editing=editing)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def unload_pipeline() -> None:
    """Force-unload pipeline from VRAM and system memory."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    log.info("Pipeline unloaded from VRAM")