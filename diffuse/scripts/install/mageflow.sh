#!/usr/bin/env bash
# Reinstall Mage-Flow-Edit-Turbo models (decommissioned 2026-09-17 for disk space)
set -e
mkdir -p models/mageflow-edit-turbo
cd models/mageflow-edit-turbo
hf download gguf-org/mageflow-gguf mageflow-edit-turbo-nvfp4.gguf
hf download gguf-org/mageflow-gguf pig_mageflow_vae_fp32-f16.gguf
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF mmproj-Qwen3VL-4B-Instruct-F16.gguf
echo "mageflow restored."
