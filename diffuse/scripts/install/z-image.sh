#!/usr/bin/env bash
# Reinstala Z-Image-Turbo no fleet diffuse.
# Baixa GGUFs do HuggingFace, compila sd-cli (branch z-image-omini-base), copia binário.
#
# Uso: bash scripts/install/z-image.sh
set -euo pipefail

DIFFUSE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="$DIFFUSE_DIR/models/z-image-turbo-q4"
BIN_DIR="$DIFFUSE_DIR/bin"
SDCLI_REPO=~/git/stable-diffusion.cpp

echo "════════════════════════════════════════════"
echo "  Reinstalação do Z-Image-Turbo no diffuse"
echo "════════════════════════════════════════════"

# ── 1. Baixar modelos do HuggingFace ──────────────────
mkdir -p "$MODELS_DIR/pipeline/vae" "$MODELS_DIR/pipeline/tokenizer"

echo ""
echo "[1/4] Baixando GGUFs do HuggingFace..."

# DiT transformer (Q3_K_S — única quantização que cabe em 6GB VRAM)
if [ ! -f "$MODELS_DIR/z_image_turbo-Q3_K_S.gguf" ]; then
    echo "  Baixando z_image_turbo-Q3_K_S.gguf (3.6 GB)..."
    hf download \
        --local-dir "$MODELS_DIR" \
        Tongyi-MAI/Z-Image-Turbo-GGUF \
        z_image_turbo-Q3_K_S.gguf
else
    echo "  z_image_turbo-Q3_K_S.gguf já existe, pulando."
fi

# Text encoder (Qwen3-4B Q4_K_M)
if [ ! -f "$MODELS_DIR/Qwen3-4B-Q4_K_M.gguf" ]; then
    echo "  Baixando Qwen3-4B-Q4_K_M.gguf (2.4 GB)..."
    hf download \
        --local-dir "$MODELS_DIR" \
        ggml-org/Qwen3-4B-GGUF \
        Qwen3-4B-Q4_K_M.gguf
else
    echo "  Qwen3-4B-Q4_K_M.gguf já existe, pulando."
fi

# VAE (Flux VAE)
if [ ! -f "$MODELS_DIR/pipeline/vae/diffusion_pytorch_model.safetensors" ]; then
    echo "  Baixando VAE (160 MB)..."
    hf download \
        --local-dir "$MODELS_DIR/pipeline/vae" \
        Tongyi-MAI/Z-Image-Turbo \
        vae/diffusion_pytorch_model.safetensors
else
    echo "  VAE já existe, pulando."
fi

# Tokenizer
if [ ! -f "$MODELS_DIR/pipeline/tokenizer/tokenizer.json" ]; then
    echo "  Baixando tokenizer..."
    hf download \
        --local-dir "$MODELS_DIR/pipeline/tokenizer" \
        Tongyi-MAI/Z-Image-Turbo \
        tokenizer/tokenizer.json \
        tokenizer/tokenizer_config.json \
        tokenizer/special_tokens_map.json \
        tokenizer/added_tokens.json
else
    echo "  Tokenizer já existe, pulando."
fi

echo "  Modelos baixados!"

# ── 2. Compilar sd-cli (branch z-image-omini-base) ───
echo ""
echo "[2/4] Compilando sd-cli (branch z-image-omini-base)..."

if [ ! -d "$SDCLI_REPO" ]; then
    echo "  Clonando stable-diffusion.cpp..."
    git clone https://github.com/leejet/stable-diffusion.cpp.git "$SDCLI_REPO"
fi

cd "$SDCLI_REPO"
git fetch origin
git checkout z-image-omini-base
git pull origin z-image-omini-base

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j"$(nproc)" sd-cli

mkdir -p "$BIN_DIR"
cp bin/sd-cli "$BIN_DIR/sd-cli-zimage"
echo "  sd-cli-zimage compilado e copiado!"

# ── 3. Verificar ──────────────────────────────────────
echo ""
echo "[3/4] Verificando instalação..."
[ -f "$MODELS_DIR/z_image_turbo-Q3_K_S.gguf" ] && echo "  ✅ DiT GGUF" || echo "  ❌ DiT GGUF faltando"
[ -f "$MODELS_DIR/Qwen3-4B-Q4_K_M.gguf" ] && echo "  ✅ Text encoder GGUF" || echo "  ❌ Text encoder faltando"
[ -f "$MODELS_DIR/pipeline/vae/diffusion_pytorch_model.safetensors" ] && echo "  ✅ VAE" || echo "  ❌ VAE faltando"
[ -x "$BIN_DIR/sd-cli-zimage" ] && echo "  ✅ sd-cli-zimage binary" || echo "  ❌ sd-cli-zimage faltando"

# ── 4. Teste rápido ───────────────────────────────────
echo ""
echo "[4/4] Teste de fumaça (512x512, 9 steps)..."
"$BIN_DIR/sd-cli-zimage" \
    --diffusion-model "$MODELS_DIR/z_image_turbo-Q3_K_S.gguf" \
    --llm "$MODELS_DIR/Qwen3-4B-Q4_K_M.gguf" \
    --vae "$MODELS_DIR/pipeline/vae/diffusion_pytorch_model.safetensors" \
    -p "a cat sitting on the moon" \
    --diffusion-fa \
    --offload-to-cpu --clip-on-cpu --vae-on-cpu --vae-tiling \
    -H 512 -W 512 --steps 9 --cfg-scale 1.0 --seed 42 \
    -o /tmp/zimage_smoke_test.png 2>&1 | tail -3

if [ -f /tmp/zimage_smoke_test.png ]; then
    echo "  ✅ Teste passou! Imagem gerada em /tmp/zimage_smoke_test.png"
    rm -f /tmp/zimage_smoke_test.png
else
    echo "  ⚠️ Teste falhou — verifique os logs acima"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  Z-Image-Turbo reinstalado com sucesso!"
echo "  Use: diffuse generate -m z-image-turbo-q4 ..."
echo "════════════════════════════════════════════"