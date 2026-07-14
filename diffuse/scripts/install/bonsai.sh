#!/usr/bin/env bash
# Reinstala Bonsai Image 4B Ternary (gemlite) no fleet diffuse.
# Cria bonsai-venv, instala deps, baixa modelo do HuggingFace.
#
# Uso: bash scripts/install/bonsai.sh
set -euo pipefail

DIFFUSE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="$DIFFUSE_DIR/models/bonsai-image-4B-ternary-gemlite"
BONSAI_VENV="$DIFFUSE_DIR/bonsai-venv"
VENDOR_STUDIO="$DIFFUSE_DIR/vendor/image-studio"

echo "════════════════════════════════════════════"
echo "  Reinstalação do Bonsai Image 4B no diffuse"
echo "════════════════════════════════════════════"

# ── 1. Criar bonsai-venv ──────────────────────────────
echo ""
echo "[1/4] Criando bonsai-venv..."

if [ -d "$BONSAI_VENV" ]; then
    echo "  bonsai-venv já existe. Deseja recriar? [y/N]"
    read -r response
    if [[ "$response" =~ ^[yY]$ ]]; then
        rm -rf "$BONSAI_VENV"
    else
        echo "  Mantendo venv existente, pulando criação."
    fi
fi

if [ ! -d "$BONSAI_VENV" ]; then
    cd "$DIFFUSE_DIR"
    uv venv bonsai-venv
    echo "  Instalando PyTorch (CUDA 12.8)..."
    uv pip install --python bonsai-venv/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    echo "  Instalando diffusers 0.39 + deps..."
    uv pip install --python bonsai-venv/bin/python \
        'diffusers>=0.38' 'transformers>=4.46' accelerate \
        hqq gemlite einops pillow safetensors sentencepiece tokenizers
    echo "  venv pronto!"
fi

# ── 2. Baixar modelo do HuggingFace ──────────────────
echo ""
echo "[2/4] Baixando modelo Bonsai (PrismML)..."

mkdir -p "$MODELS_DIR"

# Ternary gemlite int2 transformer
if [ ! -d "$MODELS_DIR/transformer-gemlite-int2" ]; then
    echo "  Baixando transformer-gemlite-int2 (1.5 GB)..."
    hf download \
        --local-dir "$MODELS_DIR/transformer-gemlite-int2" \
        prism-ml/bonsai-image-ternary-4B-gemlite-2bit \
        transformer-gemlite-int2/
else
    echo "  transformer-gemlite-int2 já existe, pulando."
fi

# Text encoder (HQQ 4-bit)
if [ ! -d "$MODELS_DIR/text_encoder-hqq-4bit" ]; then
    echo "  Baixando text_encoder-hqq-4bit (2.8 GB)..."
    hf download \
        --local-dir "$MODELS_DIR/text_encoder-hqq-4bit" \
        prism-ml/bonsai-image-ternary-4B-gemlite-2bit \
        text_encoder-hqq-4bit/
else
    echo "  text_encoder-hqq-4bit já existe, pulando."
fi

# VAE (FLUX.2)
if [ ! -d "$MODELS_DIR/vae" ]; then
    echo "  Baixando VAE (170 MB)..."
    hf download \
        --local-dir "$MODELS_DIR/vae" \
        prism-ml/bonsai-image-ternary-4B-gemlite-2bit \
        vae/
else
    echo "  VAE já existe, pulando."
fi

echo "  Modelo baixado!"

# ── 3. Verificar ──────────────────────────────────────
echo ""
echo "[3/4] Verificando instalação..."
[ -d "$MODELS_DIR/transformer-gemlite-int2" ] && echo "  ✅ Transformer" || echo "  ❌ Transformer faltando"
[ -d "$MODELS_DIR/text_encoder-hqq-4bit" ] && echo "  ✅ Text encoder" || echo "  ❌ Text encoder faltando"
[ -d "$MODELS_DIR/vae" ] && echo "  ✅ VAE" || echo "  ❌ VAE faltando"
[ -x "$BONSAI_VENV/bin/python" ] && echo "  ✅ bonsai-venv" || echo "  ❌ bonsai-venv faltando"

# Check vendored image-studio
if [ ! -d "$VENDOR_STUDIO" ]; then
    echo "  ⚠️ vendor/image-studio não encontrado — necessário para backend_gpu"
    echo "     Clone: git clone <image-studio-repo> $VENDOR_STUDIO"
else
    echo "  ✅ vendor/image-studio"
fi

# ── 4. Teste de fumaça ───────────────────────────────
echo ""
echo "[4/4] Teste de fumaça (512x512, 4 steps)..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
diffuse generate -m ternary-gemlite \
    -p "a cat sitting on the moon" \
    --size 512x512 --seed 42 \
    --output /tmp/bonsai_smoke_test.png 2>&1 | tail -5

if [ -f /tmp/bonsai_smoke_test.png ]; then
    echo "  ✅ Teste passou!"
    rm -f /tmp/bonsai_smoke_test.png
else
    echo "  ⚠️ Teste falhou — verifique os logs"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  Bonsai Image 4B reinstalado com sucesso!"
echo "  Use: diffuse generate -m ternary-gemlite ..."
echo "  Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "        é necessário para 6GB VRAM."
echo "════════════════════════════════════════════"