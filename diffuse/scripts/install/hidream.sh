#!/usr/bin/env bash
# Reinstala HiDream-O1-Image-Dev SDNQ no fleet diffuse.
# Baixa modelo SDNQ do HuggingFace, instala deps no .venv (sdnq, accelerate, triton).
#
# Uso: bash scripts/install/hidream.sh
set -euo pipefail

DIFFUSE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$DIFFUSE_DIR/.venv"
MODEL_DIR=~/.llama-models/HiDream-O1-Image-Dev-SDNQ-last8
HIDREAM_REPO=~/git/HiDream-O1-Image

echo "════════════════════════════════════════════"
echo "  Reinstalação do HiDream-O1 SDNQ no diffuse"
echo "════════════════════════════════════════════"

# ── 1. Instalar deps no .venv ─────────────────────────
echo ""
echo "[1/4] Instalando deps no .venv (sdnq, accelerate, triton)..."

cd "$DIFFUSE_DIR"
uv pip install --python .venv/bin/python sdnq accelerate triton
echo "  Deps instaladas!"

# ── 2. Clonar repo HiDream-O1-Image (tem código de inference) ──
echo ""
echo "[2/4] Clonando HiDream-O1-Image repo..."

if [ ! -d "$HIDREAM_REPO" ]; then
    git clone https://github.com/HiDream-Image/HiDream-I1.git "$HIDREAM_REPO"
else
    echo "  Repo já existe, pulando."
fi

# ── 3. Baixar modelo SDNQ do HuggingFace ─────────────
echo ""
echo "[3/4] Baixando modelo SDNQ (7.3 GB)..."

if [ ! -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
    hf download \
        --local-dir "$MODEL_DIR" \
        maidyy/HiDream-O1-Image-Dev-SDNQ-last8
    echo "  Modelo baixado!"
else
    echo "  Modelo já existe, pulando."
fi

# ── 4. Verificar ──────────────────────────────────────
echo ""
echo "[4/4] Verificando instalação..."
[ -f "$MODEL_DIR/model-00001-of-00002.safetensors" ] && echo "  ✅ Modelo SDNQ" || echo "  ❌ Modelo faltando"
[ -f "$MODEL_DIR/model-00002-of-00002.safetensors" ] && echo "  ✅ Modelo shard 2" || echo "  ❌ Shard 2 faltando"
[ -d "$HIDREAM_REPO" ] && echo "  ✅ Repo HiDream" || echo "  ❌ Repo faltando"
.venv/bin/python -c "import sdnq; print('  ✅ sdnq importable')" 2>&1
.venv/bin/python -c "import accelerate; print('  ✅ accelerate importable')" 2>&1
.venv/bin/python -c "import triton; print('  ✅ triton importable')" 2>&1

# ── Teste de fumaça ──────────────────────────────────
echo ""
echo "Teste de fumaça (512x512, seed 42)..."
diffuse generate -m hidream-sdnq \
    -p "a cat sitting on the moon" \
    --size 512x512 --seed 42 \
    --output /tmp/hidream_smoke_test.png 2>&1 | tail -8

if [ -f /tmp/hidream_smoke_test.png ]; then
    echo "  ✅ Teste passou!"
    rm -f /tmp/hidream_smoke_test.png
else
    echo "  ⚠️ Teste falhou — verifique os logs"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  HiDream reinstalado com sucesso!"
echo "  Use: diffuse generate -m hidream-sdnq ..."
echo "════════════════════════════════════════════"