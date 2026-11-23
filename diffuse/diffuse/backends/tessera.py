"""Tessera-small backend — sprites 16x16 de pixel art via ONNX, 100% CPU.

Modelo base: Tessera-small, Jagrat Patel (TheJagStudio) — thejagstudio/tessera-small,
CC BY-NC 4.0 (cobre o modelo E as imagens geradas por ele: uso pessoal/estudo,
crédito "Tessera by Jagrat Patel (TheJagStudio)"; comercial exige licença à parte).
Build ONNX int8: cfahlgren1/tessera-small-onnx. Sampler e conversão portados,
fielmente, de python/tessera_onnx.py desse repo.

Sem torch, sem CUDA: onnxruntime CPUExecutionProvider. Pipeline não toca VRAM.
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np
from PIL import Image
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

from diffuse.models import MODELS
from diffuse.paths import MODELS_DIR

GRID = 16
N = GRID * GRID
K = 512  # tokens de paleta
MASK = 512  # token "ainda mascarado"
TEXT_LEN = 24

# CFG/temperatura: receita do build ONNX (tessera_onnx.py)
DEFAULT_CFG = 3.0
CHOICE_TEMPERATURE = 2.0
NMS_THRESHOLD = 0.9


def load_pipeline_tessera(model_name: str) -> tuple:
    """Abre as sessões ONNX (CPU). Pesos vêm via require_model_dir/hf_files."""
    model_root = MODELS_DIR / MODELS[model_name]["dir"]

    t0 = time.perf_counter()
    palette = np.array(
        json.load(open(model_root / "palette.json"))["palette_rgba"], dtype=np.uint8
    )
    provider = ["CPUExecutionProvider"]
    denoiser = InferenceSession(str(model_root / "tessera_w8.onnx"), providers=provider)
    text_encoder = InferenceSession(str(model_root / "clip_text_w8.onnx"), providers=provider)
    tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
    load_time = time.perf_counter() - t0

    pipeline = {
        "denoiser": denoiser,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "palette": palette,
    }
    return pipeline, load_time


def _encode_text(pipe: dict, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    """CLIP ids com padding até 24 (BOS/EOS do post-processor do tokenizer)."""
    ids = pipe["tokenizer"].encode(re.sub(r"\s+", " ", prompt)).ids
    eos = ids[-1]
    if len(ids) > TEXT_LEN:
        ids = ids[: TEXT_LEN - 1] + [eos]
    ids = ids + [eos] * (TEXT_LEN - len(ids))
    states, pooled = pipe["text_encoder"].run(
        ["last_hidden_state", "pooler_output"],
        {"input_ids": np.array([ids], dtype=np.int64)},
    )
    return states, pooled


def _generate_tokens(
    pipe: dict, prompt: str, count: int = 4, seed: int | None = None,
    steps: int = 16, guidance: float = DEFAULT_CFG,
) -> np.ndarray:
    """Sampler masked-diffusion (schedule cosseno, CFG, unmasking por confiança
    com Gumbel anneal e NMS espacial). Portato de tessera_onnx.generate_tokens."""
    rng = np.random.default_rng(seed)
    states, pooled = _encode_text(pipe, prompt)
    b2 = count * 2  # linha condicional + incondicional por sprite
    text_states = np.repeat(states, b2, axis=0).astype(np.float32)
    text_pooled = np.repeat(pooled, b2, axis=0).astype(np.float32)
    drop_text = np.array([False] * count + [True] * count)

    x = np.full((count, N), MASK, dtype=np.int64)
    masked = np.ones((count, N), dtype=bool)
    sc = np.zeros((b2, N, K), dtype=np.float32)

    for k in range(steps):
        frac = (k + 1) / steps
        ratio = masked.mean(axis=1).astype(np.float32)
        logits = pipe["denoiser"].run(None, {
            "x": np.concatenate([x, x]),
            "text_states": text_states,
            "text_pooled": text_pooled,
            "mask_ratio": np.concatenate([ratio, ratio]),
            "sc_probs": sc,
            "drop_text": drop_text,
        })[0]
        if not np.isfinite(logits).all():
            raise RuntimeError("Tessera devolveu logits NaN/infinitos.")

        # CFG + softmax + uma amostra categórica por pixel.
        cond, uncond = logits[:count], logits[count:]
        guided = uncond + guidance * (cond - uncond)
        guided -= guided.max(axis=-1, keepdims=True)
        probs = np.exp(guided)
        probs /= probs.sum(axis=-1, keepdims=True)
        u = rng.random((count, N, 1))
        sampled = np.minimum((np.cumsum(probs, axis=-1) < u).sum(axis=-1), K - 1)
        p_sel = np.take_along_axis(probs, sampled[..., None], axis=-1)[..., 0]

        remaining = 0 if k == steps - 1 else math.floor(N * math.cos(math.pi / 2 * frac))
        for b in range(count):
            n_masked = int(masked[b].sum())
            n_commit = min(max(n_masked - remaining, 1), n_masked)
            if n_commit == 0:
                continue
            # Score de confiança com Gumbel annealed.
            gumbel = -np.log(-np.log(np.clip(rng.random(N), 1e-10, 1 - 1e-10)))
            score = np.log(np.maximum(p_sel[b], 1e-12)) + CHOICE_TEMPERATURE * (1 - frac) * gumbel

            # NMS espacial: pixels certos e máximos locais espalham pelo grid.
            certain = p_sel[b] >= NMS_THRESHOLD
            contenders = np.where(masked[b] & ~certain, score, -np.inf).reshape(GRID, GRID)
            padded = np.pad(contenders, 1, constant_values=-np.inf)
            local_max = np.max(
                [padded[1 + dr: 1 + dr + GRID, 1 + dc: 1 + dc + GRID]
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
                axis=0,
            )
            eligible = certain | (score >= local_max.reshape(N))

            key = np.where(masked[b], score + eligible * 1e4, -np.inf)
            chosen = np.argsort(-key, kind="stable")[:n_commit]
            x[b, chosen] = sampled[b, chosen]
            masked[b, chosen] = False

        # Self-conditioning: distribuição guiada deste passo nas duas linhas CFG.
        sc = np.concatenate([probs, probs]).astype(np.float32)
    return x


def reduce_palette(img: Image.Image, max_colors: int = 15, alpha_thresh: int = 128) -> Image.Image:
    """Reduz a paleta global (512 cores) para <=15 opacas + transparente (4bpp SNES)."""
    arr = np.array(img)
    opaque = arr[..., 3] >= alpha_thresh
    out = arr.copy()
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    px = arr[opaque][:, :3]
    if px.shape[0] == 0:
        return Image.fromarray(out, "RGBA")
    q = Image.fromarray(px.reshape(-1, 1, 3)).quantize(
        colors=max_colors, method=Image.Quantize.FASTOCTREE
    )
    raw_palette = q.getpalette() or [0] * (max_colors * 3)
    pal = np.array(raw_palette[: max_colors * 3], dtype=np.uint8).reshape(max_colors, 3)
    idx = np.array(q).reshape(-1)
    out[opaque, :3] = pal[idx]
    return Image.fromarray(out, "RGBA")


def generate_image_tessera(
    pipeline: dict, prompt: str, seed: int, steps: int = 16,
    count: int = 4, guidance: float = DEFAULT_CFG,
) -> tuple[list[Image.Image], float, float]:
    """Gera `count` sprites 16x16 RGBA com paleta <=15+1.

    Retorna (sprites, tempo_de_difusao, peak_hbm=0.0 — CPU).
    """
    t0 = time.perf_counter()
    tokens = _generate_tokens(
        pipeline, prompt, count=count, seed=seed, steps=steps, guidance=guidance
    )
    diffusion_time = time.perf_counter() - t0
    palette = pipeline["palette"]
    sprites = [
        reduce_palette(Image.fromarray(palette[t].reshape(GRID, GRID, 4), "RGBA"))
        for t in tokens
    ]
    return sprites, diffusion_time, 0.0
