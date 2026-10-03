#!/usr/bin/env python3
"""Filtro de saída do h3 CLI — compacta o chatter de offload do mmgp (Profile 5).

- Linhas "Loading/Unloading/Prefetching/Preloading model <submodelo>/blocks.N" não
  poluem mais o stdout: viram UMA linha viva (\\r) com step atual + bloco atual.
- Barras tqdm do denoise (linhas com %|...| N/M) passam intactas e alimentam o
  contador de passo da linha viva.
- Qualquer outra linha (warnings, erros, resumo final) passa limpa, limpando a
  linha viva antes.
- Cópia íntegra de tudo em argv[1] (log de debug) para debug sem perdas.
"""
import os
import re
import sys

DEBUG_LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/h3_debug.log"
# submodelo/bloco: transformer/blocks.N, vae/decoder.transformer_blocks.N, etc.
TRANSFER_RE = re.compile(
    r"^(Loading|Unloading|Prefetching|Preloading) model (\S*?blocks\.(\d+)|\S+) "
    r"\([^)]*\) (?:from|in) GPU")
TQDM_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[.*?,\s*([\d.]+)s/(?:it|step|s)\b")

last_step = None
last_total = None
last_sps = None
pending = False

debug = open(DEBUG_LOG, "ab", buffering=0)
bin_in = os.fdopen(0, "rb", 0)


def clear():
    global pending
    if pending:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        pending = False


def emit_status(block_label, block_num, block_max, phase):
    global pending
    parts = []
    if last_step is not None:
        parts.append(f"step {last_step}/{last_total}")
    if block_num is not None:
        parts.append(f"{block_label} {block_num}/{block_max}")
    parts.append(phase)
    if last_sps is not None:
        parts.append(f"{last_sps}s/step")
    sys.stdout.write("\r\033[K" + " · ".join(parts))
    sys.stdout.flush()
    pending = True


while True:
    raw = bin_in.readline()
    if not raw:
        break
    debug.write(raw)
    for piece in raw.decode("utf-8", "replace").split("\r"):
        piece = piece.rstrip("\n")
        m = TQDM_RE.search(piece)
        if m:
            last_step, last_total = int(m.group(2)), int(m.group(3))
            last_sps = m.group(4)
            if pending:
                clear()
            if piece.strip():
                sys.stdout.write(piece + "\n")
                sys.stdout.flush()
            continue
        t = TRANSFER_RE.match(piece.strip())
        if t:
            phase = t.group(1)
            path = t.group(2)
            # classificar por prefixo do caminho: transformer/blocks.N -> blocks N/50,
            # vae/decoder.transformer_blocks.N -> VAE N/36, text_encoder/layers.N -> enc N/50
            if path.startswith("vae/"):
                label, max_n, num = "VAE", 36, path.rsplit(".", 1)[-1]
            elif path.startswith("text_encoder/"):
                label, max_n, num = "enc", 50, path.rsplit(".", 1)[-1]
            else:
                label, max_n, num = "blocks", 50, path.rsplit(".", 1)[-1]
            try:
                emit_status(label, int(num), max_n, phase)
            except ValueError:
                pass
            continue
        if pending:
            clear()
        if piece.strip():
            sys.stdout.write(piece + "\n")
            sys.stdout.flush()

debug.close()
clear()