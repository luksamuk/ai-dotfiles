# Configuration Details

### Key Parameters

- **`--fit on`**: Automatically adjusts GPU layers and context size to fit in VRAM
- **`--fit-target 512`**: Safety margin in MiB (prevents OOM)
- **`--fit-ctx 4096`**: Minimum context size when downscaling
- **`--ctx-size`**: Maximum context length (8192 or 16384 depending on model)
- **`--flash-attn`**: Flash Attention for better performance
- **`--temp 0.7`**: Temperature for sampling diversity
- **`--top-p 0.85`**: Nucleus sampling threshold
- **`--top-k 40`**: Top-K sampling

### How `--fit` Works

The `--fit` feature automatically:

1. Detects available VRAM on each GPU
2. Calculates optimal number of GPU layers (`-ngl`)
3. Reduces context size if necessary
4. Prioritizes dense weights for MoE models
5. Leaves a safety margin (configurable)

**Note:** The flag syntax differs between binaries:
- **upstream llama.cpp:** `--fit on --fit-target 512` (target free VRAM in MiB)
- **ik_llama.cpp:** `--fit --fit-margin 512` (safety margin in MiB)

In `config.yaml`, MoE models use `${ik_llama_server}` with `--fit --fit-margin`,
while dense models use `${llama_server}` with `--fit on --fit-target`.

This is especially useful for:
- **Mixed GPU setups** - automatically balances layers
- **Memory pressure** - prevents OOM crashes
- **Different models** - no manual tuning per model

### VRAM Considerations for RTX 3050 (6GB)

With ~5GB free VRAM (after desktop environment):

| Model | Strategy |
|-------|----------|
| Qwen3.5-4B | Fits entirely in VRAM - fastest inference |
| Qwen3.5-9B | Requires partial offload - `--fit` handles automatically |
| Gemma-4-E4B | Requires partial offload - `--fit` handles automatically |

The `--fit` flag will:
1. Detect available VRAM
2. Calculate optimal number of GPU layers
3. Reduce context size if necessary
4. Move remaining layers to system RAM