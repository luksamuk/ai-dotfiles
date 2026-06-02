# parakeet-stt — Local STT with parakeet.cpp

NVIDIA Parakeet TDT 0.6B v3 (multilingual, 25 languages including Portuguese)
running natively via C++/ggml with CUDA acceleration.

## Components

- `parakeet-cli` — Built from https://github.com/mudler/parakeet.cpp (CUDA backend)
- `parakeet-stt` — Wrapper script for Hermes Agent integration
- Model: `mudler/parakeet-cpp-gguf` → `tdt-0.6b-v3-q4_k.gguf` (644MB)

## Setup

```bash
# Build from source (CUDA)
cd ~/git/parakeet.cpp
cmake -B build -DPARAKEET_GGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Install binary
cp build/examples/cli/parakeet-cli ~/.local/bin/

# Download model
mkdir -p ~/.parakeet-models
# Use huggingface_hub or direct download:
# https://huggingface.co/mudler/parakeet-cpp-gguf — get tdt-0.6b-v3-q4_k.gguf

# Install wrapper
cp parakeet-stt ~/.local/bin/ && chmod +x ~/.local/bin/parakeet-stt
```

## Hermes Agent Config

In `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: true
  provider: local_command
  local_command:
    model: parakeet-tdt
```

The `local_command` provider calls `parakeet-stt <audio_file>` which returns
plain text transcription.

## Performance (RTX 3050 6GB, Q4_K model)

- Model size: 644MB on disk
- GPU (CUDA): ~0.85s for 7.4s audio
- CPU fallback: ~0.71s for 7.4s audio (short audio; GPU wins on longer clips)
- Languages: 25 European languages including Portuguese, English, Spanish, French
- Streaming: supported via `--stream` flag (cache-aware EOU detection)
- Timestamps: `--timestamps` for word-level, `--json` for full metadata

## Parakeet.cpp CLI Direct Usage

```bash
# Basic transcription
parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav

# With timestamps
parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav --timestamps

# JSON output (timestamps + confidence)
parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav --json

# Force CPU
PARAKEET_DEVICE=cpu parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav

# Streaming (EOU model required)
parakeet-cli transcribe --model eou.gguf --input audio.wav --stream
```

## Available Models

| Model | Size (Q4_K) | Languages | Notes |
|---|---|---|---|
| tdt-0.6b-v3 | 644MB | 25 (incl. PT) | Default, multilingual |
| tdt-0.6b-v2 | ~500MB | English only | Faster, EN-only |
| tdt-1.1b | ~1GB | English | Higher accuracy |
| tdt_ctc-110m | ~100MB | English | Ultra-fast, streaming |
| rnnt-0.6b | ~600MB | English | RNNT decoder |
| rnnt-1.1b | ~1GB | English | Higher accuracy RNNT |

All available at https://huggingface.co/mudler/parakeet-cpp-gguf

## Updating

```bash
cd ~/git/parakeet.cpp
git pull --recurse-submodules
cmake -B build -DPARAKEET_GGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cp build/examples/cli/parakeet-cli ~/.local/bin/
```

## Troubleshooting

- **CUDA not found**: Set `PARAKEET_DEVICE=cpu` in env or wrapper script
- **Model not found**: Check `~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf` exists
- **Long audio**: parakeet.cpp handles arbitrarily long audio natively (no chunking needed)
- **Portuguese**: Use the v3 multilingual model (tdt-0.6b-v3) — it auto-detects language