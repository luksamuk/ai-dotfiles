---
name: audio-transcription
description: Transcribe audio files locally using parakeet.cpp (GPU, fast, short clips) or MOSS-Transcribe (CPU, diarization, long-form, subtitles). Covers when to use each, CLI commands, pitfalls, and output formats.
---

# Audio Transcription — Local STT

Two local speech-to-text engines are available. They complement each other — use the right one for the task.

**Important:** Either engine may not be installed. If a command fails or the binary is missing, inform the user and suggest the alternative.

## Engine Selection

| Use case | Engine | Why |
|---|---|---|
| Short clips (<2min), voice commands | parakeet | GPU CUDA, ~0.85s, fast |
| Multi-speaker audio (meetings, interviews) | MOSS | Built-in diarization, one pass |
| Long audio (>5min, lectures, podcasts) | MOSS | 128k context, up to 90 min (with chunking) |
| Portuguese clips | parakeet | Confirmed 25-language support, fast |
| SRT/ASS subtitles needed | MOSS | Native export (--format srt/ass/json) |
| Running alongside GPU-heavy tasks | MOSS | CPU-only, zero VRAM usage |
| Quick English transcript | Either | MOSS on CPU RTF ~0.55-0.78 |

## parakeet.cpp — Fast GPU Transcription

NVIDIA Parakeet TDT model, C++/ggml binary with CUDA backend. Best for short clips and voice commands.

### Usage

```bash
# Basic transcription
parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav

# Word-level timestamps
parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav --timestamps

# JSON output (timestamps + confidence per word)
parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav --json

# Force CPU (if GPU is busy with LLM)
PARAKEET_DEVICE=cpu parakeet-cli transcribe --model ~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf --input audio.wav

# Hermes-compatible wrapper (auto-handles CUDA/CPU fallback)
parakeet-stt audio.wav --output_dir /tmp/transcripts
```

### Models

| Model | Size | Languages | Notes |
|-------|------|-----------|-------|
| tdt-0.6b-v3 | 644MB | 25 incl. Portuguese | Default, multilingual |
| tdt-0.6b-v2 | ~500MB | English only | Faster, EN-only |
| tdt-1.1b | ~1GB | English | Higher accuracy |
| nemotron-600m | ~600MB | English only | Multilingual needs language-ID (not in parakeet.cpp yet) |

All models: https://huggingface.co/mudler/parakeet-cpp-gguf

### Performance (RTX 3050 6GB, Q4_K)

- GPU: ~0.85s for 7.4s audio (faster than real-time)
- VRAM: ~300-600MB (coexists with llama-swap models)
- Cold start: ~0.8s model load
- No audio length limit — streams cache-aware natively

### Pitfalls

- **GPU conflict**: If llama-swap has a heavy model loaded, use `PARAKEET_DEVICE=cpu` — CPU is fast enough for short clips
- **Portuguese**: Use v3 multilingual model (tdt-0.6b-v3) — it auto-detects language. v2 is English-only
- **Input must be WAV**: Non-WAV formats (MP3, OGG) need conversion first: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`
- **CUDA init messages on stderr are normal**: Lines like `ggml_cuda_init` and `[parakeet] pk::Backend using GPU device: CUDA0` appear on stderr — they don't affect output

## MOSS-Transcribe — Diarization + Long-Form

OpenMOSS MOSS-Transcribe-Diarize 0.9B, C++/ggml binary, CPU-only. Joint transcription, speaker diarization, and timestamps in a single pass.

### Usage

```bash
# Basic transcription with diarization
moss-transcribe transcribe ~/.moss-models/moss-transcribe-q5_k.gguf audio.wav

# With output format (text, srt, ass, json)
moss-transcribe transcribe ~/.moss-models/moss-transcribe-q5_k.gguf audio.wav --format srt

# Cap generated length (REQUIRED for audio >2min)
moss-transcribe transcribe ~/.moss-models/moss-transcribe-q5_k.gguf audio.wav --max-new 2048

# Control CPU threads (8 is sweet spot on 20-core)
MTD_THREADS=8 moss-transcribe transcribe ~/.moss-models/moss-transcribe-q5_k.gguf audio.wav

# Hermes-compatible wrapper (auto-converts non-WAV to WAV)
moss-stt audio.wav --output_dir /tmp/transcripts
```

### Output Format

Default text output includes speaker labels and timestamps:
```
[0.28][S01] And so, my fellow Americans
[7.71][8.12][S02] ask what you can do
[10.59]
```

Available formats: `text`, `srt`, `ass`, `json`

### Chunking Long Audio (>5min)

**CRITICAL:** Single-pass transcription of long audio causes unbounded memory growth (24GB+ RSS for 24min audio). Always chunk:

```bash
# Split into 2-minute chunks
mkdir -p chunks transcripts
ffmpeg -i input.mp3 -f segment -segment_time 120 -ar 16000 -ac 1 chunks/chunk_%02d.wav

# Transcribe each chunk
for chunk in chunks/chunk_*.wav; do
  base=$(basename "$chunk" .wav)
  MTD_THREADS=8 moss-transcribe transcribe \
    ~/.moss-models/moss-transcribe-q5_k.gguf \
    "$chunk" --max-new 2048 --format text \
    > "transcripts/${base}.txt" 2>/dev/null
  echo "Done: $base"
done

# Concatenate (adjust timestamps by chunk offset: chunk_i starts at i*120s)
cat transcripts/chunk_*.txt > full_transcript.txt
```

### Performance (CPU-only, q5_k, 20 threads)

| Audio | Duration | Wall time | RTF | Peak RSS |
|-------|----------|-----------|-----|----------|
| Short (22s) | 22.6s | 14.5s | 0.64 | ~985 MB |
| Long (174s) | 2m54s | 2m19s | 0.80 | ~985 MB |

- Zero GPU usage — safe alongside fine-tuning or LLM inference
- Thread sweet spot: 8 threads on 20-core
- VmPeak: ~2.2 GB

### Quantization

| Quant | Size | Speed vs F32 | Parity |
|-------|------|-------------|--------|
| f32 | 3.4 GB | 1.0x | byte-identical |
| f16 | 1.8 GB | 1.6x | byte-identical |
| q8_0 | 942 MB | 2.0x | byte-identical |
| q6_k | 733 MB | 1.9x | byte-identical |
| **q5_k** | **619 MB** | **1.8x** | **byte-identical** (recommended) |
| q4_k | 511 MB | 2.1x | word-identical (1 timestamp off 0.02s) |

### Portuguese

Confirmed working (07/2026). Tested on Brazilian Portuguese colloquial speech. Captures colloquialisms ("né", "pra quê"), philosophical vocabulary ("autosuficiência", "taquigrafia"). Diarization correctly identifies 2-4 speakers. No language config needed — the Whisper-Medium encoder handles PT-BR natively.

### Pitfalls

- **No GPU backend yet** — CPU-only. CUDA/Metal/Vulkan are on the roadmap
- **Input must be WAV 16kHz mono** — the `moss-stt` wrapper auto-converts via ffmpeg, but if calling `moss-transcribe` directly, pre-convert: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`
- **`--max-new 2048` is REQUIRED for audio >2min** — without it, the autoregressive decoder can enter infinite generation loops, consuming all RAM (24GB RSS observed) and producing empty output. Always set a max-new limit
- **Chunk audio >5min** — single-pass on long audio causes unbounded context growth. Split into 2-min chunks with `ffmpeg -f segment -segment_time 120`
- **Output appears only at completion** — MOSS buffers all output. Empty output file during processing is normal
- **Speaker labels are per-recording** — S01/S02/S03 reset between files, not persistent identities
- **RTF grows with audio length** — 0.64 for 22s, 0.80 for 174s. Longer audio = slower relative throughput

## File Locations

| Item | parakeet | MOSS |
|------|----------|------|
| Binary | `~/.local/bin/parakeet-cli` | `~/.local/bin/moss-transcribe` |
| Wrapper | `~/.local/bin/parakeet-stt` | `~/.local/bin/moss-stt` |
| Model | `~/.parakeet-models/tdt-0.6b-v3-q4_k.gguf` | `~/.moss-models/moss-transcribe-q5_k.gguf` |
| Source | `~/git/parakeet.cpp` | `~/git/moss-transcribe.cpp` |
| Model size | 644 MB | 619 MB |

## Audio Preparation

Both engines work best with WAV 16kHz mono. Convert any audio format:

```bash
# MP3/MP4/OGG → WAV 16kHz mono
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Extract audio from video
ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav

# Split long audio into 2-min chunks (for MOSS)
ffmpeg -i long_audio.wav -f segment -segment_time 120 -ar 16000 -ac 1 chunk_%02d.wav
```