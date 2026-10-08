#!/usr/bin/env python3
"""
Qwen3-TTS OpenAI-Compatible API Server
=======================================
Serves /v1/audio/speech and /v1/audio/voices endpoints,
compatible with the llama-swap web UI Speech tab.

Uses Qwen3-TTS 0.6B-Base via torch-direct inference (no vLLM needed).
Voice profiles loaded from the voiceclone-tui config.

Usage:
    python tts_server.py [--port 8880] [--host 0.0.0.0]
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

# ─── Compatibility patch (MUST be before qwen_tts import) ──────────────────
import transformers.utils.generic as _tgen
_tgen.check_model_inputs = lambda *args, **kwargs: (lambda func: func)

# ─── Suppress noisy output ─────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", message="flash-attn is not installed")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging as _logging
for _logger_name in ("transformers", "transformers.generation", "torch"):
    _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

# ─── Quiet qwen_tts import ──────────────────────────────────────────────────
import importlib

def _quiet_import(module_name):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return importlib.import_module(module_name)
    finally:
        sys.stdout = old_stdout

_quiet_import("qwen_tts")

# ─── Now safe to import ─────────────────────────────────────────────────────
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import uvicorn

# ─── Config ─────────────────────────────────────────────────────────────────

CONFIG_DIR = Path(os.path.expanduser("~/.config/voiceclone-tui"))
CONFIG_FILE = CONFIG_DIR / "voices.json"
BUILTIN_VOICES_DIR = Path(__file__).parent / "voices"
SAMPLES_DIR = CONFIG_DIR  # Samples live alongside config

# Default voice mapping: OpenAI-style voice names → our voice profile names
# Users can reference voices by short name (e.g. "hermes") or display name
DEFAULT_VOICES = {
    "hermes": {
        "profile": "hermes",
        "language": "Portuguese",
    },
    "lucas": {
        "profile": "lucas",
        "language": "Portuguese",
    },
    "jessica": {
        "profile": "jessica",
        "language": "Portuguese",
    },
    "bolso": {
        "profile": "bolso",
        "language": "Portuguese",
    },
    "batman": {
        "profile": "batman",
        "language": "Portuguese",
    },
    "jarvis": {
        "profile": "jarvis",
        "language": "Portuguese",
    },
    "alucard": {
        "profile": "alucard",
        "language": "English",
    },
    "dracula": {
        "profile": "dracula",
        "language": "English",
    },
}

SUPPORTED_FORMATS = {"mp3", "wav", "ogg", "opus", "flac"}

app = FastAPI(title="Qwen3-TTS API", version="1.0.0")
logger = logging.getLogger("tts-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ─── Model singleton ────────────────────────────────────────────────────────

_model = None
_model_lock = __import__("threading").Lock()
_last_used = 0
_UNLOAD_AFTER_SECONDS = int(os.environ.get("TTS_UNLOAD_TTL", "300"))  # 5 min default
_KEEP_LOADED = os.environ.get("TTS_KEEP_LOADED", "0") == "1"  # Set "1" to keep model in VRAM


def get_model():
    """Lazy-load the Qwen3-TTS model (singleton) with optional auto-unload."""
    global _model, _last_used
    with _model_lock:
        if _model is not None:
            _last_used = time.time()
            return _model

        logger.info("Loading Qwen3-TTS 0.6B-Base model...")
        t0 = time.time()
        # Suppress stdout during model loading
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            from qwen_tts import Qwen3TTSModel
            _model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map="auto",
                dtype=torch.bfloat16,
            )
        finally:
            sys.stdout = old_stdout

        elapsed = time.time() - t0
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        _last_used = time.time()
        logger.info(f"Model loaded in {elapsed:.1f}s, VRAM: {vram_mb:.0f} MB")
        return _model


def unload_model():
    """Release model from VRAM to free GPU memory for LLMs."""
    global _model
    with _model_lock:
        if _model is None:
            return
        logger.info("Unloading Qwen3-TTS model to free VRAM...")
        del _model
        _model = None
        torch.cuda.empty_cache()
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"Model unloaded. VRAM now: {vram_mb:.0f} MB")


@app.on_event("startup")
async def startup_check():
    """Start auto-unload background task if enabled."""
    if not _KEEP_LOADED and _UNLOAD_AFTER_SECONDS > 0:
        import asyncio
        async def auto_unload():
            while True:
                await asyncio.sleep(30)  # Check every 30s
                if _model is not None and _last_used > 0:
                    idle = time.time() - _last_used
                    if idle > _UNLOAD_AFTER_SECONDS:
                        unload_model()
        asyncio.create_task(auto_unload())


# ─── Voice config loading ───────────────────────────────────────────────────

def load_voice_profiles() -> dict:
    """Load voice profiles from config file + builtin fallback.
    
    Returns a dict keyed by:
      - short name (e.g. "hermes", "dracula")
      - display name (e.g. "🛡️ Hermes", "🦇 Drácula")
      - lowercase display name without emoji (e.g. "drácula")
    """
    profiles = {}

    # Also load from clone.py's VOICES dict as fallback
    _load_clone_voices(profiles)

    # Load builtin voices.json
    builtin_file = Path(__file__).parent / "voices.json"
    if builtin_file.exists():
        with open(builtin_file) as f:
            data = json.load(f)
            for name, v in data.get("voices", {}).items():
                profiles[name] = v  # Display name with emoji
                # Extract short name after emoji
                parts = name.split()
                if len(parts) > 1:
                    short = parts[-1].lower()
                    profiles[short] = v

    # Load user config (overrides builtins)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            data = json.load(f)
            for name, v in data.get("voices", {}).items():
                profiles[name] = v
                parts = name.split()
                if len(parts) > 1:
                    short = parts[-1].lower()
                    profiles[short] = v

    return profiles


def _load_clone_voices(profiles: dict):
    """Load voices from clone.py's VOICES dict as fallback.
    
    We can't easily parse Python literals, so instead we define a mapping
    of voice names → (sample_file, transcript, language, instruct) here.
    This must be kept in sync with ~/projects/ai/voiceclone/clone.py VOICES.
    """
    clone_voices = {
        "lucas": {
            "sample": "lucas.wav",
            "ref_text": "Só um minutinho que eu já vou aí. Eu tô... Só... Testando pra ver se essa aplicação consegue clonar a minha voz também.",
            "language": "Portuguese",
            "instruct": None,
            "prefix": "lucas",
        },
        "alucard": {
            "sample": "alucard.wav",
            "ref_text": "As you can see, this is a PlayStation black disc. Cut number one contains computer data, so please, don't play it. But you probably won't listen to me anyway, will you?",
            "language": "English",
            "instruct": None,
            "prefix": "alucard",
        },
        "dracula": {
            "sample": "dracula3.wav",
            "ref_text": "What is a man? A miserable little pile of secrets! But enough talk, have at you!",
            "language": "English",
            "instruct": "Show anger and raise the pitch of your voice.",
            "prefix": "dracula",
        },
        "bolso": {
            "sample": "bolso.wav",
            "ref_text": "Eu tenho uma certa liberdade pequena, quem tem muita liberdade é o meu filho Eduardo. Até que enquanto presidente, eu sabia o meu lugar ao conversar com ele.",
            "language": "Portuguese",
            "instruct": None,
            "prefix": "bolso",
        },
        "francisca": {
            "sample": "francisca.wav",
            "ref_text": "Oi, tudo bem? Eu sou a Francisca, e estou aqui pra ajudar voce com qualquer coisa que precisar. Pode contar comigo sempre, nao hesite em perguntar.",
            "language": "Portuguese",
            "instruct": None,
            "prefix": "francisca",
        },
        "thalita": {
            "sample": "thalita.wav",
            "ref_text": "Ola! Meu nome e Thalita. Prazer em conhecer voce! Estou aqui pra tornar seu dia mais facil e produtivo. Vamos comecar?",
            "language": "Portuguese",
            "instruct": None,
            "prefix": "thalita",
        },
        "jarvis": {
            "sample": "jarvis.wav",
            "ref_text": "Tive algumas experiências meio ruins, na... na... na juventude, e tal... Meio perdido.... E... Tive uma experiência, assim, de fé, que até importante, porque isso faz parte da minha vida, falar de fé, de Deus, tá entranhado, eu não separo as duas coisas, né?",
            "language": "Portuguese",
            "instruct": None,
            "prefix": "jarvis",
        },
        "batman": {
            "sample": "batman.wav",
            "ref_text": "Bom dia. Me perdoe. Quisera eu ter a... A resistência que você tem de ir até quatro, cinco da madrugada. Como eu gostaria... Viva a juventude.",
            "language": "Portuguese",
            "instruct": "Speak with deep, grave intensity. Emphasize key words with dramatic pauses. Sound seductive and passionate.",
            "prefix": "batman",
        },
        "hermes": {
            "sample": "hermes_reference.wav",
            "ref_text": "Sou Hermes, seu guardião digital. Estou aqui para proteger, orientar e cuidar do seu ambiente. Meu compromisso é com a sua segurança e bem-estar. Sempre atento, sempre vigilante.",
            "language": "Portuguese",
            "instruct": None,
            "prefix": "hermes",
        },
    }
    
    # Only add if not already present (voices.json/user config takes priority)
    for key, val in clone_voices.items():
        if key not in profiles:
            profiles[key] = val


def resolve_sample_path(sample_filename: str) -> Optional[str]:
    """Find a voice sample file in config dir or builtin voices dir."""
    # Check config dir first
    config_path = SAMPLES_DIR / sample_filename
    if config_path.exists():
        return str(config_path)

    # Check builtin voices dir
    builtin_path = BUILTIN_VOICES_DIR / sample_filename
    if builtin_path.exists():
        return str(builtin_path)

    # Check voiceclone CLI samples dir
    cli_samples = Path.home() / "projects" / "ai" / "voiceclone" / "samples" / sample_filename
    if cli_samples.exists():
        return str(cli_samples)

    return None


# ─── Generation ────────────────────────────────────────────────────────────

def generate_speech(
    text: str,
    voice: str = "hermes",
    language: Optional[str] = None,
    output_format: str = "mp3",
    speed: float = 1.0,
) -> bytes:
    """Generate speech audio and return as bytes in the requested format."""
    model = get_model()

    # Resolve voice profile — normalize accents for matching
    voice_profiles = load_voice_profiles()
    import unicodedata

    def _normalize(s):
        return unicodedata.normalize('NFKD', s.lower()).encode('ascii', 'ignore').decode('ascii')

    profile = None
    # 1. Exact match
    if voice in voice_profiles:
        profile = voice_profiles[voice]
    # 2. Lowercase match (handles "Hermes" → "hermes")
    elif voice.lower() in voice_profiles:
        profile = voice_profiles[voice.lower()]
    # 3. Accent-normalized match (handles "jessica" → "jéssica", "dracula" → "drácula")
    else:
        voice_norm = _normalize(voice)
        for key, val in voice_profiles.items():
            if _normalize(key) == voice_norm:
                profile = val
                break
    
    if profile is None:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available: {list_available_voices()}"
        )

    # Resolve sample path
    sample_path = resolve_sample_path(profile.get("sample", ""))
    if sample_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Voice sample not found: {profile.get('sample')}"
        )

    # Determine language
    if language is None:
        language = profile.get("language", "Portuguese")

    ref_text = profile.get("ref_text", profile.get("transcript", ""))
    instruct = profile.get("instruct")

    logger.info(f"Generating: voice={voice}, lang={language}, text={text[:80]}...")

    # Generate
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            gen_kwargs = {
                "text": text,
                "language": language,
                "ref_audio": sample_path,
                "ref_text": ref_text,
            }
            if instruct:
                gen_kwargs["instruct"] = instruct

            wavs, sr = model.generate_voice_clone(**gen_kwargs)
        finally:
            sys.stdout = old_stdout
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    # Convert to requested format
    # qwen_tts returns (wav_data, sample_rate)
    buf = io.BytesIO()

    if output_format in ("wav", "pcm"):
        sf.write(buf, wavs[0].numpy() if hasattr(wavs[0], 'numpy') else wavs[0], sr, format="WAV")
    elif output_format in ("ogg", "opus"):
        # Write WAV to temp, then convert with ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav, wavs[0].numpy() if hasattr(wavs[0], 'numpy') else wavs[0], sr, format="WAV")
            tmp_wav_path = tmp_wav.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_wav_path, "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1"],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:500]}")
            buf = io.BytesIO(result.stdout)
        finally:
            os.unlink(tmp_wav_path)
    else:  # mp3 (default)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav, wavs[0].numpy() if hasattr(wavs[0], 'numpy') else wavs[0], sr, format="WAV")
            tmp_wav_path = tmp_wav.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_wav_path, "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:500]}")
            buf = io.BytesIO(result.stdout)
        finally:
            os.unlink(tmp_wav_path)

    buf.seek(0)
    return buf.read()


def list_available_voices() -> list[str]:
    """List available voice names — short, clean, no duplicates."""
    profiles = load_voice_profiles()
    seen = set()
    voices = []
    for key in profiles:
        # Extract short name from emoji-prefixed display names (e.g. "🛡️ Hermes" → "hermes")
        parts = key.split()
        name = parts[-1].lower() if len(parts) > 1 else key.lower()
        # Normalize: remove accents for cleaner listing, but keep accent for match
        import unicodedata
        normalized = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        # Avoid duplicates: prefer accent-free version
        if normalized not in seen:
            seen.add(normalized)
            # Use the accent-free version if it exists as a key, otherwise use normalized
            if normalized in profiles:
                voices.append(normalized)
            else:
                voices.append(name)
    return sorted(voices)


# ─── OpenAI-Compatible Endpoints ────────────────────────────────────────────

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "opus": "audio/ogg; codecs=opus",
    "flac": "audio/flac",
}


class SpeechRequest(BaseModel):
    """OpenAI-compatible TTS request body."""
    model: str = "qwen3-tts"
    input: str
    voice: str = "hermes"
    response_format: str = "mp3"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def audio_speech(request: SpeechRequest):
    """
    OpenAI-compatible TTS endpoint.
    POST /v1/audio/speech
    Body: {"model": "qwen3-tts", "input": "text", "voice": "hermes", "response_format": "mp3"}
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="input (text) is required")

    if request.response_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {request.response_format}. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    try:
        audio_data = generate_speech(
            text=request.input,
            voice=request.voice,
            language=None,  # Let the voice profile decide, or auto-detect
            output_format=request.response_format,
            speed=request.speed,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    content_type = CONTENT_TYPES.get(request.response_format, "audio/mpeg")
    return Response(content=audio_data, media_type=content_type)


@app.get("/v1/audio/voices")
async def audio_voices(model: str = "qwen3-tts"):
    """
    List available voices for a model.
    OpenAI-compatible endpoint used by the llama-swap web UI.
    """
    voices = list_available_voices()
    return JSONResponse(content={"voices": voices})


@app.get("/v1/models")
async def list_models():
    """Minimal models endpoint for compatibility."""
    return JSONResponse(content={
        "data": [
            {
                "id": "qwen3-tts",
                "name": "Qwen3-TTS 0.6B",
                "description": "Voice cloning TTS with Portuguese support",
                "object": "model",
                "owned_by": "qwen",
            }
        ]
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "ok",
        "model_loaded": _model is not None,
        "voices": list_available_voices(),
    })


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS OpenAI-Compatible API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8880, help="Port to bind to")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")
    args = parser.parse_args()

    if args.preload:
        logger.info("Preloading model...")
        get_model()

    logger.info(f"Starting Qwen3-TTS API server on {args.host}:{args.port}")
    logger.info(f"Available voices: {list_available_voices()}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")