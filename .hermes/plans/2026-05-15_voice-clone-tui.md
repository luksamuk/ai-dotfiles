# Plano: Voice Clone TUI — Drácula & Alucard

**Data:** 2026-05-15
**Autor:** Hermes
**Status:** Rascunho, aguardando aprovação do Lucas

---

## Objetivo

Criar um mini-aplicativo TUI (não é chat — é um menu interativo) que permite ao usuário:
1. Escolher entre duas vozes (Drácula ou Alucard de *Castlevania: Symphony of the Night*)
2. Digitar o texto que a voz deve falar
3. O app gera o áudio clonado, salva em `/tmp/`, e toca automaticamente
4. Repete — o usuário pode gerar mais samples sem reiniciar

O app roda *offline*, usando o modelo Qwen3-TTS 0.6B-Base carregado localmente (não via vLLM-Omni API).

---

## Contexto e Decisões de Arquitetura

### O que já existe

- `~/projects/ai/voiceclone/` — projeto Python existente com:
  - `clone.py` — script CLI que carrega `Qwen3-TTS-12Hz-0.6B-Base` via `qwen_tts` e gera clones
  - `hermes_design.py` — VoiceDesign + clone com modelo 1.7B
  - `samples/` — arquivos WAV de referência para todas as vozes
  - `.venv/` com dependências instaladas (torch, qwen_tts, soundfile)

### Por que NÃO via vLLM-Omni API?

A ideia original era usar vLLM-Omni como backend API. Problemas:

1. **Startup de 1-2 min** — Toda vez que o llama-swap swapa pro modelo TTS, espera 1-2 min
2. **Swap exclusivo** — Com 6GB VRAM, não cabe LLM + TTS simultaneamente. Cada chamada TTS = mata o LLM ativo
3. **Complexidade** — Precisa instalar vLLM-Omni (fork do vLLM, build from source), configurar llama-swap, gerenciar ciclo de vida
4. **Latência total** — Startup vLLM-Omni (1-2min) + swap LLM (1-2min de volta) = 3-4min de ida e volta para uma frase

### Por que SIM via script local (torch direto)

- O `clone.py` já funciona — carrega o modelo 0.6B em ~10-15s, gera em ~3-5s
- **Sem swap** — não compete com o LLM ativo no llama-swap
- O modelo 0.6B cabe em VRAM (~1.2GB BF16) junto com o LLM ativo (3-4GB) com margem
- Ou, se precisar liberar VRAM, pode rodar em CPU (mais lento, ~15-30s por frase)

### Decisão: TUI standalone usando torch direto

O TUI será um script Python independente que importa `qwen_tts` e gera localmente. Não precisa de vLLM-Omni, não precisa de llama-swap, não precisa de servidor.

---

## Fluxo do TUI

```
╔══════════════════════════════════════╗
║   🧛 Castlevania Voice Clone TUI    ║
╠══════════════════════════════════════╣
║                                      ║
║  Escolha a voz:                      ║
║                                      ║
║  🦇 Drácula  — "What is a man?!"     ║
║  ⚔️ Alucard  — "The morning has come"║
║                                      ║
║  ←/→ ou 1/2 para selecionar          ║
╠══════════════════════════════════════╣
║                                      ║
║  Drácula selecionado.                ║
║  Digite o texto (Enter para gerar):  ║
║                                      ║
║  > Miserável pilha de segredos!      ║
║                                      ║
║  ⏳ Gerando áudio... (3.2s)          ║
║  ▶ Tocando: [████████░░] 72% 3.8s   ║
║                                      ║
║  Salvo em: /tmp/voice_dracula_01.ogg ║
║                                      ║
║  [Enter] novo texto | [v] trocar voz ║
║  [q] sair                            ║
╠══════════════════════════════════════╣
║  🦇 Drácula | 3 samples gerados      ║
╚══════════════════════════════════════╝
```

---

## Arquivos de Voz (Copiados para o TUI)

**Fonte:** `~/projects/ai/voiceclone/samples/`
**Destino:** `~/projects/ai/voiceclone-tui/voices/`

| Arquivo original | Arquivo destino | Voz | Idioma |
|---|---|---|---|
| `dracula3.wav` | `dracula_ref.wav` | Drácula | English |
| `alucard.wav` | `alucard_ref.wav` | Alucard | English |

Apenas estes dois arquivos são copiados. O TUI NÃO tem acesso às vozes sensíveis (lucas, jessica, bolso, etc.).

### Transcrições de referência

**Drácula** (`dracula3.wav`):
> "What is a man? A miserable little pile of secrets! But enough talk, have at you!"

**Alucard** (`alucard.wav`):
> "As you can see, this is a PlayStation black disc. Cut number one contains computer data, so please, don't play it. But you probably won't listen to me anyway, will you?"

---

## Estrutura do Projeto

```
~/projects/ai/voiceclone-tui/
├── voice_tui.py          # Main TUI app (entry point)
├── voices/
│   ├── dracula_ref.wav   # Drácula reference audio
│   └── alucard_ref.wav   # Alucard reference audio
├── pyproject.toml        # Dependencies
└── README.md
```

### Dependências (pyproject.toml)

```toml
[project]
name = "voiceclone-tui"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch",           # Already installed system-wide or in parent venv
    "qwen_tts",        # Qwen3-TTS model library
    "soundfile",       # Audio file I/O
    "rich",            # TUI rendering (progress bars, panels)
    "questionary",     # Arrow-key menus (same as testchat)
]

[project.scripts]
voice-clone = "voice_tui:main"
```

**Nota:** `torch`, `qwen_tts`, `soundfile` já estão no venv de `~/projects/ai/voiceclone/`. O TUI pode usar o mesmo venv ou criar um novo que herda os modelos já baixados (~1.2GB em `~/.cache/huggingface/`).

---

## Implementação Detalhada

### `voice_tui.py` — Módulo Principal

```python
#!/usr/bin/env python3
"""
Castlevania Voice Clone TUI
Choose Drácula or Alucard, type text, hear the clone.
Uses Qwen3-TTS 0.6B-Base locally (torch direct).
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Voice profiles — ONLY Drácula and Alucard
VOICES_DIR = Path(__file__).parent / "voices"

VOICE_PROFILES = {
    "🦇 Drácula": {
        "ref_audio": VOICES_DIR / "dracula_ref.wav",
        "ref_text": "What is a man? A miserable little pile of secrets! But enough talk, have at you!",
        "language": "Auto",  # Default — auto-detect from input text
        "instruct": "Speak with deep, dramatic intensity. Sound powerful and commanding.",
        "prefix": "dracula",
    },
    "⚔️ Alucard": {
        "ref_audio": VOICES_DIR / "alucard_ref.wav",
        "ref_text": "As you can see, this is a PlayStation black disc. Cut number one contains computer data, so please, don't play it. But you probably won't listen to me anyway, will you?",
        "language": "Auto",
        "instruct": None,
        "prefix": "alucard",
    },
}

# Language options — controls OUTPUT pronuncation/accent, NOT the reference sample
LANGUAGE_OPTIONS = ["Auto", "English", "Portuguese", "Japanese", "Spanish", "French", "German", "Korean", "Russian", "Italian"]

# Global model cache
_model = None

def get_model():
    """Load Qwen3-TTS 0.6B-Base once and cache it."""
    global _model
    if _model is None:
        console = Console()
        with console.status("[bold yellow]Loading model (0.6B-Base)...[/]"):
            from qwen_tts import Qwen3TTSModel
            import torch
            _model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map="auto",
                dtype=torch.bfloat16,
            )
    return _model

def generate(text: str, voice: str, output_path: str, language: str = "Auto") -> float:
    """Generate voice clone and return duration in seconds."""
    import soundfile as sf
    import torch

    model = get_model()
    profile = VOICE_PROFILES[voice]

    torch.cuda.empty_cache()

    # Language controls OUTPUT pronunciation/accent:
    # "Auto" → detect from text, "English" → English accent, "Portuguese" → PT accent, etc.
    # The voice timbre comes from ref_audio, independent of language.
    effective_language = language if language != "Auto" else None

    kwargs = {
        "text": text,
        "language": effective_language,  # None = Auto-detect
        "ref_audio": str(profile["ref_audio"]),
        "ref_text": profile["ref_text"],
    }
    if profile["instruct"]:
        kwargs["instruct"] = profile["instruct"]

    wavs, sr = model.generate_voice_clone(**kwargs)

    # Save as WAV first, then convert to OGG
    wav_path = output_path.replace(".ogg", ".wav")
    sf.write(wav_path, wavs[0], sr)

    if output_path.endswith(".ogg"):
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-c:a", "libopus", "-b:a", "64k", "-vbr", "on", output_path
        ], capture_output=True)
        os.remove(wav_path)

    duration = len(wavs[0]) / sr
    return duration

def play_audio(filepath: str):
    """Play audio with mpv (--no-video, terminal progress bar)."""
    subprocess.run([
        "mpv", "--no-video",
        "--term-osd-bar",
        "--term-osd-bar-chars=[=>-]",
        "--no-resume-playback",
        filepath
    ])

def main():
    console = Console()
    counter = {"dracula": 0, "alucard": 0}
    current_voice = None
    current_language = "Auto"

    # Welcome
    console.print(Panel.fit(
        "[bold]🧛 Castlevania Voice Clone TUI[/]\n\n"
        "Escolha uma voz de [bold]Symphony of the Night[/] e digite o texto.\n"
        "O áudio é gerado localmente com Qwen3-TTS 0.6B-Base.\n\n"
        "[dim]language=Auto detecta a língua do texto automaticamente.[/]",
        border_style="red",
    ))

    # Select voice
    current_voice = questionary.select(
        "Escolha a voz:",
        choices=list(VOICE_PROFILES.keys()),
    ).ask()

    if current_voice is None:
        return  # User pressed Ctrl+C

    # Select language
    current_language = questionary.select(
        "Língua de saída:",
        choices=LANGUAGE_OPTIONS,
    ).ask()

    if current_language is None:
        current_language = "Auto"

    console.print(f"[bold green]✓ {current_voice} | Língua: {current_language}[/]")
    prefix = VOICE_PROFILES[current_voice]["prefix"]

    # Main loop
    while True:
        console.rule(f"[bold]{current_voice}[/] ({current_language}) — Digite o texto (ou 'v' trocar voz, 'l' trocar língua, 'q' sair)")

        try:
            text = questionary.text("Texto>").ask()
        except KeyboardInterrupt:
            break

        if text is None or text.strip().lower() in ("q", "quit", "sair", "exit"):
            break

        if text.strip().lower() == "v":
            current_voice = questionary.select(
                "Escolha a voz:",
                choices=list(VOICE_PROFILES.keys()),
            ).ask()
            if current_voice is None:
                break
            prefix = VOICE_PROFILES[current_voice]["prefix"]
            console.print(f"[bold green]✓ Trocou para {current_voice}[/]")
            continue

        if text.strip().lower() == "l":
            current_language = questionary.select(
                "Língua de saída:",
                choices=LANGUAGE_OPTIONS,
            ).ask()
            if current_language is None:
                current_language = "Auto"
            console.print(f"[bold green]✓ Língua: {current_language}[/]")
            continue

        if not text.strip():
            continue

        counter[prefix] += 1
        output_path = f"/tmp/voice_{prefix}_{counter[prefix]:02d}.ogg"

        # Generate
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Gerando áudio...[/]]"),
            console=console,
        ) as progress:
            task = progress.add_task("generating", total=None)
            duration = generate(text.strip(), current_voice, output_path, language=current_language)
            progress.update(task, completed=1, total=1)

        lang_display = current_language if current_language != "Auto" else "detectado"
        console.print(f"[bold green]✓ Gerado:[/] {output_path} ({duration:.1f}s, língua: {lang_display})")
        console.print(f"[dim]Texto: \"{text.strip()[:60]}{'...' if len(text.strip()) > 60 else ''}\"[/]")

        # Play
        console.print("[bold]▶ Tocando...[/] (Ctrl+C para pular)")
        try:
            play_audio(output_path)
        except KeyboardInterrupt:
            console.print("[dim]Playback interrompido[/]")

        console.print(f"[dim]Salvo em: {output_path}[/]")
        console.print()

    # Summary
    total = sum(counter.values())
    console.print(f"\n[bold]Sessão encerrada.[/] {total} samples gerados.")
    for name, count in counter.items():
        if count:
            console.print(f"  {name}: {count} samples em /tmp/voice_{name}_*.ogg")

if __name__ == "__main__":
    main()
```

---

## Funcionamento do Player

### Opção escolhida: MPV subprocess

O player usa `mpv` direto como subprocess com flags mínimas:

```python
subprocess.run([
    "mpv", "--no-video",
    "--term-osd-bar",           # Barra de progresso no terminal
    "--term-osd-bar-chars=[=>-]", # Caracteres da barra
    "--no-resume-playback",     # Sempre começa do início
    filepath
])
```

**Por que MPV subprocess em vez de python-mpv?**

1. **Zero dependência extra** — mpv já está instalado no sistema (`mpv 0.41.0`)
2. **Barra de progresso nativa** — mpv mostra `▶ [=========>] 72% 3.8s/5.2s` no terminal automaticamente
3. **Controles de teclado embutidos** — espaço=pause, setas=seek, q=quit
4. **Nada sofisticado** — exatamente o que o Lucas pediu
5. **Ctrl+C** — interrompe o subprocesso e volta ao loop do TUI

### Rich Progress para geração

Enquanto o modelo gera o áudio, um spinner Rich mostra "Gerando áudio..." com animação. Após a geração, mostra a duração.

---

## Opção Futura: Player mais Integrado

Se no futuro quiser um player dentro do TUI (sem abrir mpv como subprocess), a melhor opção seria `python-mpv` + Rich Progress:

```python
import mpv
player = mpv.MPV(video=False)
player.play(filepath)
# Observar time-pos e duration para atualizar Rich Progress bar
```

Mas pra agora, subprocess com mpv é mais que suficiente e sem dependencies extras.

---

## Instalação Sob Demanda (Lazy Install)

### O problema com o venv antigo

O venv `~/projects/ai/voiceclone/.venv` está **quebrado** — `pip list` retorna vazio, `from qwen_tts import` funciona mas só porque os `.pyc` estão em cache. Não é confiável pra reusar.

### O problema com o venv do vLLM

O venv `~/.vllm/` tem tudo que a gente precisa (torch 2.11, numpy, einops, rich, torchaudio), MAS:

- **`transformers` é 5.8.0** — o `qwen-tts` demanda `==4.57.3` (downgrade!), e isso quebraria o vLLM
- **VERIFICADO**: Todos os 12 imports que o `qwen_tts` usa existem em `transformers 5.8.0`. O pin `==4.57.3` é restritivo demais.
- **`gradio`** é dependência do `qwen-tts` mas só usada em `cli/demo.py` (web UI). Não precisamos dela.
- **`accelerate`** é dependência declarada mas `qwen_tts` NÃO importa em nenhum lugar. O `device_map="auto"` vem do torch/transformers.
- **onnxruntime** é necessário pro tokenizer/codec do Qwen3-TTS.

### Estratégia: instalar no venv do vLLM com `--no-deps` + deps seletivas

Instalar `qwen-tts` com `--no-deps` pra evitar o downgrade de `transformers`, e depois instalar manualmente só as dependências que realmente faltam — sem `gradio`, sem downgradear `transformers`:

```python
#voice_tui.py — lazy install logic

VENV_PYTHON = os.path.expanduser("~/.vllm/bin/python")
REQUIREMENTS = [
    ("qwen-tts", "qwen_tts"),         # The model package (--no-deps!)
    ("soundfile",  "soundfile"),       # Audio I/O
    ("onnxruntime", "onnxruntime"),   # Tokenizer/codec inference
    ("librosa",   "librosa"),         # Audio processing
    ("sox",       "sox"),             # Audio format conversion
    ("questionary", "questionary"),   # TUI menus
]

def check_dependencies() -> bool:
    """Check if all required packages are available."""
    missing = []
    for pip_name, import_name in REQUIREMENTS:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    return len(missing) == 0

def install_dependencies(console: Console) -> bool:
    """Prompt user and install missing deps into ~/.vllm venv."""
    # List what's missing
    missing = []
    for pip_name, import_name in REQUIREMENTS:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return True

    console.print(Panel(
        f"[bold yellow]⚠ Dependências faltando![/]\n\n"
        f"Para usar Voice Clone, preciso instalar no venv do vLLM (~/.vllm/):\n\n"
        f"  {', '.join(missing)}\n\n"
        f"[dim]Isso NÃO vai downgradear o transformers (5.8.0 preservado).[/]\n"
        f"[dim]~300MB de download. O vLLM continua funcionando normalmente.[/]",
        title="Instalação Necessária",
        border_style="yellow",
    ))

    answer = questionary.confirm("Instalar as dependências?", default=True).ask()

    if not answer:
        console.print("[dim]Voltando ao menu de seleção de modelo.[/]")
        return False

    # Install qwen-tts with --no-deps to avoid transformers downgrade
    # Then install only the missing deps individually
    console.print("[bold]Instalando qwen-tts (sem deps)...")
    subprocess.run([
        VENV_PYTHON, "-m", "pip", "install", "--no-deps", "qwen-tts"
    ], check=True)

    # Install missing deps (skip what's already in vllm venv)
    for pip_name, import_name in REQUIREMENTS:
        if pip_name == "qwen-tts":
            continue  # Already installed above
        try:
            importlib.import_module(import_name)
        except ImportError:
            console.print(f"[dim]  Instalando {pip_name}...[/]")
            subprocess.run([
                VENV_PYTHON, "-m", "pip", "install", pip_name
            ], check=True)

    console.print("[bold green]✓ Dependências instaladas![/]")
    return True
```

### Ordem da instalação

1. `pip install --no-deps qwen-tts` — instala o pacote sem puxar `transformers==4.57.3`, `accelerate`, `gradio`
2. `pip install soundfile onnxruntime librosa sox questionary` — apenas as que faltam
3. O que já tá no vllm venv e NÃO precisa reinstalar: `torch`, `torchaudio`, `numpy`, `einops`, `rich`, `transformers` (5.8.0), `accelerate`

### Tamanho estimado

- ~300MB de download (onnxruntime é o maior: ~250MB)
- onnxruntime, librosa, scipy, scikit-learn, soundfile, sox, questionary

### Proteção contra quebra do vLLM

A instalação NÃO downgradeará `transformers` (5.8.0 → preservado). Todos os imports do `qwen_tts` foram verificados compatíveis com transformers 5.8.0.

Se algo quebrar no futuro: `~/.vllm/bin/pip install --force-reinstall transformers==5.8.0` restaura.

---

## Execução

```bash
# 1. Criar diretório do TUI
mkdir -p ~/projects/ai/voiceclone-tui/voices

# 2. Copiar arquivos de referência (apenas Drácula e Alucard!)
cp ~/projects/ai/voiceclone/samples/dracula3.wav ~/projects/ai/voiceclone-tui/voices/dracula_ref.wav
cp ~/projects/ai/voiceclone/samples/alucard.wav ~/projects/ai/voiceclone-tui/voices/alucard_ref.wav

# 3. Copiar o voice_tui.py para o diretório
# (implementação do código de referência acima, adaptada com lazy install)

# 4. Executar (deps serão verificadas e instaladas sob demanda)
cd ~/projects/ai/voiceclone-tui
~/.vllm/bin/python voice_tui.py
```

**Ou via tu (terminal-use):**
```bash
tu run --cwd ~/projects/ai/voiceclone-tui --name voice-clone -- ~/.vllm/bin/python voice_tui.py
```

---

## Considerações sobre VRAM

| Cenário | VRAM Usada | Viável? |
|---------|-----------|---------|
| Apenas TUI (0.6B-Base) | ~1.2GB | ✅ Sobra 4.8GB |
| TUI + LLM ativo no llama-swap | ~1.2GB + 3-4GB | ⚠️ Apertado (5.2-5.8GB) |
| Primeiro carregar modelo, depois usar | ~1.2GB após carregamento | ✅ Cache CUDA persiste |

**Estratégia recomendada:** Se o llama-swap tiver um modelo ativo, o TUI pode:
1. Carregar o modelo 0.6B-Base (~1.2GB BF16) e deixar em cache
2. Se der OOM, usar `device_map={"": "cpu"}` como fallback (mais lento, ~15-30s por frase)

---

## Passos de Implementação

1. **Criar diretório** `~/projects/ai/voiceclone-tui/` e `voices/`
2. **Copiar** `dracula3.wav` → `voices/dracula_ref.wav` e `alucard.wav` → `voices/alucard_ref.wav`
3. **Criar** `voice_tui.py` com a implementação acima
4. **Criar** `pyproject.toml` com dependências
5. **Symlink** ou reusar o `.venv` de `~/projects/ai/voiceclone/`
6. **Testar** — rodar o TUI, selecionar Drácula, digitar "What is a man?!", verificar áudio
7. **Testar** — trocar para Alucard, digitar texto, verificar áudio
8. **Ajustar** — instruções de estilo, tempo de geração, qualidade do áudio

---

## Riscos e Tradeoffs

| Risco | Mitigação |
|-------|-----------|
| `transformers` downgrade quebra vLLM | Instalar `qwen-tts` com `--no-deps`. Imports verificados compatíveis com 5.8.0. Recovery: `pip install --force-reinstall transformers==5.8.0` |
| VRAM conflita com LLM ativo | 0.6B modelo é pequeno (1.2GB). Fallback para CPU se necessário |
| Qualidade de clone pode variar | `instruct` field permite ajustar estilo. Drácula já tem `instruct` de raiva/poder |
| torch demora pra carregar | Primeira execução: ~10-15s. Depois cache aquecido: ~3-5s por geração |
| Arquivos WAV grandes (2MB+) | Converter para OGG/Opus 64kbps após geração (~10x menor) |
| mpv pode não estar disponível | Fallback para `aplay` (WAV) ou `ffplay` |
| venv antigo quebrado | Não reutilizar. Instalar no venv do vLLM com `--no-deps` |
| `qwen-tts` pin `transformers==4.57.3` | Contornado com `--no-deps`. Testado que todos os imports funcionam com 5.8.0 |
| onnxruntime grande (~250MB) | Necessário pro tokenizer. Sem alternativa mais leve no momento |

---

## Como funciona o parâmetro `language`

O `language` no Qwen3-TTS controla a **pronúncia/acento da saída**, não a língua do sample de entrada.

O model converte `language` em um **token ID** que é injetado no prompt ao lado do texto de saída. Ele funciona assim:

```python
# Do config.json do modelo:
codec_language_id = {
    "english": 2050,    # token "English" vira ID 2050 no prompt
    "german": 2053,
    "spanish": 2054,
    "chinese": 2055,
    "japanese": 2058,
    "french": 2061,
    "korean": 2064,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
}
```

- `language="Auto"` → o modelo detecta a língua do texto automaticamente (null, nenhum token de língua injetado)
- `language="English"` → força pronúncia/acento em inglês, mesmo que o texto seja em português
- `language="Portuguese"` → força pronúncia/acento em português

**O sample de referência (ref_audio + ref_text) define o *timbre da voz***. O parâmetro `language` define *como o modelo pronuncia o texto que você pediu pra falar*.

Isso significa: você PODE ter Drácula falando português com sotaque de filme de vampiro. A voz é clonada do sample, a língua é controlada separadamente.

### Implicação para o TUI

O TUI deve oferecer **escolha de língua de saída** independente da voz selecionada:

```
🦇 Drácula selecionado.
Língua: [English ▾] ← dropdown com Auto/English/Portuguese/etc.

> "Miserável pilha de segredos!"
  → Com language="Auto": pronuncia como português (detectado do texto)
  → Com language="English": pronuncia em inglês com o timbre do Drácula
  → Com language="Portuguese": garante sotajo português BR com o timbre do Drácula
```

O padrão para Drácula e Alucard será `"Auto"` — deixa o modelo detectar pela língua do texto.

---

## Perguntas Abertas (RESOLVIDAS)

1. ~~**Quer que o TUI também suporte português?**~~ ✅ RESOLVIDO: O `language` controla a pronúncia de saída, não a língua do sample. O TUI oferece dropdown de língua com "Auto" como padrão. A voz (timbre) vem do sample de referência, independente da língua escolhida.

2. ~~**Diretório de saída?**~~ ✅ `/tmp/` — samples são descartáveis, não precisa salvar.