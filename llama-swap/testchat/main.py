#!/usr/bin/env python3
"""
Chat com streaming de Reasoning e Response usando Rich Live Display.
Suporta modelos vision com envio de imagens inline no prompt.
Suporta modelos omni com envio de áudio inline no prompt.

Instalação:
    uv pip install rich questionary prompt-toolkit requests

Uso:
    uv run main.py

Imagens no prompt:
    O que tem escrito nessa imagem? ~/japanese.png
    Descreva essas imagens: /tmp/photo1.jpg ~/pics/photo2.png

Áudio no prompt (modelos omni como Nemotron-3-Nano-Omni):
    Transcreva esse áudio: ~/gravacao.wav
    O que foi dito aqui? ~/meeting.mp3 ~/clip.ogg
"""

import sys
import json
import os
import re
import base64
import hashlib
from pathlib import Path
from typing import Optional, Iterator
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich import box
import questionary
from prompt_toolkit.history import FileHistory
import requests

# Endpoint — use LLAMA_SWAP_HOST para acessar remotamente (ex: Termux via Tailscale)
# Default: localhost (rode direto na máquina)
# Remoto: export LLAMA_SWAP_HOST=100.65.187.74 (IP Tailscale)
LLAMA_SWAP_HOST = os.environ.get("LLAMA_SWAP_HOST", "127.0.0.1")
LLAMA_SWAP_PORT = int(os.environ.get("LLAMA_SWAP_PORT", "12434"))
BASE_URL = f"http://{LLAMA_SWAP_HOST}:{LLAMA_SWAP_PORT}/v1"

# Extensões de imagem suportadas pelo llama.cpp
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

# Extensões de áudio suportadas pelo llama.cpp (omni models)
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.opus'}

# Extensões de vídeo — ffmpeg extrai keyframes como imagens (omni models)
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mts', '.m4v'}
VIDEO_MAX_FRAMES = 16  # Limite de frames para não estourar contexto

# MIME types
MIME_MAP = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.bmp': 'image/bmp',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac',
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac',
    '.wma': 'audio/x-ms-wma',
    '.opus': 'audio/ogg',
    # Vídeo — não enviamos vídeo direto, extraímos frames como imagens
    '.mp4': 'video/mp4',
    '.mkv': 'video/x-matroska',
    '.webm': 'video/webm',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
}

console = Console()


def parse_media_paths(prompt: str) -> tuple[str, list[str], list[str], list[str]]:
    """
    Extrai caminhos de imagem, áudio e vídeo do prompt do usuário.
    
    Suporta:
      - Caminhos absolutos: /home/user/photo.jpg
      - Home com til: ~/pics/img.png
      - Caminhos relativos: ./images/test.jpeg
    
    Retorna:
      (texto_limpo, lista_de_caminhos_imagem, lista_de_caminhos_audio, lista_de_caminhos_video)
      O texto_limpo substitui os caminhos por [image N], [audio N], [video N].
    """
    # Pattern: caminhos que terminam com extensão de mídia
    # Captura: ~/path/img.png, /abs/path/audio.wav, ./rel/path/clip.mp3
    all_exts = '|'.join(e.lstrip('.') for e in (IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))
    pattern = rf'((?:~|[./]|\{{2}}/|/)[\w./\-]+\.({all_exts}))'
    
    matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
    if not matches:
        return prompt, [], [], []
    
    image_paths = []
    audio_paths = []
    video_paths = []
    clean_text = prompt
    
    # Processar em ordem reversa para não bagunçar os offsets
    image_counter = 0
    audio_counter = 0
    video_counter = 0
    for i, match in enumerate(reversed(matches)):
        path_str = match.group(1)
        
        # Expandir ~ para home
        expanded = os.path.expanduser(path_str)
        
        # Verificar se o arquivo existe
        if Path(expanded).is_file():
            ext = Path(expanded).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_counter += 1
                image_paths.insert(0, expanded)
                clean_text = clean_text[:match.start()] + f'[image {image_counter}]' + clean_text[match.end():]
            elif ext in AUDIO_EXTENSIONS:
                audio_counter += 1
                audio_paths.insert(0, expanded)
                clean_text = clean_text[:match.start()] + f'[audio {audio_counter}]' + clean_text[match.end():]
            elif ext in VIDEO_EXTENSIONS:
                video_counter += 1
                video_paths.insert(0, expanded)
                clean_text = clean_text[:match.start()] + f'[video {video_counter} ({VIDEO_MAX_FRAMES} frames)]' + clean_text[match.end():]
        # Se não existe, deixamos o texto original
    
    return clean_text, image_paths, audio_paths, video_paths


def encode_image_base64(image_path: str) -> tuple[str, str]:
    """
    Codifica uma imagem em base64 e retorna (base64_data_uri, mime_type).
    """
    ext = Path(image_path).suffix.lower()
    mime_type = MIME_MAP.get(ext, 'image/jpeg')
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    b64 = base64.b64encode(image_data).decode('utf-8')
    data_uri = f"data:{mime_type};base64,{b64}"
    
    return data_uri, mime_type


def encode_audio_base64(audio_path: str) -> tuple[str, str]:
    """
    Codifica um arquivo de áudio em base64 e retorna (base64_data_uri, mime_type).
    """
    ext = Path(audio_path).suffix.lower()
    mime_type = MIME_MAP.get(ext, 'audio/wav')
    
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    b64 = base64.b64encode(audio_data).decode('utf-8')
    data_uri = f"data:{mime_type};base64,{b64}"
    
    return data_uri, mime_type


def extract_video_frames(video_path: str, max_frames: int = VIDEO_MAX_FRAMES) -> tuple[list[str], str | None]:
    """
    Extrai keyframes e áudio de um vídeo com ffmpeg.
    
    Usa fps=1/2 (1 frame a cada 2 segundos) e limita ao max_frames.
    O áudio é extraído como WAV (16kHz mono) para compatibilidade com llama.cpp.
    As imagens são salvas em /tmp/video_frames_<hash>/ e removidas após o uso.
    
    Retorna:
      (lista_de_caminhos_frames, caminho_audio_ou_None)
    """
    import subprocess
    
    # Diretório temporário único por vídeo
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    frames_dir = Path(f"/tmp/video_frames_{video_hash}")
    frames_dir.mkdir(exist_ok=True)
    
    # Limpar frames antigos se existirem
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()
    
    # Extrair frames com ffmpeg (1 frame a cada 2 segundos, redimensiona p/ 1280px largura)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps=1/2,scale=1280:-1",
                "-q:v", "3",  # Qualidade JPEG-like boa
                str(frames_dir / "frame_%04d.png")
            ],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            console.print(f"[red]ffmpeg erro: {result.stderr[:200]}[/]")
            return []
    except FileNotFoundError:
        console.print("[red]ffmpeg não encontrado. Instale: pacman -S ffmpeg[/]")
        return []
    except subprocess.TimeoutExpired:
        console.print("[red]ffmpeg timeout ao extrair frames do vídeo[/]")
        return []
    
    # Coletar e ordenar os frames
    frames = sorted(frames_dir.glob("frame_*.png"))
    
    # Limitar ao máximo de frames (seleção uniforme)
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frames = [frames[i] for i in indices]
    
    # Extrair áudio como WAV 16kHz mono (compatível com llama.cpp)
    audio_path = str(frames_dir / "audio.wav")
    audio_result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                    # Sem vídeo
            "-acodec", "pcm_s16le",  # WAV PCM 16-bit
            "-ar", "16000",           # 16kHz sample rate
            "-ac", "1",               # Mono
            audio_path
        ],
        capture_output=True, text=True, timeout=60
    )
    if audio_result.returncode != 0 or not Path(audio_path).is_file():
        # vídeo sem áudio ou erro — silenciosamente ignora
        audio_path = None
    else:
        audio_size = Path(audio_path).stat().st_size / (1024 * 1024)
        if audio_size > 25:  # >25MB de áuito WAV = ~26 min, pula pra não estourar contexto
            Path(audio_path).unlink()
            audio_path = None
    
    return [str(f) for f in frames], audio_path


def build_message_content(prompt: str, image_paths: list[str] = None, audio_paths: list[str] = None) -> str | list:
    """
    Constroi o conteúdo da mensagem do usuário.
    
    Se há imagens ou áudio, retorna formato OpenAI Vision (lista com text + image_url/input_audio).
    Se não há mídia, retorna string simples (compatível com qualquer modelo).
    """
    image_paths = image_paths or []
    audio_paths = audio_paths or []
    
    if not image_paths and not audio_paths:
        return prompt
    
    content_parts = []
    
    # Adicionar o texto primeiro
    if prompt.strip():
        content_parts.append({
            "type": "text",
            "text": prompt
        })
    
    # Adicionar cada imagem
    for i, img_path in enumerate(image_paths):
        try:
            data_uri, mime_type = encode_image_base64(img_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": data_uri
                }
            })
        except Exception as e:
            # Se falhar ao ler a imagem, nota no texto
            content_parts.append({
                "type": "text",
                "text": f"[Erro ao carregar imagem {i+1}: {e}]"
            })
    
    # Adicionar cada áudio (formato OpenAI input_audio)
    for i, audio_path in enumerate(audio_paths):
        try:
            data_uri, mime_type = encode_audio_base64(audio_path)
            # OpenAI format for audio input
            content_parts.append({
                "type": "input_audio",
                "input_audio": {
                    "data": data_uri,
                    "format": mime_type.split('/')[-1]  # wav, mp3, ogg, etc.
                }
            })
        except Exception as e:
            content_parts.append({
                "type": "text",
                "text": f"[Erro ao carregar áudio {i+1}: {e}]"
            })
    
    return content_parts


def fetch_available_models():
    """Busca modelos disponíveis do endpoint /v1/models.
    
    Deduplica aliases (modelo = mesmo name), mantendo o ID mais descritivo.
    Separa variantes :think como propriedade do modelo base.
    """
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Agrupar por name — aliases compartilham o mesmo name
        # Um modelo pode aparecer várias vezes (ID principal + aliases)
        # e pode ter variante :think separada
        base_models = {}  # name -> model dict (melhor ID)
        think_variants = {}  # name -> think model dict
        seen_ids = set()  # Track IDs já processados
        
        for model in data.get("data", []):
            model_id = model.get("id", "")
            
            # Ignorar IDs duplicados (API pode retornar o mesmo alias duas vezes)
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            
            is_think = model_id.endswith(":think")
            name = model.get("name", model_id)
            description = model.get("description", "")
            
            meta = model.get("meta", {})
            llamaswap = meta.get("llamaswap", {})
            features = llamaswap.get("features", {})
            has_thinking = features.get("thinking", False)
            has_vision = features.get("vision", False)
            has_tools = features.get("tools", False)
            context = llamaswap.get("context", "")
            vram = llamaswap.get("vram_usage", "")
            size = llamaswap.get("size", "")
            warning = llamaswap.get("warning", "")
            source = llamaswap.get("source", "")
            kv_cache = llamaswap.get("kv_cache", "")
            
            model_info = {
                "id": model_id,
                "name": name,
                "description": description,
                "features": features,
                "supports_thinking": has_thinking,
                "supports_vision": has_vision,
                "supports_tools": has_tools,
                "context": context,
                "vram": vram,
                "size": size,
                "warning": warning,
                "source": source,
                "kv_cache": kv_cache,
            }
            
            if is_think:
                think_variants[name] = model_info
            else:
                # Deduplicar por name: priorizar ID canônico sobre aliases
                # Heurística: IDs com pontos e versões são mais canônicos
                # (ex: "qwen3.5-4b" > "qwen3.5-4b-q4" > "q4")
                def canonical_score(mid):
                    """Score de canonicidade: maior = mais canônico."""
                    s = 0
                    # IDs com versão (pontos) são mais canônicos
                    if '.' in mid:
                        s += 10
                    # IDs com tamanho de parâmetro (ex: -4b, -9b, -27b, -30b, -450m) são mais específicos
                    if re.search(r'-\d+[bm](?:$|-)', mid):
                        s += 8
                    # Penalizar aliases muito curtos (ex: "q4", "n4", "vl")
                    if len(mid) <= 3:
                        s -= 20
                    # Penalizar aliases que terminam em -q4 (quantização)
                    if mid.endswith('-q4') or mid.endswith('-q4_k_m'):
                        s -= 5
                    return s
                
                if name not in base_models:
                    base_models[name] = model_info
                else:
                    existing = base_models[name]
                    new_score = canonical_score(model_id)
                    old_score = canonical_score(existing["id"])
                    # Score maior vence; empate: nome mais longo vence (mais descritivo)
                    if new_score > old_score or (new_score == old_score and len(model_id) > len(existing["id"])):
                        base_models[name] = model_info
        
        # Montar lista final: base models com flag has_think_variant
        models = []
        for name, info in sorted(base_models.items(), key=lambda x: x[0].lower()):
            has_think = name in think_variants
            info["has_think_variant"] = has_think
            info["think_variant_id"] = f"{info['id']}:think" if has_think else None
            models.append(info)
        
        return models
        
    except Exception as e:
        console.print(f"[red]Erro ao buscar modelos: {e}[/]")
        return []


class StreamingChat:
    """Chat com streaming usando Rich Live."""
    
    def __init__(self):
        self.reasoning_content = ""
        self.response_content = ""
        self.has_reasoning = False
        self.has_response = False
        self.status = "Aguardando..."
        self.selected_model = ""
        self.selected_model_name = ""  # Nome amigável do modelo
        self.supports_reasoning = False
        self.use_thinking_variant = False  # True se variante :think foi selecionada
        self.supports_vision = False
        self.supports_audio = False
        self.available_models = []
        self.image_paths = []  # Caminhos das imagens extraídas do prompt
        self.audio_paths = []  # Caminhos dos áudios extraídos do prompt
        self.video_paths = []  # Caminhos dos vídeos extraídos do prompt
    
    def check_model_reasoning(self) -> bool:
        """Verifica se o modelo suporta reasoning usando o metadata da API.
        
        Para variantes :think, procura o modelo base e sempre retorna True.
        Para modelos base com thinking, retorna True.
        """
        # Se é variante :think, sabemos que suporta reasoning
        model_id_for_lookup = self.selected_model
        is_think_variant = self.selected_model.endswith(":think")
        if is_think_variant:
            model_id_for_lookup = self.selected_model.removesuffix(":think")
        
        for model in self.available_models:
            if model["id"] == model_id_for_lookup:
                supports = model["supports_thinking"]
                if is_think_variant:
                    supports = True  # Variante :think sempre tem reasoning
                self.supports_vision = model.get("supports_vision", False)
                self.supports_audio = model.get("features", {}).get("audio", False)
                features = []
                if supports:
                    features.append("reasoning")
                if self.supports_vision:
                    features.append("vision")
                if self.supports_audio:
                    features.append("audio")
                if features:
                    console.print(f"[green]✓ Modelo suporta: {', '.join(features)}[/]")
                else:
                    console.print("[green]✓ Modelo sem features especiais[/]")
                return supports
        
        console.print("[yellow]⚠ Modelo não encontrado nos metadados, assumindo sem reasoning[/]")
        return False
    
    def _format_model_label(self, model: dict) -> str:
        """Formata label multiline do modelo para o menu de seleção.
        
        Layout:
            Model Name  🤔 thinking  👁️ vision  ⚡ :think
            id • Size • VRAM • Context
            ⚠️ warning (se houver)
        """
        name = model["name"]
        model_id = model["id"]
        size = model.get("size", "")
        vram = model.get("vram", "")
        context = model.get("context", "")
        warning = model.get("warning", "")
        has_thinking = model.get("supports_thinking", False)
        has_vision = model.get("supports_vision", False)
        has_tools = model.get("supports_tools", False)
        has_think_variant = model.get("has_think_variant", False)
        
        # Linha 1: Nome + badges de features
        badges = []
        if has_thinking:
            badges.append("🤔")
        if has_vision:
            badges.append("👁️")
        if has_tools:
            badges.append("🛠️")
        if has_think_variant:
            badges.append("⚡:think")
        
        line1 = name
        if badges:
            line1 += "  " + " ".join(badges)
        
        # Linha 2: ID + size + VRAM + context
        info_parts = [model_id]
        if size:
            info_parts.append(size)
        if vram:
            info_parts.append(vram)
        if context:
            info_parts.append(f"ctx:{context}")
        line2 = " • ".join(info_parts)
        
        label = f"{line1}\n    {line2}"
        
        # Linha 3: warning (se houver)
        if warning:
            label += f"\n    ⚠️ {warning}"
        
        return label
    
    def select_model(self) -> str:
        """Mostra a lista de modelos em menu navegável com seleção de variante.
        
        Fluxo:
        1. Seleciona modelo base (lista deduplicada com descrição multiline)
        2. Se o modelo tem variante :think, pergunta se quer usar thinking
        """
        console.clear()
        
        # Busca modelos da API
        self.available_models = fetch_available_models()
        
        if not self.available_models:
            console.print("[red]Nenhum modelo disponível. Verifique se o servidor está rodando.[/]")
            sys.exit(1)
        
        questionary.print("🤖 Selecione o modelo para o chat", style="bold cyan")
        
        # Prepara as opções com label multiline
        choices = []
        for model in self.available_models:
            label = self._format_model_label(model)
            choices.append(label)
        
        try:
            selected = questionary.select(
                "Use ↑↓ para navegar e Enter para selecionar:",
                choices=choices,
                use_indicator=True,
                use_arrow_keys=True,
                style=questionary.Style([
                    ('question', 'bold #00d4ff'),       # Azul claro vibrante
                    ('selected', 'bg:#005fff #ffffff bold'),  # Fundo azul, texto branco
                    ('pointer', 'bold #00d4ff'),         # Seta azul vibrante
                    ('instruction', 'dim'),
                    ('answer', 'bold #00d4ff'),          # Resposta final em azul
                    ('highlighted', 'bg:#005fff #ffffff bold'),  # Item em destaque
                ])
            ).ask()
        except KeyboardInterrupt:
            console.print("[dim]Saindo...[/]")
            sys.exit(0)
        if not selected:
            console.print("[dim]Saindo...[/]")
            sys.exit(0)
        
        # Extrai o model_id da primeira linha do label selecionado
        # Label format: "Name  badges...\n  model-id • size • vram • ctx: ..."
        first_line = selected.split("\n")[0]
        # Mas o model_id está na segunda linha, primeiro campo
        second_line = selected.split("\n")[1] if "\n" in selected else ""
        # Parse: "  model-id • size • vram • ctx: ..."
        model_id = second_line.strip().split(" • ")[0].strip() if second_line else first_line.split(" - ")[0].strip()
        
        # Encontra o modelo na lista
        selected_model = None
        for model in self.available_models:
            if model["id"] == model_id:
                selected_model = model
                break
        
        if not selected_model:
            # Fallback: tenta pelo nome
            for model in self.available_models:
                if model["name"] in first_line:
                    selected_model = model
                    model_id = model["id"]
                    break
        
        if not selected_model:
            console.print(f"[red]Erro: modelo não encontrado na lista[/]")
            sys.exit(1)
        
        self.selected_model_name = selected_model["name"]
        self.supports_vision = selected_model.get("supports_vision", False)
        self.supports_audio = selected_model.get("features", {}).get("audio", False)
        
        # Etapa 2: Seleção de variante (se disponível)
        final_model_id = model_id
        
        if selected_model.get("has_think_variant"):
            think_id = selected_model["think_variant_id"]
            
            console.print()
            console.print(Panel(
                f"[bold]{selected_model['name']}[/]\n"
                f"[dim]{selected_model.get('description', '')}[/]",
                title="Modelo selecionado",
                border_style="cyan"
            ))
            
            variant_choices = [
                f"💬 Chat normal (sem reasoning explícito)",
                f"🤔 Reasoning/Thinking (:think — rag chain-of-thought visível)",
            ]
            
            try:
                variant = questionary.select(
                    "Qual variante?",
                    choices=variant_choices,
                    use_indicator=True,
                    use_arrow_keys=True,
                    style=questionary.Style([
                        ('question', 'bold #ffab00'),          # Âmbar vibrante
                        ('selected', 'bg:#ff8f00 #000000 bold'),  # Fundo âmbar, texto preto
                        ('pointer', 'bold #ffab00'),          # Seta âmbar
                        ('instruction', 'dim'),
                        ('answer', 'bold #ffab00'),           # Resposta final âmbar
                        ('highlighted', 'bg:#ff8f00 #000000 bold'),  # Item em destaque
                    ])
                ).ask()
            except KeyboardInterrupt:
                console.print("[dim]Saindo...[/]")
                sys.exit(0)
            
            if not variant:
                console.print("[dim]Saindo...[/]")
                sys.exit(0)
            
            if "Reasoning" in variant or ":think" in variant:
                final_model_id = think_id
                self.use_thinking_variant = True
                console.print(f"[yellow]⚡ Variante :think selecionada[/]")
            else:
                self.use_thinking_variant = False
                console.print(f"[green]💬 Variante chat normal selecionada[/]")
        
        console.print(f"[green]✓ Modelo selecionado:[/] {final_model_id}")
        
        return final_model_id
        
    def create_layout(self) -> Layout:
        """Cria o layout da interface.
        
        Só mostra o painel de raciocínio se a variante :think foi selecionada.
        Modelos com thinking capability mas em modo chat normal não mostram raciocínio.
        """
        layout = Layout()
        
        # Mostra split reasoning+response SÓ se thinking variant foi selecionada
        if self.use_thinking_variant:
            # Layout com split: reasoning (esquerda) e response (direita)
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3)
            )
            
            # Body divide em reasoning (esquerda) e response (direita)
            layout["body"].split_row(
                Layout(name="reasoning", ratio=1),
                Layout(name="response", ratio=1)
            )
        else:
            # Layout simples: só response em tela cheia
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="response"),
                Layout(name="footer", size=3)
            )
        
        return layout
    
    def _fit_markdown(self, text: str, available_height: int, title: str = "") -> "Panel":
        """Renderiza Markdown com scroll responsivo — sempre preenche 100% da área útil.
        
        Comportamento:
        - Se o conteúdo cabe inteiro: mostra Markdown completo (formatado bonito)
        - Se transborda: fatia as últimas N linhas da saída renderizada (ANSI),
          preservando cores/negrito via Text.from_ansi(). O painel sempre fica
          100% cheio, rolando suavemente linha a linha.
        
        Args:
            text: Conteúdo Markdown para renderizar
            available_height: Linhas disponíveis no painel (borda + conteúdo)
            title: Título do painel
        
        Returns:
            Panel com conteúdo que preenche exatamente o espaço disponível
        """
        # Espaço útil = altura disponível - bordas do Panel (2 top + 2 bottom) - título (1)
        usable = max(5, available_height - 5)
        
        if not text or not text.strip():
            return Panel(
                Align.center(Text("\n\nAguardando...", style="dim")),
                title=title,
                border_style="yellow" if "Raciocínio" in title else "green",
                box=box.ROUNDED
            )
        
        # Renderiza Markdown completo num Console virtual
        md = Markdown(text)
        from rich.console import Console as RichConsole
        c = RichConsole(width=self._console_width or 80, legacy_windows=False, force_terminal=True)
        with c.capture() as capture:
            c.print(md)
        ansi_output = capture.get()
        
        # Conta linhas renderizadas
        lines = ansi_output.split("\n")
        # Remove trailing empty line do c.print
        if lines and lines[-1] == "":
            lines = lines[:-1]
        
        total_lines = len(lines)
        
        # Se coube inteiro, retorna Markdown formatado (mais bonito)
        if total_lines <= usable:
            return Panel(md, title=title, border_style="yellow" if "Raciocínio" in title else "green", box=box.ROUNDED)
        
        # Transbordou: fatia as últimas `usable` linhas da saída ANSI
        # Text.from_ansi() preserva cores e formatação
        tail_lines = lines[-usable:]
        # Adiciona indicador de truncamento no topo
        tail_with_hint = ["\x1b[2m  ⬆ …\x1b[0m"] + tail_lines
        tail_ansi = "\n".join(tail_with_hint)
        tail_text = Text.from_ansi(tail_ansi)
        
        return Panel(tail_text, title=title, border_style="yellow" if "Raciocínio" in title else "green", box=box.ROUNDED)
    
    @property
    def _console_width(self):
        """Retorna a largura do terminal para o Console virtual."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80
    
    def render(self) -> Layout:
        """Renderiza a interface completa com auto-scroll responsivo.
        
        Usa Markdown rendering com truncamento inteligente por parágrafos.
        Mede a altura renderizada com Console virtual e corta do topo
        até caber no espaço disponível do terminal.
        """
        layout = self.create_layout()
        
        # Calcular altura disponível para cada painel
        try:
            term_height = os.get_terminal_size().lines
        except OSError:
            term_height = 24
        
        # Header: 3 linhas, Footer: 3 linhas
        # Layout split: cada painel (reasoning/response) usa ~45% do restante
        # Layout full: response usa ~90% do restante
        body_height = term_height - 6  # -3 header -3 footer
        if self.use_thinking_variant:
            reasoning_height = max(8, int(body_height * 0.45))
            response_height = max(8, body_height - reasoning_height)
        else:
            reasoning_height = 0
            response_height = max(10, body_height)
        
        # Header com nome do modelo e variante
        display_name = self.selected_model_name or self.selected_model or "Chat"
        if self.use_thinking_variant:
            display_name += " :think"
        header_text = Text(f"🤖 {display_name}", style="bold cyan")
        header = Align.center(header_text)
        layout["header"].update(Panel(header, border_style="cyan"))
        
        # Reasoning panel (só se :think foi selecionado)
        if self.use_thinking_variant:
            if self.reasoning_content or self.has_reasoning:
                r_content = f"## 🤔 Raciocínio\n\n{self.reasoning_content}"
            else:
                r_content = ""
            reasoning_panel = self._fit_markdown(r_content, reasoning_height, title="[yellow]Raciocínio[/]")
            layout["reasoning"].update(reasoning_panel)
        
        # Response panel (sempre presente)
        if self.response_content or self.has_response:
            resp_content = f"## 💬 Resposta\n\n{self.response_content}"
        else:
            resp_content = ""
        response_panel = self._fit_markdown(resp_content, response_height, title="[green]Resposta[/]")
        layout["response"].update(response_panel)
        
        # Footer com status
        footer_text = Text(f"⏳ {self.status}", style="dim")
        if self.has_reasoning or self.has_response:
            if self.use_thinking_variant:
                stats = f"R: {len(self.reasoning_content)} | Resp: {len(self.response_content)} chars"
            else:
                stats = f"Resp: {len(self.response_content)} chars"
            footer_text = Text(f"✅ {stats}", style="bold green")
        
        footer_panel = Panel(Align.center(footer_text), border_style="dim")
        layout["footer"].update(footer_panel)
        
        return layout
    
    def stream_chat(self, prompt: str, image_paths: list[str] = None, audio_paths: list[str] = None) -> Iterator[Layout]:
        """Realiza o streaming do chat usando requests diretamente."""
        image_paths = image_paths or []
        audio_paths = audio_paths or []
        
        try:
            # Construir conteúdo da mensagem
            message_content = build_message_content(prompt, image_paths, audio_paths)
            
            # Faz POST com streaming
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                json={
                    "model": self.selected_model,
                    "messages": [{"role": "user", "content": message_content}],
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Decode UTF-8 explicitly
                line = line.decode('utf-8')
                    
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: "
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" not in data or not data["choices"]:
                            continue
                        
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                        
                        if reasoning:
                            if not self.has_reasoning:
                                self.has_reasoning = True
                                self.status = "Raciocinando..."
                            self.reasoning_content += reasoning
                            yield self.render()
                        
                        if content:
                            if not self.has_response:
                                self.has_response = True
                                self.status = "Gerando resposta..."
                            self.response_content += content
                            yield self.render()
                    except json.JSONDecodeError:
                        continue
            
            self.status = "Concluído!"
            yield self.render()
            
        except Exception as e:
            self.status = f"Erro: {str(e)[:50]}"
            yield self.render()
    
    def run(self):
        """Executa o chat interativo."""
        # Seleciona o modelo primeiro (isso já popula available_models)
        self.selected_model = self.select_model()
        
        # Determina se suporta reasoning usando metadata da API
        self.supports_reasoning = self.check_model_reasoning()
        
        # Indica suporte a visão se disponível
        if self.supports_vision:
            console.print("[cyan]👁️ Este modelo suporta visão — você pode incluir caminhos de imagem no prompt[/]")
            console.print("[dim]   Ex: O que tem nessa imagem? ~/photo.png[/]")
        if self.supports_audio:
            console.print("[cyan]🎙️ Este modelo suporta áudio — você pode incluir caminhos de áudio no prompt[/]")
            console.print("[dim]   Ex: Transcreva esse áudio: ~/gravacao.wav[/]")
        if self.supports_vision:
            console.print("[cyan]🎬 Este modelo suporta vídeo — inclua caminhos de vídeo no prompt[/]")
            console.print("[dim]   Ex: Resuma o que acontece nesse vídeo: ~/demo.mp4[/]")
            console.print(f"[dim]   (ffmpeg extrai até {VIDEO_MAX_FRAMES} keyframes e envia como imagens)[/]")
            if not self.supports_audio:
                console.print("[yellow]⚠️  Áudio do vídeo NÃO será enviado (modelo sem suporte a áudio via GGUF)[/]")
                console.print("[dim]   Apenas os frames visuais serão processados.[/]")
        if self.use_thinking_variant:
            console.print("[yellow]🤔 Modo reasoning ativado — painel de raciocínio visível[/]")
        elif self.supports_reasoning:
            console.print("[dim]💡 Modo chat normal — raciocínio desabilitado (use variante :think para reasoning)[/]")
        console.print()
        
        console.print("[dim]Pressione Ctrl+C para sair a qualquer momento[/]")
        console.print("[dim]↑↓ no prompt para histórico[/]")
        console.print()
        
        # Configura histórico persistente
        history_file = os.path.expanduser("~/.rich_chat_history")
        history = FileHistory(history_file)
        
        try:
            prompt = questionary.text(
                "Sua pergunta:",
                history=history,
                style=questionary.Style([
                    ('question', 'bold'),
                ])
            ).ask()
        except KeyboardInterrupt:
            console.print("[dim]Saindo...[/]")
            return
        
        if prompt is None:
            console.print("[dim]Saindo...[/]")
            return
        
        if not prompt.strip():
            console.print("[red]Prompt vazio![/]")
            return
        
        # Parsear caminhos de mídia do prompt
        clean_prompt, image_paths, audio_paths, video_paths = parse_media_paths(prompt)
        
        # Processar vídeos: extrair keyframes + áudio com ffmpeg
        video_frame_count = 0
        video_audio_paths = []  # Áudio extraído dos vídeos
        if video_paths:
            console.print(f"[cyan]🎬 {len(video_paths)} vídeo(s) detectado(s), extraindo keyframes com ffmpeg...[/]")
            for i, vpath in enumerate(video_paths, 1):
                fname = Path(vpath).name
                fsize = Path(vpath).stat().st_size / (1024 * 1024)
                console.print(f"[dim]   [{i}] {fname} ({fsize:.1f} MB) — extraindo frames e áudio...[/]")
                frames, v_audio = extract_video_frames(vpath)
                if frames:
                    video_frame_count += len(frames)
                    image_paths.extend(frames)
                    console.print(f"[dim]      ✓ {len(frames)} frames extraídos[/]")
                else:
                    console.print(f"[yellow]      ⚠ Falha ao extrair frames[/]")
                if v_audio:
                    if self.supports_audio:
                        video_audio_paths.append(v_audio)
                        audio_paths.append(v_audio)
                        audio_size = Path(v_audio).stat().st_size / (1024 * 1024)
                        console.print(f"[dim]      ♪ áudio extraído ({audio_size:.1f} MB)[/]")
                    else:
                        # Modelo sem suporte a áudio — descartar áudio extraído
                        console.print(f"[dim]      ♪ áudio extraído mas DESCARTADO (modelo não suporta áudio)[/]")
            console.print(f"[dim]   Prompt: {clean_prompt}[/]")
            console.print()
        
        if image_paths:
            # Separar imagens originais dos frames de vídeo para exibição
            original_images = [p for p in image_paths if not p.startswith("/tmp/video_frames_")]
            if original_images:
                console.print(f"[cyan]🖼️  {len(original_images)} imagem(ns) detectada(s):[/]")
                for i, path in enumerate(original_images, 1):
                    fname = Path(path).name
                    fsize = Path(path).stat().st_size / 1024
                    console.print(f"[dim]   [{i}] {fname} ({fsize:.1f} KB)[/]")
            if video_frame_count:
                console.print(f"[cyan]🎬  + {video_frame_count} frames de vídeo[/]")
            console.print(f"[dim]   Total de mídia visual: {len(image_paths)} imagem(ns)[/]")
            console.print()
        
        if audio_paths:
            # Filtrar áudios se modelo não suporta
            if not self.supports_audio:
                standalone_audio = [a for a in audio_paths if not a.startswith("/tmp/video_frames_")]
                if standalone_audio:
                    console.print(f"[yellow]⚠️  {len(standalone_audio)} arquivo(s) de áudio ignorado(s) — modelo não suporta áudio via GGUF[/]")
                    audio_paths = [a for a in audio_paths if a.startswith("/tmp/video_frames_")]
                    # Se ainda tem áudios de vídeo que foram descartados, limpa também
                    audio_paths = []
                    video_audio_paths = []
            if audio_paths:
                console.print(f"[cyan]🎙️  {len(audio_paths)} áudio(s) detectado(s):[/]")
                for i, path in enumerate(audio_paths, 1):
                    fname = Path(path).name
                    fsize = Path(path).stat().st_size / 1024
                    console.print(f"[dim]   [{i}] {fname} ({fsize:.1f} KB)[/]")
                console.print(f"[dim]   Prompt: {clean_prompt}[/]")
                console.print()
        
        if not image_paths and not audio_paths and not video_paths:
            clean_prompt = prompt
        
        console.print("[dim]Iniciando streaming...[/]")
        console.print()
        
        # Inicia Live display com screen=True (tela alternativa)
        with Live(
            self.render(),
            screen=True,
            refresh_per_second=15,  # ~60fps suave
            vertical_overflow="visible"
        ) as live:
            try:
                for layout in self.stream_chat(clean_prompt, image_paths, audio_paths):
                    live.update(layout)
                
                # Mostra resultado final com prompt e modelo
                live.stop()
                console.clear()
                
                # Cabeçalho com prompt e modelo
                model_display = self.selected_model_name or self.selected_model
                if self.use_thinking_variant:
                    model_display += " :think"
                
                # Mostra o prompt original (com caminhos) se tinha imagens
                display_prompt = prompt  # prompt original
                header_parts = [f"[bold cyan]Prompt:[/] {display_prompt}"]
                if image_paths:
                    header_parts.append(f"[dim]Imagens: {len(image_paths)} anexada(s)[/]")
                if audio_paths:
                    header_parts.append(f"[dim]Áudios: {len(audio_paths)} anexado(s)[/]")
                if video_paths:
                    detail = f"Vídeos: {len(video_paths)} ({video_frame_count} frames"
                    if video_audio_paths:
                        detail += f" + {len(video_audio_paths)} áudios"
                    detail += ")"
                    header_parts.append(f"[dim]{detail}[/]")
                header_parts.append(f"[dim]Modelo:[/] {model_display}")
                
                console.print(Panel(
                    "\n".join(header_parts),
                    title="✅ Concluído",
                    border_style="green"
                ))
                console.print()
                
                if self.use_thinking_variant and self.reasoning_content:
                    console.print(Panel(
                        Markdown(f"## 🤔 Raciocínio\n\n{self.reasoning_content}"),
                        title="Raciocínio",
                        border_style="yellow"
                    ))
                
                console.print(Panel(
                    Markdown(f"## 💬 Resposta\n\n{self.response_content}"),
                    title="Resposta",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                live.stop()
                console.print("[dim]Interrompido pelo usuário[/]")
            finally:
                # Limpa frames e áudio de vídeo temporários
                if video_paths:
                    cleaned = video_frame_count + len(video_audio_paths)
                    for vpath in video_paths:
                        video_hash = hashlib.md5(vpath.encode()).hexdigest()[:8]
                        frames_dir = Path(f"/tmp/video_frames_{video_hash}")
                        if frames_dir.exists():
                            for f in frames_dir.iterdir():
                                f.unlink()
                            frames_dir.rmdir()
                    if cleaned:
                        console.print(f"[dim]🗑️  Arquivos temporários removidos ({video_frame_count} frames + {len(video_audio_paths)} áudios)[/]")


def main():
    """Entry point."""
    chat = StreamingChat()
    chat.run()


if __name__ == "__main__":
    main()