#!/usr/bin/env python3
"""
Chat com streaming de Reasoning e Response usando Rich Live Display.
Suporta modelos vision com envio de imagens inline no prompt.

Instalação:
    uv pip install rich questionary prompt-toolkit requests

Uso:
    uv run main.py

Imagens no prompt:
    O que tem escrito nessa imagem? ~/japanese.png
    Descreva essas imagens: /tmp/photo1.jpg ~/pics/photo2.png
"""

import sys
import json
import os
import re
import base64
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
}

console = Console()


def parse_image_paths(prompt: str) -> tuple[str, list[str]]:
    """
    Extrai caminhos de imagem do prompt do usuário.
    
    Suporta:
      - Caminhos absolutos: /home/user/photo.jpg
      - Home com til: ~/pics/img.png
      - Caminhos relativos: ./images/test.jpeg
    
    Retorna:
      (texto_limpo, lista_de_caminhos)
      O texto_limpo substitui os caminhos por [image 1], [image 2], etc.
    """
    # Pattern: caminhos que terminam com extensão de imagem
    # Captura: ~/path/img.png, /abs/path/img.jpg, ./rel/path/img.webp
    pattern = r'((?:~|[./]|\.{2}/|/)[\w./\-]+\.(?:jpg|jpeg|png|bmp|gif|webp|tiff|tif))'

    matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
    if not matches:
        return prompt, []

    paths = []
    clean_text = prompt
    
    # Processar em ordem reversa para não bagunçar os offsets
    for i, match in enumerate(reversed(matches)):
        original_idx = len(matches) - 1 - i
        image_num = original_idx + 1
        path_str = match.group(1)
        
        # Expandir ~ para home
        expanded = os.path.expanduser(path_str)
        
        # Verificar se o arquivo existe (silenciosamente — pode ser caminho inválido)
        if Path(expanded).is_file():
            paths.insert(0, expanded)
            # Substituir o caminho por tag [image N]
            clean_text = clean_text[:match.start()] + f'[image {image_num}]' + clean_text[match.end():]
        # Se não existe, deixamos o texto original (não é um caminho de imagem válido)
    
    return clean_text, paths


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


def build_message_content(prompt: str, image_paths: list[str]) -> str | list:
    """
    Constroi o conteúdo da mensagem do usuário.
    
    Se há imagens, retorna formato OpenAI Vision (lista com text + image_url).
    Se não há imagens, retorna string simples (compatível com qualquer modelo).
    """
    if not image_paths:
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
    
    return content_parts


def fetch_available_models():
    """Busca modelos disponíveis do endpoint /v1/models."""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            name = model.get("name", model_id)
            description = model.get("description", "")
            
            # Verifica features
            meta = model.get("meta", {})
            llamaswap = meta.get("llamaswap", {})
            features = llamaswap.get("features", {})
            has_thinking = features.get("thinking", False)
            has_vision = features.get("vision", False)
            has_tools = features.get("tools", False)
            
            # Coleta extras para mostrar
            extras = []
            if has_thinking:
                extras.append("🤔 thinking")
            if has_vision:
                extras.append("👁️ vision")
            if has_tools:
                extras.append("🛠️ tools")
            
            context = llamaswap.get("context", "")
            vram = llamaswap.get("vram_usage", "")
            
            if context:
                extras.append(f"📏 {context} context")
            if vram:
                extras.append(f"💾 {vram}")
            
            display_desc = description
            if extras:
                display_desc = f"{description} [{', '.join(extras)}]"
            
            models.append({
                "id": model_id,
                "name": name,
                "description": description,
                "display_desc": display_desc,
                "supports_thinking": has_thinking,
                "supports_vision": has_vision,
                "features": features,
                "context": context,
                "vram": vram,
            })
        
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
        self.supports_vision = False
        self.available_models = []
        self.image_paths = []  # Caminhos das imagens extraídas do prompt
    
    def check_model_reasoning(self) -> bool:
        """Verifica se o modelo suporta reasoning usando o metadata da API."""
        for model in self.available_models:
            if model["id"] == self.selected_model:
                supports = model["supports_thinking"]
                self.supports_vision = model["supports_vision"]
                features = []
                if supports:
                    features.append("reasoning")
                if self.supports_vision:
                    features.append("vision")
                if features:
                    console.print(f"[green]✓ Modelo suporta: {', '.join(features)}[/]")
                else:
                    console.print("[green]✓ Modelo não suporta reasoning/vision[/]")
                return supports
        
        console.print("[yellow]⚠ Modelo não encontrado nos metadados, assumindo sem reasoning[/]")
        return False
    
    def select_model(self) -> str:
        """Mostra a lista de modelos em menu navegável com setas (buscada da API)."""
        console.clear()
        
        # Busca modelos da API
        self.available_models = fetch_available_models()
        
        if not self.available_models:
            console.print("[red]Nenhum modelo disponível. Verifique se o servidor está rodando.[/]")
            sys.exit(1)
        
        questionary.print("🤖 Selecione o modelo para o chat", style="bold cyan")
        
        # Prepara as opções no formato "modelo - descrição"
        choices = []
        for model in self.available_models:
            label = f"{model['id']} - {model['display_desc']}"
            choices.append(label)
        
        try:
            selected = questionary.select(
                "Use ↑↓ para navegar e Enter para selecionar:",
                choices=choices,
                use_indicator=True,
                use_arrow_keys=True,
                style=questionary.Style([
                    ('question', 'bold cyan'),
                    ('selected', 'bg:#ansigreen #ansiwhite bold'),
                    ('pointer', 'bold cyan'),
                    ('instruction', 'dim'),
                ])
            ).ask()
        except KeyboardInterrupt:
            console.print("[dim]Saindo...[/]")
            sys.exit(0)
        if not selected:
            console.print("[dim]Saindo...[/]")
            sys.exit(0)
        
        # Extrai o model_id da string selecionada e encontra os dados completos
        model_id = selected.split(" - ")[0]
        for model in self.available_models:
            if model["id"] == model_id:
                self.selected_model_name = model["name"]
                self.supports_vision = model["supports_vision"]
                break
        
        console.print(f"[green]✓ Modelo selecionado:[/] {model_id}")
        
        return model_id
        
    def create_layout(self) -> Layout:
        """Cria o layout da interface."""
        layout = Layout()
        
        if self.supports_reasoning:
            # Layout com split: reasoning + response
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
    
    def render(self) -> Layout:
        """Renderiza a interface completa."""
        layout = self.create_layout()
        
        # Header com nome do modelo
        display_name = self.selected_model_name or self.selected_model or "Chat"
        header_text = Text(f"🤖 {display_name}", style="bold cyan")
        header = Align.center(header_text)
        layout["header"].update(Panel(header, border_style="cyan"))
        
        # Reasoning panel (só se suportar)
        if self.supports_reasoning:
            if self.reasoning_content or self.has_reasoning:
                r_content = f"## 🤔 Raciocínio\n\n{self.reasoning_content}"
                r_md = Markdown(r_content)
            else:
                r_md = Align.center(Text("\n\nAguardando raciocínio...", style="dim"))
            
            reasoning_panel = Panel(
                r_md,
                title="[yellow]Raciocínio[/]",
                border_style="yellow",
                box=box.ROUNDED
            )
            layout["reasoning"].update(reasoning_panel)
        
        # Response panel (sempre presente)
        if self.response_content or self.has_response:
            resp_content = f"## 💬 Resposta\n\n{self.response_content}"
            resp_md = Markdown(resp_content)
        else:
            resp_md = Align.center(Text("\n\nAguardando resposta...", style="dim"))
        
        response_panel = Panel(
            resp_md,
            title="[green]Resposta[/]",
            border_style="green",
            box=box.ROUNDED
        )
        if self.supports_reasoning:
            layout["response"].update(response_panel)
        else:
            layout["response"].update(response_panel)
        
        # Footer com status
        footer_text = Text(f"⏳ {self.status}", style="dim")
        if self.has_reasoning or self.has_response:
            if self.supports_reasoning:
                stats = f"R: {len(self.reasoning_content)} | Resp: {len(self.response_content)} chars"
            else:
                stats = f"Resp: {len(self.response_content)} chars"
            footer_text = Text(f"✅ {stats}", style="bold green")
        
        footer_panel = Panel(Align.center(footer_text), border_style="dim")
        layout["footer"].update(footer_panel)
        
        return layout
    
    def stream_chat(self, prompt: str, image_paths: list[str] = None) -> Iterator[Layout]:
        """Realiza o streaming do chat usando requests diretamente."""
        image_paths = image_paths or []
        
        try:
            # Construir conteúdo da mensagem
            message_content = build_message_content(prompt, image_paths)
            
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
        
        # Parsear caminhos de imagem do prompt
        clean_prompt, image_paths = parse_image_paths(prompt)
        
        if image_paths:
            console.print(f"[cyan]🖼️  {len(image_paths)} imagem(ns) detectada(s):[/]")
            for i, path in enumerate(image_paths, 1):
                fname = Path(path).name
                fsize = Path(path).stat().st_size / 1024
                console.print(f"[dim]   [{i}] {fname} ({fsize:.1f} KB)[/]")
            console.print(f"[dim]   Prompt: {clean_prompt}[/]")
            console.print()
        else:
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
                for layout in self.stream_chat(clean_prompt, image_paths):
                    live.update(layout)
                
                # Mostra resultado final com prompt e modelo
                live.stop()
                console.clear()
                
                # Cabeçalho com prompt e modelo
                model_display = self.selected_model_name or self.selected_model
                
                # Mostra o prompt original (com caminhos) se tinha imagens
                display_prompt = prompt  # prompt original
                header_parts = [f"[bold cyan]Prompt:[/] {display_prompt}"]
                if image_paths:
                    header_parts.append(f"[dim]Imagens: {len(image_paths)} anexada(s)[/]")
                header_parts.append(f"[dim]Modelo:[/] {model_display}")
                
                console.print(Panel(
                    "\n".join(header_parts),
                    title="✅ Concluído",
                    border_style="green"
                ))
                console.print()
                
                if self.supports_reasoning and self.reasoning_content:
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


def main():
    """Entry point."""
    chat = StreamingChat()
    chat.run()


if __name__ == "__main__":
    main()