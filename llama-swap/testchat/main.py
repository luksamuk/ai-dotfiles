#!/usr/bin/env python3
"""
Chat com streaming de Reasoning e Response usando Rich Live Display.

Instalação:
    uv pip install rich openai questionary prompt-toolkit requests

Uso:
    uv run rich_chat.py
"""

import sys
from typing import Optional
from openai import OpenAI
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
import os
import requests

# Configuração do cliente
client = OpenAI(
    base_url="http://127.0.0.1:12434/v1",
    api_key="",
)

console = Console()


def fetch_available_models():
    """Busca modelos disponíveis do endpoint /v1/models."""
    try:
        response = requests.get("http://127.0.0.1:12434/v1/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            name = model.get("name", model_id)
            description = model.get("description", "")
            
            # Verifica se tem thinking
            meta = model.get("meta", {})
            llamaswap = meta.get("llamaswap", {})
            features = llamaswap.get("features", {})
            has_thinking = features.get("thinking", False)
            
            # Coleta extras para mostrar
            extras = []
            if has_thinking:
                extras.append("🤔 thinking")
            if features.get("tools", False):
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
        self.available_models = []
    
    def check_model_reasoning(self) -> bool:
        """Verifica se o modelo suporta reasoning usando o metadata da API."""
        for model in self.available_models:
            if model["id"] == self.selected_model:
                supports = model["supports_thinking"]
                console.print(f"[green]✓ Modelo {'suporta' if supports else 'não suporta'} reasoning[/]\n")
                return supports
        
        console.print("[yellow]⚠ Modelo não encontrado nos metadados, assumindo sem reasoning[/]\n")
        return False
    
    def select_model(self) -> str:
        """Mostra a lista de modelos em menu navegável com setas (buscada da API)."""
        console.clear()
        
        # Busca modelos da API
        self.available_models = fetch_available_models()
        
        if not self.available_models:
            console.print("[red]Nenhum modelo disponível. Verifique se o servidor está rodando.[/]")
            sys.exit(1)
        
        questionary.print("🤖 Selecione o modelo para o chat\n", style="bold cyan")
        
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
            console.print("\n[dim]Saindo...[/]")
            sys.exit(0)
        
        if not selected:
            console.print("\n[dim]Saindo...[/]")
            sys.exit(0)
        
        # Extrai o model_id da string selecionada e encontra os dados completos
        model_id = selected.split(" - ")[0]
        for model in self.available_models:
            if model["id"] == model_id:
                self.selected_model_name = model["name"]
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
    
    def stream_chat(self, prompt: str):
        """Realiza o streaming do chat."""
        try:
            stream = client.chat.completions.create(
                model=self.selected_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                
                # Pega reasoning content
                reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                
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
        
        console.print("\n[dim]Pressione Ctrl+C para sair a qualquer momento[/]")
        console.print("[dim]↑↓ no prompt para histórico[/]\n")
        
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
            console.print("\n[dim]Saindo...[/]")
            return
        
        if prompt is None:
            console.print("\n[dim]Saindo...[/]")
            return
        
        if not prompt.strip():
            console.print("[red]Prompt vazio![/]")
            return
        
        console.print("\n[dim]Iniciando streaming...[/]\n")
        
        # Inicia Live display com screen=True (tela alternativa)
        with Live(
            self.render(),
            screen=True,
            refresh_per_second=15,  # ~60fps suave
            vertical_overflow="visible"
        ) as live:
            try:
                for layout in self.stream_chat(prompt):
                    live.update(layout)
                
                # Mostra resultado final com prompt e modelo
                live.stop()
                console.clear()
                
                # Cabeçalho com prompt e modelo
                model_display = self.selected_model_name or self.selected_model
                console.print(Panel(
                    f"[bold cyan]Prompt:[/] {prompt}\n[dim]Modelo:[/] {model_display}",
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
                console.print("\n[dim]Interrompido pelo usuário[/]")


def main():
    """Entry point."""
    chat = StreamingChat()
    chat.run()


if __name__ == "__main__":
    main()
