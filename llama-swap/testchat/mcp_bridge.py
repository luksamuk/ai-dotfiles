#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ponte MCP opcional para o testchat (stdlib only — o venv do testchat e
deliberadamente minimo, sem SDK de MCP).

O que faz:
  - le testchat/mcp.json (mesma pasta deste modulo); sem arquivo ou
    server inacessivel = desabilitado com aviso (nada quebra);
  - fala Streamable HTTP MCP (2025-03-26): initialize -> tools/list ->
    tools/call, parseando respostas SSE ("data: {...}");
  - converte tools MCP pro formato OpenAI function-calling (o mesmo do
    MOCK_TOOLS, que o llama-server renderiza no template do LFM3);
  - executa tools/call e devolve os blocos de texto achatados em string
    (truncado — modelo de 2B nao precisa de 100KB de resultado).

O token fica FORA do repo: no arquivo apontado por auth_file (env-style
KEY=VALUE, chave auth_key). O arquivo do searxng e o mesmo do Hermes.
"""

import json
import os
import re
from pathlib import Path

import requests

_CONFIG_PATH = Path(__file__).resolve().parent / "mcp.json"
_SESSION_ID_HEADER = "Mcp-Session-Id"
_MAX_RESULT_CHARS = 4000  # tool result truncado: 2B + context budget


def _env_file_value(path: str, key: str):
    """Le KEY=VALUE de um arquivo .env (ex: o env do MCP searxng do Hermes)."""
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        return None
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


class McpHttpServer:
    """Cliente MCP minimo sobre Streamable HTTP (um server por instancia)."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.url = cfg["url"]
        self.timeout = float(cfg.get("timeout_s", 30))
        self.allow = set(cfg.get("tools") or [])
        self.session_id = None
        self.headers = {"Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"}
        if cfg.get("auth_file") and cfg.get("auth_key"):
            tok = _env_file_value(cfg["auth_file"], cfg["auth_key"])
            if tok:
                self.headers["Authorization"] = f"Bearer {tok}"

    def _post(self, payload: dict):
        hdr = dict(self.headers)
        if self.session_id:
            hdr[_SESSION_ID_HEADER] = self.session_id
        # decode UTF-8 manual: text/event-stream sem charset => requests cai
        # em latin-1 e acentos do conteúdo PT viram mojibake
        r = requests.post(self.url, headers=hdr, data=json.dumps(payload),
                          timeout=self.timeout)
        sid = r.headers.get(_SESSION_ID_HEADER)
        if sid:
            self.session_id = sid
        body = r.content.decode('utf-8', errors='replace')
        # Streamable HTTP pode devolver JSON puro OU envelope SSE
        if body.lstrip().startswith(("event:", "data:", ":")):
            datas = [m.group(1) for m in
                     re.finditer(r"^data: (.*)$", body, re.M)]
            return [json.loads(d) for d in datas if d.strip()]
        return [json.loads(body)] if body.strip() else []

    def initialize(self):
        resp = self._post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": {"name": "testchat", "version": "0"},
            },
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def list_tools(self) -> list:
        resp = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        for msg in resp:
            tools = (msg.get("result") or {}).get("tools")
            if tools is not None:
                return tools
        return []

    def call_tool(self, name: str, args: dict) -> str:
        resp = self._post({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                           "params": {"name": name, "arguments": args}})
        for msg in resp:
            result = msg.get("result")
            if result is None:
                err = msg.get("error")
                if err:
                    return f"MCP error: {err.get('message', err)}"
                continue
            parts = []
            for c in result.get("content", []):
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
            text = "\n".join(parts) or json.dumps(result, ensure_ascii=False)
            if len(text) > _MAX_RESULT_CHARS:
                text = text[:_MAX_RESULT_CHARS] + "\n[... truncado por tamanho]"
            return text
        return "MCP error: resposta vazia do servidor"


class McpBridge:
    """Merge de tools MCP no registro do testchat, com fallback gracioso."""

    def __init__(self, config_path: Path = None):
        self.config_path = Path(config_path or _CONFIG_PATH)
        self.servers: list[McpHttpServer] = []
        self.tools: list[dict] = []       # formato OpenAI function-calling
        self.by_name: dict[str, tuple[McpHttpServer, dict]] = {}
        self.status = "desligado"

    def load(self):
        try:
            cfg = json.loads(self.config_path.read_text())
        except FileNotFoundError:
            self.status = "sem mcp.json — MCP desligado"
            return self
        for name, scfg in (cfg.get("mcpServers") or {}).items():
            if not scfg.get("enabled", True):
                continue
            try:
                srv = McpHttpServer(name, scfg)
                srv.initialize()
                tools = srv.list_tools()
            except Exception as e:  # noqa: BLE001 — MCP é opcional por design
                self.status = f"server '{name}' inacessivel ({str(e)[:60]})"
                continue
            self.servers.append(srv)
            for t in tools:
                name_t = t.get("name", "")
                if not name_t or (srv.allow and name_t not in srv.allow):
                    continue
                if name_t in self.by_name:
                    continue  # primeira definição vence
                self.by_name[name_t] = (srv, t)
                self.tools.append({
                    "type": "function",
                    "function": {
                        "name": name_t,
                        "description": f"[{name}] {t.get('description', '')}",
                        "parameters": t.get("inputSchema",
                                            {"type": "object", "properties": {}}),
                    },
                })
        if self.tools:
            self.status = f"OK ({len(self.tools)} tools de {len(self.servers)} server)"
        elif not self.servers:
            self.status = self.status or "nenhum server habilitado"
        return self

    def is_mcp_tool(self, name: str) -> bool:
        return name in self.by_name

    def execute(self, name: str, args: dict) -> str:
        srv, tool = self.by_name[name]
        try:
            return srv.call_tool(name, args)
        except Exception as e:  # noqa: BLE001 — tool call nunca derruba o chat
            return f"MCP error em {name}: {str(e)[:200]}"


# Singleton do processo: importado por main.py e duplex.py
BRIDGE = McpBridge()


def load_tools() -> list:
    """Chame uma vez no boot. Retorna tools OpenAI (mocks + MCP)."""
    from tools import MOCK_TOOLS
    BRIDGE.load()
    return MOCK_TOOLS + BRIDGE.tools


# Tools BUILTIN do harness (nao-mock, nao-MCP): executam aqui, antes do
# fallback de mock. O fluxo do switch_language: o modelo ANUNCIA a troca
# em fala, chama a tool, e o RETORNO viaja como user item (instrucao
# in-context; session.update com novas instrucoes e REJEITADO pelo server
# realtime apos a primeira fala/mensagem, server-realtime.cpp:443).
_LANG_NAMES = {"pt": "Portuguese (pt)", "en": "English (en)",
               "es": "Spanish (es)"}


def _builtin_switch_language(args: dict) -> str:
    lang = str(args.get("lang", "")).strip().lower()
    if lang not in _LANG_NAMES:
        return (f"switch_language error: unknown lang '{lang}'. "
                f"Valid: {', '.join(_LANG_NAMES)}")
    return (f"[Language switch acknowledged. From now on, speak and write "
            f"in {_LANG_NAMES[lang]}. Acknowledge and continue naturally; "
            f"do not call switch_language again unless the language changes "
            f"again.]")


BUILTIN_TOOLS = {"switch_language": _builtin_switch_language}


def execute_tool(name: str, args: dict):
    """Executor unificado: builtin do harness, depois MCP, depois mocks —
    mesma assinatura do get_mock_response."""
    if name in BUILTIN_TOOLS:
        return BUILTIN_TOOLS[name](args)
    if BRIDGE.is_mcp_tool(name):
        return BRIDGE.execute(name, args)
    from tools import get_mock_response
    return get_mock_response(name, args)