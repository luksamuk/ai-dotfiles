#!/usr/bin/env python3
"""Navi status server — provides clean JSON endpoints for Homepage widgets.
Runs on port 8083 alongside llama-swap (12434), Ollama (11434), and Open Design (7456)."""

import json
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

LLAMA_SWAP = "http://127.0.0.1:12434"
OLLAMA = "http://127.0.0.1:11434"
OPEN_DESIGN = "http://127.0.0.1:7456"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/llama-swap":
            self._llama_swap_status()
        elif self.path == "/ollama":
            self._ollama_status()
        elif self.path == "/open-design":
            self._open_design_status()
        elif self.path == "/open-webui":
            self._open_webui_status()
        elif self.path == "/hermes":
            self._hermes_status()
        else:
            self.send_response(404)
            self.end_headers()

    def _llama_swap_status(self):
        try:
            req = urllib.request.Request(f"{LLAMA_SWAP}/v1/models")
            resp = urllib.request.urlopen(req, timeout=5)
            models = json.loads(resp.read()).get("data", [])

            req2 = urllib.request.Request(f"{LLAMA_SWAP}/running")
            resp2 = urllib.request.urlopen(req2, timeout=5)
            running = json.loads(resp2.read()).get("running", [])

            loaded = [r.get("model", "?") for r in running]
            self.send_json({
                "available": str(len(models)),
                "running": str(len(loaded)),
                "status": "loaded" if loaded else "standby",
            })
        except Exception:
            self.send_json({"available": "?", "running": "0", "status": "error"})

    def _ollama_status(self):
        try:
            req = urllib.request.Request(f"{OLLAMA}/api/tags")
            resp = urllib.request.urlopen(req, timeout=5)
            models = json.loads(resp.read()).get("models", [])

            req2 = urllib.request.Request(f"{OLLAMA}/api/ps")
            resp2 = urllib.request.urlopen(req2, timeout=5)
            running = json.loads(resp2.read()).get("models", [])

            self.send_json({
                "available": str(len(models)),
                "running": str(len(running)),
                "status": "loaded" if running else "standby",
            })
        except Exception:
            self.send_json({"available": "?", "running": "0", "status": "error"})

    def _open_design_status(self):
        try:
            req = urllib.request.Request(f"{OPEN_DESIGN}/api/agents")
            resp = urllib.request.urlopen(req, timeout=5)
            agents = json.loads(resp.read()).get("agents", [])
            detected = sum(1 for a in agents if a.get("available"))
            self.send_json({
                "agents": str(detected),
                "status": "online",
            })
        except Exception:
            self.send_json({"agents": "?", "status": "error"})

    def _open_webui_status(self):
        try:
            req = urllib.request.Request("http://127.0.0.1:3111/api/config")
            resp = urllib.request.urlopen(req, timeout=5)
            config = json.loads(resp.read())
            self.send_json({
                "version": config.get("version", "?"),
                "status": "online",
            })
        except Exception:
            self.send_json({"version": "?", "status": "error"})

    def _hermes_status(self):
        try:
            req = urllib.request.Request("http://127.0.0.1:9119/")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                self.send_json({"status": "online"})
            else:
                self.send_json({"status": "error"})
        except Exception:
            self.send_json({"status": "error"})

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8083), Handler)
    server.serve_forever()