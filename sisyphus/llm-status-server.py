#!/usr/bin/env python3
"""Mini HTTP server that provides status endpoints for Homepage widgets.
Runs on port 8082 alongside llama-swap (port 8081) and ChromaDB (port 8100)."""

import json
import re
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

LLAMA_SWAP = "http://127.0.0.1:8081"
CHROMADB = "http://127.0.0.1:8100"
CHROMADB_TOKEN="***"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/gemma4-e2b":
            self._model_status("gemma4-e2b")
        elif self.path == "/lfm2.5-vl-450m":
            self._model_status("lfm2.5-vl-450m")
        elif self.path == "/chromadb":
            self._chromadb_status()
        elif self.path == "/embedding":
            self._embedding_status()
        else:
            self.send_response(404)
            self.end_headers()

    def _model_status(self, model_id):
        try:
            req = urllib.request.Request(f"{LLAMA_SWAP}/v1/models")
            resp = urllib.request.urlopen(req, timeout=5)
            models = json.loads(resp.read()).get("data", [])
            model = next((m for m in models if m["id"] == model_id), None)

            req2 = urllib.request.Request(f"{LLAMA_SWAP}/running")
            resp2 = urllib.request.urlopen(req2, timeout=5)
            running = json.loads(resp2.read()).get("running", [])
            is_running = any(r["model"] == model_id for r in running)

            if not model:
                self.send_json({"name": model_id, "status": "unavailable", "size": "?"})
                return

            meta = model.get("meta", {}).get("llamaswap", {})
            size_str = meta.get("size", "?")
            size_str = re.sub(r'(\d+)\.(\d+)', r'\1,\2', size_str)
            features = meta.get("features", {})
            self.send_json({
                "name": model.get("name", model_id),
                "status": "loaded" if is_running else "standby",
                "size": size_str,
                "thinking": "sim" if features.get("thinking") else "não",
                "vision": "sim" if features.get("vision") else "não",
                "context": meta.get("context", "?"),
            })
        except Exception:
            self.send_json({"name": model_id, "status": "error", "size": "?"})

    def _chromadb_status(self):
        try:
            req = urllib.request.Request(
                f"{CHROMADB}/api/v2/tenants/default_tenant/databases/default_database/collections",
                headers={"Authorization": f"Bearer {CHROMADB_TOKEN}"}
            )
            resp = urllib.request.urlopen(req, timeout=5)
            collections = json.loads(resp.read())
            count = len(collections) if isinstance(collections, list) else 0
            self.send_json({
                "collections": str(count),
                "embedding": "MiniLM-L6",
            })
        except Exception:
            self.send_json({"collections": "error", "embedding": "?"})

    def _embedding_status(self):
        try:
            req = urllib.request.Request(
                f"{CHROMADB}/api/v2/tenants/default_tenant/databases/default_database/collections",
                headers={"Authorization": f"Bearer {CHROMADB_TOKEN}"}
            )
            urllib.request.urlopen(req, timeout=5)
            self.send_json({"status": "online", "size": "80 MB"})
        except Exception:
            self.send_json({"status": "offline", "size": "80 MB"})

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
    server = HTTPServer(("127.0.0.1", 8082), Handler)
    server.serve_forever()
