"""Loopback-only dashboard. Static files are explicitly allowlisted."""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .evaluate import rollout
from .policy import Policy


def make_handler(root: Path):
    root = root.resolve()
    models = {}
    lock = threading.Lock()
    static_files = {
        "/": root / "web/index.html",
        "/index.html": root / "web/index.html",
        "/style.css": root / "web/style.css",
        "/app.js": root / "web/app.js",
        "/api/report": root / "artifacts/report.json",
        "/api/replays": root / "artifacts/results/replays.json",
        "/media/demo.gif": root / "artifacts/media/demo.gif",
        "/media/demo.json": root / "artifacts/media/demo.json",
        "/media/scene.png": root / "artifacts/media/scene.png",
    }

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, data: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, payload: dict):
            self.send(
                status,
                json.dumps(payload, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/health":
                self.send_json(200, {"status": "ok"})
                return
            if url.path == "/api/rollout":
                query = parse_qs(url.query)
                name = query.get("policy", ["dagger"])[0]
                distribution = query.get("distribution", ["id"])[0]
                try:
                    seed = int(query.get("seed", ["42000"])[0])
                    if not 0 <= seed < 2**31:
                        raise ValueError("seed must be in [0, 2147483647]")
                    if name not in ("random", "expert", "bc32", "bc256", "dagger"):
                        raise ValueError("Unknown policy")
                    if distribution not in ("id", "ood"):
                        raise ValueError("distribution must be id or ood")
                except ValueError as error:
                    self.send_json(400, {"error": str(error)})
                    return
                if not lock.acquire(blocking=False):
                    self.send_json(429, {"error": "A rollout is already running; retry shortly"})
                    return
                try:
                    if name in ("random", "expert"):
                        policy = name
                    else:
                        if name not in models:
                            models[name] = Policy.load(root / f"artifacts/checkpoints/{name}.pt")
                        policy = models[name]
                    result = rollout(policy, seed, distribution, record=True)
                    result.update({"policy": name, "id": f"fresh-{name}-{distribution}-{seed}"})
                    self.send_json(200, result)
                except FileNotFoundError:
                    self.send_json(
                        404, {"error": "Checkpoint not found; run the training pipeline first"}
                    )
                except Exception:
                    logging.getLogger(__name__).exception("Rollout failed")
                    self.send_json(500, {"error": "Simulation failed; check the server log"})
                finally:
                    lock.release()
                return
            path = static_files.get(url.path)
            if path is None or not path.is_file():
                self.send_json(404, {"error": "Not found"})
                return
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send(200, path.read_bytes(), kind)

    return Handler


def serve(root: Path, port: int = 8765):
    torch.set_num_threads(1)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    print(f"Pickplace Lab: http://127.0.0.1:{server.server_port} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
