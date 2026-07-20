"""A dependency-free localhost dashboard (stdlib http.server, no Torch, no flask).

Computes the investor sequence once at startup and serves it as one page plus a
JSON endpoint. `morrow demo` opens it.
"""

from __future__ import annotations

import json
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ..pipeline import investor_sequence
from .app import render_page


def runtime_info() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(terse=True),
        "cores": os.cpu_count() or 1,
    }


def _handler(sequence: dict, runtime: dict):
    page = render_page(sequence, runtime).encode()
    payload = json.dumps(sequence).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/api/sequence"):
                self._send(payload, "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def log_message(self, *a) -> None:  # quiet
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, benchmark_n: int = 60) -> None:
    print("computing investor sequence ...", flush=True)
    seq = investor_sequence(benchmark_n=benchmark_n)
    httpd = ThreadingHTTPServer((host, port), _handler(seq, runtime_info()))
    print(f"morrow dashboard on http://{host}:{port}  (ctrl-c to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
