#!/usr/bin/env python3
"""
server.py — Serve the WebGPU demo locally.

Usage:
    .venv/bin/python webgpu-demo/server.py

Then open http://localhost:8000 in a WebGPU-capable browser.
"""
import http.server
import socketserver
import sys
from pathlib import Path

PORT = 8000
DIR = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIR), **kwargs)

    def end_headers(self):
        # CORS + cross-origin isolation for SharedArrayBuffer
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

if __name__ == "__main__":
    print(f"Serving {DIR} at http://localhost:{PORT}")
    print(f"Open in Chrome 113+ / Edge 113+ with WebGPU support")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nBye!")
            sys.exit(0)
