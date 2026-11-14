"""Custom robust HTTP server for Antisymmetric Attention Head and Activation Visualizers.

Seamlessly maps root web requests to:
- /index.html        -> outputs/antisymmetric_only_head_visualization/report/index.html
- /activations.html  -> outputs/antisymmetric_only_head_visualization/report/activations.html
- /comparisons/...   -> outputs/antisymmetric_only_head_visualization/comparisons/...
- /initialization/.. -> outputs/antisymmetric_only_head_visualization/initialization/...
- /trained/...       -> outputs/antisymmetric_only_head_visualization/trained/...
- /data/...          -> outputs/antisymmetric_only_head_visualization/data/...
"""

from __future__ import annotations

import http.server
import socketserver
import sys
from pathlib import Path
import urllib.parse

PORT = 8080
ROOT_DIR = Path("outputs/antisymmetric_only_head_visualization").resolve()
REPORT_DIR = (ROOT_DIR / "report").resolve()


class VisualizationRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPORT_DIR), **kwargs)

    def translate_path(self, path):
        # Strip query parameters and fragments
        parsed = urllib.parse.urlparse(path)
        clean_path = urllib.parse.unquote(parsed.path).lstrip("/")

        # First directory segment
        parts = clean_path.split("/")
        first_segment = parts[0] if parts else ""

        if first_segment in ["comparisons", "initialization", "trained", "data"]:
            target = ROOT_DIR.joinpath(*parts)
            return str(target)

        return super().translate_path(path)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def run_server(port: int = PORT):
    # Allow address reuse to avoid 'address already in use' during quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), VisualizationRequestHandler) as httpd:
        print(f"================================================================================")
        print(f"⚡ Interactive Visualizer Web Server Running at: http://localhost:{port}/")
        print(f"   - Head Visualization Gallery: http://localhost:{port}/index.html")
        print(f"   - Interactive Activations Studio: http://localhost:{port}/activations.html")
        print(f"================================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
