"""A local image upload demo. Images stay in memory on this computer."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock

from PIL import UnidentifiedImageError
import torch

from .predict import DEFAULT_RUN, Predictor

ASSETS = Path(__file__).resolve().parents[1] / "demo"
MAX_BYTES = 10 * 1024 * 1024


def make_server(predictor, port=7860):
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            if isinstance(body, dict):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; "
                "style-src 'self'; script-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/model":
                metadata = predictor.metadata
                self.respond(
                    200,
                    {
                        "architecture": metadata["architecture"],
                        "classes": metadata["class_names"],
                        "parameters": metadata["parameters"],
                        "best_epoch": metadata["best_epoch"],
                        "training_images": sum(
                            metadata["dataset"]["counts"]["train"].values()
                        ),
                        "temperature": metadata["temperature"],
                    },
                )
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "application/javascript; charset=utf-8"),
            }
            if self.path not in assets:
                self.respond(404, {"error": "Page not found."})
                return
            filename, content_type = assets[self.path]
            self.respond(200, (ASSETS / filename).read_bytes(), content_type)

        def do_POST(self):
            if self.path != "/api/predict":
                self.respond(404, {"error": "Endpoint not found."})
                return
            # Same-origin UI only; no cross-origin image submission.
            origin = self.headers.get("Origin")
            allowed = f"http://127.0.0.1:{self.server.server_port}"
            if origin and origin != allowed:
                self.respond(
                    403, {"error": "Open the demo at its printed local address."}
                )
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_BYTES:
                self.respond(413, {"error": "Choose an image smaller than 10 MB."})
                return
            try:
                self.connection.settimeout(20)
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ValueError("The upload was incomplete.")
                with lock:
                    result = predictor.predict(payload)
                self.respond(200, result)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                message = (
                    str(error)
                    if isinstance(error, ValueError)
                    else "Use a readable JPG, PNG or WebP image."
                )
                self.respond(400, {"error": message})
            except Exception:
                self.log_error("Prediction failed")
                self.respond(
                    500,
                    {"error": "Prediction failed. Check the local server terminal."},
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(4)
    server = make_server(Predictor(args.run), args.port)
    print(f"Open http://127.0.0.1:{server.server_port} — Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
