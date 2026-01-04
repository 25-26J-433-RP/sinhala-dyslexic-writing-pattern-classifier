import os
import threading
import time
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from train_model import main as run_training


PORT = int(os.environ.get("PORT", "8080"))

state = {"running": False, "done": False, "error": None, "started_at": None}


def training_worker():
    state["running"] = True
    state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        run_training()
        state["done"] = True
    except Exception as e:
        state["error"] = str(e)
    finally:
        state["running"] = False


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/status"):
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(state, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence default logging to keep container logs cleaner
        return


def run_server():
    server = HTTPServer(("0.0.0.0", PORT), StatusHandler)
    print(f"Server listening on port {PORT} — returning training status at / or /status")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    # Start training in a background thread so the container stays alive for health checks
    t = threading.Thread(target=training_worker, daemon=True)
    t.start()

    # Run HTTP server to respond to health checks
    run_server()
