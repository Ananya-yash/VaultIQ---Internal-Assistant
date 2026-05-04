import logging
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

API_HOST = "127.0.0.1"
API_PORT = 8000
HEALTH_URL = f"http://{API_HOST}:{API_PORT}/health"
STARTUP_TIMEOUT = 30  # seconds to wait for API before giving up
POLL_INTERVAL = 1  # seconds between health check attempts
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"
API_CMD = [
    "uvicorn",
    "api.main:app",
    "--host",
    API_HOST,
    "--port",
    str(API_PORT),
    "--log-level",
    "warning",
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
api_proc: subprocess.Popen | None


def shutdown(sig, frame) -> None:
    logging.info("Shutting down...")
    if api_proc is not None:
        api_proc.terminate()
        try:
            api_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_proc.kill()
    sys.exit(0)


def wait_for_api() -> bool:
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        try:
            response = requests.get(HEALTH_URL, timeout=2)
            if response.status_code == 200:
                logging.info("API is ready at http://localhost:8000")
                return True
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(POLL_INTERVAL)
            continue
    return False


def main() -> None:
    global api_proc

    if not FRONTEND_PATH.exists():
        logging.error("frontend/index.html not found. Please ensure the frontend file exists.")
        sys.exit(1)

    try:
        response = requests.get(HEALTH_URL, timeout=1)
        if response.ok:
            logging.warning("Port 8000 is already in use by a running service. Please stop it first.")
            sys.exit(1)
    except (requests.ConnectionError, requests.Timeout):
        pass

    logging.info("Starting API server on port 8000...")
    api_proc = subprocess.Popen(API_CMD)
    signal.signal(signal.SIGINT, shutdown)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, shutdown)

    if not wait_for_api():
        logging.error("API failed to start within 30 seconds. Check for errors above.")
        api_proc.terminate()
        sys.exit(1)

    try:
        ready_response = requests.get(f"http://{API_HOST}:{API_PORT}/ready", timeout=2)
        if ready_response.ok:
            ready_payload = ready_response.json()
            qdrant_ok = bool(ready_payload.get("qdrant", False))
            groq_ok = bool(ready_payload.get("groq", False))
            if not qdrant_ok or not groq_ok:
                logging.warning("Warning: API started but some services are degraded:")
                logging.warning("  Qdrant: %s", "OK" if qdrant_ok else "DEGRADED")
                logging.warning("  Groq:   %s", "OK" if groq_ok else "DEGRADED")
    except (requests.ConnectionError, requests.Timeout, ValueError):
        pass

    webbrowser.open(Path("frontend/index.html").resolve().as_uri())
    logging.info("Browser opened — navigate to frontend/index.html if it did not open automatically.")
    logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logging.info("  Internal RAG Chatbot is running")
    logging.info("  API:      http://localhost:8000")
    logging.info("  API Docs: http://localhost:8000/docs")
    logging.info("  Frontend: frontend/index.html")
    logging.info("  Press Ctrl+C to stop")
    logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    api_proc.wait()


api_proc = None


if __name__ == "__main__":
    main()
