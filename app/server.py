#!/usr/bin/env python3
"""Dependency-free local server for the Sign2Text prototype."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import tempfile
import uuid
import time
import importlib.util
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))
from ml.gestures import Translator, LiveRecognizer, installed, download_status, MODEL_ID, WARNING
from ml.asl_letters import installed as asl_letters_installed

translator = Translator()
live_recognizer = LiveRecognizer()
job_lock = threading.Lock()
jobs = {}
jobs_guard = threading.Lock()


def runtime_ready():
    return all(importlib.util.find_spec(name) for name in ("mediapipe", "cv2"))


def letters_runtime_ready():
    return importlib.util.find_spec("torch") is not None


def purge_jobs():
    with jobs_guard:
        for key in list(jobs):
            if jobs[key]["status"] != "running" and time.time() - jobs[key].get("finished", jobs[key]["created"]) > 1800:
                jobs.pop(key)


def housekeeping(stop):
    while not stop.wait(10):
        purge_jobs()


def perform_job(job_id, path, mode="gestures"):
    def progress(message):
        with jobs_guard:
            jobs[job_id].update(message=message)
    try:
        result = translator.translate(path, progress) if mode == "gestures" else translator.translate(path, progress, mode=mode)
        with jobs_guard:
            jobs[job_id].update(status="complete", result=result, finished=time.time())
    except Exception as exc:
        with jobs_guard:
            jobs[job_id].update(status="error", message=str(exc), finished=time.time())
    finally:
        try:
            path.unlink(missing_ok=True)
        finally:
            job_lock.release()


class Handler(SimpleHTTPRequestHandler):
    def json_response(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def local_request(self):
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"} and self.headers.get("Origin", f"http://{host}") == f"http://{host}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self.local_request():
            self.json_response(403, {"error": "Local requests only"})
            return
        if self.path == "/api/model":
            dependencies = runtime_ready()
            self.json_response(200, {"installed": installed(), "aslLettersInstalled": asl_letters_installed(), "aslLettersRuntimeReady": letters_runtime_ready(), "runtimeReady": dependencies, "loaded": translator.model is not None,
                                    "id": MODEL_ID, "experimental": True, "warning": WARNING, "download": download_status()})
            return
        if self.path.startswith("/api/jobs/"):
            purge_jobs()
            with jobs_guard:
                job = dict(jobs.get(self.path.rsplit("/", 1)[-1], {}))
            self.json_response(200 if job else 404, job or {"error": "Job not found"})
            return
        if self.path == "/health":
            payload = json.dumps({"status": "ok", "offline": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def do_POST(self):
        if not self.local_request():
            self.json_response(403, {"error": "Local requests only"})
            return
        if self.path == "/api/live":
            self.handle_live_frame()
            return
        if self.path != "/api/translate":
            self.json_response(404, {"error": "Not found"})
            return
        mode = self.headers.get("X-Sign-Mode", "gestures")
        if mode not in {"gestures", "asl", "asl_letters"}:
            self.json_response(400, {"error": "Unknown recognition mode"})
            return
        if self.headers.get("Content-Type") != "application/octet-stream":
            self.json_response(415, {"error": "Expected a video body"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= 32 * 1024 * 1024:
            self.json_response(413, {"error": "Choose a video smaller than 32 MB"})
            return
        if not installed():
            self.json_response(503, {"error": "Model download is not complete yet"})
            return
        if mode == "asl_letters" and (not asl_letters_installed() or not letters_runtime_ready()):
            self.json_response(503, {"error": "ASL letter model is not ready"})
            return
        if not runtime_ready():
            self.json_response(503, {"error": "Start the server with python3 start.py to use the model environment"})
            return
        if not job_lock.acquire(blocking=False):
            self.json_response(409, {"error": "A translation is already running"})
            return
        path = None
        try:
            self.connection.settimeout(30)
            payload = self.rfile.read(size)
            if len(payload) != size:
                raise ValueError("Incomplete upload")
            folder = APP_DIR.parent / "work" / "incoming"
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".video", delete=False) as video:
                path = Path(video.name)
                video.write(payload)
            job_id = uuid.uuid4().hex
            with jobs_guard:
                # Keep only recent results in process memory; never write translations to disk.
                for key in list(jobs):
                    if time.time() - jobs[key]["created"] > 1800 or len(jobs) >= 20:
                        jobs.pop(key)
                jobs[job_id] = {"status": "running", "message": "Preparing video…", "created": time.time()}
            threading.Thread(target=perform_job, args=(job_id, path, mode), daemon=True).start()
        except Exception as exc:
            if path:
                path.unlink(missing_ok=True)
            job_lock.release()
            self.json_response(400, {"error": str(exc)})
            return
        self.json_response(202, {"jobId": job_id})

    def handle_live_frame(self):
        mode = self.headers.get("X-Sign-Mode", "gestures")
        if mode not in {"gestures", "asl", "asl_letters"}:
            self.json_response(400, {"error": "Unknown recognition mode"})
            return
        if self.headers.get("Content-Type") != "image/jpeg":
            self.json_response(415, {"error": "Expected a JPEG camera frame"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= 512 * 1024:
            self.json_response(413, {"error": "Camera frame must be smaller than 512 KB"})
            return
        if not installed() or not runtime_ready():
            self.json_response(503, {"error": "Gesture recognition is not ready"})
            return
        if mode == "asl_letters" and (not asl_letters_installed() or not letters_runtime_ready()):
            self.json_response(503, {"error": "ASL letter model is not ready"})
            return
        try:
            self.connection.settimeout(5)
            payload = self.rfile.read(size)
            if len(payload) != size:
                raise ValueError("Incomplete camera frame")
            prediction = live_recognizer.recognize_jpeg(payload) if mode == "gestures" else live_recognizer.recognize_jpeg(payload, mode=mode)
            self.json_response(200, prediction)
        except ValueError as exc:
            self.json_response(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", default=4173, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    stopped = threading.Event()
    threading.Thread(target=housekeeping, args=(stopped,), daemon=True).start()
    print(f"Sign2Text Offline: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        server.server_close()


if __name__ == "__main__":
    main()
