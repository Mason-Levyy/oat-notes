"""Local web UI: stdlib HTTP server + Server-Sent Events, no web framework.

Single-user, localhost-only by design. The transcriber loads once at server
start; each meeting is a Session (session.py) driven through a small JSON
API. Transcript lines stream to the page over SSE.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from .config import Config
from .output import format_timestamp
from .session import Session, SessionEvents, SessionOptions
from .types import TranscriptSegment


class EventHub:
    """Fan-out of JSON events to every connected SSE client."""

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=512)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass


class AppState:
    """Everything the handlers share. One meeting at a time."""

    def __init__(self, config: Config, transcriber) -> None:
        self.config = config
        self.transcriber = transcriber
        self.hub = EventHub()
        self.lock = threading.Lock()
        self.session: Session | None = None
        self.speakers: tuple[str, ...] = ()
        self.remote_name: str | None = None
        self.lines: list[dict] = []
        self.last_saved: str | None = None
        self.shutdown = threading.Event()

    def status(self) -> dict:
        with self.lock:
            recording = self.session is not None
            return {
                "recording": recording,
                "speakers": list(self.speakers),
                "active_speaker": (
                    self.session.active_speaker if recording else 0
                ),
                "elapsed": self.session.elapsed() if recording else 0.0,
                "channels": (
                    [channel.value for channel in self.session.channels]
                    if recording
                    else []
                ),
                "backend": self.config.backend,
                "remote_name": self.remote_name,
                "lines": self.lines,
                "last_saved": self.last_saved,
            }

    def start_session(self, body: dict) -> dict:
        with self.lock:
            if self.session is not None:
                return {"error": "already recording"}
            self.speakers = tuple(
                str(name).strip() for name in body.get("speakers", []) if str(name).strip()
            )
            self.remote_name = (body.get("remote_name") or "").strip() or None
            self.lines = []
            self.last_saved = None

            def on_segment(segment: TranscriptSegment, label: str, latency: float) -> None:
                line = {
                    "time": format_timestamp(segment.start),
                    "label": label,
                    "text": segment.text,
                    "channel": segment.channel.value,
                }
                self.lines.append(line)
                self.hub.publish({"type": "line", **line})

            def on_speaker(index: int) -> None:
                self.hub.publish({"type": "speaker", "index": index})

            self.session = Session(
                self.config,
                SessionOptions(
                    speakers=self.speakers,
                    remote_name=self.remote_name,
                    use_loopback=bool(body.get("loopback", True)),
                ),
                self.transcriber,
                SessionEvents(on_segment=on_segment, on_speaker=on_speaker),
            )
            self.session.start()
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def stop_session(self) -> dict:
        with self.lock:
            if self.session is None:
                return {"error": "not recording"}
            session = self.session
            self.session = None
        saved = session.stop()
        with self.lock:
            self.last_saved = str(saved) if saved else None
        self.hub.publish(
            {"type": "status", "recording": False, "saved": self.last_saved}
        )
        return self.status()

    def switch_speaker(self, index: int) -> dict:
        with self.lock:
            session = self.session
        if session is not None:
            session.switch_speaker(index)
        return self.status()


def _load_asset(name: str) -> bytes:
    return (resources.files("oat_notes") / "web" / name).read_bytes()


class Handler(BaseHTTPRequestHandler):
    state: AppState  # assigned by serve()

    def log_message(self, format: str, *log_args) -> None:
        pass  # keep the console clean; transcript lines matter more

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict) -> None:
        status = (
            HTTPStatus.CONFLICT if "error" in payload else HTTPStatus.OK
        )
        self._send(
            status, json.dumps(payload).encode(), "application/json"
        )

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, _load_asset("index.html"), "text/html; charset=utf-8")
        elif self.path == "/pixel.woff2":
            self._send(HTTPStatus.OK, _load_asset("pixel.woff2"), "font/woff2")
        elif self.path == "/api/state":
            self._send_json(self.state.status())
        elif self.path == "/api/events":
            self._serve_events()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/start":
            self._send_json(self.state.start_session(body))
        elif self.path == "/api/stop":
            self._send_json(self.state.stop_session())
        elif self.path == "/api/switch":
            self._send_json(self.state.switch_speaker(int(body.get("index", 0))))
        elif self.path == "/api/quit":
            self._send_json({"ok": True})
            self.state.shutdown.set()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        subscriber = self.state.hub.subscribe()
        try:
            while not self.state.shutdown.is_set():
                try:
                    event = subscriber.get(timeout=15.0)
                    payload = f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    payload = ": keepalive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (ConnectionAbortedError, ConnectionResetError, OSError):
            pass
        finally:
            self.state.hub.unsubscribe(subscriber)


def serve(config: Config, args: argparse.Namespace, open_browser: bool = True) -> None:
    from .cli import warm_up
    from .transcriber import create_transcriber

    print(f"Loading transcription model ({config.backend})…", flush=True)
    transcriber = create_transcriber(config)
    warm_up(config, transcriber)

    state = AppState(config, transcriber)
    Handler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"oat-notes UI: {url}  (Ctrl+C to quit)", flush=True)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        webbrowser.open(url)
    try:
        while not state.shutdown.wait(timeout=0.3):
            pass
    except KeyboardInterrupt:
        pass
    print("Shutting down…", flush=True)
    if state.session is not None:
        state.stop_session()
    server.shutdown()
