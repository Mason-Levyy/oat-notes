"""Localhost-only web UI: stdlib HTTP + SSE, one meeting at a time. Each
connected SSE client holds a handler thread."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable

from .attribution import Speaker
from .config import Config
from .output import format_timestamp
from .settings import AppSettings, SettingsStore
from .types import Channel, TranscriptSegment


class EventHub:
    """Fans JSON events out to every connected SSE client."""

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

    def __init__(
        self,
        config: Config,
        transcriber,
        out_dir: Path = Path("transcripts"),
        settings: AppSettings = AppSettings(),
        save_settings: Callable[[AppSettings], None] | None = None,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self._out_dir = out_dir
        self.settings = settings
        self._save_settings = save_settings
        self.model_status = "ready" if transcriber is not None else "loading"
        self.model_error: str | None = None
        self.model_load_seconds: float | None = None
        self.hub = EventHub()
        self.lock = threading.Lock()
        self.session: Session | None = None
        self.awaiting_backfill: Session | None = None
        self.roster: tuple[Speaker, ...] = ()
        self.meeting_name: str = "meeting"
        self.lines: list[dict] = []
        self.last_saved: str | None = None
        self.shutdown = threading.Event()

    def status(self) -> dict:
        with self.lock:
            recording = self.session is not None
            actives = (
                self.session.active
                if recording
                else {Channel.MIC: None, Channel.LOOPBACK: None}
            )
            pending_guests = (
                _guests_with_lines(self.awaiting_backfill)
                if self.awaiting_backfill is not None
                else None
            )
            return {
                "recording": recording,
                "speakers": [
                    {"name": speaker.name, "remote": speaker.remote}
                    for speaker in self.roster
                ],
                "actives": {
                    "mic": actives[Channel.MIC],
                    "loopback": actives[Channel.LOOPBACK],
                },
                "pending_backfill": pending_guests,
                "meeting_name": self.meeting_name,
                "elapsed": self.session.elapsed() if recording else 0.0,
                "channels": (
                    [channel.value for channel in self.session.channels]
                    if recording
                    else []
                ),
                "backend": self.config.backend,
                "model_status": self.model_status,
                "model_error": self.model_error,
                "model_load_seconds": self.model_load_seconds,
                "settings": self.settings.to_dict(),
                "lines": self.lines,
                "last_saved": self.last_saved,
            }

    def model_ready(self, transcriber, elapsed: float) -> None:
        with self.lock:
            self.transcriber = transcriber
            self.model_status = "ready"
            self.model_error = None
            self.model_load_seconds = elapsed
        self.hub.publish({"type": "status", "recording": False})

    def model_failed(self, error: Exception, elapsed: float) -> None:
        with self.lock:
            self.transcriber = None
            self.model_status = "error"
            self.model_error = f"{type(error).__name__}: {error}"
            self.model_load_seconds = elapsed
        self.hub.publish({"type": "status", "recording": False})

    def update_settings(self, body: dict) -> dict:
        try:
            settings = AppSettings.from_dict(body)
        except (TypeError, ValueError) as error:
            return {"error": str(error)}
        with self.lock:
            if self.session is not None:
                return {"error": "end the meeting before changing hotkeys"}
            try:
                if self._save_settings is not None:
                    self._save_settings(settings)
            except OSError as error:
                return {"error": f"could not save settings: {error}"}
            self.settings = settings
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def start_session(self, body: dict) -> dict:
        with self.lock:
            if self.session is not None:
                return {"error": "already recording"}
            if self.transcriber is None:
                if self.model_status == "error":
                    return {"error": self.model_error or "model failed to load"}
                return {"error": "transcription model is still loading"}
            from .session import Session, SessionEvents, SessionOptions

            self.roster = tuple(
                Speaker(
                    name=str(entry.get("name", "")).strip(),
                    remote=bool(entry.get("remote", False)),
                )
                for entry in body.get("speakers", [])
                if str(entry.get("name", "")).strip()
            )
            self.meeting_name = str(body.get("name") or "meeting").strip() or "meeting"
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
                # Reads the live session roster without the state lock —
                # this fires from inside switch calls that may hold it.
                session = self.session
                if session is None or index >= len(session.roster):
                    return
                if len(session.roster) != len(self.roster):
                    self.roster = tuple(session.roster)
                    self.hub.publish({"type": "status", "recording": True})
                self.hub.publish(
                    {
                        "type": "speaker",
                        "index": index,
                        "channel": (
                            "loopback" if session.roster[index].remote else "mic"
                        ),
                    }
                )

            self.awaiting_backfill = None
            self.session = Session(
                self.config,
                SessionOptions(
                    speakers=self.roster,
                    meeting_name=self.meeting_name,
                    use_loopback=bool(body.get("loopback", True)),
                    out_dir=self._out_dir,
                    hotkey_modifiers=self.settings.hotkey_modifiers,
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
        session.stop()
        needs_backfill = bool(_guests_with_lines(session)) and not session.log.is_empty
        with self.lock:
            if needs_backfill:
                self.awaiting_backfill = session
            else:
                saved = session.save()
                self.last_saved = str(saved) if saved else None
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def finalize(self, body: dict) -> dict:
        with self.lock:
            if self.awaiting_backfill is None:
                return {"error": "nothing awaiting backfill"}
            session = self.awaiting_backfill
            self.awaiting_backfill = None
            renames = {
                str(old): str(new)
                for old, new in (body.get("renames") or {}).items()
                if str(new).strip()
            }
            saved = session.save(renames)
            self.last_saved = str(saved) if saved else None
            for line in self.lines:
                if line["label"] in renames:
                    line["label"] = renames[line["label"]]
        self.hub.publish({"type": "status", "recording": False})
        return self.status()

    def add_guest(self, body: dict) -> dict:
        with self.lock:
            session = self.session
            if session is None:
                return {"error": "not recording"}
            session.add_guest(remote=bool(body.get("remote", False)))
            self.roster = tuple(session.roster)
        self.hub.publish({"type": "status", "recording": True})
        return self.status()

    def switch_speaker(self, index: int) -> dict:
        with self.lock:
            session = self.session
        if session is not None:
            session.switch_speaker(index)
        return self.status()


def is_trusted_request(host_header: str | None, origin_header: str | None, port: int) -> bool:
    """Blocks DNS-rebinding (Host must name this loopback port) and
    cross-site POSTs (a present Origin must match too)."""
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if host_header not in allowed_hosts:
        return False
    if origin_header is None:
        return True
    allowed_origins = {f"http://{host}" for host in allowed_hosts}
    return origin_header in allowed_origins


def _guests_with_lines(session: Session) -> list[str]:
    spoke = session.log.speakers_with_lines()
    return [
        session.roster[index].name
        for index in session.guest_indices
        if session.roster[index].name in spoke
    ]


def _load_asset(name: str) -> bytes:
    return (resources.files("oat_notes") / "web" / name).read_bytes()


class Handler(BaseHTTPRequestHandler):
    state: AppState  # assigned by serve()
    port: int  # assigned by serve()

    def log_message(self, format: str, *log_args) -> None:
        pass  # keep the console clean; transcript lines matter more

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus | None = None) -> None:
        if status is None:
            status = HTTPStatus.CONFLICT if "error" in payload else HTTPStatus.OK
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _is_trusted(self) -> bool:
        return is_trusted_request(
            self.headers.get("Host"), self.headers.get("Origin"), self.port
        )

    def do_GET(self) -> None:
        if not self._is_trusted():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        if self.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, _load_asset("index.html"), "text/html; charset=utf-8")
        elif self.path == "/favicon.svg":
            self._send(HTTPStatus.OK, _load_asset("favicon.svg"), "image/svg+xml")
        elif self.path == "/pixel.woff2":
            self._send(HTTPStatus.OK, _load_asset("pixel.woff2"), "font/woff2")
        elif self.path == "/api/state":
            self._send_json(self.state.status())
        elif self.path == "/api/events":
            self._serve_events()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self._is_trusted():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
        except ValueError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/start":
            self._send_json(self.state.start_session(body))
        elif self.path == "/api/stop":
            self._send_json(self.state.stop_session())
        elif self.path == "/api/switch":
            index = body.get("index")
            if not isinstance(index, int):
                self._send_json(
                    {"error": "index must be an integer"}, status=HTTPStatus.BAD_REQUEST
                )
                return
            self._send_json(self.state.switch_speaker(index))
        elif self.path == "/api/add_guest":
            self._send_json(self.state.add_guest(body))
        elif self.path == "/api/finalize":
            self._send_json(self.state.finalize(body))
        elif self.path == "/api/settings":
            self._send_json(self.state.update_settings(body))
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


def _initialize_model(state: AppState, config: Config) -> None:
    """Load and warm the expensive transcription backend off the UI thread."""
    started = time.perf_counter()
    try:
        from .cli import warm_up
        from .transcriber import create_transcriber

        print(f"Loading transcription model ({config.backend})…", flush=True)
        transcriber = create_transcriber(config)
        warm_up(config, transcriber)
    except Exception as error:
        elapsed = time.perf_counter() - started
        print(f"Model initialization failed after {elapsed:.1f}s: {error}", file=sys.stderr)
        state.model_failed(error, elapsed)
        return
    elapsed = time.perf_counter() - started
    print(f"Transcription model ready in {elapsed:.1f}s", flush=True)
    state.model_ready(transcriber, elapsed)


def serve(config: Config, args: argparse.Namespace, open_browser: bool = True) -> None:
    started = time.perf_counter()
    settings_store = SettingsStore()
    state = AppState(
        config,
        transcriber=None,
        out_dir=args.out_dir,
        settings=settings_store.load(),
        save_settings=settings_store.save,
    )
    Handler.state = state
    Handler.port = args.port
    url = f"http://127.0.0.1:{args.port}"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        if getattr(error, "winerror", None) == 10048 or error.errno == 98:
            print(f"oat-notes is already running at {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            return
        raise
    server.daemon_threads = True

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"oat-notes UI ready in {time.perf_counter() - started:.2f}s: {url}",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    # OpenVINO construction can hold the interpreter lock briefly. Give the
    # browser enough time to fetch and paint the tiny UI before it begins.
    model_thread = threading.Timer(
        0.75,
        _initialize_model,
        args=(state, config),
    )
    model_thread.name = "model-loader"
    model_thread.daemon = True
    model_thread.start()
    try:
        while not state.shutdown.wait(timeout=0.3):
            pass
    except KeyboardInterrupt:
        pass
    print("Shutting down…", flush=True)
    if state.session is not None:
        state.stop_session()
    server.shutdown()
