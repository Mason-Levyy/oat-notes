"""Localhost-only HTTP: static assets, the JSON API, and the SSE stream.
Each connected SSE client holds a handler thread."""

from __future__ import annotations

import json
import logging
import queue
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib import resources

from ..log import error_kind
from . import library, meeting, settings_api
from .requests import ApiError, Body
from .state import AppState

log = logging.getLogger(__name__)

PostHandler = Callable[[AppState, Body], dict]

MAX_BODY_BYTES = 1_000_000
EVENT_KEEPALIVE_SECONDS = 15.0

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/pixel.woff2": ("pixel.woff2", "font/woff2"),
}

GET_ROUTES: dict[str, Callable[[AppState], dict]] = {
    "/api/state": AppState.status,
    "/api/dictation/history": settings_api.dictation_history,
    "/api/library": AppState.library,
}

POST_ROUTES: dict[str, PostHandler] = {
    "/api/start": meeting.start_session,
    "/api/stop": meeting.stop_session,
    "/api/switch": meeting.switch_speaker,
    "/api/add_guest": meeting.add_guest,
    "/api/roster/add": meeting.add_roster_speaker,
    "/api/roster/remove": meeting.remove_roster_speaker,
    "/api/roster/rename": meeting.rename_roster_speaker,
    "/api/roster/reset_profile": meeting.reset_roster_profile,
    "/api/lines/assign": meeting.assign_line,
    "/api/profile_learning/cancel": meeting.cancel_profile_learning,
    "/api/notes/add": meeting.add_note,
    "/api/hotkey_bank": meeting.page_hotkeys,
    "/api/finalize": meeting.finalize,
    "/api/settings": settings_api.update_settings,
    "/api/dictation/toggle": settings_api.toggle_dictation,
    "/api/dictation/history/clear": settings_api.clear_dictation_history,
    "/api/dictation/copy": settings_api.copy_dictation,
    "/api/library/speakers/create": library.create_speaker,
    "/api/library/speakers/update": library.update_speaker,
    "/api/library/speakers/delete": library.delete_speaker,
    "/api/library/speakers/reset": library.reset_speaker,
    "/api/library/groups/create": library.create_group,
    "/api/library/groups/update": library.update_group,
    "/api/library/groups/delete": library.delete_group,
    "/api/library/speakers/record/start": library.start_voice_recording,
    "/api/library/speakers/record/stop": library.stop_voice_recording,
}


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


def load_asset(name: str) -> bytes:
    return (resources.files("oat_notes") / "web" / name).read_bytes()


class Handler(BaseHTTPRequestHandler):
    state: AppState
    port: int

    def log_message(self, format: str, *log_args) -> None:
        pass

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send_error(self, error: ApiError) -> None:
        self._send_json({"error": error.message}, error.status)

    def _is_trusted(self) -> bool:
        return is_trusted_request(
            self.headers.get("Host"), self.headers.get("Origin"), self.port
        )

    def do_GET(self) -> None:
        if not self._is_trusted():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        if self.path in ASSETS:
            name, content_type = ASSETS[self.path]
            self._send(HTTPStatus.OK, load_asset(name), content_type)
        elif self.path in GET_ROUTES:
            self._send_json(GET_ROUTES[self.path](self.state))
        elif self.path == "/api/events":
            self._serve_events()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self._is_trusted():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        try:
            body = self._read_body()
        except ApiError as error:
            self._send_error(error)
            return
        if self.path == "/api/quit":
            self._send_json({"ok": True})
            self.state.shutdown.set()
            return
        route = POST_ROUTES.get(self.path)
        if route is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        try:
            self._send_json(route(self.state, body))
        except ApiError as error:
            self._send_error(error)

    def _read_body(self) -> Body:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as error:
            raise ApiError("Content-Length must be a number", HTTPStatus.BAD_REQUEST) from error
        if length > MAX_BODY_BYTES:
            raise ApiError("request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as error:
            raise ApiError(str(error), HTTPStatus.BAD_REQUEST) from error
        if not isinstance(body, dict):
            raise ApiError("request body must be a JSON object", HTTPStatus.BAD_REQUEST)
        return body

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        subscriber = self.state.hub.subscribe()
        try:
            while not self.state.shutdown.is_set():
                try:
                    event = subscriber.get(timeout=EVENT_KEEPALIVE_SECONDS)
                    payload = f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    payload = ": keepalive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except OSError as error:
            log.debug("event stream closed: %s", error_kind(error))
        finally:
            self.state.hub.unsubscribe(subscriber)
