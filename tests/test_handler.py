"""The HTTP layer end to end: a real ThreadingHTTPServer, a bare AppState."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from oat_notes.config import Config
from oat_notes.webapp.handler import Handler
from oat_notes.webapp.state import AppState


@pytest.fixture
def server():
    Handler.state = AppState(Config(), transcriber=None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.port = httpd.server_address[1]
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()


def call(server, path, body=None, headers=None):
    port = server.server_address[1]
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="GET" if body is None else "POST",
        headers={"Host": f"127.0.0.1:{port}", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        payload = error.read()
        return error.code, json.loads(payload) if payload.startswith(b"{") else payload


def test_state_is_served(server):
    status, payload = call(server, "/api/state")
    assert status == 200
    assert payload["recording"] is False


def test_validation_errors_are_400(server):
    status, payload = call(server, "/api/switch", {"index": "two"})
    assert (status, payload) == (400, {"error": "index must be an integer"})


def test_state_conflicts_are_409(server):
    status, payload = call(server, "/api/stop", {})
    assert (status, payload) == (409, {"error": "not recording"})


def test_unavailable_features_are_409_and_unknown_routes_404(server):
    status, payload = call(server, "/api/dictation/copy", {"id": 1})
    assert status == 409
    assert payload == {"error": "dictation is unavailable"}
    status, _ = call(server, "/api/nowhere", {})
    assert status == 404


def test_malformed_bodies_are_400(server):
    port = server.server_address[1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/stop",
        data=b"[1, 2]",
        method="POST",
        headers={"Host": f"127.0.0.1:{port}"},
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=5)
    assert raised.value.code == 400
    assert json.loads(raised.value.read()) == {"error": "request body must be a JSON object"}


def test_oversized_bodies_are_refused(server):
    port = server.server_address[1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/stop",
        data=b"{}",
        method="POST",
        headers={"Host": f"127.0.0.1:{port}", "Content-Length": "5000000"},
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=5)
    assert raised.value.code == 413


def test_foreign_hosts_are_forbidden(server):
    status, _ = call(server, "/api/state", headers={"Host": "evil.example:80"})
    assert status == 403
