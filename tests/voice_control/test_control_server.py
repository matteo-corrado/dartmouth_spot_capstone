# tests/voice_control/test_control_server.py
"""Tests for the localhost control server + spotctl CLI (Task 4)."""
import importlib.util
import json
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.voice_control.control_server import (
    COMMANDS, ControlServer, drain, parse_command,
)

_SPOTCTL = Path(__file__).resolve().parents[2] / "scripts" / "spotctl.py"
_spec = importlib.util.spec_from_file_location("spotctl", _SPOTCTL)
spotctl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spotctl)

# Proxy-less opener: never route test traffic through an http_proxy env var.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(port, name, arg=None):
    body = json.dumps({"arg": arg} if arg is not None else {}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/cmd/{name}", data=body,
                                 method="POST")
    try:
        with _OPENER.open(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def server():
    q, calls = queue.Queue(), []

    def on_immediate(n, a):
        calls.append((n, a))
        if n == "stow":
            raise RuntimeError("boom")
        return {"ok": True, "name": n}

    srv = ControlServer(port=0, on_immediate=on_immediate, queue=q)
    srv.start()
    yield srv, q, calls
    srv.stop()


def test_command_table():
    assert set(COMMANDS) == set(
        "mute unmute listen release reset hush say persona stop freeze sit "
        "stand stow status crowd voiceprint".split())
    imm = {n for n, c in COMMANDS.items() if c.immediate}
    assert imm == set("hush stop freeze sit stand stow status mute unmute".split())
    assert {n for n, c in COMMANDS.items() if c.takes_arg} == {"say", "persona"}


def test_parse_known_and_arg():
    assert parse_command("/cmd/say", {"arg": "hello"}) == ("say", "hello")
    assert parse_command("/cmd/hush", {}) == ("hush", None)


def test_parse_unknown_raises():
    with pytest.raises(ValueError):
        parse_command("/cmd/dance", {})


def test_parse_missing_arg_raises():
    with pytest.raises(ValueError):
        parse_command("/cmd/say", {})
    with pytest.raises(ValueError):
        parse_command("/cmd/persona", {"arg": "  "})


def test_immediate_commands_bypass_queue():
    q = queue.Queue(); calls = []
    srv = ControlServer(port=0, on_immediate=lambda n, a: calls.append(n) or {"ok": True}, queue=q); srv.start()
    _post(srv.port, "hush"); _post(srv.port, "listen")
    assert calls == ["hush"] and q.get_nowait() == ("listen", None); srv.stop()


def test_immediate_runs_while_main_thread_blocked():
    # Main loop never drains; an immediate command must still answer.
    q = queue.Queue(); gate = threading.Event()
    srv = ControlServer(port=0, on_immediate=lambda n, a: gate.set() or {"ok": True}, queue=q)
    srv.start()
    try:
        for _ in range(3):
            _post(srv.port, "listen")
        assert _post(srv.port, "stop") == (200, {"ok": True})
        assert gate.is_set() and q.qsize() == 3
    finally:
        srv.stop()


def test_binds_localhost_only():
    srv = ControlServer(port=0, on_immediate=lambda n, a: {}, queue=queue.Queue()); srv.start()
    assert srv.address[0] == "127.0.0.1"; srv.stop()


def test_http_errors_and_status(server):
    srv, q, calls = server
    assert _post(srv.port, "dance")[0] == 400
    assert _post(srv.port, "say")[0] == 400
    code, body = _post(srv.port, "stow")
    assert code == 500 and "boom" in body["error"]
    assert _post(srv.port, "say", "hi there") == (200, {"queued": True})
    assert q.get_nowait() == ("say", "hi there")
    with _OPENER.open(f"http://127.0.0.1:{srv.port}/status?x=1", timeout=5) as r:
        assert json.loads(r.read())["name"] == "status"
    assert ("status", None) in calls


def test_drain_calls_handler_and_survives_errors():
    q = queue.Queue(); seen = []
    for item in [("listen", None), ("say", "x"), ("reset", None)]:
        q.put(item)

    def h(n, a):
        seen.append((n, a))
        if n == "say":
            raise RuntimeError("tts down")

    assert drain(q, h) == 3
    assert seen == [("listen", None), ("say", "x"), ("reset", None)]
    assert q.empty()


def test_spotctl_one_shot(server, capsys):
    srv, q, calls = server
    assert spotctl.main(["--port", str(srv.port), "hush"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "hush"
    assert spotctl.main(["--port", str(srv.port), "say", "hello", "world"]) == 0
    assert q.get_nowait() == ("say", "hello world")
    assert spotctl.main(["--port", str(srv.port), "status"]) == 0
    assert spotctl.main(["--port", str(srv.port), "dance"]) == 1


def test_spotctl_unreachable_is_readable(capsys):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    assert spotctl.main(["--port", str(port), "hush"]) == 2
    err = capsys.readouterr().err
    assert "cannot reach control server" in err and "running" in err


def test_parse_strips_query_string():
    assert parse_command("/cmd/hush?verbose=1", {}) == ("hush", None)


def test_stop_without_start_and_double_stop():
    srv = ControlServer(port=0, on_immediate=lambda n, a: {}, queue=queue.Queue())
    t0 = time.monotonic(); srv.stop(); srv.stop()
    assert time.monotonic() - t0 < 1.0
    srv2 = ControlServer(port=0, on_immediate=lambda n, a: {}, queue=queue.Queue())
    srv2.start(); srv2.stop(); srv2.stop()


def test_spotctl_timeout_is_not_unreachable(capsys):
    gate = threading.Event()
    srv = ControlServer(port=0, on_immediate=lambda n, a: gate.wait(3) or {}, queue=queue.Queue())
    srv.start()
    try:
        assert spotctl.main(["--port", str(srv.port), "--timeout", "0.3", "hush"]) == 3
        err = capsys.readouterr().err
        assert "no reply within 0.3s" in err and "cannot reach" not in err
    finally:
        gate.set(); srv.stop()


def test_spotctl_non_json_reply():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200); self.send_header("Content-Length", "5"); self.end_headers()
            self.wfile.write(b"hello")
        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    try:
        assert spotctl.send("hush", port=httpd.server_address[1]) == {"error": "non-JSON reply"}
    finally:
        httpd.shutdown(); httpd.server_close()
