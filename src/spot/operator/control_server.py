# src/voice_control/control_server.py
"""Localhost control endpoint for the operator's laptop CLI (``scripts/spotctl``).

The operator ssh's into the Jetson (Tailscale) and runs ``spotctl <name> [arg]``,
which POSTs to this server. The server binds 127.0.0.1 ONLY.

HTTP API
--------
- ``POST /cmd/<name>`` with optional JSON body ``{"arg": ...}``.
    * immediate command -> ``on_immediate(name, arg)`` is called on the HTTP
      thread and its dict is returned as JSON (200). An exception -> 500.
    * queued command    -> ``(name, arg)`` is put on ``queue``; returns
      ``{"queued": true}`` (200).
    * unknown name / missing arg -> 400 ``{"error": "..."}``.
- ``GET /status`` -> ``on_immediate("status", None)``.

Integration interface (for client_mic.py)
-----------------------------------------
::

    import queue
    from src.voice_control.control_server import ControlServer, drain

    ctl_q = queue.Queue()
    srv = ControlServer(port=8770, on_immediate=_on_immediate, queue=ctl_q)
    srv.start()                       # daemon thread, returns immediately
    ...
    while running:                    # main loop, once per iteration:
        drain(ctl_q, _on_queued)      # calls _on_queued(name, arg) per item
    ...
    srv.stop()

- ``_on_immediate(name: str, arg: str | None) -> dict`` must handle every name
  in ``IMMEDIATE`` (hush stop freeze sit stand stow status mute unmute).
  Threading contract: it runs on HTTP worker threads, possibly SEVERAL AT ONCE
  (ThreadingHTTPServer, one thread per request), concurrently with the main
  loop. So it must be re-entrant, must only touch thread-safe things (player,
  mouth, dispatcher), and must NEVER hold a lock that stop/freeze also need
  (a slow ``status`` or ``say`` must not be able to delay a ``stop``).
  ``status`` must not block: read plain attributes, take no long-held locks.
- ``_on_queued(name: str, arg: str | None) -> None`` handles every name in
  ``QUEUED`` (listen release reset say persona crowd voiceprint) on the main
  thread, between frames.
"""
from __future__ import annotations

import json
import logging
import queue as _queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

log = logging.getLogger(__name__)

DEFAULT_PORT = 8770


@dataclass(frozen=True)
class Command:
    name: str
    immediate: bool
    takes_arg: bool


_IMMEDIATE_NAMES = ("hush", "stop", "freeze", "sit", "stand", "stow",
                    "status", "mute", "unmute")
_QUEUED_NAMES = ("listen", "release", "reset", "say", "persona", "crowd",
                 "voiceprint")
_ARG_NAMES = ("say", "persona")
# Toggles: an optional "on"/"off" arg is forwarded; no arg means flip.
# voiceprint also takes "threshold <x>" (parsed by client_mic.apply_voiceprint_arg).
_OPTIONAL_ARG_NAMES = ("crowd", "voiceprint")

COMMANDS: dict[str, Command] = {
    **{n: Command(n, True, n in _ARG_NAMES) for n in _IMMEDIATE_NAMES},
    **{n: Command(n, False, n in _ARG_NAMES) for n in _QUEUED_NAMES},
}
IMMEDIATE = frozenset(_IMMEDIATE_NAMES)
QUEUED = frozenset(_QUEUED_NAMES)


def parse_command(path: str, body: dict) -> tuple[str, Optional[str]]:
    """Map ``/cmd/<name>`` + JSON body to ``(name, arg)``.

    Raises ValueError for an unknown name or a missing/empty arg. An arg sent
    to a command that takes none is ignored (returned as None), except the
    toggles (crowd, voiceprint), which forward an optional on/off arg.
    """
    prefix = "/cmd/"
    if not path.startswith(prefix):
        raise ValueError(f"bad path: {path!r}")
    name = path[len(prefix):].split("?", 1)[0].strip("/").lower()
    cmd = COMMANDS.get(name)
    if cmd is None:
        raise ValueError(f"unknown command: {name!r}")
    arg = (body or {}).get("arg")
    if name in _OPTIONAL_ARG_NAMES:
        return name, (str(arg).strip() if arg is not None and str(arg).strip() else None)
    if not cmd.takes_arg:
        return name, None
    if arg is None or not str(arg).strip():
        raise ValueError(f"command {name!r} needs an arg")
    return name, str(arg)


def drain(q: _queue.Queue, handler: Callable[[str, Optional[str]], object]) -> int:
    """Call ``handler(name, arg)`` for every queued command; return count."""
    n = 0
    while True:
        try:
            name, arg = q.get_nowait()
        except _queue.Empty:
            return n
        try:
            handler(name, arg)
        except Exception:
            log.exception("control: queued command %r failed", name)
        n += 1


class ControlServer:
    """ThreadingHTTPServer on a daemon thread, bound to 127.0.0.1 only."""

    def __init__(self, port: int = DEFAULT_PORT,
                 on_immediate: Callable[[str, Optional[str]], dict] = None,
                 queue: _queue.Queue = None):
        if on_immediate is None or queue is None:
            raise ValueError("ControlServer needs on_immediate and queue")
        self._on_immediate = on_immediate
        self._queue = queue
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        self._httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    @property
    def address(self) -> tuple:
        return self._httpd.server_address

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="control-server", daemon=True)
        self._thread.start()
        log.info("control server on http://127.0.0.1:%d", self.port)

    def stop(self) -> None:
        """Idempotent; safe even if start() was never called."""
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            # shutdown() blocks until serve_forever exits, so only call it
            # when serve_forever is actually running.
            self._httpd.shutdown()
            t.join(timeout=2.0)
        if not self._closed:
            self._closed = True
            self._httpd.server_close()

    def _dispatch(self, name: str, arg: Optional[str]) -> dict:
        if COMMANDS[name].immediate:
            return self._on_immediate(name, arg)
        self._queue.put((name, arg))
        return {"queued": True}

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, code: int, payload: dict) -> None:
                data = json.dumps(payload, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _run(self, name: str, arg: Optional[str]) -> None:
                try:
                    result = server._dispatch(name, arg)
                except Exception as e:
                    log.exception("control: command %r failed", name)
                    self._reply(500, {"error": f"{name} failed: {e}"})
                    return
                self._reply(200, result if isinstance(result, dict) else {"result": result})

            def do_GET(self):
                if self.path.split("?", 1)[0].rstrip("/") == "/status":
                    self._run("status", None)
                else:
                    self._reply(404, {"error": f"not found: {self.path}"})

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(length) if length else b""
                    body = json.loads(raw) if raw.strip() else {}
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                    name, arg = parse_command(self.path, body)
                except ValueError as e:  # includes JSONDecodeError
                    self._reply(400, {"error": str(e)})
                    return
                self._run(name, arg)

            def log_message(self, fmt, *args):
                log.debug("control: " + fmt, *args)

        return Handler
