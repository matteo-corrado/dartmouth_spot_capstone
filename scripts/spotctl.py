#!/usr/bin/env python3
"""spotctl — operator control CLI for the running voice pipeline.

Talks to the localhost control server (src/voice_control/control_server.py),
default http://127.0.0.1:8770. Run on the Jetson, e.g. ``ssh jetson spotctl hush``.
Stdlib only.

  spotctl <name> [arg...]      one-shot, prints the JSON reply
  spotctl                      interactive hotkeys (raw tty), q quits

Names: mute unmute listen release reset hush say persona stop freeze sit
stand stow status crowd voiceprint   (say/persona need an arg;
crowd/voiceprint take an optional on|off, no arg flips)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

HOTKEYS = {
    "m": "mute", "u": "unmute", "l": "listen", "r": "release", "x": "reset",
    "h": "hush", "s": "stop", "f": "freeze", "w": "stow", "c": "crowd",
    "v": "voiceprint", "?": "status",
}

HELP = ("keys: m mute  u unmute  l listen  r release  x reset  h hush  "
        "s stop  f freeze  w stow  c crowd  v voiceprint  ? status  q quit")


class Unreachable(Exception):
    pass


class NoReply(Exception):
    """Connected, but the server did not answer within the timeout."""


# Never route localhost control traffic through an http_proxy env var.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def send(name: str, arg: str | None = None, host: str = DEFAULT_HOST,
         port: int = DEFAULT_PORT, timeout: float = 5.0) -> dict:
    """POST /cmd/<name> (or GET /status) and return the decoded JSON reply.

    HTTP errors from the server (400/500) are returned as their JSON body.
    Raises Unreachable if nothing is listening, NoReply on a read timeout.
    """
    base = f"http://{host}:{port}"
    if name == "status":
        req = urllib.request.Request(base + "/status", method="GET")
    else:
        data = json.dumps({"arg": arg} if arg is not None else {}).encode()
        req = urllib.request.Request(base + "/cmd/" + name, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read() or b"{}")
        except ValueError:
            return {"error": f"HTTP {e.code}"}
    except urllib.error.URLError as e:  # connect failed (refused, no route, connect timeout)
        raise Unreachable(
            f"spotctl: cannot reach control server at {base} ({e.reason}). "
            "Is the voice pipeline (client_mic.py) running?") from None
    except TimeoutError:  # connected, but no reply in time
        raise NoReply(f"spotctl: {name}: no reply within {timeout:g}s "
                      "— the command may still be executing") from None
    except (ConnectionError, OSError) as e:
        raise Unreachable(f"spotctl: connection to {base} failed ({e})") from None
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {"error": "non-JSON reply"}


def _interactive(host: str, port: int, timeout: float) -> int:
    import termios
    import tty

    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        print("spotctl: interactive mode needs a tty (use ssh -t), or pass a command",
              file=sys.stderr)
        return 2
    print(HELP)
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # keypress at a time; Ctrl-C still works
        while True:
            ch = sys.stdin.read(1)
            if ch in ("q", "\x04", ""):
                return 0
            name = HOTKEYS.get(ch)
            if name is None:
                print(HELP, flush=True)
                continue
            try:
                reply = send(name, host=host, port=port, timeout=timeout)
            except (Unreachable, NoReply) as e:
                print(e, flush=True)
                # drop keys mashed while we waited so they don't replay
                termios.tcflush(fd, termios.TCIFLUSH)
                continue
            print(f"{name}: {json.dumps(reply)}", flush=True)
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="spotctl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for a reply")
    p.add_argument("name", nargs="?", help="command name; omit for hotkey mode")
    p.add_argument("arg", nargs="*", help="argument (say/persona); words are joined")
    a = p.parse_args(argv)

    if a.name is None:
        return _interactive(a.host, a.port, a.timeout)
    arg = " ".join(a.arg) if a.arg else None
    try:
        reply = send(a.name.lower(), arg, host=a.host, port=a.port, timeout=a.timeout)
    except Unreachable as e:
        print(e, file=sys.stderr)
        return 2
    except NoReply as e:
        print(e, file=sys.stderr)
        return 3
    print(json.dumps(reply, indent=2))
    return 1 if "error" in reply else 0


if __name__ == "__main__":
    sys.exit(main())
