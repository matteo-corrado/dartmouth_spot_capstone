#!/usr/bin/env python3
"""Web control panel for Spot — E-Stop + voice pipeline management.

Mobile-friendly web UI accessible via Tailscale or local network.
No external dependencies (stdlib only).

Usage:
    python scripts/web_panel.py                  # default port 8080
    python scripts/web_panel.py --port 9090      # custom port
    python scripts/web_panel.py --no-spot        # UI-only mode (no Spot connection)
"""
import sys
import pathlib
import argparse
import json
import signal
import subprocess
import threading
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Spot E-Stop manager
# ---------------------------------------------------------------------------
class EStopManager:
    """Manages hardware E-Stop keepalive for Spot."""

    def __init__(self):
        self._robot = None
        self._estop_client = None
        self._endpoint = None
        self._keepalive = None
        self._thread = None
        self._running = False
        self._stopped = False  # True = motors cut
        self._lock = threading.Lock()

    @property
    def status(self):
        with self._lock:
            if self._keepalive is None:
                return "unclaimed"
            if self._stopped:
                return "stopped"
            return "active"

    def claim(self):
        """Connect to Spot and claim E-Stop."""
        with self._lock:
            if self._keepalive is not None:
                return True, "E-Stop already claimed"

        try:
            from bosdyn.client import create_standard_sdk
            from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
            from bosdyn.client.time_sync import TimeSyncClient
            from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD

            sdk = create_standard_sdk("dartmouth_spot_web_estop")
            robot = sdk.create_robot(BOSDYN_ROBOT_IP)
            robot.authenticate(BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD)

            ts = robot.ensure_client(TimeSyncClient.default_service_name)
            for _ in range(5):
                try:
                    ts.get_time_sync_update()
                    break
                except Exception:
                    time.sleep(0.2)

            estop_client = robot.ensure_client(EstopClient.default_service_name)
            endpoint = EstopEndpoint(estop_client, name="dartmouth_web_estop",
                                     estop_timeout=3.0)
            endpoint.force_simple_setup()
            keepalive = EstopKeepAlive(endpoint)
            keepalive.allow()

            with self._lock:
                self._robot = robot
                self._estop_client = estop_client
                self._endpoint = endpoint
                self._keepalive = keepalive
                self._stopped = False
                self._running = True

            # Start heartbeat thread
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()

            return True, "E-Stop claimed"

        except Exception as e:
            return False, f"Failed to claim E-Stop: {e}"

    def stop(self):
        """Cut motors — stop the keepalive."""
        with self._lock:
            if self._keepalive is None:
                return False, "E-Stop not claimed"
            if self._stopped:
                return True, "Already stopped"
            try:
                self._running = False
                self._keepalive.stop()
                self._keepalive.shutdown()
                self._keepalive = None
                self._stopped = True
                return True, "E-STOP TRIGGERED — motors will cut"
            except Exception as e:
                return False, f"Stop failed: {e}"

    def release(self):
        """Resume — re-create keepalive so robot can move."""
        with self._lock:
            if not self._stopped:
                if self._keepalive:
                    return True, "E-Stop already active"
                return False, "E-Stop not claimed"

        # Re-claim from scratch (endpoint was invalidated by stop)
        try:
            from bosdyn.client.estop import EstopEndpoint, EstopKeepAlive

            with self._lock:
                endpoint = EstopEndpoint(self._estop_client,
                                         name="dartmouth_web_estop",
                                         estop_timeout=3.0)
                endpoint.force_simple_setup()
                keepalive = EstopKeepAlive(endpoint)
                keepalive.allow()
                self._endpoint = endpoint
                self._keepalive = keepalive
                self._stopped = False
                self._running = True

            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()

            return True, "E-Stop released — robot can move"
        except Exception as e:
            return False, f"Release failed: {e}"

    def _heartbeat(self):
        """Background keepalive loop."""
        while self._running:
            try:
                with self._lock:
                    if self._keepalive and not self._stopped:
                        self._keepalive.allow()
            except Exception:
                with self._lock:
                    self._running = False
                break
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Voice pipeline manager
# ---------------------------------------------------------------------------
class VoicePipelineManager:
    """Manages the voice control subprocess."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    @property
    def status(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return "running"
            return "stopped"

    def start(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return True, "Voice pipeline already running"

        try:
            script = str(PROJECT_ROOT / "scripts" / "run_voice_control.py")
            self._proc = subprocess.Popen(
                [sys.executable, script],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            return True, f"Voice pipeline started (PID {self._proc.pid})"
        except Exception as e:
            return False, f"Failed to start: {e}"

    def stop(self):
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                return True, "Voice pipeline not running"

        try:
            # Kill the entire process group
            import os
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.wait(timeout=5)
            return True, "Voice pipeline stopped"
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            return True, "Voice pipeline force-killed"
        except Exception as e:
            return False, f"Failed to stop: {e}"


# ---------------------------------------------------------------------------
# Service status checks
# ---------------------------------------------------------------------------
def _port_open(port, host="127.0.0.1"):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        ok = s.connect_ex((host, port)) == 0
        s.close()
        return ok
    except Exception:
        return False


def get_service_status():
    return {
        "riva": "running" if _port_open(50051) else "stopped",
        "ollama": "running" if _port_open(11434) else "stopped",
    }


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Spot Control</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #eee;
    min-height: 100vh; padding: 16px;
  }
  h1 { text-align: center; font-size: 1.4em; margin-bottom: 12px; color: #e0e0e0; }

  .status-bar {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-bottom: 16px;
  }
  .status-item {
    font-size: 0.85em; padding: 4px 10px;
    border-radius: 12px; background: #16213e;
  }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  .dot-green { background: #00c853; }
  .dot-red { background: #ff1744; }
  .dot-yellow { background: #ffd600; }

  .section { margin-bottom: 18px; }
  .section-title {
    font-size: 0.9em; color: #888; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; text-align: center;
  }

  .btn {
    display: block; width: 100%; padding: 18px;
    border: none; border-radius: 12px;
    font-size: 1.2em; font-weight: 700;
    cursor: pointer; transition: opacity 0.2s;
    color: #fff; text-align: center;
  }
  .btn:active { opacity: 0.7; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-estop { background: #d32f2f; font-size: 1.6em; padding: 28px; margin-bottom: 10px; }
  .btn-claim { background: #1565c0; }
  .btn-release { background: #2e7d32; }
  .btn-start { background: #00838f; }
  .btn-stop { background: #e65100; }

  .btn-row { display: flex; gap: 8px; }
  .btn-row .btn { flex: 1; }

  .log {
    background: #0d1117; border: 1px solid #333;
    border-radius: 8px; padding: 10px;
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.8em; max-height: 150px;
    overflow-y: auto; color: #8b949e;
  }
  .log-entry { margin-bottom: 2px; }
  .log-ok { color: #3fb950; }
  .log-err { color: #f85149; }
  .log-warn { color: #d29922; }
</style>
</head>
<body>

<h1>Spot Control Panel</h1>

<div class="status-bar">
  <span class="status-item"><span class="dot" id="dot-riva"></span>Riva: <span id="s-riva">--</span></span>
  <span class="status-item"><span class="dot" id="dot-ollama"></span>Ollama: <span id="s-ollama">--</span></span>
  <span class="status-item"><span class="dot" id="dot-estop"></span>E-Stop: <span id="s-estop">--</span></span>
  <span class="status-item"><span class="dot" id="dot-voice"></span>Voice: <span id="s-voice">--</span></span>
</div>

<div class="section">
  <div class="section-title">Emergency Stop</div>
  <button class="btn btn-estop" id="btn-estop" onclick="action('/estop/stop')">E-STOP</button>
  <div class="btn-row">
    <button class="btn btn-claim" onclick="action('/estop/claim')">Claim</button>
    <button class="btn btn-release" onclick="action('/estop/release')">Release</button>
  </div>
</div>

<div class="section">
  <div class="section-title">Voice Pipeline</div>
  <div class="btn-row">
    <button class="btn btn-start" onclick="action('/voice/start')">Start</button>
    <button class="btn btn-stop" onclick="action('/voice/stop')">Stop</button>
  </div>
</div>

<div class="section">
  <div class="section-title">Log</div>
  <div class="log" id="log"></div>
</div>

<script>
const logEl = document.getElementById('log');

function log(msg, cls) {
  const d = document.createElement('div');
  d.className = 'log-entry ' + (cls || '');
  const t = new Date().toLocaleTimeString();
  d.textContent = t + ' ' + msg;
  logEl.prepend(d);
  // Keep log size manageable
  while (logEl.children.length > 50) logEl.lastChild.remove();
}

function setDot(id, color) {
  const el = document.getElementById(id);
  el.className = 'dot dot-' + color;
}

function updateStatus(data) {
  function set(id, val) {
    document.getElementById('s-' + id).textContent = val;
    const color = val === 'running' || val === 'active' ? 'green' :
                  val === 'stopped' ? 'red' : 'yellow';
    setDot('dot-' + id, color);
  }
  set('riva', data.riva);
  set('ollama', data.ollama);
  set('estop', data.estop);
  set('voice', data.voice);
}

async function pollStatus() {
  try {
    const r = await fetch('/status');
    if (r.ok) updateStatus(await r.json());
  } catch(e) {}
}

async function action(path) {
  try {
    log('>> ' + path, 'log-warn');
    const r = await fetch(path, {method: 'POST'});
    const data = await r.json();
    log(data.message, data.ok ? 'log-ok' : 'log-err');
    pollStatus();
  } catch(e) {
    log('Request failed: ' + e, 'log-err');
  }
}

// Poll every 2 seconds
pollStatus();
setInterval(pollStatus, 2000);
</script>
</body>
</html>
"""

import os

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class SpotHandler(BaseHTTPRequestHandler):
    """Handle GET/POST requests for the control panel."""

    def __init__(self, estop_mgr, voice_mgr, no_spot, *args, **kwargs):
        self.estop_mgr = estop_mgr
        self.voice_mgr = voice_mgr
        self.no_spot = no_spot
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        # Suppress default access logs (noisy with polling)
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._html(HTML_PAGE)
        elif self.path == "/status":
            svc = get_service_status()
            svc["estop"] = self.estop_mgr.status
            svc["voice"] = self.voice_mgr.status
            self._json(svc)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.no_spot and self.path.startswith("/estop"):
            self._json({"ok": False, "message": "No-Spot mode: E-Stop disabled"})
            return

        if self.path == "/estop/claim":
            ok, msg = self.estop_mgr.claim()
            self._json({"ok": ok, "message": msg})
        elif self.path == "/estop/stop":
            ok, msg = self.estop_mgr.stop()
            self._json({"ok": ok, "message": msg})
        elif self.path == "/estop/release":
            ok, msg = self.estop_mgr.release()
            self._json({"ok": ok, "message": msg})
        elif self.path == "/voice/start":
            ok, msg = self.voice_mgr.start()
            self._json({"ok": ok, "message": msg})
        elif self.path == "/voice/stop":
            ok, msg = self.voice_mgr.stop()
            self._json({"ok": ok, "message": msg})
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Spot web control panel")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    parser.add_argument("--no-spot", action="store_true",
                        help="UI-only mode (no Spot connection, E-Stop disabled)")
    args = parser.parse_args()

    estop_mgr = EStopManager()
    voice_mgr = VoicePipelineManager()

    handler = partial(SpotHandler, estop_mgr, voice_mgr, args.no_spot)
    server = HTTPServer(("0.0.0.0", args.port), handler)

    # Get local IPs for display
    ips = []
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=3
        )
        ips = result.stdout.strip().split()[:3]
    except Exception:
        pass

    print("=" * 50)
    print("  Spot Web Control Panel")
    print("=" * 50)
    if args.no_spot:
        print("  Mode: UI-only (no Spot connection)")
    print(f"\n  Listening on port {args.port}")
    print(f"  Local:     http://localhost:{args.port}")
    for ip in ips:
        print(f"  Network:   http://{ip}:{args.port}")
    print(f"\n  Open this URL on your phone to control Spot.")
    print("  Press Ctrl+C to stop.\n")

    def handle_shutdown(sig, frame):
        print("\nShutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        server.serve_forever()
    except Exception:
        pass
    finally:
        # Clean up E-Stop
        if estop_mgr.status == "active":
            print("Releasing E-Stop...")
            estop_mgr.stop()


if __name__ == "__main__":
    main()
