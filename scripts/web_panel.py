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
import os
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
import io
import socketserver
from urllib.parse import urlparse, parse_qs

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


# Threading server to handle concurrent connections (MJPEG stream + polling)
class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


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
            from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
            from src.session import quick_robot

            robot = quick_robot("dartmouth_spot_web_estop")
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
        """Resume — re-create keepalive so robot can move.

        Holds the lock through the entire mutate-state operation. The
        previous version released the lock between the early-return
        checks and the keepalive re-creation, which left a window where
        a concurrent stop() could re-cut motors while we were halfway
        through allowing them. Web panel is single-operator so the
        practical race is small, but the wider lock is essentially free.
        """
        with self._lock:
            if not self._stopped:
                if self._keepalive:
                    return True, "E-Stop already active"
                return False, "E-Stop not claimed"

            try:
                # Re-claim from scratch (endpoint was invalidated by stop)
                from bosdyn.client.estop import EstopEndpoint, EstopKeepAlive

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
            except Exception as e:
                return False, f"Release failed: {e}"

        # Heartbeat thread starts outside the lock so it can take its own
        # lock immediately on first iteration without recursive contention.
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

        return True, "E-Stop released — robot can move"

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
VOICE_LOG_PATH = "/tmp/spot-voice.log"
# Manual one-shot rotation: when the log exceeds this size, the .1 backup is
# overwritten and we reopen a fresh file. Avoids unbounded growth on the
# Jetson's tight /tmp without pulling in logging.handlers.RotatingFileHandler
# (web_panel uses plain print everywhere else).
VOICE_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def _rotate_voice_log():
    """Rotate /tmp/spot-voice.log if it exceeds VOICE_LOG_MAX_BYTES.

    Renames the existing file to ``<path>.1`` (clobbering any prior backup).
    Best-effort: any failure is logged and ignored — we'd rather keep the
    pipeline running than crash on a missing /tmp permission.
    """
    try:
        size = os.path.getsize(VOICE_LOG_PATH)
    except FileNotFoundError:
        return
    except OSError as e:
        print(f"[web] log rotation stat failed: {e}")
        return
    if size < VOICE_LOG_MAX_BYTES:
        return
    backup = VOICE_LOG_PATH + ".1"
    try:
        os.replace(VOICE_LOG_PATH, backup)
        print(f"[web] rotated voice log ({size} bytes -> {backup})")
    except OSError as e:
        print(f"[web] log rotation rename failed: {e}")


class VoicePipelineManager:
    """Manages the voice control subprocess."""

    def __init__(self):
        self._proc = None
        self._log_file = None
        self._lock = threading.Lock()
        # Cache for get_logs(). The frontend polls every ~2s and the log
        # can grow to 10s of MB during a long session — re-reading the
        # whole file each poll would saturate the eMMC. We key the cache
        # on (st_size, st_mtime); if neither changed since the last call,
        # we return the cached lines instead of re-reading.
        self._log_cache_key = None        # (size, mtime) tuple
        self._log_cache_lines = []

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
            # Rotate before opening so each pipeline session starts with
            # bounded growth. Append mode preserves prior shutdown traces
            # within a session — needed to debug device-busy races where
            # the previous instance is the one holding the mic.
            _rotate_voice_log()
            self._log_file = open(VOICE_LOG_PATH, "a")
            script = str(PROJECT_ROOT / "scripts" / "run_voice_control.py")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._proc = subprocess.Popen(
                [sys.executable, script],
                cwd=str(PROJECT_ROOT),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            return True, f"Voice pipeline started (PID {self._proc.pid})"
        except Exception as e:
            # If Popen raised after the log was opened, close the fd so
            # repeated failed starts don't leak file descriptors.
            self._close_log()
            return False, f"Failed to start: {e}"

    def _close_log(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def stop(self):
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                return True, "Voice pipeline not running"
            proc = self._proc

        try:
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                # Race: child exited between the poll() check above and
                # getpgid(). Treat as already-stopped instead of bubbling
                # the exception up to the user.
                self._close_log()
                return True, "Voice pipeline stopped"

            # Send SIGINT first — client_mic.py's _shutdown_signal_handler
            # converts it to KeyboardInterrupt and runs the graceful path
            # (cancel nav, sit, power off, drain audio). 30s gives the
            # bosdyn blocking_sit + power_off pair enough headroom for a
            # healthy robot; the _cancel_nav step in cleanup_spot keeps
            # them from fighting an in-progress follow_me / visual_nav
            # daemon thread, which used to push cleanup past the old 20s
            # window and made the stop button look broken.
            os.killpg(pgid, signal.SIGINT)
            try:
                proc.wait(timeout=30)
                self._close_log()
                return True, "Voice pipeline stopped (robot sat down)"
            except subprocess.TimeoutExpired:
                pass

            # SIGTERM as fallback. client_mic.py treats this as the
            # second signal and hard-exits via os._exit() without
            # waiting for any wedged bosdyn calls.
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
                self._close_log()
                return True, "Voice pipeline stopped"
            except subprocess.TimeoutExpired:
                pass

            # Last resort
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=3)
            self._close_log()
            return True, "Voice pipeline force-killed"
        except Exception as e:
            return False, f"Failed to stop: {e}"

    def get_logs(self, max_lines=200):
        """Read recent voice pipeline log lines (cached + tail-only)."""
        try:
            stat = os.stat(VOICE_LOG_PATH)
        except FileNotFoundError:
            self._log_cache_key = None
            self._log_cache_lines = []
            return []
        except OSError:
            return self._log_cache_lines

        key = (stat.st_size, stat.st_mtime)
        if key == self._log_cache_key:
            return self._log_cache_lines

        # Tail-read: seek to roughly the last `max_lines * 200 bytes` chunk
        # so we never read more than ~40 KB even when the file is hundreds
        # of MB. The 200 bytes/line estimate is generous for our log
        # format (mostly short status lines).
        tail_window = max_lines * 200
        try:
            with open(VOICE_LOG_PATH, "rb") as f:
                if stat.st_size > tail_window:
                    f.seek(stat.st_size - tail_window)
                    f.readline()  # discard the partial first line
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()[-max_lines:]
        except OSError:
            return self._log_cache_lines

        self._log_cache_key = key
        self._log_cache_lines = lines
        return lines


# ---------------------------------------------------------------------------
# Camera streamer (read-only SDK connection)
# ---------------------------------------------------------------------------
class CameraStreamer:
    """Captures images from Spot's cameras for web streaming."""

    CAMERA_SOURCES = {
        "front": "frontleft_fisheye_image",
        "front_right": "frontright_fisheye_image",
        "left": "left_fisheye_image",
        "right": "right_fisheye_image",
        "back": "back_fisheye_image",
    }

    def __init__(self):
        self._image_client = None
        self._lock = threading.Lock()

    def _ensure_connection(self):
        """Lazy SDK connection on first use."""
        if self._image_client is not None:
            return True
        try:
            from bosdyn.client.image import ImageClient
            from src.session import quick_robot

            robot = quick_robot("dartmouth_spot_web_camera")
            self._image_client = robot.ensure_client(
                ImageClient.default_service_name
            )
            print("[Camera] Connected to Spot (read-only)")
            return True
        except Exception as e:
            print(f"[Camera] Connection failed: {e}")
            return False

    def capture_jpeg(self, camera="front"):
        """Capture a JPEG frame, rotated 90 CW for upright display."""
        source = self.CAMERA_SOURCES.get(camera, self.CAMERA_SOURCES["front"])
        with self._lock:
            if not self._ensure_connection():
                return None
            try:
                resps = self._image_client.get_image_from_sources([source])
                if not resps:
                    return None
                data = resps[0].shot.image.data
                from PIL import Image
                img = Image.open(io.BytesIO(data))
                img = img.transpose(Image.Transpose.ROTATE_270)  # 90 CW
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                return buf.getvalue()
            except Exception as e:
                print(f"[Camera] Capture failed ({camera}): {e}")
                self._image_client = None  # force reconnect next time
                return None


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
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Spot</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #09090b;
    --card: #18181b;
    --card-border: rgba(255,255,255,0.06);
    --card-hover: #1f1f23;
    --text: #fafafa;
    --text-muted: #a1a1aa;
    --text-dim: #52525b;
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.15);
    --green: #22c55e;
    --green-dim: rgba(34,197,94,0.12);
    --red: #ef4444;
    --red-dim: rgba(239,68,68,0.12);
    --red-glow: rgba(239,68,68,0.25);
    --orange: #f97316;
    --orange-dim: rgba(249,115,22,0.12);
    --yellow: #eab308;
    --yellow-dim: rgba(234,179,8,0.12);
    --radius: 14px;
    --radius-sm: 10px;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'JetBrains Mono', 'SF Mono', Monaco, monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { -webkit-tap-highlight-color: transparent; }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    min-height: 100dvh;
    padding: 0 16px 32px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* ---- Header ---- */
  .header {
    display: flex; align-items: center; justify-content: center;
    gap: 10px; padding: 20px 0 16px;
  }
  .header svg { width: 28px; height: 28px; }
  .header h1 {
    font-size: 1.25em; font-weight: 700;
    letter-spacing: -0.03em; color: var(--text);
  }

  /* ---- Status pills ---- */
  .status-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin-bottom: 20px;
  }
  .pill {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    font-size: 0.8em; font-weight: 500;
    color: var(--text-muted);
    transition: border-color 0.3s;
  }
  .pill .label { flex: 1; }
  .pill .val {
    font-family: var(--mono); font-size: 0.85em;
    font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .pill .indicator {
    width: 7px; height: 7px; border-radius: 50%;
    flex-shrink: 0; transition: background 0.3s, box-shadow 0.3s;
  }
  .ind-green { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .ind-red { background: var(--red); box-shadow: 0 0 6px rgba(239,68,68,0.4); }
  .ind-yellow { background: var(--yellow); box-shadow: 0 0 6px rgba(234,179,8,0.4); }

  /* ---- Cards ---- */
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 14px;
  }
  .card-label {
    font-size: 0.7em; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-dim); margin-bottom: 14px;
  }

  /* ---- E-Stop ---- */
  .estop-btn {
    width: 100%; padding: 22px;
    background: var(--red);
    border: none; border-radius: var(--radius-sm);
    font-family: var(--font);
    font-size: 1.3em; font-weight: 700;
    letter-spacing: 0.06em;
    color: #fff; cursor: pointer;
    transition: transform 0.1s, box-shadow 0.2s;
    box-shadow: 0 0 30px var(--red-glow), inset 0 1px 0 rgba(255,255,255,0.1);
    text-transform: uppercase;
  }
  .estop-btn:active { transform: scale(0.97); }

  .estop-secondary {
    display: flex; gap: 8px; margin-top: 10px;
  }

  /* ---- Shared button styles ---- */
  .btn {
    flex: 1; padding: 13px 0;
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    font-family: var(--font);
    font-size: 0.85em; font-weight: 600;
    color: var(--text); cursor: pointer;
    transition: background 0.15s, border-color 0.15s, transform 0.1s;
    background: transparent;
    text-align: center;
  }
  .btn:active { transform: scale(0.97); }
  .btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }

  .btn-accent {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .btn-accent:active { background: #2563eb; }

  .btn-outline-green { border-color: rgba(34,197,94,0.3); color: var(--green); }
  .btn-outline-green:active { background: var(--green-dim); }

  .btn-outline-red { border-color: rgba(239,68,68,0.3); color: var(--red); }
  .btn-outline-red:active { background: var(--red-dim); }

  .btn-outline-orange { border-color: rgba(249,115,22,0.3); color: var(--orange); }
  .btn-outline-orange:active { background: var(--orange-dim); }

  /* ---- Voice pipeline ---- */
  .pipeline-row { display: flex; gap: 8px; }

  /* ---- Voice progress ---- */
  .progress-bar {
    width: 100%; height: 4px; background: var(--bg);
    border-radius: 2px; overflow: hidden; margin-bottom: 10px;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, var(--accent), var(--green));
    border-radius: 2px; transition: width 0.5s ease; width: 0%;
  }
  .progress-label {
    font-size: 0.8em; font-weight: 500; color: var(--text-muted);
    text-align: center; margin-bottom: 10px;
  }
  .voice-log {
    background: var(--bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 0.68em; line-height: 1.6;
    max-height: 150px;
    overflow-y: auto; color: var(--text-dim);
    white-space: pre-wrap; word-break: break-all;
    -webkit-overflow-scrolling: touch;
  }

  /* ---- Camera ---- */
  .cam-bar { display: flex; gap: 8px; margin-bottom: 12px; }
  .cam-select {
    flex: 1; padding: 12px 14px;
    background: var(--bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    font-family: var(--font);
    font-size: 0.85em; font-weight: 500;
    color: var(--text);
    -webkit-appearance: none; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23a1a1aa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 12px center;
    padding-right: 32px;
  }
  .cam-frame {
    width: 100%;
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--bg);
    border: 1px solid var(--card-border);
    aspect-ratio: 4/3;
    display: flex; align-items: center; justify-content: center;
  }
  .cam-feed { width: 100%; height: 100%; object-fit: cover; display: none; }
  .cam-placeholder {
    color: var(--text-dim); font-size: 0.85em; font-weight: 500;
    display: flex; flex-direction: column; align-items: center; gap: 8px;
  }
  .cam-placeholder svg { opacity: 0.3; }

  /* ---- Log ---- */
  .log-container {
    background: var(--bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    max-height: 140px;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
  }
  .log-entry {
    font-family: var(--mono);
    font-size: 0.72em; line-height: 1.7;
    color: var(--text-dim);
  }
  .log-entry .ts { color: var(--text-dim); margin-right: 6px; }
  .log-ok .msg { color: var(--green); }
  .log-err .msg { color: var(--red); }
  .log-warn .msg { color: var(--yellow); }
  .log-empty {
    font-family: var(--mono); font-size: 0.72em;
    color: var(--text-dim); opacity: 0.5; text-align: center;
    padding: 8px 0;
  }

  /* ---- Subtle loading state for buttons ---- */
  .btn-loading { position: relative; color: transparent !important; }
  .btn-loading::after {
    content: ''; position: absolute;
    width: 16px; height: 16px;
    top: 50%; left: 50%;
    margin: -8px 0 0 -8px;
    border: 2px solid rgba(255,255,255,0.2);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---- Safe area for notched phones ---- */
  @supports (padding-top: env(safe-area-inset-top)) {
    body { padding-top: env(safe-area-inset-top); padding-bottom: env(safe-area-inset-bottom); }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <path d="M12 6v6l4 2"/>
  </svg>
  <h1>Spot Control</h1>
</div>

<!-- Status pills -->
<div class="status-row">
  <div class="pill"><span class="indicator" id="dot-riva"></span><span class="label">Riva</span><span class="val" id="s-riva">--</span></div>
  <div class="pill"><span class="indicator" id="dot-ollama"></span><span class="label">Ollama</span><span class="val" id="s-ollama">--</span></div>
  <div class="pill"><span class="indicator" id="dot-estop"></span><span class="label">E-Stop</span><span class="val" id="s-estop">--</span></div>
  <div class="pill"><span class="indicator" id="dot-voice"></span><span class="label">Voice</span><span class="val" id="s-voice">--</span></div>
</div>

<!-- E-Stop -->
<div class="card">
  <div class="card-label">Emergency Stop</div>
  <button class="estop-btn" id="btn-estop" onclick="action('/estop/stop', this)">E-STOP</button>
  <div class="estop-secondary">
    <button class="btn btn-accent" onclick="action('/estop/claim', this)">Claim</button>
    <button class="btn btn-outline-green" onclick="action('/estop/release', this)">Release</button>
  </div>
</div>

<!-- Voice Pipeline -->
<div class="card">
  <div class="card-label">Voice Pipeline</div>
  <div class="pipeline-row">
    <button class="btn btn-accent" onclick="action('/voice/start', this)">Start</button>
    <button class="btn btn-outline-orange" onclick="action('/voice/stop', this)">Stop</button>
  </div>
  <div id="voice-progress" style="display:none; margin-top:14px;">
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <div class="progress-label" id="progress-label">Initializing...</div>
    <div class="voice-log" id="voice-log"></div>
  </div>
</div>

<!-- Camera -->
<div class="card">
  <div class="card-label">Camera</div>
  <div class="cam-bar">
    <select id="cam-select" class="cam-select">
      <option value="front">Front</option>
      <option value="front_right">Front Right</option>
      <option value="left">Left</option>
      <option value="right">Right</option>
      <option value="back">Back</option>
    </select>
    <button class="btn btn-accent" style="flex:0 0 auto; padding:12px 20px;" onclick="startCam()">Stream</button>
    <button class="btn btn-outline-red" id="btn-cam-stop" style="flex:0 0 auto; padding:12px 20px;" onclick="stopCam()" disabled>Stop</button>
  </div>
  <div class="cam-frame">
    <img id="cam-feed" class="cam-feed" alt="Camera feed">
    <div id="cam-placeholder" class="cam-placeholder">
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
      No feed
    </div>
  </div>
</div>

<!-- Log -->
<div class="card">
  <div class="card-label">Activity</div>
  <div class="log-container" id="log">
    <div class="log-empty">Waiting for events...</div>
  </div>
</div>

<script>
const logEl = document.getElementById('log');
let logStarted = false;

function log(msg, cls) {
  if (!logStarted) { logEl.innerHTML = ''; logStarted = true; }
  const d = document.createElement('div');
  d.className = 'log-entry ' + (cls || '');
  const t = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  d.innerHTML = '<span class="ts">' + t + '</span><span class="msg">' + msg.replace(/</g,'&lt;') + '</span>';
  logEl.prepend(d);
  while (logEl.children.length > 50) logEl.lastChild.remove();
}

function setIndicator(id, color) {
  const el = document.getElementById(id);
  if (el) el.className = 'indicator ind-' + color;
}

let prevVoice = null;
let logInterval = null;

const STAGES = [
  { p: 'Checking Riva', l: 'Checking Riva ASR...', pct: 10 },
  { p: 'Riva is already running', l: 'Riva ASR ready', pct: 20 },
  { p: 'Checking Ollama', l: 'Checking Ollama...', pct: 25 },
  { p: 'Ollama is already running', l: 'Ollama ready', pct: 35 },
  { p: 'Starting ASR bridge', l: 'Starting ASR bridge...', pct: 45 },
  { p: 'Server is running', l: 'ASR bridge ready', pct: 55 },
  { p: 'Initializing LLM', l: 'Loading LLM...', pct: 60 },
  { p: 'Brain] Ready', l: 'LLM ready', pct: 75 },
  { p: 'Initializing TTS', l: 'Loading TTS...', pct: 80 },
  { p: 'TTS] Ready', l: 'TTS ready', pct: 90 },
  { p: 'LISTENING', l: 'Ready!', pct: 100 },
];

async function pollLogs() {
  try {
    const r = await fetch('/voice/logs');
    if (!r.ok) return;
    const data = await r.json();
    if (!data.lines.length) return;

    const el = document.getElementById('voice-log');
    el.textContent = data.lines.join('\n');
    el.scrollTop = el.scrollHeight;

    const text = data.lines.join('\n');
    let pct = 5, label = 'Initializing...';
    for (const s of STAGES) {
      if (text.includes(s.p)) { pct = s.pct; label = s.l; }
    }
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-label').textContent = label;

    if (pct >= 100 && logInterval) {
      clearInterval(logInterval);
      logInterval = setInterval(pollLogs, 5000);
    }
  } catch(e) {}
}

function updateStatus(data) {
  function set(id, val) {
    const el = document.getElementById('s-' + id);
    if (el) el.textContent = val;
    const color = val === 'running' || val === 'active' ? 'green' :
                  val === 'stopped' ? 'red' : 'yellow';
    setIndicator('dot-' + id, color);
  }
  set('riva', data.riva);
  set('ollama', data.ollama);
  set('estop', data.estop);
  set('voice', data.voice);

  if (data.voice === 'running' && prevVoice !== 'running') {
    document.getElementById('voice-progress').style.display = 'block';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-label').textContent = 'Initializing...';
    document.getElementById('voice-log').textContent = '';
    if (logInterval) clearInterval(logInterval);
    logInterval = setInterval(pollLogs, 1000);
    pollLogs();
  } else if (data.voice !== 'running' && prevVoice === 'running') {
    if (logInterval) { clearInterval(logInterval); logInterval = null; }
    document.getElementById('progress-label').textContent = 'Pipeline stopped';
    document.getElementById('progress-fill').style.width = '0%';
    setTimeout(function() {
      document.getElementById('voice-progress').style.display = 'none';
    }, 5000);
  }
  prevVoice = data.voice;
}

async function pollStatus() {
  try {
    const r = await fetch('/status');
    if (r.ok) updateStatus(await r.json());
  } catch(e) {}
}

async function action(path, btnEl) {
  if (btnEl && !btnEl.classList.contains('estop-btn')) {
    btnEl.classList.add('btn-loading');
    btnEl.disabled = true;
  }
  try {
    log(path, 'log-warn');
    const r = await fetch(path, {method: 'POST'});
    const data = await r.json();
    log(data.message, data.ok ? 'log-ok' : 'log-err');
    pollStatus();
  } catch(e) {
    log('Request failed: ' + e, 'log-err');
  } finally {
    if (btnEl && !btnEl.classList.contains('estop-btn')) {
      btnEl.classList.remove('btn-loading');
      btnEl.disabled = false;
    }
  }
}

pollStatus();
setInterval(pollStatus, 2000);

let camActive = false;

function startCam() {
  const source = document.getElementById('cam-select').value;
  const img = document.getElementById('cam-feed');
  const ph = document.getElementById('cam-placeholder');
  img.src = '/camera/stream?source=' + source;
  img.style.display = 'block';
  ph.style.display = 'none';
  document.getElementById('btn-cam-stop').disabled = false;
  camActive = true;
  log('Camera: ' + source, 'log-ok');
}

function stopCam() {
  const img = document.getElementById('cam-feed');
  const ph = document.getElementById('cam-placeholder');
  img.src = '';
  img.style.display = 'none';
  ph.style.display = 'block';
  document.getElementById('btn-cam-stop').disabled = true;
  camActive = false;
  log('Camera off', 'log-warn');
}

document.getElementById('cam-select').addEventListener('change', function() {
  if (camActive) startCam();
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class SpotHandler(BaseHTTPRequestHandler):
    """Handle GET/POST requests for the control panel."""

    def __init__(self, estop_mgr, voice_mgr, camera_streamer, no_spot, *args, **kwargs):
        self.estop_mgr = estop_mgr
        self.voice_mgr = voice_mgr
        self.camera_streamer = camera_streamer
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
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._html(HTML_PAGE)
        elif path == "/status":
            svc = get_service_status()
            svc["estop"] = self.estop_mgr.status
            svc["voice"] = self.voice_mgr.status
            self._json(svc)
        elif path == "/voice/logs":
            lines = self.voice_mgr.get_logs()
            self._json({"lines": lines})
        elif path == "/camera/stream":
            if self.no_spot or self.camera_streamer is None:
                self._json({"ok": False, "message": "Camera unavailable"}, 503)
                return
            source = params.get("source", ["front"])[0]
            self._stream_camera(source)
        elif path == "/camera/snapshot":
            if self.no_spot or self.camera_streamer is None:
                self._json({"ok": False, "message": "Camera unavailable"}, 503)
                return
            source = params.get("source", ["front"])[0]
            self._snapshot_camera(source)
        else:
            self.send_error(404)

    def _stream_camera(self, source):
        """MJPEG multipart stream."""
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

        failures = 0
        try:
            while True:
                jpeg = self.camera_streamer.capture_jpeg(source)
                if jpeg is None:
                    failures += 1
                    if failures > 10:
                        break
                    time.sleep(1)
                    continue
                failures = 0
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.3)  # ~3 FPS
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _snapshot_camera(self, source):
        """Single JPEG frame."""
        jpeg = self.camera_streamer.capture_jpeg(source)
        if jpeg is None:
            self._json({"ok": False, "message": "Camera capture failed"}, 503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg)

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
    camera_streamer = None if args.no_spot else CameraStreamer()

    handler = partial(SpotHandler, estop_mgr, voice_mgr, camera_streamer, args.no_spot)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)

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
        # Tear down the voice subprocess first — it runs in its own
        # process group (preexec_fn=os.setsid) and would otherwise keep
        # holding the mic after the panel exits. The 20s SIGINT path
        # gives client_mic.py time to sit Spot down cleanly.
        if voice_mgr.status == "running":
            print("Stopping voice pipeline...")
            voice_mgr.stop()
        # Clean up E-Stop
        if estop_mgr.status == "active":
            print("Releasing E-Stop...")
            estop_mgr.stop()


if __name__ == "__main__":
    main()
