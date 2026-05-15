#!/usr/bin/env python3
"""Integrated voice control for Spot.

    Mic -> VAD -> ASR (server.py backend) -> LLM Brain -> Action + TTS Response

Auto-starts all required services (Ollama, ASR bridge).
Only prerequisite: E-Stop must be running separately.

Usage:
    python scripts/run_voice_control.py
    python scripts/run_voice_control.py --no-tts       # silent mode
    python scripts/run_voice_control.py --server-only   # ASR server only
    python scripts/run_voice_control.py --skip-services  # don't touch Ollama
"""
import sys
import pathlib
import subprocess
import argparse
import time
import socket
import os
import signal
import threading
import queue as queue_module

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOICE_DIR = PROJECT_ROOT / "src" / "voice_control"

OLLAMA_PORT = 11434
ASR_PORT = 50055


# ---------------------------------------------------------------------------
# Service helpers
# ---------------------------------------------------------------------------
def _port_open(port, host="127.0.0.1", timeout=1):
    """Check if a TCP port is accepting connections."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        ok = s.connect_ex((host, port)) == 0
        s.close()
        return ok
    except Exception as e:
        # Surface the underlying reason once instead of swallowing it.
        # Otherwise the user gets a cryptic service-failure message with
        # no hint that e.g. the loopback adapter is misconfigured.
        print(f"[svc] _port_open({host}:{port}) failed: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return False


def _wait_for_port(port, label, timeout=60):
    """Wait until a port is open, with progress dots."""
    print(f"   Waiting for {label} (port {port})...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            print(" ready")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    print(" TIMEOUT")
    return False


def _docker_container_running(name):
    """Check if a Docker container is running by name."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() == "true"
    except Exception as e:
        # Most common cause: docker socket permission denied
        # ("permission denied while trying to connect"). The old silent
        # except left the user staring at a cryptic failure message with
        # no hint that they need to add themselves to the docker group.
        print(f"[svc] docker inspect {name} (Running) failed: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return False


def _docker_container_exists(name):
    """Check if a Docker container exists (running or stopped)."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except Exception as e:
        print(f"[svc] docker inspect {name} (Status) failed: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return False


def ensure_ollama():
    """Ensure Ollama is running."""
    print("\n[Services] Checking Ollama...")

    if _port_open(OLLAMA_PORT):
        print("   Ollama is already running on port", OLLAMA_PORT)
        return True

    # Try systemctl first (preferred on Jetson)
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "ollama"],
            capture_output=True, text=True, timeout=5
        )
        if r.stdout.strip() == "active":
            # Service says active but port not ready — wait briefly
            return _wait_for_port(OLLAMA_PORT, "Ollama", timeout=15)

        print("   Starting Ollama via systemctl...")
        subprocess.run(
            ["sudo", "systemctl", "start", "ollama"],
            capture_output=True, text=True, timeout=15
        )
        return _wait_for_port(OLLAMA_PORT, "Ollama", timeout=30)
    except Exception:
        pass

    print("   ERROR: Could not start Ollama.")
    print("   Try manually: sudo systemctl start ollama")
    return False


# ---------------------------------------------------------------------------
# ASR bridge server
# ---------------------------------------------------------------------------
def start_asr_server():
    """Start the gRPC ASR bridge server in the background."""
    print("\n[ASR] Starting ASR bridge server...")

    # Kill stale processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", "voice_control/server.py"],
            capture_output=True, text=True
        )
        for pid_str in result.stdout.strip().split('\n'):
            if pid_str.strip():
                pid = int(pid_str.strip())
                print(f"   Killing stale ASR server (PID {pid})...")
                os.kill(pid, signal.SIGTERM)
        if result.stdout.strip():
            time.sleep(1)
    except Exception:
        pass

    output_queue = queue_module.Queue()

    def read_output(pipe, q):
        for line in iter(pipe.readline, b''):
            q.put(line.decode('utf-8', errors='replace').rstrip())
        pipe.close()

    server_proc = subprocess.Popen(
        [sys.executable, str(VOICE_DIR / "server.py")],
        cwd=str(VOICE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )

    output_thread = threading.Thread(target=read_output, args=(server_proc.stdout, output_queue))
    output_thread.daemon = True
    output_thread.start()

    print(f"   ASR server started (PID: {server_proc.pid})")
    print("   Waiting for server to initialize...")

    server_ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            while True:
                line = output_queue.get_nowait()
                print(f"   [Server] {line}")
                if "listening on" in line.lower() or "ready for connections" in line.lower():
                    server_ready = True
        except queue_module.Empty:
            pass

        if server_proc.poll() is not None:
            print("   ERROR: ASR server exited!")
            time.sleep(0.5)
            try:
                while True:
                    line = output_queue.get_nowait()
                    print(f"   [Server] {line}")
            except queue_module.Empty:
                pass
            return None

        if server_ready:
            break

    if not server_ready:
        print("   WARNING: Server may not be ready, but continuing...")
    else:
        print("   Server is running and ready")

    # Verify port
    if _wait_for_port(ASR_PORT, "ASR bridge", timeout=10):
        pass
    else:
        print("   WARNING: ASR bridge port not responding, client may fail")

    return server_proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Spot voice control system")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start ASR server (assumes it's already running)")
    parser.add_argument("--server-only", action="store_true",
                        help="Only start ASR server, don't start client")
    parser.add_argument("--no-brain", action="store_true",
                        help="Disable LLM brain (regex-only)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech")
    parser.add_argument("--debug-audio", action="store_true",
                        help="Print audio levels for mic diagnostics")
    parser.add_argument("--skip-services", action="store_true",
                        help="Don't auto-start Ollama (assume already running)")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Always listening (skip wake word)")
    parser.add_argument("--debug-crash", action="store_true",
                        help="Enable native-crash diagnostics for the client "
                             "(MALLOC_CHECK_=3 + PYTHONFAULTHANDLER=1). Use this "
                             "to capture 'double free / corruption' aborts with a "
                             "Python stack trace at the moment of the abort.")
    parser.add_argument("--device", type=int, default=None,
                        help="Mic input device index (auto-detected from XVF3800)")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Speaker output device index (auto-detected from UACDemoV1.0)")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="TTS + beep output gain (0.0-1.5, default 1.0). "
                             "Voice control accepts 1-100%% at runtime.")
    parser.add_argument("--map", type=str, default=None,
                        help="Path to a GraphNav map directory to upload at "
                             "startup (default: don't upload).")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spot Voice Control System")
    print("=" * 60)
    print("\n  Prerequisite: E-Stop must be running separately")
    print("    python scripts/estop_run.py")
    print("    (or scripts/web_panel.py for phone control)")

    # --- Auto-start services ---
    if not args.skip_services:
        if not args.no_brain:
            if not ensure_ollama():
                print("\n  WARNING: Ollama not available. LLM brain will be disabled.")
                print("  Falling back to regex-only mode.\n")
                args.no_brain = True
    else:
        print("\n[Services] Skipping service checks (--skip-services)")

    # --- ASR bridge server ---
    server_proc = None
    if not args.no_server:
        server_proc = start_asr_server()
        if server_proc is None:
            return 1

    if args.server_only:
        print("\n[Server-only mode] ASR server is running.")
        print("Press Ctrl+C to stop...")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server_proc.terminate()
            server_proc.wait()
        return 0

    # --- Voice client ---
    step = "2/2" if server_proc else "1/1"
    print(f"\n[{step}] Starting voice client (microphone listener)...")
    if args.no_server:
        print("   (ASR server should be running on localhost:50055)")
    print("   Speak naturally -- Spot will understand and respond.")
    print("   Safety commands (stop/freeze/estop) are always instant.")
    print("   Press Ctrl+C to stop.\n")

    client_cmd = [sys.executable, str(VOICE_DIR / "client_mic.py")]
    if args.no_brain:
        client_cmd.append("--no-brain")
    if args.no_tts:
        client_cmd.append("--no-tts")
    if args.debug_audio:
        client_cmd.append("--debug-audio")
    if args.no_wake_word:
        client_cmd.append("--no-wake-word")
    if args.device is not None:
        client_cmd.extend(["--device", str(args.device)])
    if args.output_device is not None:
        client_cmd.extend(["--output-device", str(args.output_device)])
    if args.volume != 1.0:
        client_cmd.extend(["--volume", str(args.volume)])
    if args.map:
        client_cmd.extend(["--map", args.map])

    client_env = os.environ.copy()
    if args.debug_crash:
        # MALLOC_CHECK_=3 makes glibc abort *at* the bad free/corruption instead
        # of much later, so the backtrace points at the actual culprit.
        # PYTHONFAULTHANDLER=1 dumps the Python stack on SIGABRT/SIGSEGV so we
        # can see which Python frame was active when the C extension blew up.
        client_env["MALLOC_CHECK_"] = "3"
        client_env["PYTHONFAULTHANDLER"] = "1"
        print("[Debug] Crash diagnostics ON for client subprocess")
        print("        MALLOC_CHECK_=3 PYTHONFAULTHANDLER=1")
        print("        (expect a glibc 'malloc: ...' line and a Python traceback on abort)")

    # Use Popen instead of subprocess.run so we can manually wait, escalate
    # the kill on a second Ctrl+C, and avoid the wedge where the parent's
    # subprocess.run sits forever in process.wait() while the child is hung
    # in cleanup. Without this escape hatch, Ctrl+C in the terminal looked
    # to the user like it did nothing — see ``client_mic.py``'s
    # ``_shutdown_signal_handler`` for the matching child-side fix.
    client_proc = subprocess.Popen(client_cmd, cwd=str(VOICE_DIR), env=client_env)

    # Shutdown signal forwarding.
    #
    # There are three ways this script gets a shutdown signal, and the
    # signal-routing differs in each case:
    #
    #   1. CLI Ctrl+C: terminal sends SIGINT to the foreground process
    #      group. Both this parent and ``client_mic.py`` receive it
    #      simultaneously. The child runs its own graceful shutdown.
    #      We just need to wait.
    #
    #   2. Web panel stop button: ``web_panel.py`` calls
    #      ``os.killpg(pgid, SIGINT)``. Same as case 1 — process-group
    #      delivery. We just need to wait.
    #
    #   3. wakespot orchestrator: ``wakespot.py`` puts us in our own
    #      session via ``start_new_session=True`` and then sends
    #      ``proc.send_signal(SIGTERM)`` — to the PARENT ONLY, not the
    #      group. Without explicit forwarding, ``client_mic.py`` never
    #      hears about the shutdown and keeps running its main loop
    #      while we sit here in ``client_proc.wait()`` with nothing to
    #      wait for. This was the bug that made wakespot's graceful
    #      stop look completely broken.
    #
    # Cases 1 and 2 deliver SIGINT and the child has already received it
    # via the process group. We must NOT re-forward SIGINT to the child,
    # because ``client_mic.py``'s second-signal handler treats a second
    # signal as the "force quit" escape hatch and would call os._exit()
    # immediately, skipping cleanup. So SIGINT we just absorb and wait.
    #
    # Case 3 delivers SIGTERM and the child got nothing. We forward
    # SIGINT to it explicitly so its main loop wakes up into cleanup.
    def _on_sigterm(signum, frame):
        if client_proc.poll() is None:
            try:
                client_proc.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        try:
            client_proc.wait()
        except KeyboardInterrupt:
            # First Ctrl+C: the SIGINT already went to the child via the
            # foreground process group, so client_mic.py is doing its
            # graceful shutdown right now. Wait a generous-but-bounded
            # window for it to finish (sit + power_off can legitimately
            # take ~30s on a healthy robot). A second Ctrl+C escalates.
            print("\n\nShutting down... (press Ctrl+C again to force quit)")
            try:
                client_proc.wait(timeout=40)
            except KeyboardInterrupt:
                print("\n[run_voice_control] second Ctrl+C — terminating client")
                client_proc.terminate()
                try:
                    client_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("[run_voice_control] client did not exit — sending SIGKILL")
                    client_proc.kill()
                    client_proc.wait()
            except subprocess.TimeoutExpired:
                print("[run_voice_control] client cleanup exceeded 40s — terminating")
                client_proc.terminate()
                try:
                    client_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    client_proc.kill()
                    client_proc.wait()
        return client_proc.returncode
    finally:
        if server_proc and server_proc.poll() is None:
            print("Stopping ASR server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[run_voice_control] ASR server did not exit — SIGKILL")
                server_proc.kill()
                server_proc.wait()


if __name__ == "__main__":
    sys.exit(main())
