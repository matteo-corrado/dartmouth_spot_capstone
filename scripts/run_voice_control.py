#!/usr/bin/env python3
"""Integrated voice control for Spot.

    Mic -> VAD -> Riva ASR (Canary-Qwen-2.5B) -> LLM Brain -> Action + TTS Response

Auto-starts all required services (Riva Docker, Ollama, ASR bridge).
Only prerequisite: E-Stop must be running separately.

Usage:
    python scripts/run_voice_control.py
    python scripts/run_voice_control.py --no-tts       # silent mode
    python scripts/run_voice_control.py --server-only   # ASR server only
    python scripts/run_voice_control.py --skip-services  # don't touch Riva/Ollama
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

RIVA_CONTAINER = "riva-speech"
RIVA_PORT = 50051
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
    except Exception:
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
    except Exception:
        return False


def _docker_container_exists(name):
    """Check if a Docker container exists (running or stopped)."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


def ensure_riva():
    """Ensure Riva ASR Docker container is running."""
    print("\n[Services] Checking Riva ASR...")

    if _docker_container_running(RIVA_CONTAINER):
        if _port_open(RIVA_PORT):
            print("   Riva is already running on port", RIVA_PORT)
            return True
        # Container running but port not ready yet — wait
        return _wait_for_port(RIVA_PORT, "Riva ASR", timeout=120)

    if _docker_container_exists(RIVA_CONTAINER):
        print("   Riva container exists but is stopped. Starting...")
        try:
            subprocess.run(
                ["docker", "start", RIVA_CONTAINER],
                capture_output=True, text=True, timeout=30
            )
        except Exception as e:
            print(f"   ERROR: docker start failed: {e}")
            return False
        return _wait_for_port(RIVA_PORT, "Riva ASR", timeout=120)

    print("   ERROR: Riva container not found.")
    print("   Run the initial setup first:")
    print("     cd ~/riva_quickstart_arm64_v2.17.0 && bash riva_start.sh")
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
                        help="Don't auto-start Riva/Ollama (assume already running)")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Always listening (skip wake word)")
    parser.add_argument("--device", type=int, default=None,
                        help="Mic input device index (default: 25 = XVF3800)")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Speaker output device index (default: 24 = UACDemoV1.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spot Voice Control System")
    print("=" * 60)
    print("\n  Prerequisite: E-Stop must be running separately")
    print("    python scripts/estop_run.py")
    print("    (or scripts/web_panel.py for phone control)")

    # --- Auto-start services ---
    if not args.skip_services:
        if not ensure_riva():
            print("\n  Riva is required for ASR. Exiting.")
            return 1

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

    try:
        client_proc = subprocess.run(client_cmd, cwd=str(VOICE_DIR))
        return client_proc.returncode
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if server_proc and server_proc.poll() is None:
            print("Stopping ASR server...")
            server_proc.terminate()
            server_proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
