#!/usr/bin/env python3
"""Integrated voice control for Spot.

    Mic → VAD → Riva ASR (Canary-Qwen-2.5B) → LLM Brain → Action + TTS Response

Requires:
1. E-Stop running (scripts/estop_run.py in separate terminal)
2. Riva server running (scripts/setup_riva.sh start)
3. Ollama running with qwen2.5:7b pulled
4. ASR bridge server (starts automatically)

Usage:
    python scripts/run_voice_control.py
    python scripts/run_voice_control.py --no-tts      # silent mode
    python scripts/run_voice_control.py --server-only  # ASR server only
"""
import sys
import pathlib
import subprocess
import argparse
import time
import socket

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

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
    args = parser.parse_args()
    
    project_root = pathlib.Path(__file__).resolve().parents[1]
    voice_dir = project_root / "src" / "voice_control"
    
    print("=" * 60)
    print("Spot Voice Control System")
    print("=" * 60)
    print("\n  IMPORTANT: Make sure these are running:")
    print("  1. E-Stop:      python scripts/estop_run.py")
    print("  2. Riva server:  scripts/setup_riva.sh start")
    if not args.no_brain:
        print("  3. Ollama:       sudo systemctl start ollama")

    # Quick check: is Riva server reachable on port 50051?
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        riva_ok = sock.connect_ex(('127.0.0.1', 50051)) == 0
        sock.close()
    except Exception:
        riva_ok = False

    if riva_ok:
        print("\n  ✓ Riva server detected on port 50051")
    else:
        print("\n  ✗ Riva server NOT detected on port 50051")
        print("    Start it with: scripts/setup_riva.sh start")

    print("\nPress Enter when ready...")
    input()
    
    if not args.no_server and not args.server_only:
        print("\n[1/2] Starting ASR server...")

        # Kill any stale ASR server processes from previous runs
        import os, signal
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
                time.sleep(1)  # Let old processes die
        except Exception:
            pass

        # Run server in background but capture output to see errors
        import threading
        import queue as queue_module
        
        output_queue = queue_module.Queue()
        
        def read_output(pipe, queue):
            for line in iter(pipe.readline, b''):
                queue.put(line.decode('utf-8', errors='replace').rstrip())
            pipe.close()
        
        server_proc = subprocess.Popen(
            [sys.executable, str(voice_dir / "server.py")],
            cwd=str(voice_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        
        # Start thread to read output
        output_thread = threading.Thread(target=read_output, args=(server_proc.stdout, output_queue))
        output_thread.daemon = True
        output_thread.start()
        
        print("   ASR server started (PID: {})".format(server_proc.pid))
        print("   Waiting for server to initialize...")
        
        # Wait and check for "Server listening" message or process death
        server_ready = False
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            
            # Check for output
            try:
                while True:
                    line = output_queue.get_nowait()
                    print(f"   [Server] {line}")
                    if "listening on" in line.lower() or "ready for connections" in line.lower():
                        server_ready = True
            except queue_module.Empty:
                pass
            
            # Check if process died
            if server_proc.poll() is not None:
                print("   ERROR: Server exited!")
                # Get remaining output
                time.sleep(0.5)
                try:
                    while True:
                        line = output_queue.get_nowait()
                        print(f"   [Server] {line}")
                except queue_module.Empty:
                    pass
                return 1
            
            if server_ready:
                break
        
        if not server_ready:
            print("   WARNING: Server may not be ready, but continuing...")
        else:
            print("   ✓ Server is running and ready")
        
        # Verify server is actually listening on port 50055
        print("   Verifying server is listening on port 50055...")
        for i in range(5):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', 50055))
                sock.close()
                if result == 0:
                    print("   ✓ Server is listening on port 50055")
                    break
            except Exception as e:
                pass
            time.sleep(1)
        else:
            print("   ✗ WARNING: Server may not be listening on port 50055")
            print("   Continuing anyway, but client may fail to connect...")
    
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
    
    if not args.no_server:
        print("\n[2/2] Starting voice client (microphone listener)...")
    else:
        print("\n[1/1] Starting voice client (microphone listener)...")
        print("   (ASR server should be running on localhost:50055)")

    print("   Speak naturally — Spot will understand and respond.")
    print("   Safety commands (stop/freeze/estop) are always instant.")
    print("   Press Ctrl+C to stop.\n")

    # Build client command
    client_cmd = [sys.executable, str(voice_dir / "client_mic.py")]
    if args.no_brain:
        client_cmd.append("--no-brain")
    if args.no_tts:
        client_cmd.append("--no-tts")
    if args.debug_audio:
        client_cmd.append("--debug-audio")

    try:
        client_proc = subprocess.run(client_cmd, cwd=str(voice_dir))
        return client_proc.returncode
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if not args.no_server and 'server_proc' in locals():
            print("Stopping ASR server...")
            server_proc.terminate()
            server_proc.wait()
        return 0

if __name__ == "__main__":
    sys.exit(main())

