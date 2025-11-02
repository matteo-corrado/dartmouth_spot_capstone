#!/usr/bin/env python3
"""Integrated voice control script for Spot.

This script provides a convenient way to run the voice control system.
It requires:
1. E-Stop running (via scripts/estop_run.py in separate terminal)
2. ASR server running (starts automatically or can run separately)
3. Client microphone listening for voice commands

Usage:
    python scripts/run_voice_control.py [--no-server] [--server-only]
    
    --no-server: Don't start ASR server (assumes it's running elsewhere)
    --server-only: Only start ASR server, don't start client
"""
import sys
import pathlib
import subprocess
import argparse
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description="Run Spot voice control system")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start ASR server (assumes it's already running)")
    parser.add_argument("--server-only", action="store_true",
                        help="Only start ASR server, don't start client")
    args = parser.parse_args()
    
    project_root = pathlib.Path(__file__).resolve().parents[1]
    voice_dir = project_root / "src" / "voice_control"
    
    print("=" * 60)
    print("Spot Voice Control System")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Make sure E-Stop is running!")
    print("   Run this in a separate terminal:")
    print("   python scripts/estop_run.py")
    print("\nPress Enter when E-Stop is running...")
    input()
    
    if not args.no_server and not args.server_only:
        print("\n[1/2] Starting ASR server...")
        server_proc = subprocess.Popen(
            [sys.executable, str(voice_dir / "server.py")],
            cwd=str(voice_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("   ASR server started (PID: {})".format(server_proc.pid))
        print("   Waiting for server to initialize...")
        time.sleep(3)  # Give server time to load model
        
        if server_proc.poll() is not None:
            print("   ERROR: Server exited immediately!")
            stdout, stderr = server_proc.communicate()
            print(stdout.decode())
            print(stderr.decode())
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
    
    if not args.no_server:
        print("\n[2/2] Starting voice client (microphone listener)...")
        print("   Speak commands when prompted.")
        print("   Commands: 'stop', 'follow me', 'come here', 'turn left 45', 'turn right 90', etc.")
        print("   Press Ctrl+C to stop.\n")
    else:
        print("\n[1/1] Starting voice client (microphone listener)...")
        print("   (ASR server should be running on localhost:50055)")
        print("   Press Ctrl+C to stop.\n")
    
    try:
        client_proc = subprocess.run(
            [sys.executable, str(voice_dir / "client_mic.py")],
            cwd=str(voice_dir)
        )
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

