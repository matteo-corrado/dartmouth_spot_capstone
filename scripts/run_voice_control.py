#!/usr/bin/env python3
"""Integrated voice control for Spot.

    Mic -> VAD -> Dartmouth STT -> LLM Brain (ChatDartmouth) -> Action + TTS Response

Checks Dartmouth API connectivity on startup, then launches the voice client.
Only prerequisite: E-Stop must be running separately.

Usage:
    python scripts/run_voice_control.py
    python scripts/run_voice_control.py --no-tts             # silent mode
    python scripts/run_voice_control.py --output-device 7    # Bluetooth speaker
    python scripts/run_voice_control.py --no-brain           # regex-only
"""
import sys
import pathlib
import subprocess
import argparse
import os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOICE_DIR = PROJECT_ROOT / "src" / "voice_control"


# ---------------------------------------------------------------------------
# API connectivity check
# ---------------------------------------------------------------------------
def check_dartmouth_api() -> bool:
    """Verify that the Dartmouth API is reachable and keys are configured."""
    from src.config import DARTMOUTH_API_KEY, DARTMOUTH_CHAT_API_KEY

    ok = True

    if not DARTMOUTH_API_KEY:
        print("   WARNING: DARTMOUTH_API_KEY is not set (needed for STT).")
        print("   Get one at https://developer.dartmouth.edu/keys")
        ok = False

    if not DARTMOUTH_CHAT_API_KEY:
        print("   WARNING: DARTMOUTH_CHAT_API_KEY is not set (needed for LLM/VLM).")
        print("   Get one at https://chat.dartmouth.edu")
        ok = False

    if not ok:
        return False

    # Quick connectivity test: try JWT exchange
    try:
        import requests
        from src.config import DARTMOUTH_JWT_URL
        resp = requests.post(
            DARTMOUTH_JWT_URL,
            headers={"Authorization": DARTMOUTH_API_KEY},
            timeout=5,
        )
        if resp.status_code == 200:
            print("   Dartmouth API reachable (JWT OK)")
        else:
            print(f"   WARNING: JWT exchange returned {resp.status_code}")
            print(f"   Check DARTMOUTH_API_KEY in .env")
            return False
    except Exception as e:
        print(f"   WARNING: Cannot reach Dartmouth API: {e}")
        print("   Ensure you are on campus WiFi.")
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Spot voice control system")
    parser.add_argument("--no-brain", action="store_true",
                        help="Disable LLM brain (regex-only)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech")
    parser.add_argument("--debug-audio", action="store_true",
                        help="Print audio levels for mic diagnostics")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Always listening (skip wake word)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (default: 24 = XVF3800)")
    parser.add_argument("--output-device", type=str, default=None,
                        help="Audio output device name or index for TTS and beeps")
    parser.add_argument("--skip-api-check", action="store_true",
                        help="Don't verify Dartmouth API connectivity at startup")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spot Voice Control System (Dartmouth Cloud APIs)")
    print("=" * 60)
    print("\n  Prerequisite: E-Stop must be running separately")
    print("    python scripts/estop_run.py")
    print("    (or scripts/web_panel.py for phone control)")

    # --- Show configuration ---
    from src.config import LLM_MODEL, VLM_MODEL
    print(f"\n  STT:  Dartmouth speech-recognition API")
    print(f"  LLM:  {LLM_MODEL} (via ChatDartmouth)")
    print(f"  VLM:  {VLM_MODEL} (via ChatDartmouth)")
    print(f"  TTS:  Kokoro (local, CPU)")

    # --- API check ---
    if not args.skip_api_check:
        print("\n[API] Checking Dartmouth API connectivity...")
        if not check_dartmouth_api():
            print("\n  Dartmouth API is required. Fix keys / network and retry.")
            print("  Or use --skip-api-check to bypass this check.")
            return 1
    else:
        print("\n[API] Skipping connectivity check (--skip-api-check)")

    # --- Voice client ---
    print(f"\nStarting voice client (microphone listener)...")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
