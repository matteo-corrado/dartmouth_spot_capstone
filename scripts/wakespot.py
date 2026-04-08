#!/usr/bin/env python3
"""wakespot — one-word launch for E-Stop + voice control + map.

Usage:
    wakespot                  # last-used map (or latest by mtime)
    wakespot jackson          # bare name -> resolves to maps/jackson
    wakespot /path/to/map     # absolute or relative path
    wakespot --no-map         # skip startup map upload
    wakespot --no-tts jackson # pass-through to voice control

The orchestrator:
  1. Acquires a /tmp/wakespot.pid lockfile (refuses if another wakespot is alive).
  2. Resolves the requested map (if any) via src.map_loader.resolve_map_path
     and records it in maps/.last_used.
  3. Claims the E-Stop via src.estop.estop_session — its EstopKeepAlive runs
     its own heartbeat thread, so we just hold the context manager open.
  4. Execs scripts/run_voice_control.py in the foreground, forwarding any
     pass-through flags plus --map <resolved_path>.
  5. The estop_session context manager releases the E-Stop on exit.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = Path("/tmp/wakespot.pid")

# Make `import src.*` work when wakespot.py is run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import map_loader
from src.estop import estop_session


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """Return True if a process with `pid` is currently alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — treat as alive.
        return True
    except OSError:
        return False
    return True


def _acquire_lockfile() -> bool:
    """Try to acquire /tmp/wakespot.pid. Returns True on success."""
    if LOCKFILE.exists():
        try:
            existing = int(LOCKFILE.read_text().strip())
        except (ValueError, OSError):
            existing = None
        if existing and _pid_alive(existing):
            print(
                f"[wakespot] Another wakespot is already running "
                f"(PID {existing}). Refusing to start."
            )
            return False
        try:
            LOCKFILE.unlink()
        except OSError:
            pass
    try:
        LOCKFILE.write_text(str(os.getpid()))
    except OSError as e:
        print(f"[wakespot] Failed to write lockfile {LOCKFILE}: {e}")
        return False
    return True


def _release_lockfile() -> None:
    """Remove our lockfile if it still belongs to us."""
    try:
        if not LOCKFILE.exists():
            return
        if LOCKFILE.read_text().strip() == str(os.getpid()):
            LOCKFILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wakespot",
        description=(
            "One-word launch for Spot's E-Stop, voice control, and a "
            "GraphNav map. Voice-control flags are forwarded as-is to "
            "scripts/run_voice_control.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  wakespot                   # last-used map\n"
            "  wakespot jackson           # named map under maps/\n"
            "  wakespot /path/to/map      # explicit path\n"
            "  wakespot --no-map          # skip startup upload\n"
            "  wakespot --no-tts jackson  # forward --no-tts to voice control\n"
        ),
    )
    parser.add_argument(
        "map",
        nargs="?",
        default=None,
        help=(
            "Map to upload at startup. Bare name (resolves to maps/<name>), "
            "an absolute/relative path, or 'latest'. Omit to use the "
            "last-deployed map (maps/.last_used)."
        ),
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip the startup map upload entirely.",
    )
    # Pass-through flags forwarded to scripts/run_voice_control.py
    parser.add_argument("--no-tts", action="store_true",
                        help="Forwarded to voice control: disable text-to-speech.")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Forwarded to voice control: always-listening mode.")
    parser.add_argument("--volume", type=float, default=None,
                        help="Forwarded to voice control: TTS + beep output gain (0.0-1.5).")
    parser.add_argument("--debug-audio", action="store_true",
                        help="Forwarded to voice control: print mic levels for diagnostics.")
    parser.add_argument("--device", type=int, default=None,
                        help="Forwarded to voice control: mic input device index.")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Forwarded to voice control: speaker output device index.")
    parser.add_argument("--debug-crash", action="store_true",
                        help="Forwarded to voice control: enable native-crash diagnostics "
                             "(MALLOC_CHECK_=3 + PYTHONFAULTHANDLER=1).")
    parser.add_argument("--skip-services", action="store_true",
                        help="Forwarded to voice control: do not auto-start Riva/Ollama.")
    return parser


def _build_voice_control_cmd(args: argparse.Namespace, resolved_map: Path | None) -> list[str]:
    """Construct the argv for scripts/run_voice_control.py from parsed args."""
    cmd: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_voice_control.py"),
    ]
    if args.no_tts:
        cmd.append("--no-tts")
    if args.no_wake_word:
        cmd.append("--no-wake-word")
    if args.volume is not None:
        cmd.extend(["--volume", str(args.volume)])
    if args.debug_audio:
        cmd.append("--debug-audio")
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.output_device is not None:
        cmd.extend(["--output-device", str(args.output_device)])
    if args.debug_crash:
        cmd.append("--debug-crash")
    if args.skip_services:
        cmd.append("--skip-services")
    if resolved_map is not None:
        cmd.extend(["--map", str(resolved_map)])
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not _acquire_lockfile():
        return 1

    try:
        # ----- Resolve map (skip if --no-map) -----
        resolved_map: Path | None = None
        if not args.no_map:
            try:
                resolved_map = map_loader.resolve_map_path(args.map, PROJECT_ROOT)
            except FileNotFoundError as e:
                print(f"[wakespot] {e}")
                print(
                    "[wakespot] Hint: list maps under "
                    f"{PROJECT_ROOT / 'maps'} or run with --no-map."
                )
                return 1
            print(f"[wakespot] Map resolved: {resolved_map}")

            # Record .last_used now — once the user has chosen a map and it
            # exists on disk, that IS the current map for this session.
            # Writing here (rather than after voice control exits) ensures the
            # right slice is used by save_location calls and survives a crash.
            try:
                map_loader.write_last_used(PROJECT_ROOT, resolved_map.name)
            except OSError as e:
                print(f"[wakespot] WARN: could not write .last_used: {e}")
        else:
            print("[wakespot] Skipping startup map upload (--no-map).")

        # ----- E-Stop + voice control -----
        # estop_session opens the EstopClient, claims the endpoint, and starts
        # an internal EstopKeepAlive heartbeat thread. The context manager
        # releases the E-Stop on exit (issues STOP, then shuts down the
        # keepalive). subprocess.run blocks the main thread for the duration
        # of voice control while the heartbeat thread keeps the E-Stop alive.
        cmd = _build_voice_control_cmd(args, resolved_map)
        with estop_session(name="wakespot_estop"):
            print("[wakespot] E-Stop active.")
            try:
                completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
                return completed.returncode
            except KeyboardInterrupt:
                print("\n[wakespot] Interrupted — shutting down.")
                return 130
    finally:
        print("[wakespot] Released E-Stop and cleaned up.")
        _release_lockfile()


if __name__ == "__main__":
    # Make Ctrl+C in the main thread propagate cleanly through to our finally.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
