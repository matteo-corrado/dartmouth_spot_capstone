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
  2. Resolves the requested map (if any) via src.map_loader.resolve_map_path.
  3. Claims the E-Stop in-process (a daemon thread runs the keepalive loop) so
     there is no buffered-stdout / signal-race issue with subprocess approach.
  4. Execs scripts/run_voice_control.py in the foreground, forwarding any
     pass-through flags plus --map <resolved_path>.
  5. On clean exit (or KeyboardInterrupt), records the deployed map name in
     maps/.last_used and tears down the E-Stop thread.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = Path("/tmp/wakespot.pid")

# Make `import src.*` work when wakespot.py is run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        # Stale lockfile — remove it.
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
        contents = LOCKFILE.read_text().strip()
        if contents == str(os.getpid()):
            LOCKFILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# E-Stop in-process
# ---------------------------------------------------------------------------
def _import_estop_run():
    """Import scripts/estop_run.py as a module despite its sys.path side-effects.

    `scripts/estop_run.py` does `sys.path.append(...)` at the top, so it can't
    be imported via the normal `from scripts.estop_run import ...` (no __init__,
    plus path magic). Use importlib's spec_from_file_location to load it
    directly from disk.
    """
    estop_path = PROJECT_ROOT / "scripts" / "estop_run.py"
    spec = importlib.util.spec_from_file_location("estop_run", estop_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {estop_path}")
    estop_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(estop_run)
    return estop_run


def _start_estop_thread(stop_event: threading.Event):
    """Claim the E-Stop in a daemon thread and run the keepalive loop.

    Returns (thread, holder) where `holder` is a dict containing the
    EstopKeepAlive object once the claim succeeds (so the caller can call
    .shutdown() on cleanup). The thread exits when stop_event is set.
    """
    estop_run = _import_estop_run()
    holder: dict = {"keepalive": None, "error": None, "ready": threading.Event()}

    def _runner():
        try:
            print("[wakespot] Claiming E-Stop...")
            _, _, _, keepalive = estop_run.start_estop(claim=True)
            holder["keepalive"] = keepalive
            holder["ready"].set()
            print("[wakespot] E-Stop active.")
            while not stop_event.is_set():
                try:
                    keepalive.allow()
                except Exception:
                    print(
                        "[wakespot] E-Stop endpoint lost — another client "
                        "took ownership. Stopping keepalive."
                    )
                    break
                time.sleep(0.5)
        except Exception as e:
            holder["error"] = e
            holder["ready"].set()
            print(f"[wakespot] E-Stop claim failed: {e}")

    thread = threading.Thread(target=_runner, name="wakespot-estop", daemon=True)
    thread.start()
    return thread, holder


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
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Forwarded to voice control: disable text-to-speech.",
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Forwarded to voice control: always-listening mode.",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=None,
        help="Forwarded to voice control: TTS + beep output gain (0.0-1.5).",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Forwarded to voice control: print mic levels for diagnostics.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Forwarded to voice control: mic input device index.",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Forwarded to voice control: speaker output device index.",
    )
    parser.add_argument(
        "--debug-crash",
        action="store_true",
        help=(
            "Forwarded to voice control: enable native-crash diagnostics "
            "(MALLOC_CHECK_=3 + PYTHONFAULTHANDLER=1)."
        ),
    )
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="Forwarded to voice control: do not auto-start Riva/Ollama.",
    )
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

    # ----- Lockfile -----
    if not _acquire_lockfile():
        return 1

    estop_thread = None
    estop_holder = None
    stop_event = threading.Event()
    return_code = 0

    try:
        # ----- Resolve map (skip if --no-map) -----
        resolved_map: Path | None = None
        if not args.no_map:
            try:
                from src import map_loader

                resolved_map = map_loader.resolve_map_path(args.map, PROJECT_ROOT)
                print(f"[wakespot] Map resolved: {resolved_map}")
            except FileNotFoundError as e:
                print(f"[wakespot] {e}")
                print(
                    "[wakespot] Hint: list maps under "
                    f"{PROJECT_ROOT / 'maps'} or run with --no-map."
                )
                return 1
            except NotImplementedError:
                # W1 hasn't landed yet — graceful fallback.
                print(
                    "[wakespot] map_loader not yet implemented "
                    "(W1 not merged). Re-run with --no-map to skip "
                    "startup map upload."
                )
                return 1
        else:
            print("[wakespot] Skipping startup map upload (--no-map).")

        # ----- E-Stop in daemon thread -----
        estop_thread, estop_holder = _start_estop_thread(stop_event)
        # Wait for the claim to finish (success or failure) before launching
        # voice control, so we never start voice control with an unclaimed
        # e-stop.
        estop_holder["ready"].wait(timeout=30)
        if estop_holder["error"] is not None:
            print(
                f"[wakespot] E-Stop could not be claimed: "
                f"{estop_holder['error']}"
            )
            return 2
        if estop_holder["keepalive"] is None:
            print("[wakespot] E-Stop did not become ready within timeout.")
            return 2

        # ----- Voice control in foreground -----
        cmd = _build_voice_control_cmd(args, resolved_map)
        try:
            completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            return_code = completed.returncode
        except KeyboardInterrupt:
            print("\n[wakespot] Interrupted — shutting down.")
            return_code = 130

        # ----- Record .last_used on a clean voice-control exit -----
        if resolved_map is not None and return_code == 0:
            try:
                from src import map_loader

                map_loader.write_last_used(PROJECT_ROOT, resolved_map.name)
                print(f"[wakespot] Recorded {resolved_map.name} in maps/.last_used")
            except NotImplementedError:
                # W1 not landed yet — non-fatal.
                pass
            except Exception as e:
                print(f"[wakespot] Could not write .last_used: {e}")

        return return_code

    except KeyboardInterrupt:
        print("\n[wakespot] Interrupted before voice control started.")
        return 130
    finally:
        # ----- Cleanup -----
        stop_event.set()
        if estop_thread is not None:
            estop_thread.join(timeout=3.0)
        if estop_holder is not None and estop_holder.get("keepalive") is not None:
            try:
                estop_holder["keepalive"].stop()
            except Exception:
                pass
            try:
                estop_holder["keepalive"].shutdown()
            except Exception:
                pass
        print("[wakespot] Stopped E-Stop and cleaned up.")
        _release_lockfile()


if __name__ == "__main__":
    # Make Ctrl+C in the main thread propagate cleanly through to our finally.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
