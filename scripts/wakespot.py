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
  4. Spawns scripts/run_voice_control.py in its own process group, forwarding
     pass-through flags plus --map <resolved_path>.
  5. Drives a graceful-or-emergency shutdown state machine:
       * Ctrl+C  → graceful: ask voice control to sit Spot and power off
                   motors, wait, then release the E-Stop endpoint WITHOUT
                   issuing a CUT.
       * Ctrl+C twice OR Ctrl+\\ (SIGQUIT) → emergency: issue an immediate
                   E-Stop CUT, then SIGKILL voice control.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = Path("/tmp/wakespot.pid")

# Upper bound on how long we wait for voice control to finish blocking_sit +
# power_off after we ask it to shut down. Sized to comfortably cover
# close_spot_session()'s blocking_sit(timeout_sec=15) + power_off(timeout_sec=20)
# plus audio teardown slack. Past this, we escalate to the emergency path so
# the user is never stuck with a hung child holding the lease.
GRACEFUL_TIMEOUT_SEC = 45

# ---------------------------------------------------------------------------
# Shutdown signalling state (set by signal handlers, read by the wait loop).
# ---------------------------------------------------------------------------
_shutdown_requested = threading.Event()
_emergency_requested = threading.Event()
_sigint_count = 0


def _on_sigint(signum, frame):
    """First Ctrl+C → graceful. Second Ctrl+C → escalate to emergency."""
    global _sigint_count
    _sigint_count += 1
    if _sigint_count == 1:
        print(
            "\n[wakespot] Graceful shutdown requested — sit, power off, then "
            "release E-Stop. Hit Ctrl+C again or Ctrl+\\ for an emergency cut.",
            flush=True,
        )
        _shutdown_requested.set()
    else:
        print(
            "\n[wakespot] EMERGENCY E-Stop (second Ctrl+C) — cutting motors NOW.",
            flush=True,
        )
        _emergency_requested.set()


def _on_sigquit(signum, frame):
    """Ctrl+\\ is always an immediate emergency cut, no second-press needed."""
    print(
        "\n[wakespot] EMERGENCY E-Stop (SIGQUIT) — cutting motors NOW.",
        flush=True,
    )
    _emergency_requested.set()

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
    parser.add_argument("--tts-backend", choices=["kokoro", "elevenlabs"], default=None,
                        help="Forwarded to voice control: TTS backend "
                             "(overrides SPOT_TTS_BACKEND env).")
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
    parser.add_argument("--enable-arm", action="store_true",
                        help="Forwarded to voice control: deploy the arm + enable "
                             "the gripper mouth at startup.")
    return parser


def _build_voice_control_cmd(args: argparse.Namespace, resolved_map: Path | None) -> list[str]:
    """Construct the argv for scripts/run_voice_control.py from parsed args."""
    cmd: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_voice_control.py"),
    ]
    if args.no_tts:
        cmd.append("--no-tts")
    if args.tts_backend is not None:
        cmd.extend(["--tts-backend", args.tts_backend])
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
    if args.enable_arm:
        cmd.append("--enable-arm")
    if resolved_map is not None:
        cmd.extend(["--map", str(resolved_map)])
    return cmd


# ---------------------------------------------------------------------------
# Shutdown state machine
# ---------------------------------------------------------------------------
def _emergency_cut(proc: subprocess.Popen, keepalive) -> int:
    """Issue an immediate E-Stop CUT and SIGKILL the voice control subprocess.

    Order matters: cut motors FIRST, then kill the process. The opposite
    order would leave a brief window in which a runaway voice control
    process could send another command before motors are de-energized.
    """
    try:
        keepalive.stop()  # E-Stop CUT
        print("[wakespot] E-Stop CUT issued — motors de-energized.", flush=True)
    except Exception as e:
        print(f"[wakespot] keepalive.stop() failed: {e}", flush=True)

    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print("[wakespot] Voice control SIGKILLed.", flush=True)
        except (ProcessLookupError, OSError) as e:
            print(f"[wakespot] killpg failed: {e}", flush=True)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[wakespot] WARN: voice control did not reap after SIGKILL.",
                  flush=True)
    return 137  # 128 + SIGKILL(9)


def _graceful_shutdown(proc: subprocess.Popen, keepalive) -> int:
    """Ask voice control to sit Spot and power off motors, then return.

    Sends SIGTERM to the child (which client_mic.py converts to a
    KeyboardInterrupt and runs cleanup_spot → blocking_sit → power_off),
    then waits up to GRACEFUL_TIMEOUT_SEC for the child to exit. If the
    user escalates to emergency mid-wait, OR the timeout expires, we
    escalate to _emergency_cut.
    """
    print("[wakespot] Asking voice control to sit Spot and power off motors...",
          flush=True)
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        # Child already dead — nothing to wait on, fall through to release.
        pass

    deadline = time.monotonic() + GRACEFUL_TIMEOUT_SEC
    last_progress = time.monotonic()
    while proc.poll() is None and time.monotonic() < deadline:
        if _emergency_requested.is_set():
            print("[wakespot] Emergency requested mid-graceful — escalating.",
                  flush=True)
            return _emergency_cut(proc, keepalive)
        # Heartbeat so the user can see we are waiting on the child, not hung.
        if time.monotonic() - last_progress >= 5:
            remaining = int(deadline - time.monotonic())
            print(f"[wakespot]   ...waiting for voice control "
                  f"({remaining}s before escalation)", flush=True)
            last_progress = time.monotonic()
        time.sleep(0.1)

    if proc.poll() is None:
        print(
            f"[wakespot] Voice control did not exit within {GRACEFUL_TIMEOUT_SEC}s "
            "— escalating to emergency cut.",
            flush=True,
        )
        return _emergency_cut(proc, keepalive)

    print("[wakespot] Voice control exited cleanly. Releasing E-Stop endpoint "
          "(no cut).", flush=True)
    return proc.returncode if proc.returncode is not None else 0


def _wait_for_child(proc: subprocess.Popen, keepalive) -> int:
    """Idle wait loop. Watches for shutdown / emergency events and the child.

    Runs entirely on the main thread so signal handlers (which Python only
    invokes on the main thread) can deliver their state via the
    threading.Events. Polling cadence is 100ms — fast enough that an
    emergency-cut request feels instant to a human, slow enough to be
    invisible on the CPU profile.
    """
    while proc.poll() is None:
        if _emergency_requested.is_set():
            return _emergency_cut(proc, keepalive)
        if _shutdown_requested.is_set():
            return _graceful_shutdown(proc, keepalive)
        time.sleep(0.1)
    # Voice control exited on its own (e.g. user said "shutdown" via voice).
    print(f"[wakespot] Voice control exited (rc={proc.returncode}).", flush=True)
    return proc.returncode if proc.returncode is not None else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not _acquire_lockfile():
        return 1

    # Install our shutdown handlers BEFORE doing any robot work, so a Ctrl+C
    # during early bring-up (auth, time-sync, map upload) is still observed.
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGQUIT, _on_sigquit)

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
        # an internal EstopKeepAlive heartbeat thread. We pass cut_on_exit=False
        # so the context manager only releases the endpoint on exit — wakespot
        # itself decides whether to issue keepalive.stop() (the CUT) based on
        # whether the user requested a graceful or emergency shutdown.
        cmd = _build_voice_control_cmd(args, resolved_map)
        with estop_session(name="wakespot_estop", cut_on_exit=False) as estop_ctx:
            keepalive = estop_ctx["keepalive"]
            print("[wakespot] E-Stop active.")
            print("[wakespot]   Ctrl+C  = graceful shutdown (sit, power off, "
                  "then release)")
            print("[wakespot]   Ctrl+C ×2 / Ctrl+\\ = emergency E-Stop CUT")

            # start_new_session=True puts the child in its OWN process group,
            # so the terminal's Ctrl+C / Ctrl+\ only hit wakespot. Wakespot
            # then forwards SIGTERM (graceful) or SIGKILL (emergency) on its
            # own terms. Without this, both processes would race to handle
            # the same signal and the e-stop release could fire before the
            # child finished sitting Spot.
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), start_new_session=True
            )
            return _wait_for_child(proc, keepalive)
    finally:
        print("[wakespot] Released E-Stop and cleaned up.")
        _release_lockfile()


if __name__ == "__main__":
    sys.exit(main())
