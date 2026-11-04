"""Operator control (spotctl): status snapshot, immediate commands, toggles
and the localhost control server.

Moved from the old client_mic.py (Task 10), logic unchanged. The
flags it flips are owned by the Conversation (operator mute, crowd mode,
voiceprint mode/threshold) and the Ring (repaint pending); the HTTP threads
read/flip them without locks so a slow request can never delay a stop.
"""
import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
import src.spot.mic.ring_control as _ring
import src.spot.safety.listener as _sl

_MOTION_COMMANDS = ("stop", "freeze", "sit", "stand", "stow")
VOICEPRINT_THRESHOLD_MAX = 0.9


def _status_snapshot() -> dict:
    """Plain-attribute read; never blocks."""
    C = _loop.CONVERSATION
    s = dict(C._status)
    s.update(muted=C._operator_muted, crowd=C._crowd_mode, lock=C._lock_enabled, voiceprint=C._voiceprint_mode,
             voiceprint_threshold=round(C._voiceprint_threshold, 3),
             voiceprint_enrolled=C._voice.enrolled,
             voiceprint_last_sim=(None if C._voice.last_sim is None else round(C._voice.last_sim, 3)),
             voiceprint_ready=C._voice.embedder is not None,
             mic_ok=_cap._mic_usable(), safety_kws_dropped=_sl.LISTENER._drops["total"],
             safety_kws_ok=_sl.LISTENER.health())
    return s


def _on_immediate(name: str, arg=None) -> dict:
    """spotctl immediate commands, run on HTTP worker threads.

    Re-entrant and lock-free: touches only the player, the mouth and the
    dispatcher (via execute_on_spot, which honors dry-run), plus plain flags.
    Motion commands also set _abort_chain so a chain in progress stops.
    """
    if name == "status":
        return _status_snapshot()
    if name == "hush":
        _sl._mark_safety()
        _sl._hush()
        return {"ok": True}
    if name in _MOTION_COMMANDS:
        _sl._mark_safety()
        if name in ("stop", "freeze"):
            _sl._start_hush()  # same path as a spoken halt: cut speech too
        ok = _loop.execute_on_spot({"intent": name, "params": {}, "raw": f"spotctl {name}"})
        return {"ok": bool(ok)}
    if name in ("mute", "unmute"):
        _loop.CONVERSATION._operator_muted = (name == "mute")
        _ring.RING._ring_pending = True  # main loop repaints (red while muted)
        return {"ok": True, "muted": _loop.CONVERSATION._operator_muted}
    return {"error": f"unhandled immediate command {name!r}"}


def apply_voiceprint_arg(arg, mode: bool, threshold: float) -> tuple:
    """spotctl voiceprint [on|off|threshold <x>] -> (mode, threshold).
    The threshold is clamped to [0.0, 0.9]; a bad number raises ValueError."""
    a = (arg or "").strip().lower().split()
    if a and a[0] in ("threshold", "thr", "t"):
        if len(a) < 2:
            raise ValueError("threshold needs a value, e.g. 'voiceprint threshold 0.25'")
        try:
            x = float(a[1])
        except ValueError:
            raise ValueError(f"bad threshold {a[1]!r}") from None
        if x != x:  # NaN
            raise ValueError("bad threshold 'nan'")
        return mode, min(VOICEPRINT_THRESHOLD_MAX, max(0.0, x))
    return _toggle(mode, arg), threshold


def _toggle(current: bool, arg) -> bool:
    """spotctl toggle: explicit on/off arg wins, otherwise flip."""
    a = (arg or "").strip().lower()
    if a in ("1", "on", "true", "yes"):
        return True
    if a in ("0", "off", "false", "no"):
        return False
    return not current


def _start_control_server(ctl_q):
    """Start the localhost control server; None if the port is taken etc."""
    try:
        from src.spot.operator.control_server import ControlServer
        srv = ControlServer(port=8770, on_immediate=_on_immediate, queue=ctl_q)
        srv.start()
        return srv
    except OSError as e:
        print(f"[Control] server not started (port 8770: {e}) — continuing without spotctl")
    except Exception as e:
        print(f"[Control] server failed: {type(e).__name__}: {e} — continuing without spotctl")
    return None
