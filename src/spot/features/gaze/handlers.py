"""Gaze toward the locked speaker (Task 19, operator decision 4). Moved from
the old spot_dispatch.py (Task 11), behaviour unchanged.

The gaze worker thread stays in client_mic (Task 10 owns it). The arm poses,
the gaze hold and the stop/stow plumbing stay in the robot core (spec 7.1b);
ARM._gaze_deg lives on the Arm owner. The robot owners are reached through
`import src.spot.robot as _r` at call time only (never at import).
"""
import time

_gaze_t = 0.0               # monotonic time of the last gaze command

GAZE_MIN_CHANGE_DEG = 15.0  # re-aim only when the target moved more than this
GAZE_MIN_INTERVAL_S = 2.0   # ... and at most this often
GAZE_BEHIND_DEG = 150.0     # a speaker this far behind is ambiguous: ignore
GAZE_HARD_MAX_DEG = 60.0    # code cap on tuning gaze_max_deg (arm reach, safety review)
GAZE_RPC_TIMEOUT_S = 0.5    # < SAFETY_LOCK_TIMEOUT_S: a stop's stow never loses the lock race


def gaze_sh0_deg(body_deg, max_deg: float):
    """Shoulder-yaw target (deg) for a body-frame bearing (0 = ahead, +90 =
    left), clamped to +/-max_deg. None for a non-finite bearing, one almost
    straight behind, or max_deg <= 0 (gaze off). Pure."""
    import math
    try:
        b = float(body_deg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(b) or max_deg <= 0:
        return None
    b = ((b + 180.0) % 360.0) - 180.0
    if abs(b) > GAZE_BEHIND_DEG:
        return None
    return max(-max_deg, min(max_deg, b))


def _gaze_shutting_down() -> bool:
    """True once main's shutdown has stopped the gaze worker
    (conversation.loop.stop_gaze_worker). Read lock-free from sys.modules (no
    import under ARM._lock): a request the worker already picked up must not
    move the arm after shutdown has begun. Never raises."""
    import sys
    loop = sys.modules.get("src.spot.conversation.loop")
    ev = getattr(loop, "_gaze_stop", None) if loop is not None else None
    try:
        return bool(ev is not None and ev.is_set())
    except Exception:  # noqa: BLE001
        return True  # fail closed: no gaze


def gaze_at_bearing(body_deg: float | None, gen=None) -> bool:
    """Turn the folded arm (shoulder yaw sh0, slow joint move) toward a
    body-frame bearing. None -> back to centre.

    Only in the "talk" pose while standing still (not walking, dancing or
    closing); clamped to +/-tuning.gaze_max_deg; rate-limited (target moved
    > 15 deg and >= 2 s since the last gaze). Sent under _arm_lock after
    re-checking the pose and the safety generation, so a stop always wins and
    no gaze follows a stop's stow. Non-blocking; never brings up a session;
    never raises.
    """
    global _gaze_t
    import src.spot.robot as _r
    sess = _r.CONNECTION._session
    if sess is None or _r.CONNECTION._closing or _r.ARM._pose != "talk" or _r.NAV.is_navigating():
        return False
    try:
        import math
        from src.spot.robot import state as _state
        from src.spot.robot.session import joints_pose_command
        from src.spot.core.tuning import get_tuning
        tun = get_tuning()
        max_deg = min(float(tun.gaze_max_deg), GAZE_HARD_MAX_DEG)
        if max_deg <= 0:
            return False
        if gen is None:
            gen = _r.SAFETY_GEN.current()
        target = 0.0 if body_deg is None else gaze_sh0_deg(body_deg, max_deg)
        if target is None:
            return False
        now = time.monotonic()
        if now < _r.ARM._settle_until:
            return False  # the arm is still folding / re-centring
        last = _r.ARM._gaze_deg
        if last is not None:
            if abs(target - last) <= GAZE_MIN_CHANGE_DEG and not (target == 0.0 and last != 0.0):
                return False
            if now - _gaze_t < GAZE_MIN_INTERVAL_S:
                return False
        if not _state.standing_and_powered(sess):
            return False
        sh0 = math.radians(target) * (1 if tun.gaze_sh0_sign >= 0 else -1)
        with _r.ARM._lock:
            if (_r.SAFETY_GEN.aborted(gen) or _r.CONNECTION._closing or _r.ARM._pose != "talk" or _r.NAV.is_navigating()
                    or time.monotonic() < _r.ARM._settle_until or _gaze_shutting_down()):
                return False
            sess["cmd"].robot_command(joints_pose_command(sh0), timeout=GAZE_RPC_TIMEOUT_S)
            _r.ARM._gaze_deg, _gaze_t = target, now
        bearing = "centre" if body_deg is None else f"{((float(body_deg) + 180) % 360) - 180:+.0f} deg (body, +left)"
        print(f"[Spot] gaze: speaker {bearing} -> shoulder yaw {target:+.0f} deg (limit {max_deg:.0f})")
        return True
    except Exception as e:
        print(f"[Spot] gaze_at_bearing failed: {e}")
        return False


def on_speaker_locked(body_deg, gen):
    """`speaker_locked` subscriber (declared, not subscribed: Task 15 kept the
    conversation loop's gaze worker calling gaze_at_bearing directly).

    No defaults on purpose: a payload without the request-time gen raises
    (the bus logs it) instead of moving the arm with a late gen, and a None
    gen is skipped exactly as client_mic._gaze_at_speaker skips it. Runs the
    RPC on the publisher's thread, so subscribing it needs a hand-off to the
    gaze worker (off the audio loop) first."""
    if gen is None:
        return False
    return gaze_at_bearing(body_deg, gen)
