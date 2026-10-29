"""Mouth feature handlers: enable/disable the gripper mouth and the automatic
fold + mouth once standing (operator decision 3). Moved verbatim from
src/voice_control/spot_dispatch.py.

The robot owners (src.spot.robot) are imported at call time only: this module
is loaded while src.spot.features builds REGISTRY, so a module-level robot
import could cycle. The core (dispatcher / Connection) still calls
_cancel_auto_arm, _auto_arm_after_stand, start_auto_arm and
_handle_disable_mouth through spot_dispatch (src.spot.robot.legacy_features),
which re-exports them from here.
"""
import threading
import time

# Feature state (mouth auto-arm). spot_dispatch forwards reads of these names
# here (spot_dispatch._FEATURE_FORWARDS), so there is one live copy.
_auto_arm = False           # operator decision 3: fold the arm + mouth on once standing
_auto_arm_pending = False   # an auto fold is still owed (startup / after a stand)
_auto_arm_thread = None


def _handle_enable_mouth(params, gen=None):
    """Deploy the arm and enable gripper mouth animation (Stage 2E.2).

    Lazily creates the MouthDriver on first enable using the live session's
    arm command client, so it works regardless of session bring-up timing."""
    import src.spot.robot as _r
    from src.spot.robot import state as _state
    global _auto_arm, _auto_arm_pending
    try:
        from src.voice_control.spot_tts import get_tts
        from src.voice_control.animation.mouth import MouthDriver
        tts = get_tts()
        if tts is None:
            print("[Spot] enable_mouth: TTS not initialized")
            return False
        session = _r.CONNECTION.ensure()
        robot = session["robot"]
        if not robot.has_arm():
            print("[Spot] enable_mouth: no arm installed")
            return False
        # Safety gate: never deploy the arm unless motors are powered. An
        # e-stop cut also drops motor power, so this blocks deploy while
        # e-stopped too.
        if not robot.is_powered_on():
            print("[Spot] enable_mouth: motors not powered (or e-stopped) — refusing arm deploy")
            return False
        if not _state.standing_and_powered(session):
            print("[Spot] enable_mouth: not standing — refusing arm deploy")
            return False
        if _r.NAV.is_navigating():
            print("[Spot] enable_mouth: moving — not deploying the arm while walking")
            return False
        if not _r.ARM.send(session["cmd"], "talk", timeout_s=5.0, gen=gen):
            print("[Spot] enable_mouth: arm did not reach the talk pose — not enabling")
            return False
        _r.ARM._pose_before_walk = None
        _r.ARM._mouth_resume = False
        _r.ARM.set_talk_resume(False)
        _auto_arm = True  # keep folding back in after a sit/stand this session
        _auto_arm_pending = False
        if tts.mouth is None:
            tts.mouth = MouthDriver(cmd_client=session["cmd"])
        with _r.ARM._lock:  # a stop since the send wins: the claw stays shut
            if _r.SAFETY_GEN.aborted(gen) or _r.ARM._pose != "talk":
                print("[Spot] enable_mouth: stopped while raising — mouth stays off")
                return False
            tts.mouth.enable()
        print("[Spot] ✓ Mouth enabled (arm folded)")
        return True
    except Exception as e:
        print(f"[Spot] ✗ enable_mouth failed: {e}")
        return False


def _handle_disable_mouth(params, session=None):
    """Disable mouth animation, close the gripper, and stow the arm (Stage 2E.2).
    Also opts out of the automatic fold for the rest of the session."""
    import src.spot.robot as _r
    global _auto_arm, _auto_arm_pending
    _r.ARM._mouth_resume = False
    _auto_arm = False
    _auto_arm_pending = False
    _r.ARM.set_talk_resume(False)
    try:
        from src.voice_control.spot_tts import get_tts
        tts = get_tts()
        if tts is not None and tts.mouth is not None:
            tts.mouth.disable()
        if session is None:
            session = _r.CONNECTION.ensure()
        if _r.NAV.is_navigating():
            # Arm is tucked for walking; stow when the walk ends instead.
            _r.ARM._pose_before_walk = "stowed"
        else:
            _r.ARM.send(session["cmd"], "stowed", timeout_s=5.0)
        print("[Spot] ✓ Mouth disabled (arm stowed)")
        return True
    except Exception as e:
        print(f"[Spot] ✗ disable_mouth failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Automatic fold + mouth once standing (operator decision 3)
# ---------------------------------------------------------------------------
AUTO_ARM_STAND_WAIT_S = 8.0  # how long a stand may take before we give up


def set_auto_arm(on: bool):
    """client_mic startup: fold the arm + enable the mouth once the robot is
    standing and powered (False = opt-out, e.g. --no-arm / tuning auto_arm)."""
    import src.spot.robot as _r
    global _auto_arm, _auto_arm_pending
    _auto_arm = bool(on)
    _auto_arm_pending = bool(on)
    if on and _r.CONNECTION._session is not None:  # e.g. --map brought it up at launch
        start_auto_arm(_r.SAFETY_GEN.current(), wait_s=AUTO_ARM_STAND_WAIT_S)


def _cancel_auto_arm():
    global _auto_arm_pending
    _auto_arm_pending = False


def auto_arm_if_ready(gen, wait_s: float = 0.0) -> bool:
    """Fold the arm into the talk pose and enable the mouth if an automatic
    fold is owed. Needs a live session, a standing + powered robot (polled
    for up to wait_s), no motion, the arm stowed and no stop since `gen`.
    Serialised with motion dispatches via _dispatch_lock. Never raises."""
    import src.spot.robot as _r
    from src.spot.robot import state as _state
    from src.spot.robot.dispatcher import _dispatch_lock
    global _auto_arm_pending
    if not _auto_arm_pending or _r.CONNECTION._closing or gen is None or _r.SAFETY_GEN.aborted(gen):
        return False
    sess = _r.CONNECTION._session
    if sess is None:
        return False
    try:
        deadline = time.monotonic() + wait_s
        while not _state.standing_and_powered(sess):
            if time.monotonic() >= deadline or _r.SAFETY_GEN.aborted(gen) or _r.CONNECTION._closing:
                return False
            time.sleep(0.25)
        with _dispatch_lock:
            if not _auto_arm_pending or _r.CONNECTION._closing or _r.SAFETY_GEN.aborted(gen):
                return False
            if _r.ARM._pose != "stowed":
                _auto_arm_pending = False  # already folded (e.g. enable_mouth)
                return False
            if _r.NAV.is_navigating() or _r.ARM._talk_resume:
                return False  # moving, or the halt resume owns the arm: retry later
            print("[Spot] Standing: folding the arm and turning the mouth on")
            ok = _handle_enable_mouth({}, gen=gen)
            if ok:
                _auto_arm_pending = False
            return ok
    except Exception as e:
        print(f"[Spot] ⚠️  automatic arm fold failed: {e}")
        return False


def start_auto_arm(gen, wait_s: float = 0.0):
    """auto_arm_if_ready on a daemon thread (one at a time). Returns the
    thread, or None when nothing is owed."""
    import src.spot.robot as _r
    global _auto_arm_thread
    if not _auto_arm_pending or _r.CONNECTION._session is None:
        return None
    t = _auto_arm_thread
    if t is not None and t.is_alive():
        return None
    t = threading.Thread(target=auto_arm_if_ready, args=(gen, wait_s),
                         name="auto-arm", daemon=True)
    _auto_arm_thread = t
    t.start()
    return t


def _auto_arm_after_stand():
    """A stand succeeded: owe an automatic fold (if enabled) once it is up."""
    import src.spot.robot as _r
    global _auto_arm_pending
    if not _auto_arm:
        return
    _auto_arm_pending = True
    start_auto_arm(_r.SAFETY_GEN.current(), wait_s=AUTO_ARM_STAND_WAIT_S)


# ---------------------------------------------------------------------------
# Registered actions (the legacy _dispatch_core branches)
# ---------------------------------------------------------------------------
def enable_mouth_action(params, ctx):
    """Legacy branch order: bring up the session first (an ensure failure
    touches no arm/auto-arm state), then the handler with the dispatch gen."""
    import src.spot.robot as _r
    _r.CONNECTION.ensure()
    return _handle_enable_mouth(params, gen=_r.SAFETY_GEN.dispatch_current())


def disable_mouth_action(params, ctx):
    """Legacy branch order: ensure the session, then the handler (which, as
    before, is called without the session and looks it up again)."""
    import src.spot.robot as _r
    _r.CONNECTION.ensure()
    return _handle_disable_mouth(params)


def start(ctx):
    """Feature start hook: the automatic fold after a stand. A no-op unless
    auto-arm was opted in (set_auto_arm); nothing calls it until Task 15."""
    _auto_arm_after_stand()
