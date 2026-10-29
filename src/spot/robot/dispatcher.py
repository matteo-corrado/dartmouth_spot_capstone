"""Spot command dispatcher: the arm policy around every intent, the safety
path, and the core branches (stop, freeze, estop, stand, sit, power_off,
stow, selfright, status, battery_status).

Feature intents go to registry_dispatch(); an unregistered (or disabled) one
gets a session, prints "Unknown intent" and returns False, as the old
spot_dispatch fall-through did. The intent sets other than the safety family
are generated from core_intents + the enabled features at call time. The
safety path never looks at the registry or the event bus.
"""
import threading
import time

from bosdyn.client.robot_command import RobotCommandBuilder

import src.spot.core.events as _ev
import src.spot.robot as _r
from src.spot.robot import motion as _motion
from src.spot.robot import state as _state
from src.spot.robot.state import _ESTOP_STATE_NAMES
from src.spot.core.core_intents import CORE_INTENTS, intent_names
from src.spot.robot.arm import (  # noqa: F401  (re-exported constant sets)
    ABORTING_INTENTS,
    RPC_TIMEOUT_S,
    SAFETY_INTENTS,
    SAFETY_LOCK_TIMEOUT_S,
    SHUTDOWN_INTENTS,
    arm_policy_for,
)

# Generated sets (intent_names, read at call time):
# - "unlocked": never takes _dispatch_lock (safety must never wait on it; the
#   rest touch no robot motion or can take long, e.g. add_persona).
# - "info": speaks "I can't reach my sensors" when the robot is unreachable.
# - "locomotion": tucks the arm first (arm_policy_for).
# CORE_INTENTS (core_intents.py) are the branches handled here.

# Spoken lines never contain stop/halt/freeze/e-stop: the always-on safety
# KWS hears Spot's own voice and would halt it.
_ESTOP_SPOKEN = {"not_cut": "released", "cut": "engaged", "soft_stop": "in a soft hold"}

_dispatch_lock = threading.RLock()  # serialises non-safety robot dispatches


def _say(speak, text: str):
    """Speak a short line via the caller's speak callable; print if None."""
    print(f"[Spot] {text}")
    if speak is not None:
        try:
            speak(text)
        except Exception as e:
            print(f"[Spot] speak failed: {e}")


def dispatch_intent(intent, brain=None, speak=None):
    """Execute a Spot command based on parsed intent. Thread-safe.

    Args:
        intent: Dict with keys "intent" (str) and "params" (dict)
        brain: Optional LLMBrain instance — required only for set_persona.
        speak: Optional callable(str) used to say result/refusal lines
            (e.g. tts.speak). None -> lines are only printed.

    Returns:
        bool: True if command executed successfully, False otherwise

    Arm policy (arm_policy_for):
    - stop/freeze/estop never take a dispatch lock: motion stop first, then
      the original branch, then a no-wait stow. With no session they do
      nothing (never power on a parked robot).
    - stop/freeze/estop/sit/power_off/stow abort any locomotion still being
      set up on another thread (the safety generation).
    - Locomotion tucks a deployed arm (refusing to move if the tuck fails) and
      restores it only after a natural finish.
    """
    if not intent:
        return False
    if brain is not None:
        _r.ARM._last_brain = brain
        try:
            brain.session_state.arm_pose = _r.ARM._pose
        except Exception:
            pass
    name = intent.get("intent")

    if name in ABORTING_INTENTS:
        _r.SAFETY_GEN.bump()
        _r.ARM.hold_gaze()  # no gaze captures the post-stop gen before the stow lands
    if name in SAFETY_INTENTS:
        return dispatch_safety(intent, brain, speak)
    if name in intent_names("unlocked"):
        return _dispatch_core(intent, brain, speak)
    held = _r.CONNECTION._session
    if name in ABORTING_INTENTS and held is None:
        print(f"[Spot] {name}: no robot session — not connecting just to {name}")
        return False
    with _dispatch_lock:
        outer = _r.SAFETY_GEN.dispatch_bump()
        try:
            if name in intent_names("locomotion"):
                return _dispatch_locomotion(intent, brain, speak)
            if name in ABORTING_INTENTS:  # sit/power_off/stow never bring up a session
                return _dispatch_core(intent, brain, speak, session=held)
            return _dispatch_core(intent, brain, speak)
        finally:
            _r.SAFETY_GEN.dispatch_restore(outer)


def dispatch_safety(intent, brain=None, speak=None):
    name = intent.get("intent")
    sess = _r.CONNECTION._session
    if sess is None:
        print(f"[Spot] {name}: no robot session — nothing to stop (not connecting)")
        _r.ARM.close_mouth()
        _r.NAV.cancel()
        return False
    cmd_client = sess["cmd"]
    was_emote = _r.NAV._mode == "emote"
    _motion.send_stop(cmd_client)  # FIRST, before any thread join
    was_talk = _r.ARM.in_talk_mode()  # read-only; before the stop branch latches the mouth
    ok = _dispatch_core(intent, brain, speak, session=sess)  # original branch (stops again)
    _r.ARM.stow_now(cmd_client, resend_stop=was_emote,
                 then_stand=(was_emote and name == "freeze" and _state.is_standing(sess)))
    if name == "estop":
        _r.ARM.set_talk_resume(False)  # an e-stop cancels any pending talk-pose restore
        _r.mouth_handlers._cancel_auto_arm()  # ... and any pending automatic fold (until a stand)
    elif was_talk:
        _r.ARM.set_talk_resume(True)  # the next motion-allowed turn brings the arm back
    return ok


_dispatch_safety = dispatch_safety  # old spot_dispatch name


def current_safety_gen() -> int:
    """Snapshot for resume_talk_after_halt(gen): any stop after it wins."""
    return _r.SAFETY_GEN.current()


def _aborted(gen) -> bool:
    """True if an aborting intent arrived since generation `gen` was taken."""
    return _r.SAFETY_GEN.aborted(gen)


def resume_talk_after_halt(gen):
    """Bring the talk pose back after a safety halt (see Arm.resume_talk_after_halt)."""
    return _r.ARM.resume_talk_after_halt(gen)


def _dispatch_locomotion(intent, brain=None, speak=None):
    """Tuck, move, and (via the motion thread) restore after a natural finish."""
    name = intent.get("intent")
    try:
        session = _r.CONNECTION.ensure()
    except Exception as e:
        print(f"[Spot] Error executing intent '{name}': {e}")
        return False
    cmd_client = session["cmd"]
    gen = _r.SAFETY_GEN.dispatch_current()
    _r.ARM.hold_gaze()  # no gaze may land while this walk starts (incl. the join below)
    _r.NAV.cancel()  # a new motion supersedes the old one (its restore is skipped)
    current = _r.ARM._pose
    before, after = arm_policy_for(name, current)
    if current != "stowed":
        if _r.ARM._pose_before_walk is not None and current == (_r.ARM._pose_during_walk or "walk"):
            after = _r.ARM._pose_before_walk  # superseded walk: keep its pending restore
        _r.ARM._pose_before_walk = after if after != "walk" else "stowed"
        _r.ARM._pose_during_walk = before
        if current == before == "talk" and _r.ARM._gaze_deg not in (None, 0.0):
            # Folded pose kept for the walk; just re-centre a gaze yaw (no wait).
            try:
                _r.ARM.send(cmd_client, "talk", timeout_s=0, require_pose="talk", gen=gen)
            except Exception as e:
                print(f"[Spot] gaze re-centre before walking failed: {e}")
        if current != before:
            _r.ARM.mouth_suspend()  # claw closed and still for the whole walk
            try:
                tucked = _r.ARM.send(cmd_client, before, timeout_s=5.0)
            except Exception as e:
                print(f"[Spot] Arm tuck failed: {e}")
                tucked = False
            if _r.SAFETY_GEN.aborted(gen):
                return _motion.abort_motion(cmd_client, name)
            if not tucked:
                _say(speak, "I couldn't tuck my arm in, so I'll stay put.")
                _r.ARM.restore_after_motion()
                return False
    if _r.SAFETY_GEN.aborted(gen):
        return _motion.abort_motion(cmd_client, name)
    _r.ARM._last_traj = None
    _r.ARM.hold_gaze()
    ok = _dispatch_core(intent, brain, speak)
    if _r.SAFETY_GEN.aborted(gen):
        return _motion.abort_motion(cmd_client, name)
    if _r.NAV.is_navigating():
        return ok  # the nav thread restores the arm after a natural finish
    if ok and _r.ARM._last_traj is not None:
        if not _motion.start_restore_watch(cmd_client, *_r.ARM._last_traj, gen=gen):
            return _motion.abort_motion(cmd_client, name)
    else:
        _r.ARM.restore_after_motion()  # nothing moved
    return ok


# ---------------------------------------------------------------- features
def registry_dispatch(name, params, ctx):
    """Run a registered feature action. None when `name` is not registered,
    otherwise the handler's result (a False result is final: never
    re-dispatched to the legacy branches)."""
    from src.spot.features import REGISTRY
    spec = REGISTRY.action(name)
    if spec is None:
        return None
    return spec.handler(params, ctx)


def _dispatch_feature(intent, brain=None, speak=None, session=None):
    from src.spot.features import REGISTRY
    name = intent["intent"]
    if REGISTRY.action(name) is not None:
        # Same catch-all as every legacy branch: a raising handler (or Ctx /
        # tuning build) returns False and never escapes dispatch_intent, so the
        # locomotion caller still runs its arm restore.
        try:
            from src.spot.core.tuning import get_tuning
            from src.spot.features.ctx import Ctx
            ctx = Ctx(connection=_r.CONNECTION, arm=_r.ARM, safety=_r.SAFETY_GEN, nav=_r.NAV,
                      speak=speak, brain=brain, bus=None, tuning=get_tuning())
            return registry_dispatch(name, intent.get("params", {}), ctx)  # final: never legacy
        except Exception as e:
            print(f"[Spot] Error executing intent '{name}': {e}")
            if name in intent_names("info"):
                _say(speak, "I can't reach my sensors right now.")
            import traceback
            traceback.print_exc()
            return False
    return _unknown_intent(name, session)


def _unknown_intent(name, session=None):
    """An unregistered or disabled intent: the old spot_dispatch fall-through
    (R22). The session comes up first, inside the catch-all, then the intent
    is refused with the same False as any unknown intent."""
    try:
        if session is None:
            session = _r.CONNECTION.ensure()
        print(f"[Spot] Unknown intent: {name}")
        return False
    except Exception as e:
        print(f"[Spot] Error executing intent '{name}': {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------- core branches
def _dispatch_core(intent, brain=None, speak=None, session=None):
    """The per-intent branches. Called only through dispatch_intent (which
    applies the arm policy) or for nested re-dispatch."""
    if not intent:
        return False

    name = intent["intent"]
    params = intent.get("params", {})

    if name not in CORE_INTENTS:
        return _dispatch_feature(intent, brain, speak, session=session)

    try:
        if session is None:  # safety/aborting callers pass the session they hold
            session = _r.CONNECTION.ensure()
        cmd_client = session["cmd"]

        if name == "stow":  # spotctl stow: mouth off, arm stowed
            # NOTE: stow also interrupts a running walk/patrol/tour/follow/emote
            # (it is an ABORTING_INTENT): the robot is stopped first, then the
            # nav cancelled, then the arm stowed. Fails safe.
            if _r.NAV.is_navigating():
                was_emote = _r.NAV._mode == "emote"
                _motion.send_stop(cmd_client)
                _r.NAV.cancel()
                if was_emote:
                    # Re-stop under the arm lock so a choreography request already
                    # in flight can't start after this stop.
                    _r.ARM.stow_now(cmd_client, resend_stop=True)
                else:
                    # A navigate_to / visual-nav move sent while the join ran
                    # lands after the first stop: stop again (as stop/freeze do).
                    _motion.send_stop(cmd_client)
            return _r.mouth_handlers._handle_disable_mouth(params, session)

        if name == "stop":
            # Stop current movement/action
            print("[Spot] Stopping current action...")
            _r.ARM.close_mouth()
            _r.NAV.cancel()
            try:
                # RPC timeout: a stalled link must not wedge the safety thread (M2)
                cmd_client.robot_command(RobotCommandBuilder.stop_command(), timeout=RPC_TIMEOUT_S)
                print("[Spot] ✓ Stop command sent")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Stop failed: {e}")
                return False

        elif name == "freeze":
            # Freeze robot - stop all movement and hold current position
            print("[Spot] Freezing robot in place...")
            _r.ARM.close_mouth()
            _r.NAV.cancel()
            try:
                # First stop any current movement
                cmd_client.robot_command(RobotCommandBuilder.stop_command(), timeout=RPC_TIMEOUT_S)
                # Then hold position with a stand — ONLY if already standing, so
                # freeze never stands up a sitting robot.
                if _state.is_standing(session):
                    cmd = RobotCommandBuilder.synchro_stand_command()
                    cmd_client.robot_command(cmd, timeout=RPC_TIMEOUT_S)
                print("[Spot] ✓ Robot frozen - holding position")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Freeze failed: {e}")
                return False

        elif name == "stand":
            # Stand the robot up (clear behavior faults if needed)
            print("[Spot] Standing up...")
            try:
                cmd = RobotCommandBuilder.synchro_stand_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Stand command sent")
                _ev.publish("robot_stood")
                _r.mouth_handlers._auto_arm_after_stand()
                return True
            except Exception as e:
                if "BehaviorFault" in str(e):
                    print("[Spot] Behavior faults detected — clearing and retrying...")
                    try:
                        state_client = session["state"]
                        robot_state = state_client.get_robot_state()
                        for f in robot_state.behavior_fault_state.faults:
                            try:
                                cmd_client.clear_behavior_fault(behavior_fault_id=f.behavior_fault_id)
                            except Exception:
                                pass
                        time.sleep(1)
                        # No self-right here: that is its own explicit intent.
                        cmd_client.robot_command(RobotCommandBuilder.synchro_stand_command())
                        print("[Spot] ✓ Faults cleared, standing")
                        _ev.publish("robot_stood")
                        _r.mouth_handlers._auto_arm_after_stand()
                        return True
                    except Exception as e2:
                        print(f"[Spot] ✗ Recovery failed: {e2}")
                        return False
                print(f"[Spot] ✗ Stand failed: {e}")
                return False

        elif name == "sit":
            # Sit the robot down
            print("[Spot] Sitting down...")
            _motion.send_stop(cmd_client)  # stop first, then stow, then sit
            _r.ARM.set_talk_resume(False)
            _r.mouth_handlers._cancel_auto_arm()  # only an explicit stand folds the arm again
            _r.NAV.cancel()
            _r.ARM._pose_before_walk = None
            if _r.ARM._pose != "stowed":
                try:
                    _r.ARM.send(cmd_client, "stowed", timeout_s=3.0)  # stow BEFORE sitting
                except Exception as e:
                    print(f"[Spot] ⚠️  Arm stow before sit failed: {e}")
            try:
                cmd = RobotCommandBuilder.synchro_sit_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Sit command sent")
                _ev.publish("robot_sat")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Sit failed: {e}")
                return False

        elif name == "selfright":
            # Self-right: recover from fall (clear behavior faults first)
            print("[Spot] Attempting self-right (recovery from fall)...")
            try:
                # Clear behavior faults first — required before any command after a fall
                state_client = session["state"]
                robot_state = state_client.get_robot_state()
                faults = robot_state.behavior_fault_state.faults
                if faults:
                    print(f"[Spot] Clearing {len(faults)} behavior fault(s)...")
                    for f in faults:
                        try:
                            cmd_client.clear_behavior_fault(behavior_fault_id=f.behavior_fault_id)
                        except Exception:
                            pass
                    time.sleep(1)

                cmd = RobotCommandBuilder.selfright_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Self-right command sent")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Self-right failed: {e}")
                return False

        elif name == "power_off":
            # Safe power off
            print("[Spot] Initiating safe power off...")
            _motion.send_stop(cmd_client)  # stop first, then stow, then power off
            _r.ARM.set_talk_resume(False)
            _r.mouth_handlers._cancel_auto_arm()  # only an explicit stand folds the arm again
            _r.ARM.close_mouth()
            _r.NAV.cancel()
            try:
                _r.ARM.send(cmd_client, "stowed", timeout_s=3.0)  # stow BEFORE sitting
            except Exception as e:
                print(f"[Spot] ⚠️  Arm stow before power off failed: {e}")
            try:
                cmd = RobotCommandBuilder.safe_power_off_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Safe power off command sent")
                print("[Spot]   Robot will sit, then power off motors")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Power off failed: {e}")
                return False

        elif name == "battery_status":
            # Report battery status
            print("[Spot] Checking battery status...")
            try:
                state_client = session["state"]
                robot_state = state_client.get_robot_state()
                battery = robot_state.power_state.locomotion_charge_percentage.value
                runtime_est = robot_state.power_state.locomotion_estimated_runtime.seconds

                # Format runtime
                mins = runtime_est // 60
                hours = mins // 60
                mins = mins % 60

                print(f"[Spot] ✓ Battery: {battery:.0f}%")
                if hours > 0:
                    print(f"[Spot]   Estimated runtime: {hours}h {mins}m")
                else:
                    print(f"[Spot]   Estimated runtime: {mins}m")
                parts = []
                if hours > 0:
                    parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if mins > 0:
                    parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
                line = f"My battery is at {battery:.0f} percent"
                if parts:
                    line += f", with about {' and '.join(parts)} of walking left"
                _say(speak, line + ".")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Battery status failed: {e}")
                _say(speak, "I couldn't read my battery just now.")
                return False

        elif name == "status":
            # Report overall robot status
            print("[Spot] Checking robot status...")
            try:
                state_client = session["state"]
                robot_state = state_client.get_robot_state()

                # Battery
                battery = robot_state.power_state.locomotion_charge_percentage.value

                # E-stop status (uses module-level _ESTOP_STATE_NAMES)
                estop = _ESTOP_STATE_NAMES.get(
                    robot_state.estop_states[0].state if robot_state.estop_states else 0,
                    "unknown",
                )

                print(f"[Spot] ✓ Status Report:")
                print(f"[Spot]   Battery: {battery:.0f}%")
                print(f"[Spot]   E-Stop: {estop}")
                motors = "on" if (robot_state.power_state.motor_power_state
                                  == robot_state.power_state.STATE_ON) else "off"
                estop_words = _ESTOP_SPOKEN.get(estop, "in an unknown state")
                ok_word = "All good. " if (motors == "on" and estop == "not_cut") else ""
                _say(speak, f"{ok_word}My battery is at {battery:.0f} percent, "
                            f"my motors are {motors}, and my safety switch is {estop_words}.")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Status check failed: {e}")
                _say(speak, "I couldn't check my status just now.")
                return False

        elif name == "estop":
            # Emergency stop (software e-stop)
            print("[Spot] ⚠️  EMERGENCY STOP triggered!")
            _r.ARM.close_mouth()
            _r.NAV.cancel()
            try:
                cmd_client.robot_command(RobotCommandBuilder.stop_command(), timeout=RPC_TIMEOUT_S)
                print("[Spot] ✓ Emergency stop executed")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Emergency stop failed: {e}")
                return False

    except Exception as e:
        print(f"[Spot] Error executing intent '{name}': {e}")
        # the stop path never reads the registry (no safety intent is info)
        if name not in SAFETY_INTENTS and name in intent_names("info"):
            _say(speak, "I can't reach my sensors right now.")
        import traceback
        traceback.print_exc()
        return False
