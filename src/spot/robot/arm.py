"""Arm poses, the arm policy, the mouth latch and the talk-pose resume after a
safety halt (spec 7.1b: these stay in the robot core).

Moved out of spot_dispatch globals ``_arm_pose``, ``_pose_before_walk``,
``_pose_during_walk``, ``_arm_lock``, ``_arm_settle_until``, ``_last_traj``,
``_mouth_resume``, ``_talk_resume`` (plus ``_gaze_deg`` and ``_last_brain``,
which set_pose writes / reads).
"""
import threading
import time

from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives

import src.spot.robot.session as _poses
from src.spot.robot.session import send_arm_pose
import src.spot.robot as _r
from src.spot.robot import state as _state

# ---------------------------------------------------------------------------
# Arm policy (Task 14)
# ---------------------------------------------------------------------------
# The arm tucks into the walking pose before any locomotion and returns to its
# previous pose only after the motion finished naturally. stop/freeze/estop
# send a motion stop FIRST, then run their original branch, then stow without
# waiting. The only arm motions while walking are that tuck and that restore,
# and the gripper mouth is latched off for the whole walk / routine.
LOCOMOTION_INTENTS = frozenset({
    "walk_to", "turn", "walk", "strafe", "go_to", "tour", "patrol",
    "come_back", "go_to_object", "follow_me",
})
SAFETY_INTENTS = frozenset({"stop", "freeze", "estop"})
SHUTDOWN_INTENTS = frozenset({"power_off", "shutdown", "sit"})  # stop, stow, then act
# Intents that abort any locomotion still being set up on another thread.
ABORTING_INTENTS = SAFETY_INTENTS | SHUTDOWN_INTENTS | {"stow"}

SAFETY_LOCK_TIMEOUT_S = 1.0  # stop/close paths never wait longer on the arm lock
RPC_TIMEOUT_S = 2.0          # per-RPC timeout for commands sent under the arm lock


def arm_policy_for(intent_name: str, current_pose: str) -> tuple[str, str]:
    """(pose before, pose after) for an intent. Pure."""
    if intent_name in SAFETY_INTENTS or intent_name in SHUTDOWN_INTENTS:
        return ("stowed", "stowed")
    if intent_name in LOCOMOTION_INTENTS:
        if current_pose == "talk" and _talk_is_walk_pose():
            return ("talk", "talk")  # same folded pose: no re-pose, mouth stays on
        return ("walk", current_pose)
    if intent_name == "emote":
        return ("stowed", current_pose)
    return (current_pose, current_pose)


def _talk_is_walk_pose() -> bool:
    """True when the talk pose IS the walking pose (folded joints)."""
    return _poses.WALK_POSE == "joints"


class Arm:
    def __init__(self):
        self._pose = "stowed"            # live arm pose: "stowed" | "talk" | "walk"
        self._pose_before_walk = None    # pose to restore once the current locomotion ends
        self._pose_during_walk = None    # live arm pose while that locomotion runs ("walk" | "talk")
        self._lock = threading.Lock()    # serialises "check cancelled + send arm cmd"
        self._settle_until = 0.0         # monotonic: a pose move is still in flight (no gaze before)
        self._last_traj = None           # (cmd_id, deadline) of the last trajectory sent
        self._mouth_resume = False       # re-enable the mouth when the arm returns to "talk"
        self._talk_resume = False        # a safety halt stowed the talk pose: restore on the next turn
        self._gaze_deg = None            # sh0 (deg) last commanded in the talk pose; None = unknown
        self._last_brain = None          # brain whose session_state mirrors the arm pose

    # ------------------------------------------------------------ pose
    def pose(self) -> str:
        return self._pose

    def policy_for(self, intent_name: str) -> tuple[str, str]:
        return arm_policy_for(intent_name, self._pose)

    def hold_gaze(self, secs: float = None):
        """Block gaze for `secs` (default JOINT_MOVE_SECS) from now."""
        secs = _poses.JOINT_MOVE_SECS if secs is None else secs
        self._settle_until = max(self._settle_until, time.monotonic() + secs)

    def set_pose(self, pose: str):
        self._pose = pose
        self._gaze_deg = 0.0 if pose == "talk" else None  # every talk send centres sh0
        # A gaze must not supersede a pose move still in flight (its sender
        # waits on that cmd id before turning the mouth on).
        self._settle_until = time.monotonic() + _poses.JOINT_MOVE_SECS + 0.5
        st = getattr(self._last_brain, "session_state", None)
        if st is not None:
            try:
                st.arm_pose = pose
            except Exception:
                pass

    # ------------------------------------------------------------ mouth
    def mouth_suspend(self):
        """Latch the gripper mouth off (claw closed) before a tuck or emote stow,
        remembering whether to turn it back on when the arm returns to "talk"."""
        try:
            from src.spot.tts.speaker import get_tts
            t = get_tts()
            m = t.mouth if t is not None else None
            if m is not None and m.enabled:
                self._mouth_resume = True
                m.disable()
        except Exception as e:
            print(f"[Spot] ⚠️  mouth suspend failed: {e}")

    def mouth_restore(self):
        if not self._mouth_resume or self._pose != "talk":
            return
        self._mouth_resume = False
        try:
            from src.spot.tts.speaker import get_tts
            t = get_tts()
            if t is not None and t.mouth is not None:
                t.mouth.enable()
        except Exception as e:
            print(f"[Spot] ⚠️  mouth restore failed: {e}")

    def close_mouth(self):
        """Stage 2E.2 safety path: latch the gripper mouth OFF and close it.
        Uses disable() (not close()) so a TTS chunk still queued/playing can't
        re-open the gripper via its play callback after a stop/estop. Non-blocking
        (the gripper command runs on the MouthDriver worker). Safe no-op if
        TTS/mouth isn't initialized. Called from stop/freeze/estop."""
        try:
            from src.spot.tts.speaker import get_tts
            _t = get_tts()
            if _t is not None and _t.mouth is not None:
                _t.mouth.disable()
        except Exception as e:
            # Safety path — surface a failed gripper-close instead of hiding it.
            print(f"[Spot] ⚠️  _close_mouth failed (gripper may be open): {e}")

    # ------------------------------------------------------------ sends
    def send(self, cmd_client, pose: str, stop_event=None, timeout_s: float = 5.0,
             require_pose=None, gen=None) -> bool:
        """Send an arm pose. Under the arm lock, skip if stop_event is set (or the
        live pose is not require_pose, or a stop arrived since `gen`), so a
        cancelled thread can never move the arm after a stop's stow. Waits for
        arrival outside the lock."""
        with self._lock:
            if stop_event is not None and stop_event.is_set():
                return False
            if _r.SAFETY_GEN.aborted(gen):
                return False
            if require_pose is not None and self._pose != require_pose:
                return False
            cmd_id = send_arm_pose(cmd_client, pose, rpc_timeout=RPC_TIMEOUT_S)
            self.set_pose(pose)
        if timeout_s is None or timeout_s <= 0:
            return True
        if stop_event is None:
            return bool(block_until_arm_arrives(cmd_client, cmd_id, timeout_s))
        # Wait in short slices so a stop's nav-cancel join is never held up.
        deadline = time.time() + timeout_s
        while not stop_event.is_set() and time.time() < deadline:
            if block_until_arm_arrives(cmd_client, cmd_id, 0.3):
                return True
        return False

    def stow_now(self, cmd_client, resend_stop=False, then_stand=False, force=False):
        """Safety-path stow. Never waits for the arm; waits at most
        SAFETY_LOCK_TIMEOUT_S for the arm lock and sends without it on timeout.
        resend_stop re-sends the motion stop under the lock (overrides a
        choreography request that raced the first stop)."""
        got = self._lock.acquire(timeout=SAFETY_LOCK_TIMEOUT_S)
        if not got:
            print("[Spot] ⚠️  arm lock busy — stowing without it")
        try:
            if resend_stop:
                cmd_client.robot_command(RobotCommandBuilder.stop_command(), timeout=RPC_TIMEOUT_S)
                if then_stand:
                    cmd_client.robot_command(RobotCommandBuilder.synchro_stand_command(),
                                             timeout=RPC_TIMEOUT_S)
            self._pose_before_walk = None
            self._pose_during_walk = None
            self._mouth_resume = False
            if self._pose == "stowed" and not force:
                return  # already stowed: never send arm commands to e.g. a sitting robot
            send_arm_pose(cmd_client, "stowed", rpc_timeout=RPC_TIMEOUT_S)
            self.set_pose("stowed")
            print("[Spot] Arm stowing after stop")
        except Exception as e:
            print(f"[Spot] ⚠️  Arm stow after stop failed: {e}")
        finally:
            if got:
                self._lock.release()

    def restore_after_motion(self, stop_event=None):
        """Return the arm to its pre-walk pose (only after a natural finish, or
        when the motion never started)."""
        target = self._pose_before_walk
        if target is None or _r.CONNECTION._session is None:
            return
        if stop_event is not None and stop_event.is_set():
            return
        try:
            # Only from the pose the walk ran in: after a stop's stow it is
            # "stowed", so no restore can follow a stop.
            during = self._pose_during_walk or "walk"
            if self._pose != target and self.send(_r.CONNECTION._session["cmd"], target, stop_event,
                                                  timeout_s=5.0, require_pose=during):
                print(f"[Spot] Arm restored to '{target}'")
            if self._pose == target:
                self._pose_before_walk = None
                self._pose_during_walk = None
                self.mouth_restore()
        except Exception as e:
            print(f"[Spot] ⚠️  Arm restore failed: {e}")

    # ------------------------------------------------------------ talk resume
    def in_talk_mode(self) -> bool:
        """True if the arm is in (or will return to) the "talk" pose with the mouth on."""
        return self._pose == "talk" or self._pose_before_walk == "talk" or self._mouth_resume

    def set_talk_resume(self, on: bool):
        self._talk_resume = bool(on)

    def resume_talk_after_halt(self, gen):
        """Final-review I1: a stop/freeze stowed the arm and latched the mouth off.
        Called by client_mic at the start of the next turn that may move; sends the
        arm back to "talk" without waiting, and a daemon thread re-enables the
        mouth once it arrives. Needs a standing, powered, idle robot; otherwise the
        flag stays for a later turn. `gen` is current_safety_gen() taken at the
        start of the turn; any stop since then wins. Returns the thread or False."""
        if not self._talk_resume or _r.CONNECTION._closing or gen is None or _r.SAFETY_GEN.aborted(gen):
            return False
        sess = _r.CONNECTION._session
        if sess is None or _r.NAV.is_navigating() or self._pose != "stowed":
            return False
        try:
            if not _state.standing_and_powered(sess):
                self.set_talk_resume(False)  # sat / powered off / e-stopped: don't keep it for later
                return False
            from src.spot.tts.speaker import get_tts
            tts = get_tts()
            if tts is None or tts.mouth is None:
                self.set_talk_resume(False)
                return False
            cmd = sess["cmd"]
            with self._lock:
                if _r.SAFETY_GEN.aborted(gen) or self._pose != "stowed":
                    return False
                cmd_id = send_arm_pose(cmd, "talk", rpc_timeout=RPC_TIMEOUT_S)
                self.set_pose("talk")
                self.set_talk_resume(False)

            def _enable_when_up():
                # Like enable_mouth: the claw only animates once the arm is up.
                try:
                    if not block_until_arm_arrives(cmd, cmd_id, 5.0):
                        print("[Spot] talk pose not reached after the halt — mouth stays off")
                        return
                    with self._lock:  # a stop since the send wins
                        if _r.SAFETY_GEN.aborted(gen) or self._pose != "talk":
                            return
                        tts.mouth.enable()
                    print("[Spot] Arm back in talk pose after the halt")
                except Exception as e:
                    print(f"[Spot] ⚠️  mouth re-enable after the halt failed: {e}")

            t = threading.Thread(target=_enable_when_up, name="talk-resume", daemon=True)
            t.start()
            return t
        except Exception as e:
            print(f"[Spot] ⚠️  talk-pose resume failed: {e}")
            return False


# ---------------------------------------------------------------- old names
def _arm():
    return _r.ARM


def get_arm_pose() -> str:
    return _arm().pose()


def _set_arm_pose(pose: str):
    _arm().set_pose(pose)


def _hold_gaze(secs: float = None):
    _arm().hold_gaze(secs)


def _arm_send(cmd_client, pose: str, stop_event=None, timeout_s: float = 5.0,
              require_pose=None, gen=None) -> bool:
    return _arm().send(cmd_client, pose, stop_event, timeout_s, require_pose, gen)


def _stow_now(cmd_client, resend_stop=False, then_stand=False, force=False):
    _arm().stow_now(cmd_client, resend_stop=resend_stop, then_stand=then_stand, force=force)


def _restore_after_motion(stop_event=None):
    _arm().restore_after_motion(stop_event)


def _mouth_suspend():
    _arm().mouth_suspend()


def _mouth_restore():
    _arm().mouth_restore()


def _close_mouth():
    _arm().close_mouth()


def _in_talk_mode() -> bool:
    return _arm().in_talk_mode()


def _set_talk_resume(on: bool):
    _arm().set_talk_resume(on)
