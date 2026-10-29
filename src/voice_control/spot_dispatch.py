"""Spot command dispatcher for voice intents (legacy feature half).

The robot core (connection, arm, motion, state, nav runner, dispatcher) moved
to src/spot/robot and src/spot/nav/runner.py (Task 9). This module keeps the
feature handler functions and the feature branches of _dispatch_core until
Wave 4 moves them into src/spot/features, and re-exports every moved name.

State that moved into the owner objects (CONNECTION, ARM, NAV, SAFETY_GEN) is
NOT a module global here any more: reads of the old names are forwarded by
__getattr__ below; writes must go to the owners.
"""
import os
import re
import sys
import pathlib
import time
import threading

# Add project root to path to enable imports
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives  # noqa: F401
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.client.image import ImageClient  # noqa: F401
from src.session import spot_session, send_arm_pose, joints_pose_command  # noqa: F401
import src.session as _poses  # noqa: F401
from src.location_manager import (
    list_locations as _list_saved_locations,
    normalize_location_name as _normalize_location_name,
)
from src.map_loader import list_available_maps, read_last_used, resolve_map_path, write_last_used
from src.graph_nav_utils import (  # noqa: F401
    upload_graph_and_snapshots,
    initialize_localization,
    capped_mobility_params,
    capped_travel_params,
    walk_speed_cap,
)

# ---------------------------------------------------------------------------
# Robot core (moved in Task 9): owners + re-exported functions and constants
# ---------------------------------------------------------------------------
import src.spot.robot as _r
from src.spot.robot import motion as _motion
from src.spot.robot import state as _state
from src.spot.robot import dispatcher as _dispatcher
from src.spot.robot.connection import ensure_spot_session, close_spot_session  # noqa: F401
from src.spot.robot.state import (  # noqa: F401
    _ESTOP_STATE_NAMES,
    CAMERA_SOURCES,
    _current_tts_volume_percent,
    get_robot_state_dict,
    capture_frame,
)
from src.spot.robot.state import is_standing as _is_standing  # noqa: F401
from src.spot.robot.state import standing_and_powered as _standing_and_powered  # noqa: F401
from src.spot.robot.arm import (  # noqa: F401
    LOCOMOTION_INTENTS,
    SAFETY_INTENTS,
    SHUTDOWN_INTENTS,
    ABORTING_INTENTS,
    SAFETY_LOCK_TIMEOUT_S,
    RPC_TIMEOUT_S,
    arm_policy_for,
    _talk_is_walk_pose,
    _hold_gaze,
    get_arm_pose,
    _set_arm_pose,
    _mouth_suspend,
    _mouth_restore,
    _arm_send,
    _stow_now,
    _restore_after_motion,
    _close_mouth,
    _in_talk_mode,
    _set_talk_resume,
)
from src.spot.robot.motion import (  # noqa: F401
    send_stop as _send_stop,
    finish_motion as _finish_motion,
    thread_then_restore as _thread_then_restore,
    wait_trajectory_done as _wait_trajectory_done,
    start_restore_watch as _start_restore_watch,
    abort_motion as _abort_motion,
)
from src.spot.nav.runner import (  # noqa: F401
    is_navigating,
    is_follow_mode,
    _spawn_nav,
    _cancel_nav,
    _ensure_localized,
    _force_relocalize,
    _save_home_waypoint,
    _navigate_waypoint_sequence,
    _start_nav_thread,
)
from src.spot.robot.dispatcher import (  # noqa: F401
    _UNLOCKED_INTENTS,
    INFO_INTENTS,
    _dispatch_lock,
    _say,
    dispatch_intent,
    _dispatch_safety,
    current_safety_gen,
    _aborted,
    resume_talk_after_halt,
    _dispatch_locomotion,
)

# Old module-global name -> (owner, attribute). Reads only (PEP 562).
_FORWARDS = {
    "_spot_session": ("CONNECTION", "_session"),
    "_session_context": ("CONNECTION", "_context"),
    "_session_lock": ("CONNECTION", "_lock"),
    "_closing": ("CONNECTION", "_closing"),
    "_arm_pose": ("ARM", "_pose"),
    "_pose_before_walk": ("ARM", "_pose_before_walk"),
    "_pose_during_walk": ("ARM", "_pose_during_walk"),
    "_arm_lock": ("ARM", "_lock"),
    "_arm_settle_until": ("ARM", "_settle_until"),
    "_last_traj": ("ARM", "_last_traj"),
    "_mouth_resume": ("ARM", "_mouth_resume"),
    "_talk_resume": ("ARM", "_talk_resume"),
    "_gaze_deg": ("ARM", "_gaze_deg"),
    "_last_brain": ("ARM", "_last_brain"),
    "_safety_gen": ("SAFETY_GEN", "_gen"),
    "_dispatch_gen": ("SAFETY_GEN", "_dispatch_gen"),
    "_nav_thread": ("NAV", "_thread"),
    "_nav_stop_event": ("NAV", "_stop_event"),
    "_nav_mode": ("NAV", "_mode"),
    "_nav_lock": ("NAV", "_lock"),
    "_home_waypoint": ("NAV", "_home_waypoint"),
}


_FEATURE_FORWARDS = {}  # old global name -> feature module path (live reads, Wave 4)


def __getattr__(name):
    if name in _FEATURE_FORWARDS:
        import importlib
        return getattr(importlib.import_module(_FEATURE_FORWARDS[name]), name)
    try:
        owner, attr = _FORWARDS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(getattr(_r, owner), attr)  # owner looked up at call time


# --- mouth (moved to src/spot/features/mouth) ---
from src.spot.features.mouth.handlers import _handle_enable_mouth  # noqa: F401,E402
from src.spot.features.mouth.handlers import _handle_disable_mouth  # noqa: F401,E402
from src.spot.features.mouth.handlers import AUTO_ARM_STAND_WAIT_S  # noqa: F401,E402
from src.spot.features.mouth.handlers import set_auto_arm  # noqa: F401,E402
from src.spot.features.mouth.handlers import _cancel_auto_arm  # noqa: F401,E402
from src.spot.features.mouth.handlers import auto_arm_if_ready  # noqa: F401,E402
from src.spot.features.mouth.handlers import start_auto_arm  # noqa: F401,E402
from src.spot.features.mouth.handlers import _auto_arm_after_stand  # noqa: F401,E402
_FEATURE_FORWARDS.update({n: "src.spot.features.mouth.handlers"
                          for n in ("_auto_arm", "_auto_arm_pending", "_auto_arm_thread")})

# --- settings (moved to src/spot/features/settings) ---
from src.spot.features.settings.handlers import _handle_set_volume  # noqa: F401,E402
from src.spot.features.settings.handlers import _handle_set_backend  # noqa: F401,E402
from src.spot.features.settings.handlers import do_set_persona  # noqa: F401,E402
from src.spot.features.settings.handlers import do_add_persona  # noqa: F401,E402
from src.spot.features.settings.handlers import KOKORO_GENDER_FALLBACK  # noqa: F401,E402
from src.spot.features.settings.handlers import PROMPT_PREFIX_TEMPLATE  # noqa: F401,E402


# Feature state that stays here until Wave 4 (mouth auto-arm, gaze timing).
_gaze_t = 0.0               # monotonic time of the last gaze command



def _is_named_location(name: str) -> bool:
    """Return True if the location has a meaningful human-assigned name.

    Filters out auto-generated names like 'waypoint_6' or 'localize_-_1'.
    """
    import re
    return not re.match(r"^(waypoint_\d+|localize_)", name)


# ---------------------------------------------------------------------------
# Gaze toward the locked speaker (Task 19, operator decision 4)
# ---------------------------------------------------------------------------
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
    sess = _r.CONNECTION._session
    if sess is None or _r.CONNECTION._closing or _r.ARM._pose != "talk" or _r.NAV.is_navigating():
        return False
    try:
        import math
        from src.voice_control.tuning import get_tuning
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
                    or time.monotonic() < _r.ARM._settle_until):
                return False
            sess["cmd"].robot_command(joints_pose_command(sh0), timeout=GAZE_RPC_TIMEOUT_S)
            _r.ARM._gaze_deg, _gaze_t = target, now
        bearing = "centre" if body_deg is None else f"{((float(body_deg) + 180) % 360) - 180:+.0f} deg (body, +left)"
        print(f"[Spot] gaze: speaker {bearing} -> shoulder yaw {target:+.0f} deg (limit {max_deg:.0f})")
        return True
    except Exception as e:
        print(f"[Spot] gaze_at_bearing failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Emotes / dances (Task 15)
# ---------------------------------------------------------------------------
_last_routine = None  # last routine run, so "dance again" never repeats it
EMOTE_TIMEOUT_S = 60.0
EMOTE_STALE_GRACE_S = 4.0
# ChoreographyStatusResponse statuses that mean "still going"
_CHOREO_RUNNING = {1, 3, 4, 5}  # DANCING, PREPPING, WAITING_FOR_START_TIME, VALIDATING
_CHOREO_COMPLETED = 2           # STATUS_COMPLETED_SEQUENCE
NO_ROOM_LINE = "I need a little more room to dance."
NO_GRID_LINE = "I can't check my surroundings right now, so I'll skip the dance."


def _clearance_refusal(robot):
    """None if the floor around Spot is clear, else the line to speak.
    Fails closed: no usable local grid -> refuse."""
    from src.voice_control.tuning import get_tuning
    from src.voice_control.perception.obstacle_query import clearance_scan
    clearance = get_tuning().dance_clearance_m
    try:
        usable, hit = clearance_scan(robot, max_radius_m=clearance)
    except Exception as e:
        print(f"[Spot] emote: obstacle check failed: {e}")
        return NO_GRID_LINE
    if not usable:
        print("[Spot] emote: no usable local grid")
        return NO_GRID_LINE
    if hit is not None and hit.distance_m < clearance:
        print(f"[Spot] emote: obstacle at {hit.distance_m:.2f} m < {clearance} m")
        return NO_ROOM_LINE
    return None



def _emote_routine(session, routine, prev_pose, stop_event, speak=None, gen=None):
    """Background thread: stow arm (mouth latched off), re-check clearance,
    run the routine, restore the arm only after it completed. Every robot
    command is gated on stop_event under _arm_lock, so a stop/freeze/estop
    (which sets the event before its own stop) is never followed by a routine
    start or an arm restore from here."""
    global _last_routine
    from bosdyn.choreography.client.choreography import ChoreographyClient
    cmd = session["cmd"]
    outcome = "aborted"  # "done" | "timeout" | "abnormal" | "aborted"
    started = False
    try:
        _r.ARM.mouth_suspend()  # the claw never moves during a routine
        if prev_pose != "stowed":
            if not _r.ARM.send(cmd, "stowed", stop_event, timeout_s=5.0):
                print("[Spot] emote: arm did not stow — not dancing")
                outcome = "done"  # restore the previous pose explicitly
                return
        refusal = _clearance_refusal(session["robot"])  # re-check right before
        if refusal is not None:
            _say(speak, refusal)
            outcome = "done"  # nothing ran; put the arm back
            return
        choreo = session["robot"].ensure_client(ChoreographyClient.default_service_name)
        with _r.ARM._lock:
            if stop_event.is_set() or _r.SAFETY_GEN.aborted(gen):
                return
            started = True
            choreo.execute_choreography(routine, client_start_time=time.time() + 1.0,
                                        choreography_starting_slice=0, timeout=RPC_TIMEOUT_S)
        _last_routine = routine  # only a routine that actually started counts as "last"
        print(f"[Spot] ✓ Routine '{routine}' started")
        t0 = time.time()
        deadline = t0 + EMOTE_TIMEOUT_S
        seen_running = False
        outcome = "timeout"
        stop_event.wait(1.0)
        while not stop_event.is_set() and time.time() < deadline:
            try:
                status, _ = choreo.get_choreography_status()
                st = status.status
                if st in _CHOREO_RUNNING:
                    seen_running = True
                elif seen_running:
                    outcome = "done" if st == _CHOREO_COMPLETED else "abnormal"
                    break
                elif time.time() > t0 + EMOTE_STALE_GRACE_S:
                    outcome = "abnormal"  # never saw it run: don't trust the status
                    break
            except Exception as e:
                print(f"[Spot] emote status poll failed: {e}")
            stop_event.wait(0.5)
        if stop_event.is_set():
            outcome = "aborted"
        print(f"[Spot] Routine '{routine}' ended: {outcome}")
    except Exception as e:
        print(f"[Spot] ✗ emote '{routine}' failed: {e}")
        outcome = "abnormal" if started else "done"
    finally:
        if stop_event.is_set():
            return  # the stop owns the robot and the arm
        try:
            if outcome in ("timeout", "abnormal") and started:
                _motion.send_stop(cmd)
            if outcome in ("done", "timeout") and prev_pose != "stowed":
                # also covers a failed stow: put the arm back in a known pose
                _r.ARM.send(cmd, prev_pose, stop_event, timeout_s=5.0)
            _r.ARM.mouth_restore()
        except Exception as e:
            print(f"[Spot] ⚠️  Arm restore after emote failed: {e}")


def _handle_emote(params, session, speak=None):
    """emote {"name": str}: refuse while moving, sitting or crowded, then run
    the routine on a background thread registered as the nav thread (so stop
    cancels it)."""
    from src.voice_control.emotes import resolve_routine
    if _r.NAV._mode == "emote" and _r.NAV.is_navigating():
        _say(speak, "One dance at a time!")
        return False
    if _r.NAV.is_navigating() or _r.ARM._pose == "walk":
        _say(speak, "I can't dance while I'm on the move.")
        return False
    try:
        name = (params or {}).get("name")
        if isinstance(name, str):
            name = name.strip().lower().replace(" ", "_")  # "Happy Wiggle" -> happy_wiggle
        routine = resolve_routine(name or None, _last_routine)
    except ValueError:
        _say(speak, "I don't know that one yet.")
        return False
    if not _state.standing_and_powered(session):
        _say(speak, "Ask me to stand up first.")
        return False
    refusal = _clearance_refusal(session["robot"])
    if refusal is not None:
        _say(speak, refusal)
        return False
    gen = _r.SAFETY_GEN.dispatch_current()
    if _r.SAFETY_GEN.aborted(gen):
        return False
    _r.NAV.cancel()
    prev = _r.ARM._pose
    if not _r.NAV.spawn(lambda ev: _emote_routine(session, routine, prev, ev, speak, gen),
                      mode="emote", gen=gen):
        return False
    print(f"[Spot] emote: running '{routine}'")
    return True


# Spoken lines never contain stop/halt/freeze/e-stop: the always-on safety
# KWS hears Spot's own voice and would halt it.
_ESTOP_SPOKEN = {"not_cut": "released", "cut": "engaged", "soft_stop": "in a soft hold"}


def _spoken_list(names) -> str:
    """["a_b", "c", "d"] -> "a b, c and d"."""
    items = [str(n).replace("_", " ") for n in names]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _handle_list_maps(speak=None):
    """Spoken list of maps on disk. No robot needed."""
    try:
        maps = list_available_maps(project_root)
    except OSError as e:
        print(f"[Spot] ✗ list_maps failed: {e}")
        _say(speak, "I couldn't read my maps folder.")
        return False
    if not maps:
        _say(speak, "I don't have any maps saved yet.")
        return True
    print(f"[Spot] Available maps ({len(maps)}): {', '.join(maps)}")
    line = f"I have {len(maps)} map{'s' if len(maps) != 1 else ''}: {_spoken_list(maps)}."
    current = read_last_used(project_root)
    if current:
        print(f"[Spot] Currently loaded: {current}")
        line += f" The current one is {current.replace('_', ' ')}."
    _say(speak, line)
    return True


def _handle_list_locations(speak=None):
    """Spoken list of saved locations. No robot needed."""
    try:
        locations = _list_saved_locations()
    except Exception as e:
        print(f"[Spot] ✗ Failed to list locations: {e}")
        _say(speak, "I couldn't read my saved places.")
        return False
    names = [n for n in locations if _is_named_location(n)]
    if not names:
        _say(speak, "I don't have any saved places yet.")
        return True
    for n in names:
        print(f"[Spot]   - {n}")
    _say(speak, f"I know {len(names)} place{'s' if len(names) != 1 else ''}: "
                f"{_spoken_list(names)}.")
    return True



def _dispatch_core(intent, brain=None, speak=None, session=None):
    """The per-intent branches. Called only through dispatch_intent (which
    applies the arm policy) or for nested re-dispatch."""
    if not intent:
        return False

    name = intent["intent"]
    params = intent.get("params", {})

    if name in _dispatcher.CORE_INTENTS:  # core branches moved to src/spot/robot/dispatcher.py
        return _dispatcher._dispatch_core(intent, brain, speak, session=session)

    if name == "list_maps":
        return _handle_list_maps(speak)
    if name == "list_locations":
        return _handle_list_locations(speak)

    try:
        if session is None:  # safety/aborting callers pass the session they hold
            session = _r.CONNECTION.ensure()
        cmd_client = session["cmd"]
        
        if name == "emote":
            return _handle_emote(params, session, speak)
        if name == "walk_to":
            # Walk to relative position
            relative = params.get("relative", [0.0, 0.0, 0.0])
            if len(relative) >= 2:
                x, y = relative[0], relative[1]
                yaw = relative[2] if len(relative) > 2 else 0.0
                print(f"[Spot] Walking to relative position: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}rad")
                
                # Use SE2Pose for relative movement in body frame
                goal_pose = SE2Pose(x, y, yaw)
                cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                    goal_se2=goal_pose.to_proto(),
                    frame_name=ODOM_FRAME_NAME,
                    params=capped_mobility_params(),
                )
                cmd_id = cmd_client.robot_command(cmd, end_time_secs=None)
                _r.ARM._last_traj = (cmd_id, time.time() + 30.0)
                return True
            else:
                print(f"[Spot] Invalid relative position: {relative}")
                return False
                
        elif name == "turn":
            # Rotate in place
            deg = params.get("deg", 0)
            direction = params.get("dir", "left")

            # Convert degrees to radians (negative for left, positive for right)
            rad = -abs(deg) if direction == "left" else abs(deg)
            rad = rad * 3.14159 / 180.0  # deg to rad

            print(f"[Spot] Turning {direction} {deg} degrees ({rad:.3f} rad)")

            # Get current position and add rotation
            state_client = session["state"]
            robot_state = state_client.get_robot_state()
            odom_tform_body = get_odom_tform_body(
                robot_state.kinematic_state.transforms_snapshot
            )
            current_x = odom_tform_body.position.x
            current_y = odom_tform_body.position.y
            # Extract yaw angle from rotation
            current_yaw = odom_tform_body.rotation.to_yaw()

            # Add rotation to current yaw
            new_yaw = current_yaw + rad

            # Create trajectory command to rotate in place
            goal_pose = SE2Pose(current_x, current_y, new_yaw)
            cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                goal_se2=goal_pose.to_proto(),
                frame_name=ODOM_FRAME_NAME,
                params=capped_mobility_params(),
            )
            # Send command with timeout to allow execution
            end_t = time.time() + 5.0
            cmd_id = cmd_client.robot_command(cmd, end_time_secs=end_t)
            _r.ARM._last_traj = (cmd_id, end_t + 1.0)
            print(f"[Spot] ✓ Turn command sent")
            return True
            
        elif name == "walk":
            # Walk forward or backward a specific distance
            direction = params.get("direction", "forward")
            distance = float(params.get("distance", 1.0))

            # Clamp distance for safety (0.5 to 5 meters)
            distance = max(0.5, min(5.0, distance))

            # Forward is positive X, backward is negative X
            x = distance if direction == "forward" else -distance

            print(f"[Spot] Walking {direction} {distance:.1f}m...")
            try:
                state_client = session["state"]
                robot_state = state_client.get_robot_state()
                frame_tree = robot_state.kinematic_state.transforms_snapshot

                cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    goal_x_rt_body=x,
                    goal_y_rt_body=0.0,
                    goal_heading_rt_body=0.0,
                    frame_tree_snapshot=frame_tree,
                    params=capped_mobility_params(),
                )
                # Speed cap: allow enough time to cover the distance at the cap.
                end_t = time.time() + max(10.0, distance / walk_speed_cap() + 3.0)
                cmd_id = cmd_client.robot_command(cmd, end_time_secs=end_t)
                _r.ARM._last_traj = (cmd_id, end_t + 1.0)
                print(f"[Spot] ✓ Walk {direction} command sent")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Walk failed: {e}")
                return False

        elif name == "strafe":
            # Strafe left or right a specific distance
            direction = params.get("direction", "left")
            distance = float(params.get("distance", 0.5))

            # Clamp distance (0.25 to 2 meters)
            distance = max(0.25, min(2.0, distance))

            # Left is positive Y, right is negative Y
            y = distance if direction == "left" else -distance

            print(f"[Spot] Strafing {direction} {distance:.1f}m...")
            try:
                state_client = session["state"]
                robot_state = state_client.get_robot_state()
                frame_tree = robot_state.kinematic_state.transforms_snapshot

                cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    goal_x_rt_body=0.0,
                    goal_y_rt_body=y,
                    goal_heading_rt_body=0.0,
                    frame_tree_snapshot=frame_tree,
                    params=capped_mobility_params(),
                )
                end_t = time.time() + max(10.0, distance / walk_speed_cap() + 3.0)
                cmd_id = cmd_client.robot_command(cmd, end_time_secs=end_t)
                _r.ARM._last_traj = (cmd_id, end_t + 1.0)
                print(f"[Spot] ✓ Strafe {direction} command sent")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Strafe failed: {e}")
                return False

        elif name == "body_height":
            # Adjust body height (crouch/stand tall)
            height = float(params.get("height", 0.0))

            # Clamp height (-0.15 to 0.1 meters relative to nominal)
            height = max(-0.15, min(0.1, height))

            height_desc = "normal"
            if height < -0.05:
                height_desc = "low (crouching)"
            elif height > 0.05:
                height_desc = "tall"

            print(f"[Spot] Setting body height to {height_desc} ({height:+.2f}m)...")
            try:
                cmd = RobotCommandBuilder.synchro_stand_command(body_height=height)
                cmd_client.robot_command(cmd)
                print(f"[Spot] ✓ Body height set to {height_desc}")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Body height adjustment failed: {e}")
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

        elif name == "go_to_object":
            # Generalized "go to X" — smart dispatch to three backends:
            # 1. Saved location in the graph_nav map → delegate to go_to
            # 2. WorldObject match ("fiducial 5" / "dock") → native AprilTag /
            #    dock service, SE2 trajectory. No YOLO involvement.
            # 3. Anything else → YOLO-World visual nav (the original path).
            description = params.get("description", "")
            if not description:
                print("[Spot] No object description provided")
                return False

            # (1) saved location shortcut
            saved = _list_saved_locations()
            desc_normalized = description.strip().lower().replace(" ", "_")
            if desc_normalized in saved:
                print(f"[Spot] '{description}' is a saved location — using map navigation")
                return _dispatch_core({"intent": "go_to", "params": {"location": desc_normalized}})

            # (2) world-object shortcut — fiducials and docks
            try:
                from src.voice_control.world_objects import find_navigation_target
                robot_obj = session["robot"]
                target = find_navigation_target(robot_obj, description)
            except Exception as e:
                print(f"[Spot] world_objects lookup failed, falling through to YOLO: {e}")
                target = None

            if target is not None:
                print(f"[Spot] '{description}' → {target.source} "
                      f"at ({target.x:.2f}, {target.y:.2f}, "
                      f"yaw={target.yaw:.2f}) in {target.frame_name}")
                try:
                    goal_pose = SE2Pose(target.x, target.y, target.yaw)
                    cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                        goal_se2=goal_pose.to_proto(),
                        frame_name=target.frame_name,
                        params=capped_mobility_params(),
                    )
                    end_t = time.time() + 15.0
                    cmd_id = cmd_client.robot_command(cmd, end_time_secs=end_t)
                    _r.ARM._last_traj = (cmd_id, end_t + 1.0)
                    print(f"[Spot] ✓ SE2 trajectory sent toward {target.source}")
                    return True
                except Exception as e:
                    print(f"[Spot] ✗ World-object approach failed: {e}")
                    # fall through to YOLO as a last resort
                    pass

            # (3) YOLO fallback
            print(f"[Spot] Looking for '{description}' with YOLO-World...")

            def _visual_nav_thread(ev):
                try:
                    from src.voice_control.visual_nav import navigate_to_object
                    result = navigate_to_object(description, session, ev)
                    if result:
                        print(f"[Spot] ✓ Arrived near '{description}'")
                    else:
                        print(f"[Spot] ✗ Could not reach '{description}'")
                    return bool(result)
                except ImportError:
                    print("[Spot] ✗ ultralytics not installed. Run: pip install ultralytics")
                except Exception as e:
                    print(f"[Spot] ✗ Visual nav failed: {e}")

            _r.NAV.cancel()
            return _r.NAV.spawn(
                lambda ev: _motion.thread_then_restore(lambda: _visual_nav_thread(ev), ev),
                mode="visual_nav", gen=_r.SAFETY_GEN.dispatch_current())

        elif name == "check_obstacles":
            # Query Spot's native LocalGrid obstacle field via the
            # perception module. Read-only — never commands motion.
            # Returns structured (distance, direction) so the LLM can
            # verbalize it. On a base Spot the grid may not be available
            # at all (probe script reports this) — in that case we log
            # and return True since "no obstacles reported" is a valid
            # answer, not a failure.
            try:
                from src.voice_control.perception.obstacle_query import (
                    nearest_obstacle_in_body_frame,
                )
            except Exception as e:
                print(f"[Spot] check_obstacles: perception module unavailable: {e}")
                _say(speak, "I can't check my surroundings right now.")
                return False

            robot_obj = session["robot"]
            try:
                hit = nearest_obstacle_in_body_frame(robot_obj, max_radius_m=3.0)
            except Exception as e:
                print(f"[Spot] check_obstacles: query failed: {e}")
                _say(speak, "I couldn't check my surroundings just now.")
                return False

            if hit is None:
                print("[Spot] check_obstacles: no obstacles within 3m "
                      "(or obstacle-grid unavailable on this firmware)")
                _say(speak, "I don't see anything close to me right now.")
                return True

            # Convert bearing (radians, 0=forward, +pi/2=left) to a
            # human-friendly direction label. 45°-wide bins.
            import math as _m
            deg = _m.degrees(hit.bearing_rad)
            if -22.5 <= deg <= 22.5:
                direction = "directly ahead"
            elif 22.5 < deg <= 67.5:
                direction = "ahead and to the left"
            elif 67.5 < deg <= 112.5:
                direction = "to the left"
            elif 112.5 < deg <= 157.5:
                direction = "behind and to the left"
            elif deg > 157.5 or deg < -157.5:
                direction = "directly behind"
            elif -157.5 <= deg < -112.5:
                direction = "behind and to the right"
            elif -112.5 <= deg < -67.5:
                direction = "to the right"
            else:  # -67.5 <= deg < -22.5
                direction = "ahead and to the right"
            print(
                f"[Spot] check_obstacles: nearest obstacle {hit.distance_m:.2f}m "
                f"{direction} ({deg:.0f}°, source={hit.grid_name})"
            )
            _say(speak, f"The closest thing is about {hit.distance_m:.1f} meters {direction}.")
            return True

        elif name == "follow_me":
            # Follow the nearest person using YOLO person detection
            print("[Spot] Starting follow mode...")

            def _follow_thread(ev):
                try:
                    from src.voice_control.visual_nav import follow_person
                    result = follow_person(session, ev)
                    if result:
                        print("[Spot] ✓ Follow mode ended cleanly")
                    else:
                        print("[Spot] Lost the person — follow mode ended")
                    return bool(result)
                except ImportError:
                    print("[Spot] ✗ ultralytics not installed. Run: pip install ultralytics")
                except Exception as e:
                    print(f"[Spot] ✗ Follow failed: {e}")

            _r.NAV.cancel()
            return _r.NAV.spawn(
                lambda ev: _motion.thread_then_restore(lambda: _follow_thread(ev), ev),
                mode="follow", gen=_r.SAFETY_GEN.dispatch_current())

        elif name == "set_speed":
            # Set movement speed (affects subsequent commands)
            speed = params.get("speed", "normal")

            # Speed mappings (approximate m/s limits)
            speed_map = {
                "slow": 0.5,
                "normal": 1.0,
                "fast": 1.6
            }
            velocity = speed_map.get(speed, 1.0)

            print(f"[Spot] Speed mode set to: {speed} ({velocity} m/s)")
            # Note: Speed is applied per-command via mobility params
            # This is informational - actual speed control happens in movement commands
            print("[Spot] ✓ Speed mode updated (applies to next movement)")
            return True

        elif name == "load_map":
            # Failure to localize after upload is non-fatal — the robot ends up
            # graph-loaded-but-not-localized; the user can drive in front of a
            # fiducial and retry.
            map_name_or_none = (params.get("map") or "").strip() or None
            try:
                map_path = resolve_map_path(map_name_or_none, project_root)
            except FileNotFoundError as e:
                print(f"[Spot] ✗ Map not found: {e}")
                return False

            try:
                _r.NAV.cancel()
                robot = _r.CONNECTION.ensure()["robot"]

                print(f"[Spot] Loading map: {map_path.name}")
                if not upload_graph_and_snapshots(robot, str(map_path)):
                    print("[Spot] ✗ Map upload failed.")
                    return False

                # Record the new current map BEFORE attempting localization, so
                # save_location calls during this session land in the right
                # slice even if the fiducial isn't visible yet.
                write_last_used(project_root, map_path.name)
                localized = initialize_localization(robot, use_fiducial=True)

                if localized:
                    print(f"[Spot] ✓ Loaded {map_path.name} and re-localized.")
                else:
                    print(
                        f"[Spot] Loaded {map_path.name}, but no fiducial visible. "
                        f"Drive in front of a fiducial and say 'load map' again."
                    )
                return True
            except Exception as e:
                print(f"[Spot] ✗ load_map failed: {e}")
                return False

        elif name == "save_location":
            # Save current position as a named location. Use the same
            # canonical form (strip + lowercase + underscores) that
            # tour/patrol use when looking up locations, so "save location
            # my desk" → "my_desk" matches a later "go to my desk".
            location_name = _normalize_location_name(params.get("location", ""))
            if not location_name:
                print("[Spot] No location name provided")
                return False
            
            print(f"[Spot] Saving current position as '{location_name}'...")
            try:
                from bosdyn.client.graph_nav import GraphNavClient
                from src.location_manager import save_location
                
                # Get current localization state to find waypoint
                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)
                localization = graph_nav_client.get_localization_state()
                
                if not localization.localization.waypoint_id:
                    print("[Spot] ✗ Not localized to a map.")
                    print("[Spot]   Please localize first (robot must see a fiducial or be at a known waypoint)")
                    return False
                
                current_waypoint = localization.localization.waypoint_id
                save_location(location_name, current_waypoint)
                print(f"[Spot] ✓ Location '{location_name}' saved at waypoint {current_waypoint}")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Save location failed: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        elif name == "go_to":
            # Navigate to a named location using GraphNav
            location_name = _normalize_location_name(params.get("location", ""))
            if not location_name:
                print("[Spot] No location name provided")
                return False

            print(f"[Spot] Navigating to '{location_name}'...")
            try:
                from bosdyn.client.graph_nav import GraphNavClient
                from src.location_manager import load_location, list_locations

                waypoint_id = load_location(location_name)
                if not waypoint_id:
                    available = list(list_locations().keys())
                    print(f"[Spot] Location '{location_name}' not found.")
                    if available:
                        print(f"[Spot]   Available locations: {', '.join(available)}")
                    else:
                        print(f"[Spot]   No locations saved yet. Say 'save location {location_name}' at the desired position.")
                    return False

                print(f"[Spot] Found: '{location_name}' -> waypoint {waypoint_id}")

                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)

                if not _r.NAV.ensure_localized(graph_nav_client, session):
                    print("[Spot]   Please run 'python scripts/setup_map.py --waypoint-init' to localize.")
                    return False

                _r.NAV.save_home_waypoint(graph_nav_client)
                if not _r.NAV.start_thread(graph_nav_client, [waypoint_id], [location_name]):
                    return False

                print(f"[Spot] Navigation started to '{location_name}'")
                return True
            except Exception as e:
                print(f"[Spot] Navigation failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        elif name == "tour":
            # Visit locations in sequence (one pass)
            locations_param = params.get("locations", "all")
            print("[Spot] Starting tour...")
            try:
                from bosdyn.client.graph_nav import GraphNavClient
                from src.location_manager import load_location, list_locations
                from src.graph_nav_utils import get_graph_waypoint_order

                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)

                if not _r.NAV.ensure_localized(graph_nav_client, session):
                    print("[Spot]   Please localize first.")
                    return False

                if locations_param == "all":
                    saved = list_locations()
                    if not saved:
                        print("[Spot] No saved locations. Save some first.")
                        return False
                    # Filter to named locations only (skip waypoint_XX, localize_, etc.)
                    named = {n: wid for n, wid in saved.items() if _is_named_location(n)}
                    if not named:
                        print("[Spot] No named locations found (only generic waypoints).")
                        return False
                    # Order by graph recording order for natural traversal
                    graph_order = get_graph_waypoint_order(session["robot"])
                    wp_to_name = {wid: name for name, wid in named.items()}
                    waypoint_ids = []
                    location_names = []
                    for wp_id in graph_order:
                        if wp_id in wp_to_name:
                            waypoint_ids.append(wp_id)
                            location_names.append(wp_to_name[wp_id])
                    # Include any named locations not in graph order
                    for loc_name, wid in named.items():
                        if wid not in waypoint_ids:
                            waypoint_ids.append(wid)
                            location_names.append(loc_name)
                else:
                    if isinstance(locations_param, str):
                        locations_param = [locations_param]
                    waypoint_ids = []
                    location_names = []
                    for loc_name in locations_param:
                        loc_name = _normalize_location_name(loc_name)
                        wid = load_location(loc_name)
                        if not wid:
                            print(f"[Spot] Location '{loc_name}' not found, skipping.")
                            continue
                        waypoint_ids.append(wid)
                        location_names.append(loc_name)

                if not waypoint_ids:
                    print("[Spot] No valid locations to visit.")
                    return False

                _r.NAV.save_home_waypoint(graph_nav_client)
                if not _r.NAV.start_thread(graph_nav_client, waypoint_ids, location_names, repeat=False):
                    return False

                print(f"[Spot] Tour started: {', '.join(location_names)}")
                return True
            except Exception as e:
                print(f"[Spot] Tour failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        elif name == "patrol":
            # Visit all named locations and loop continuously until stopped.
            # The LLM system prompt advertises patrol as the looping variant
            # of tour, so the underlying _start_nav_thread call below uses
            # repeat=True. Stop with an explicit "stop" command.
            locations_param = params.get("locations", "all")
            print("[Spot] Starting patrol...")
            try:
                from bosdyn.client.graph_nav import GraphNavClient
                from src.location_manager import load_location, list_locations
                from src.graph_nav_utils import get_graph_waypoint_order

                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)

                if not _r.NAV.ensure_localized(graph_nav_client, session):
                    print("[Spot]   Please localize first.")
                    return False

                if locations_param == "all":
                    saved = list_locations()
                    if not saved:
                        print("[Spot] No saved locations. Save some first.")
                        return False
                    # Filter to named locations only
                    named = {n: wid for n, wid in saved.items() if _is_named_location(n)}
                    if not named:
                        print("[Spot] No named locations found (only generic waypoints).")
                        return False
                    graph_order = get_graph_waypoint_order(session["robot"])
                    wp_to_name = {wid: name for name, wid in named.items()}
                    waypoint_ids = []
                    location_names = []
                    for wp_id in graph_order:
                        if wp_id in wp_to_name:
                            waypoint_ids.append(wp_id)
                            location_names.append(wp_to_name[wp_id])
                    for loc_name, wid in named.items():
                        if wid not in waypoint_ids:
                            waypoint_ids.append(wid)
                            location_names.append(loc_name)
                else:
                    if isinstance(locations_param, str):
                        locations_param = [locations_param]
                    waypoint_ids = []
                    location_names = []
                    for loc_name in locations_param:
                        loc_name = _normalize_location_name(loc_name)
                        wid = load_location(loc_name)
                        if not wid:
                            print(f"[Spot] Location '{loc_name}' not found, skipping.")
                            continue
                        waypoint_ids.append(wid)
                        location_names.append(loc_name)

                if not waypoint_ids:
                    print("[Spot] No valid locations to patrol.")
                    return False

                _r.NAV.save_home_waypoint(graph_nav_client)
                if not _r.NAV.start_thread(graph_nav_client, waypoint_ids, location_names, repeat=True):
                    return False

                print(f"[Spot] Patrol started: {', '.join(location_names)}")
                return True
            except Exception as e:
                print(f"[Spot] Patrol failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        elif name == "come_back":
            # Return to position before last navigation
            print("[Spot] Heading back...")
            try:
                if not _r.NAV._home_waypoint:
                    print("[Spot] No departure point saved. Navigate somewhere first.")
                    return False

                from bosdyn.client.graph_nav import GraphNavClient
                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)

                if not _r.NAV.ensure_localized(graph_nav_client, session):
                    print("[Spot]   Please localize first.")
                    return False

                # Reverse-lookup name for display
                locs = _list_saved_locations()
                home_name = next((n for n, wid in locs.items() if wid == _r.NAV._home_waypoint), _r.NAV._home_waypoint[:12])

                if not _r.NAV.start_thread(graph_nav_client, [_r.NAV._home_waypoint], [home_name]):
                    return False

                print(f"[Spot] Heading back to '{home_name}'")
                return True
            except Exception as e:
                print(f"[Spot] Come back failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        else:
            print(f"[Spot] Unknown intent: {name}")
            return False
            
    except Exception as e:
        print(f"[Spot] Error executing intent '{name}': {e}")
        if name in INFO_INTENTS:
            _say(speak, "I can't reach my sensors right now.")
        import traceback
        traceback.print_exc()
        return False