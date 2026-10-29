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

# --- navigation (moved to src/spot/features/navigation) ---
from src.spot.features.navigation.handlers import _is_named_location, _spoken_list, _handle_list_maps, _handle_list_locations  # noqa: F401

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


# --- dance (moved to src/spot/features/dance) ---
from src.spot.features.dance.handlers import _clearance_refusal  # noqa: E402,F401
from src.spot.features.dance.handlers import _emote_routine  # noqa: E402,F401
from src.spot.features.dance.handlers import _handle_emote  # noqa: E402,F401
from src.spot.features.dance.handlers import NO_ROOM_LINE, NO_GRID_LINE  # noqa: E402,F401
from src.spot.features.dance.handlers import EMOTE_TIMEOUT_S, EMOTE_STALE_GRACE_S  # noqa: E402,F401
_FEATURE_FORWARDS.update({"_last_routine": "src.spot.features.dance.handlers"})

# --- gaze (moved to src/spot/features/gaze) ---
from src.spot.features.gaze.handlers import gaze_sh0_deg, gaze_at_bearing  # noqa: E402,F401
from src.spot.features.gaze.handlers import (  # noqa: E402,F401
    GAZE_MIN_CHANGE_DEG, GAZE_MIN_INTERVAL_S, GAZE_BEHIND_DEG, GAZE_HARD_MAX_DEG, GAZE_RPC_TIMEOUT_S,
)
_FEATURE_FORWARDS.update({"_gaze_t": "src.spot.features.gaze.handlers"})

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



# Spoken lines never contain stop/halt/freeze/e-stop: the always-on safety
# KWS hears Spot's own voice and would halt it.
_ESTOP_SPOKEN = {"not_cut": "released", "cut": "engaged", "soft_stop": "in a soft hold"}


def _dispatch_core(intent, brain=None, speak=None, session=None):
    """The per-intent branches. Called only through dispatch_intent (which
    applies the arm policy) or for nested re-dispatch."""
    if not intent:
        return False

    name = intent["intent"]
    params = intent.get("params", {})

    if name in _dispatcher.CORE_INTENTS:  # core branches moved to src/spot/robot/dispatcher.py
        return _dispatcher._dispatch_core(intent, brain, speak, session=session)

    try:
        if session is None:  # safety/aborting callers pass the session they hold
            session = _r.CONNECTION.ensure()
        cmd_client = session["cmd"]
        
        if name == "battery_status":
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

            return True

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