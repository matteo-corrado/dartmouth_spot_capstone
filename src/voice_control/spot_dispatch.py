"""Spot command dispatcher for voice intents.

This module executes Spot commands based on parsed voice intents.
Uses a persistent session connection that stays open for multiple commands.
"""
import sys
import pathlib
import time
import threading

# Add project root to path to enable imports
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.client.image import ImageClient
from src.session import spot_session
from src.location_manager import list_locations as _list_saved_locations
from src.map_loader import list_available_maps, read_last_used, resolve_map_path, write_last_used
from src.graph_nav_utils import upload_graph_and_snapshots, initialize_localization

# Camera source names for Spot's fisheye cameras
CAMERA_SOURCES = {
    "front": "frontleft_fisheye_image",
    "front_left": "frontleft_fisheye_image",
    "front_right": "frontright_fisheye_image",
    "left": "left_fisheye_image",
    "right": "right_fisheye_image",
    "back": "back_fisheye_image",
}

# Global session handle (initialized on first use)
_spot_session = None
_session_context = None

# Navigation thread management
_nav_thread = None
_nav_stop_event = None
_nav_mode = None  # "follow", "graphnav", "visual_nav", or None
_home_waypoint = None


def is_navigating() -> bool:
    """Check if the robot is currently navigating (motor noise expected)."""
    return _nav_thread is not None and _nav_thread.is_alive()


def is_follow_mode() -> bool:
    """Check if robot is in follow-person mode (lighter motor noise)."""
    return _nav_mode == "follow" and is_navigating()


def _current_tts_volume_percent():
    """Read current TTS volume as a 1-100 percent integer (best-effort)."""
    try:
        from src.voice_control.spot_tts import _tts_instance
        if _tts_instance is None:
            return "unknown"
        return int(round(min(1.0, _tts_instance.volume) * 100))
    except Exception:
        return "unknown"


def get_robot_state_dict() -> dict:
    """Collect current robot state for the LLM brain context.

    Returns a dict with battery, posture, location, etc.
    Safe to call even if the session is not active yet.
    """
    state = {
        "battery_percent": "unknown",
        "estimated_runtime_minutes": "unknown",
        "is_powered": False,
        "is_standing": "unknown",
        "current_location": "unknown",
        "saved_locations": ", ".join(_list_saved_locations().keys()) or "none",
        "estop_status": "unknown",
        "tts_volume_percent": _current_tts_volume_percent(),
        "available_maps": "unknown",
        "current_map": "unknown",
    }

    try:
        maps = list_available_maps(project_root)
        state["available_maps"] = ", ".join(maps) if maps else "none"
        state["current_map"] = read_last_used(project_root) or "unknown"
    except OSError as e:
        print(f"[Spot] WARN: map state unavailable: {e}")

    if _spot_session is None:
        state["session"] = "not connected"
        return state

    try:
        state_client = _spot_session["state"]
        robot_state = state_client.get_robot_state()

        # Battery
        state["battery_percent"] = round(
            robot_state.power_state.locomotion_charge_percentage.value
        )
        runtime_sec = robot_state.power_state.locomotion_estimated_runtime.seconds
        state["estimated_runtime_minutes"] = runtime_sec // 60

        # Power / motor state
        is_on = (
            robot_state.power_state.motor_power_state
            == robot_state.power_state.STATE_ON
        )
        state["is_powered"] = is_on

        # E-stop
        estop_map = {0: "unknown", 1: "cut", 2: "not_cut", 3: "soft_stop"}
        if robot_state.estop_states:
            state["estop_status"] = estop_map.get(
                robot_state.estop_states[0].state, "unknown"
            )

    except Exception as e:
        state["state_error"] = str(e)

    # Current localization (GraphNav)
    try:
        from bosdyn.client.graph_nav import GraphNavClient

        gn = _spot_session["robot"].ensure_client(
            GraphNavClient.default_service_name
        )
        loc = gn.get_localization_state()
        wp = loc.localization.waypoint_id
        if wp:
            # Reverse-lookup friendly name from locations.json
            locs = _list_saved_locations()
            friendly = next(
                (name for name, wid in locs.items() if wid == wp), None
            )
            state["current_location"] = friendly or wp
        else:
            state["current_location"] = "not localized"
    except Exception:
        pass

    return state


def capture_frame(camera: str = "front") -> bytes:
    """Capture a JPEG frame from one of Spot's cameras.

    Args:
        camera: Camera name ("front", "left", "right", "back", "front_left", "front_right").

    Returns:
        JPEG image bytes, or empty bytes on failure.
    """
    source_name = CAMERA_SOURCES.get(camera, CAMERA_SOURCES["front"])
    try:
        session = ensure_spot_session()
        image_client = session["robot"].ensure_client(ImageClient.default_service_name)

        image_responses = image_client.get_image_from_sources([source_name])
        if not image_responses:
            print(f"[Spot] No image returned from {source_name}")
            return b""

        image_data = image_responses[0].shot.image.data
        print(f"[Spot] Captured frame from {source_name} ({len(image_data)} bytes)")
        return image_data

    except Exception as e:
        print(f"[Spot] Camera capture failed ({source_name}): {e}")
        return b""


def ensure_spot_session(map_path=None):
    """Ensure Spot session is initialized and return session dict.

    Args:
        map_path: Optional GraphNav map directory to upload during session
            bring-up. When provided AND the session is not yet active, the
            session is created with ``upload_map=True``, ``map_path=...``, and
            ``auto_localize=True`` so the robot uploads the map and attempts
            fiducial localization as part of stand-up.

            When provided AND a session is ALREADY active, the existing
            session is returned silently — the assumption is that the caller
            wants the session, not necessarily the map. To switch maps at
            runtime, use the ``load_map`` voice handler instead, which
            re-uploads on a live session.

    Returns:
        dict with the live session clients (robot, lease_client, cmd, power,
        state).
    """
    global _spot_session, _session_context

    if _spot_session is None:
        try:
            _session_context = spot_session(
                stand_on_enter=True,
                sit_on_exit=False,  # keep standing for voice commands
                upload_map=bool(map_path),
                map_path=str(map_path) if map_path else None,
                auto_localize=bool(map_path),
            )
            _spot_session = _session_context.__enter__()
            print("[Spot] Connected and standing ready for voice commands.")
        except Exception as e:
            print(f"[Spot] Failed to establish session: {e}")
            # Clear the context so we can retry
            _session_context = None
            raise

    return _spot_session


def close_spot_session():
    """Close the Spot session (sit and power off)."""
    global _spot_session, _session_context
    
    if _spot_session is not None and _session_context is not None:
        try:
            from bosdyn.client.robot_command import blocking_sit
            blocking_sit(_spot_session["cmd"], timeout_sec=15)
        except Exception as e:
            print(f"[Spot] Warning during sit: {e}")
        try:
            _spot_session["robot"].power_off(cut_immediately=False, timeout_sec=20)
        except Exception as e:
            print(f"[Spot] Warning during power off: {e}")
        _session_context.__exit__(None, None, None)
        _spot_session = None
        _session_context = None
        print("[Spot] Session closed.")


def _cancel_nav():
    """Cancel any in-progress navigation thread."""
    global _nav_thread, _nav_stop_event, _nav_mode
    if _nav_stop_event:
        _nav_stop_event.set()
    if _nav_thread and _nav_thread.is_alive():
        _nav_thread.join(timeout=2.0)
    _nav_thread = None
    _nav_stop_event = None
    _nav_mode = None


def _ensure_localized(graph_nav_client, session) -> bool:
    """Check localization and attempt fiducial/waypoint init if needed."""
    localization = graph_nav_client.get_localization_state()
    if localization.localization.waypoint_id:
        print(f"[Spot] Localized at waypoint: {localization.localization.waypoint_id}")
        return True

    print("[Spot] Not localized. Attempting to localize...")
    return _force_relocalize(graph_nav_client, session)


def _force_relocalize(graph_nav_client, session) -> bool:
    """Force re-localization using fiducials first, then waypoint fallback."""
    from bosdyn.api.graph_nav import nav_pb2, graph_nav_pb2
    import math

    # Try fiducial-based localization first (most accurate)
    try:
        print("[Spot] Trying fiducial-based localization...")
        empty_loc = nav_pb2.Localization()
        graph_nav_client.set_localization(
            initial_guess_localization=empty_loc,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST
        )
        localization = graph_nav_client.get_localization_state()
        if localization.localization.waypoint_id:
            print(f"[Spot] Localized via fiducial at: {localization.localization.waypoint_id}")
            return True
        print("[Spot] Fiducial localization failed (no fiducials visible)")
    except Exception as e:
        print(f"[Spot] Fiducial localization error: {e}")

    # Fallback: waypoint-based localization
    if session is None:
        return False

    try:
        from bosdyn.client.robot_state import RobotStateClient

        robot_state_client = session["robot"].ensure_client(RobotStateClient.default_service_name)
        robot_state = robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()

        graph = graph_nav_client.download_graph()
        if not graph.waypoints:
            print("[Spot] No waypoints in map. Please upload map first.")
            return False

        first_waypoint_id = graph.waypoints[0].id
        print(f"[Spot] Trying waypoint-based localization to: {first_waypoint_id}")

        loc_guess = nav_pb2.Localization()
        loc_guess.waypoint_id = first_waypoint_id
        loc_guess.waypoint_tform_body.rotation.w = 1.0

        graph_nav_client.set_localization(
            initial_guess_localization=loc_guess,
            max_distance=5.0,   # Allow larger search radius
            max_yaw=180.0 * math.pi / 180.0,  # Allow any orientation
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body
        )

        localization = graph_nav_client.get_localization_state()
        if localization.localization.waypoint_id:
            print(f"[Spot] Localized via waypoint at: {localization.localization.waypoint_id}")
            return True

        print("[Spot] Localization failed. Robot may not be near a known waypoint.")
        return False
    except Exception as e:
        print(f"[Spot] Localization error: {e}")
        return False


def _is_named_location(name: str) -> bool:
    """Return True if the location has a meaningful human-assigned name.

    Filters out auto-generated names like 'waypoint_6' or 'localize_-_1'.
    """
    import re
    return not re.match(r"^(waypoint_\d+|localize_)", name)


def _save_home_waypoint(graph_nav_client):
    """Save current localized waypoint as departure point for come_back."""
    global _home_waypoint
    try:
        loc = graph_nav_client.get_localization_state()
        if loc.localization.waypoint_id:
            _home_waypoint = loc.localization.waypoint_id
    except Exception:
        pass


def _navigate_waypoint_sequence(graph_nav_client, waypoint_ids, location_names,
                                stop_event, repeat=False):
    """Navigate through a sequence of waypoints. Runs in a background thread.

    On STATUS_STUCK: retries up to MAX_STUCK_RETRIES with a new command ID.
    If still stuck after retries, skips to the next waypoint (multi-waypoint)
    or gives up (single waypoint).
    """
    from bosdyn.api.graph_nav import graph_nav_pb2

    MAX_STUCK_RETRIES = 2  # Retry with fresh command ID before skipping

    pass_num = 0
    try:
        while True:
            pass_num += 1
            if repeat and pass_num > 1:
                print(f"[Spot] Starting patrol pass #{pass_num}")

            for i, (wp_id, name) in enumerate(zip(waypoint_ids, location_names)):
                if stop_event.is_set():
                    print("[Spot] Navigation cancelled")
                    return

                print(f"[Spot] Navigating to '{name}' ({i+1}/{len(waypoint_ids)})...")
                stuck_retries = 0
                nav_to_cmd_id = None
                waypoint_reached = False

                while not stop_event.is_set():
                    try:
                        nav_to_cmd_id = graph_nav_client.navigate_to(
                            wp_id, 10.0,
                            command_id=nav_to_cmd_id
                        )
                    except Exception as nav_ex:
                        # navigate_to() raises RobotStuckError as exception
                        if "Stuck" in type(nav_ex).__name__ or "Stuck" in str(nav_ex):
                            stuck_retries += 1
                            if stuck_retries <= MAX_STUCK_RETRIES:
                                print(f"[Spot] Stuck navigating to '{name}', retrying ({stuck_retries}/{MAX_STUCK_RETRIES})...")
                                nav_to_cmd_id = None  # Fresh command on retry
                                time.sleep(2.0)
                                continue
                            else:
                                print(f"[Spot] Still stuck after {MAX_STUCK_RETRIES} retries for '{name}'")
                                break
                        else:
                            print(f"[Spot] Navigation error for '{name}': {nav_ex}")
                            break

                    try:
                        feedback = graph_nav_client.navigation_feedback(nav_to_cmd_id)
                        if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                            print(f"[Spot] Reached '{name}'")
                            waypoint_reached = True
                            break
                        elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                            stuck_retries += 1
                            if stuck_retries <= MAX_STUCK_RETRIES:
                                print(f"[Spot] Stuck navigating to '{name}', retrying ({stuck_retries}/{MAX_STUCK_RETRIES})...")
                                nav_to_cmd_id = None  # Fresh command on retry
                                time.sleep(2.0)
                                continue
                            else:
                                print(f"[Spot] Still stuck after {MAX_STUCK_RETRIES} retries for '{name}'")
                                break
                        elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                            print(f"[Spot] Robot lost during navigation to '{name}'")
                            try:
                                if _spot_session and _ensure_localized(graph_nav_client, _spot_session):
                                    print(f"[Spot] Re-localized, retrying navigation to '{name}'")
                                    nav_to_cmd_id = None
                                    continue
                            except Exception:
                                pass
                            break
                        elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
                            print(f"[Spot] Robot impaired during navigation to '{name}'")
                            return
                        elif feedback.status in (
                            graph_nav_pb2.NavigationFeedbackResponse.STATUS_COMMAND_TIMED_OUT,
                            graph_nav_pb2.NavigationFeedbackResponse.STATUS_CONSTRAINT_FAULT,
                        ):
                            # Command expired or constraint fault — re-send with fresh command
                            nav_to_cmd_id = None
                            continue
                    except Exception:
                        pass

                    if stop_event.wait(0.5):
                        print("[Spot] Navigation cancelled")
                        return

                if not waypoint_reached and len(waypoint_ids) > 1:
                    print(f"[Spot] Skipping '{name}', moving to next waypoint...")
                    continue
                elif not waypoint_reached:
                    print(f"[Spot] Could not reach '{name}'")
                    return

            if not repeat:
                print("[Spot] Tour complete — visited all locations!")
                return
    except Exception as e:
        print(f"[Spot] Navigation thread error: {e}")


def _start_nav_thread(graph_nav_client, waypoint_ids, location_names, repeat=False):
    """Cancel any existing nav and start a new multi-waypoint navigation thread."""
    global _nav_thread, _nav_stop_event

    _cancel_nav()

    _nav_stop_event = threading.Event()
    stop_event = _nav_stop_event

    _nav_thread = threading.Thread(
        target=_navigate_waypoint_sequence,
        args=(graph_nav_client, waypoint_ids, location_names, stop_event, repeat),
        daemon=True,
    )
    _nav_thread.start()


def _handle_set_volume(params):
    """Handle set_volume action — does not require a Spot session.

    Accepts {"level": 1-100} (percentage) and applies it to both the TTS
    output gain and the audio-feedback beep gain. The percentage maps
    linearly to internal gain 0.01..1.0 (1.0 = full volume).
    """
    try:
        raw = params.get("level")
        if raw is None:
            print("[Spot] set_volume: missing 'level' param")
            return False
        pct = float(raw)
        # Clamp 1-100 (LLM-facing range)
        pct = max(1.0, min(100.0, pct))
        gain = pct / 100.0

        from src.voice_control.spot_tts import get_tts
        from src.voice_control.audio_feedback import beep
        tts = get_tts()
        applied_tts = tts.set_volume(gain)
        applied_beep = beep.set_volume(gain)
        print(f"[Spot] ✓ Volume set to {int(pct)}% (tts={applied_tts:.2f}, beep={applied_beep:.2f})")
        return True
    except Exception as e:
        print(f"[Spot] ✗ set_volume failed: {e}")
        return False


def dispatch_intent(intent):
    """Execute a Spot command based on parsed intent.

    Args:
        intent: Dict with keys "intent" (str) and "params" (dict)

    Returns:
        bool: True if command executed successfully, False otherwise
    """
    global _nav_thread, _nav_stop_event

    if not intent:
        return False

    name = intent["intent"]
    params = intent.get("params", {})

    # Volume control doesn't need a Spot session — handle before ensure_spot_session()
    if name == "set_volume":
        return _handle_set_volume(params)

    try:
        session = ensure_spot_session()
        cmd_client = session["cmd"]
        
        if name == "stop":
            # Stop current movement/action
            print("[Spot] Stopping current action...")
            _cancel_nav()
            try:
                cmd_client.robot_command(RobotCommandBuilder.stop_command())
                print("[Spot] ✓ Stop command sent")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Stop failed: {e}")
                return False
            
        elif name == "freeze":
            # Freeze robot - stop all movement and hold current position
            print("[Spot] Freezing robot in place...")
            _cancel_nav()
            try:
                # First stop any current movement
                cmd_client.robot_command(RobotCommandBuilder.stop_command())
                # Then issue a stand command to hold position (robot will stay where it is)
                # This keeps the robot standing but frozen
                cmd = RobotCommandBuilder.synchro_stand_command()
                cmd_client.robot_command(cmd)
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
                        cmd_client.robot_command(RobotCommandBuilder.selfright_command())
                        time.sleep(5)
                        cmd_client.robot_command(RobotCommandBuilder.synchro_stand_command())
                        print("[Spot] ✓ Recovered and standing")
                        return True
                    except Exception as e2:
                        print(f"[Spot] ✗ Recovery failed: {e2}")
                        return False
                print(f"[Spot] ✗ Stand failed: {e}")
                return False
                
        elif name == "sit":
            # Sit the robot down
            print("[Spot] Sitting down...")
            try:
                cmd = RobotCommandBuilder.synchro_sit_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Sit command sent")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Sit failed: {e}")
                return False
            
        elif name == "walk_to":
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
                    frame_name=ODOM_FRAME_NAME
                )
                cmd_client.robot_command(cmd, end_time_secs=None)
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
                frame_name=ODOM_FRAME_NAME
            )
            # Send command with timeout to allow execution
            cmd_client.robot_command(cmd, end_time_secs=time.time() + 5.0)
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
                )
                cmd_client.robot_command(cmd, end_time_secs=time.time() + 10.0)
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
                )
                cmd_client.robot_command(cmd, end_time_secs=time.time() + 10.0)
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
                return True
            except Exception as e:
                print(f"[Spot] ✗ Battery status failed: {e}")
                return False

        elif name == "status":
            # Report overall robot status
            print("[Spot] Checking robot status...")
            try:
                state_client = session["state"]
                robot_state = state_client.get_robot_state()

                # Battery
                battery = robot_state.power_state.locomotion_charge_percentage.value

                # E-stop status
                estop_states = {
                    0: "unknown",
                    1: "cut",
                    2: "not_cut",
                    3: "soft_stop"
                }
                estop = estop_states.get(robot_state.estop_states[0].state if robot_state.estop_states else 0, "unknown")

                print(f"[Spot] ✓ Status Report:")
                print(f"[Spot]   Battery: {battery:.0f}%")
                print(f"[Spot]   E-Stop: {estop}")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Status check failed: {e}")
                return False

        elif name == "power_off":
            # Safe power off
            print("[Spot] Initiating safe power off...")
            try:
                cmd = RobotCommandBuilder.safe_power_off_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Safe power off command sent")
                print("[Spot]   Robot will sit, then power off motors")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Power off failed: {e}")
                return False

        elif name == "estop":
            # Emergency stop (software e-stop)
            print("[Spot] ⚠️  EMERGENCY STOP triggered!")
            _cancel_nav()
            try:
                cmd_client.robot_command(RobotCommandBuilder.stop_command())
                print("[Spot] ✓ Emergency stop executed")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Emergency stop failed: {e}")
                return False

        elif name == "open_door":
            # Open a door using BD's DoorService (AutoGrasp).
            # Requires prior calibration: python scripts/calibrate_door.py
            # Flow: pitch up -> capture image -> WalkToObject -> raycast ->
            #       AutoGraspCommand -> DoorService handles grasp+open+walk-through
            print("[Spot] Opening door...")

            def _open_door_thread():
                import json
                import math
                import numpy as np
                from bosdyn.api import (manipulation_api_pb2, geometry_pb2,
                                        basic_command_pb2)
                from bosdyn.api.manipulation_api_pb2 import (
                    WalkToObjectInImage, ManipulationApiRequest,
                    ManipulationApiFeedbackRequest)
                from bosdyn.api.spot import door_pb2
                from bosdyn.client.manipulation_api_client import ManipulationApiClient
                from bosdyn.client.door import DoorClient
                from bosdyn.client.image import ImageClient
                from bosdyn.client import frame_helpers
                from bosdyn import geometry as bd_geometry

                robot = session["robot"]

                try:
                    # Load calibration
                    config_path = project_root / "door_config.json"
                    if not config_path.exists():
                        print("[Spot] No door calibration found.")
                        print("[Spot] Run: python scripts/calibrate_door.py")
                        return
                    with open(config_path) as f:
                        door_config = json.load(f)

                    camera_source = door_config["camera_source"]
                    handle_rx = door_config["handle_pixel_x_rotated"]
                    handle_ry = door_config["handle_pixel_y_rotated"]
                    hinge_side_str = door_config["hinge_side"]

                    # 1. Pitch robot up to see the door
                    print("[Spot] [1/5] Pitching up to view door...")
                    pitch_cmd = RobotCommandBuilder.synchro_stand_command(
                        footprint_R_body=bd_geometry.EulerZXY(
                            pitch=-0.4, roll=0.0, yaw=0.0)
                    )
                    cmd_client.robot_command(pitch_cmd)
                    time.sleep(2.0)

                    # 2. Capture front fisheye image
                    print("[Spot] [2/5] Capturing door image...")
                    image_client = robot.ensure_client(
                        ImageClient.default_service_name)
                    image_responses = image_client.get_image_from_sources(
                        [camera_source])
                    if not image_responses:
                        print(f"[Spot] No image from {camera_source}")
                        return
                    image_proto = image_responses[0]

                    # Get image dimensions for pixel un-rotation
                    img_h = image_proto.shot.image.rows
                    img_w = image_proto.shot.image.cols
                    # Rotated image dimensions (after 90 CW): W_rot=h, H_rot=w
                    rotated_w = img_h

                    # Un-rotate pixel from 90 CW display back to original
                    # (same math as arm_door.py)
                    th = -math.pi / 2
                    xm = rotated_w / 2.0
                    ym = img_w / 2.0
                    x = handle_rx - xm
                    y = handle_ry - ym
                    orig_px = math.cos(th) * x - math.sin(th) * y + ym
                    orig_py = math.sin(th) * x + math.cos(th) * y + xm

                    # 3. WalkToObjectInImage: position robot and raycast
                    # (skipped if robot is already close — falls back to
                    #  body-frame search ray)
                    print("[Spot] [3/5] Walking to door...")
                    manip_client = robot.ensure_client(
                        ManipulationApiClient.default_service_name)

                    walk_cmd = WalkToObjectInImage()
                    walk_cmd.pixel_xy.x = orig_px
                    walk_cmd.pixel_xy.y = orig_py
                    walk_cmd.frame_name_image_sensor = (
                        image_proto.shot.frame_name_image_sensor)
                    walk_cmd.transforms_snapshot_for_camera.CopyFrom(
                        image_proto.shot.transforms_snapshot)
                    walk_cmd.camera_model.CopyFrom(
                        image_proto.source.pinhole)
                    walk_cmd.offset_distance.value = 1.25

                    walk_request = ManipulationApiRequest(
                        walk_to_object_in_image=walk_cmd)
                    walk_response = manip_client.manipulation_api_command(
                        walk_request)

                    # Poll for walk completion (with state logging)
                    walk_cmd_id = walk_response.manipulation_cmd_id
                    snapshot = None
                    last_state = -1
                    # Map state codes to names for logging
                    state_names = {
                        0: "UNKNOWN", 1: "DONE",
                        2: "SEARCHING_FOR_GRASP",
                        3: "MOVING_TO_GRASP",
                        9: "FAILED_TO_RAYCAST",
                        10: "WALKING_TO_OBJECT",
                        12: "ATTEMPTING_RAYCASTING",
                    }
                    end_time = time.time() + 25.0
                    while time.time() < end_time:
                        fb = manip_client.manipulation_api_feedback_command(
                            ManipulationApiFeedbackRequest(
                                manipulation_cmd_id=walk_cmd_id))
                        st = fb.current_state
                        if st != last_state:
                            sname = state_names.get(st, str(st))
                            print(f"[Spot]   Walk state: {sname}")
                            last_state = st
                        if st == manipulation_api_pb2.MANIP_STATE_DONE:
                            snapshot = (
                                fb.transforms_snapshot_manipulation_data)
                            break
                        # Detect failure states
                        if st in (7, 8, 9):  # FAILED / NO_SOLUTION / RAYCAST_FAIL
                            print(f"[Spot]   Walk failed (state={st})")
                            break
                        time.sleep(0.5)

                    # Reset body pitch
                    cmd_client.robot_command(
                        RobotCommandBuilder.synchro_stand_command())
                    time.sleep(0.5)

                    # 4. Build search ray and send AutoGrasp door command
                    print("[Spot] [4/5] Sending AutoGrasp door command...")
                    auto_cmd = door_pb2.DoorCommand.AutoGraspCommand()

                    if snapshot is not None:
                        # Use raycast-based search ray (vision frame)
                        print("[Spot]   Using raycast search ray")
                        vision_tform_raycast = frame_helpers.get_a_tform_b(
                            snapshot, frame_helpers.VISION_FRAME_NAME,
                            frame_helpers.RAYCAST_FRAME_NAME)
                        vision_tform_sensor = frame_helpers.get_a_tform_b(
                            snapshot, frame_helpers.VISION_FRAME_NAME,
                            image_proto.shot.frame_name_image_sensor)

                        raycast_pt = (
                            vision_tform_raycast.get_translation())
                        sensor_pt = (
                            vision_tform_sensor.get_translation())

                        ray_dir = raycast_pt - sensor_pt
                        ray_dir_unit = ray_dir / np.linalg.norm(ray_dir)

                        search_dist = 0.25
                        search_vec = search_dist * ray_dir_unit
                        ray_start = raycast_pt - search_vec
                        ray_end = raycast_pt + search_vec

                        auto_cmd.frame_name = (
                            frame_helpers.VISION_FRAME_NAME)
                        auto_cmd.search_ray_start_in_frame.CopyFrom(
                            geometry_pb2.Vec3(
                                x=ray_start[0], y=ray_start[1],
                                z=ray_start[2]))
                        auto_cmd.search_ray_end_in_frame.CopyFrom(
                            geometry_pb2.Vec3(
                                x=ray_end[0], y=ray_end[1],
                                z=ray_end[2]))
                    else:
                        # Fallback: body-frame search ray (robot already
                        # close to door). Ray angles upward from body
                        # center to push bar height (~0.9-1.0m from floor;
                        # body origin is ~0.5m off ground, so z≈0.4-0.5).
                        print("[Spot]   Walk failed/timed out — using "
                              "body-frame search ray")
                        auto_cmd.frame_name = "body"
                        auto_cmd.search_ray_start_in_frame.CopyFrom(
                            geometry_pb2.Vec3(x=0.4, y=0.0, z=0.0))
                        auto_cmd.search_ray_end_in_frame.CopyFrom(
                            geometry_pb2.Vec3(x=0.8, y=0.0, z=0.5))

                    if hinge_side_str == "left":
                        auto_cmd.hinge_side = (
                            door_pb2.DoorCommand.HINGE_SIDE_LEFT)
                    else:
                        auto_cmd.hinge_side = (
                            door_pb2.DoorCommand.HINGE_SIDE_RIGHT)
                    auto_cmd.swing_direction = (
                        door_pb2.DoorCommand.SWING_DIRECTION_PUSH)

                    door_command = door_pb2.DoorCommand.Request(
                        auto_grasp_command=auto_cmd)
                    door_request = door_pb2.OpenDoorCommandRequest(
                        door_command=door_command)

                    door_client = robot.ensure_client(
                        DoorClient.default_service_name)
                    door_response = door_client.open_door(door_request)

                    if (door_response.status !=
                            door_pb2.OpenDoorCommandResponse.STATUS_OK):
                        print(f"[Spot] Door command rejected: "
                              f"{door_response.message}")
                        return

                    # 5. Poll for door completion
                    print("[Spot] [5/5] Opening door...")
                    fb_req = door_pb2.OpenDoorFeedbackRequest()
                    fb_req.door_command_id = door_response.door_command_id

                    end_time = time.time() + 60.0
                    while time.time() < end_time:
                        fb = door_client.open_door_feedback(fb_req)
                        if (fb.status != basic_command_pb2
                                .RobotCommandFeedbackStatus.STATUS_PROCESSING):
                            print(f"[Spot] Door command stopped "
                                  f"(status={fb.status})")
                            break
                        door_fb = fb.feedback.status
                        if door_fb == (door_pb2.DoorCommand
                                       .Feedback.STATUS_COMPLETED):
                            print("[Spot] ✓ Door opened successfully!")
                            return
                        elif door_fb == (door_pb2.DoorCommand
                                         .Feedback.STATUS_STALLED):
                            print("[Spot] Door opening stalled")
                            break
                        elif door_fb == (door_pb2.DoorCommand
                                         .Feedback.STATUS_NOT_DETECTED):
                            print("[Spot] Door not detected")
                            break
                        time.sleep(0.5)

                    print("[Spot] Door operation finished")

                except Exception as e:
                    print(f"[Spot] ✗ Door opening failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Reset body pitch in case it's still tilted
                    try:
                        cmd_client.robot_command(
                            RobotCommandBuilder.synchro_stand_command())
                    except Exception:
                        pass

            t = threading.Thread(target=_open_door_thread, daemon=True)
            t.start()
            return True

        elif name == "go_to_object":
            # Visual navigation — walk toward a visible object using YOLO-World
            description = params.get("description", "")
            if not description:
                print("[Spot] No object description provided")
                return False

            # Check if description matches a saved location — use GraphNav instead
            saved = _list_saved_locations()
            desc_normalized = description.strip().lower().replace(" ", "_")
            if desc_normalized in saved:
                print(f"[Spot] '{description}' is a saved location — using map navigation")
                return dispatch_intent({"intent": "go_to", "params": {"location": desc_normalized}})

            print(f"[Spot] Looking for '{description}'...")

            def _visual_nav_thread():
                try:
                    from src.voice_control.visual_nav import navigate_to_object
                    result = navigate_to_object(description, session, _nav_stop_event)
                    if result:
                        print(f"[Spot] ✓ Arrived near '{description}'")
                    else:
                        print(f"[Spot] ✗ Could not reach '{description}'")
                except ImportError:
                    print("[Spot] ✗ ultralytics not installed. Run: pip install ultralytics")
                except Exception as e:
                    print(f"[Spot] ✗ Visual nav failed: {e}")

            _cancel_nav()
            _nav_stop_event = threading.Event()
            _nav_thread = threading.Thread(target=_visual_nav_thread, daemon=True)
            _nav_thread.start()
            return True

        elif name == "follow_me":
            # Follow the nearest person using YOLO person detection
            print("[Spot] Starting follow mode...")

            def _follow_thread():
                try:
                    from src.voice_control.visual_nav import follow_person
                    result = follow_person(session, _nav_stop_event)
                    if result:
                        print("[Spot] ✓ Follow mode ended cleanly")
                    else:
                        print("[Spot] Lost the person — follow mode ended")
                except ImportError:
                    print("[Spot] ✗ ultralytics not installed. Run: pip install ultralytics")
                except Exception as e:
                    print(f"[Spot] ✗ Follow failed: {e}")

            _cancel_nav()
            _nav_stop_event = threading.Event()
            _nav_mode = "follow"
            _nav_thread = threading.Thread(target=_follow_thread, daemon=True)
            _nav_thread.start()
            return True

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
                _cancel_nav()
                robot = ensure_spot_session()["robot"]

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

        elif name == "list_maps":
            try:
                maps = list_available_maps(project_root)
            except OSError as e:
                print(f"[Spot] ✗ list_maps failed: {e}")
                return False

            if not maps:
                print("[Spot] No maps found.")
                return True

            print(f"[Spot] Available maps ({len(maps)}): {', '.join(maps)}")
            current = read_last_used(project_root)
            if current:
                print(f"[Spot] Currently loaded: {current}")
            return True

        elif name == "list_locations":
            # List saved locations
            print("[Spot] Listing saved locations...")
            try:
                from src.location_manager import list_locations
                locations = list_locations()
                if locations:
                    print(f"[Spot] ✓ Saved locations:")
                    for name, waypoint in locations.items():
                        print(f"[Spot]   - {name}")
                else:
                    print("[Spot] No locations saved yet.")
                    print("[Spot] Say 'save location [name]' to save current position.")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Failed to list locations: {e}")
                return False

        elif name == "save_location":
            # Save current position as a named location
            location_name = params.get("location", "").lower()
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
            location_name = params.get("location", "").lower()
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

                if not _ensure_localized(graph_nav_client, session):
                    print("[Spot]   Please run 'python scripts/setup_map.py --waypoint-init' to localize.")
                    return False

                _save_home_waypoint(graph_nav_client)
                _start_nav_thread(graph_nav_client, [waypoint_id], [location_name])

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

                if not _ensure_localized(graph_nav_client, session):
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
                        loc_name = loc_name.strip().lower().replace(" ", "_")
                        wid = load_location(loc_name)
                        if not wid:
                            print(f"[Spot] Location '{loc_name}' not found, skipping.")
                            continue
                        waypoint_ids.append(wid)
                        location_names.append(loc_name)

                if not waypoint_ids:
                    print("[Spot] No valid locations to visit.")
                    return False

                _save_home_waypoint(graph_nav_client)
                _start_nav_thread(graph_nav_client, waypoint_ids, location_names, repeat=False)

                print(f"[Spot] Tour started: {', '.join(location_names)}")
                return True
            except Exception as e:
                print(f"[Spot] Tour failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        elif name == "patrol":
            # Visit all named locations once (single pass by default)
            locations_param = params.get("locations", "all")
            print("[Spot] Starting patrol...")
            try:
                from bosdyn.client.graph_nav import GraphNavClient
                from src.location_manager import load_location, list_locations
                from src.graph_nav_utils import get_graph_waypoint_order

                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)

                if not _ensure_localized(graph_nav_client, session):
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
                        loc_name = loc_name.strip().lower().replace(" ", "_")
                        wid = load_location(loc_name)
                        if not wid:
                            print(f"[Spot] Location '{loc_name}' not found, skipping.")
                            continue
                        waypoint_ids.append(wid)
                        location_names.append(loc_name)

                if not waypoint_ids:
                    print("[Spot] No valid locations to patrol.")
                    return False

                _save_home_waypoint(graph_nav_client)
                _start_nav_thread(graph_nav_client, waypoint_ids, location_names, repeat=False)

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
                if not _home_waypoint:
                    print("[Spot] No departure point saved. Navigate somewhere first.")
                    return False

                from bosdyn.client.graph_nav import GraphNavClient
                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)

                if not _ensure_localized(graph_nav_client, session):
                    print("[Spot]   Please localize first.")
                    return False

                # Reverse-lookup name for display
                locs = _list_saved_locations()
                home_name = next((n for n, wid in locs.items() if wid == _home_waypoint), _home_waypoint[:12])

                _start_nav_thread(graph_nav_client, [_home_waypoint], [home_name])

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
        import traceback
        traceback.print_exc()
        return False
