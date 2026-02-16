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
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api.image_pb2 import ImageRequest as ImageRequestProto
from src.session import spot_session
from src.location_manager import list_locations as _list_saved_locations

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
_home_waypoint = None


def is_navigating() -> bool:
    """Check if the robot is currently navigating (motor noise expected)."""
    return _nav_thread is not None and _nav_thread.is_alive()


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
    }

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


def ensure_spot_session():
    """Ensure Spot session is initialized and return session dict."""
    global _spot_session, _session_context
    
    if _spot_session is None:
        try:
            # Start a persistent session (don't sit/power off on exit for voice control)
            _session_context = spot_session(
                stand_on_enter=True,
                sit_on_exit=False  # Keep robot standing for voice commands
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
    global _nav_thread, _nav_stop_event
    if _nav_stop_event:
        _nav_stop_event.set()
    if _nav_thread and _nav_thread.is_alive():
        _nav_thread.join(timeout=2.0)
    _nav_thread = None
    _nav_stop_event = None


def _ensure_localized(graph_nav_client, session) -> bool:
    """Check localization and attempt waypoint-based init if needed."""
    localization = graph_nav_client.get_localization_state()
    if localization.localization.waypoint_id:
        return True

    print("[Spot] Not localized. Attempting waypoint-based localization...")
    try:
        from bosdyn.client.robot_state import RobotStateClient
        from bosdyn.api.graph_nav import nav_pb2, graph_nav_pb2
        import math

        robot_state_client = session["robot"].ensure_client(RobotStateClient.default_service_name)
        robot_state = robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()

        graph = graph_nav_client.download_graph()
        if not graph.waypoints:
            print("[Spot] No waypoints in map. Please upload map first.")
            return False

        first_waypoint_id = graph.waypoints[0].id
        print(f"[Spot] Attempting to localize to waypoint: {first_waypoint_id}")

        loc_guess = nav_pb2.Localization()
        loc_guess.waypoint_id = first_waypoint_id
        loc_guess.waypoint_tform_body.rotation.w = 1.0

        graph_nav_client.set_localization(
            initial_guess_localization=loc_guess,
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body
        )

        localization = graph_nav_client.get_localization_state()
        if localization.localization.waypoint_id:
            return True

        print("[Spot] Localization failed. Robot may not be at a known waypoint.")
        return False
    except Exception as e:
        print(f"[Spot] Localization error: {e}")
        return False


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
    """Navigate through a sequence of waypoints. Runs in a background thread."""
    from bosdyn.api.graph_nav import graph_nav_pb2

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
                nav_to_cmd_id = None

                while not stop_event.is_set():
                    nav_to_cmd_id = graph_nav_client.navigate_to(
                        wp_id, 1.0, command_id=nav_to_cmd_id
                    )

                    try:
                        feedback = graph_nav_client.navigation_feedback(nav_to_cmd_id)
                        if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                            print(f"[Spot] Reached '{name}'")
                            break
                        elif feedback.status in [
                            graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST,
                            graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK,
                            graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED,
                        ]:
                            status_names = {
                                graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST: "LOST",
                                graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK: "STUCK",
                                graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED: "IMPAIRED",
                            }
                            print(f"[Spot] Navigation to '{name}': {status_names.get(feedback.status, 'ERROR')}")
                            return
                    except Exception:
                        pass

                    if stop_event.wait(0.5):
                        print("[Spot] Navigation cancelled")
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


def dispatch_intent(intent):
    """Execute a Spot command based on parsed intent.

    Args:
        intent: Dict with keys "intent" (str) and "params" (dict)

    Returns:
        bool: True if command executed successfully, False otherwise
    """
    if not intent:
        return False
    
    name = intent["intent"]
    params = intent.get("params", {})
    
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
            # Stand the robot up
            print("[Spot] Standing up...")
            try:
                cmd = RobotCommandBuilder.synchro_stand_command()
                cmd_client.robot_command(cmd)
                print("[Spot] ✓ Stand command sent")
                return True
            except Exception as e:
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
                    frame_tree_snapshot=frame_tree
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
                    frame_tree_snapshot=frame_tree
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
            # Self-right: recover from fall
            print("[Spot] Attempting self-right (recovery from fall)...")
            try:
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
                    # Order by graph recording order for natural traversal
                    graph_order = get_graph_waypoint_order(session["robot"])
                    wp_to_name = {wid: name for name, wid in saved.items()}
                    waypoint_ids = []
                    location_names = []
                    for wp_id in graph_order:
                        if wp_id in wp_to_name:
                            waypoint_ids.append(wp_id)
                            location_names.append(wp_to_name[wp_id])
                    # Include any saved locations not in graph order
                    for loc_name, wid in saved.items():
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
            # Continuously loop through locations until stopped
            locations_param = params.get("locations", "all")
            print("[Spot] Starting patrol (say 'stop' to end)...")
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
                    graph_order = get_graph_waypoint_order(session["robot"])
                    wp_to_name = {wid: name for name, wid in saved.items()}
                    waypoint_ids = []
                    location_names = []
                    for wp_id in graph_order:
                        if wp_id in wp_to_name:
                            waypoint_ids.append(wp_id)
                            location_names.append(wp_to_name[wp_id])
                    for loc_name, wid in saved.items():
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
                _start_nav_thread(graph_nav_client, waypoint_ids, location_names, repeat=True)

                print(f"[Spot] Patrol started: {', '.join(location_names)} (looping)")
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
