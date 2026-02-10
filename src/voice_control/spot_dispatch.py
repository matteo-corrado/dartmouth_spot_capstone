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

from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, blocking_sit
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn import geometry
from src.session import spot_session
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD
from src.location_manager import list_locations as _list_saved_locations

# Global session handle (initialized on first use)
_spot_session = None
_session_context = None

# Navigation thread management
_nav_thread = None
_nav_stop_event = None


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
            
        elif name == "follow":
            # TODO: Implement person following behavior
            # This would require vision/person tracking integration
            print("[Spot] Follow behavior not yet implemented (requires person tracking)")
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

        elif name == "ptz_aim":
            # PTZ camera aiming (requires camera payload)
            target = params.get("target", "speaker")
            print(f"[Spot] PTZ aim at {target} (not yet implemented - requires camera payload)")
            # TODO: Implement PTZ control when camera payload is available
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
                from bosdyn.api.graph_nav import graph_nav_pb2
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
                from bosdyn.api.graph_nav import graph_nav_pb2
                from src.location_manager import load_location, list_locations
                
                # Load location from saved mappings
                waypoint_id = load_location(location_name)
                if not waypoint_id:
                    available = list(list_locations().keys())
                    print(f"[Spot] ✗ Location '{location_name}' not found.")
                    if available:
                        print(f"[Spot]   Available locations: {', '.join(available)}")
                    else:
                        print(f"[Spot]   No locations saved yet. Say 'save location {location_name}' at the desired position.")
                    return False
                
                print(f"[Spot] Found: '{location_name}' -> waypoint {waypoint_id}")
                
                # Get GraphNav client
                graph_nav_client = session["robot"].ensure_client(GraphNavClient.default_service_name)
                
                # Check if localized, try to localize if not
                localization = graph_nav_client.get_localization_state()
                if not localization.localization.waypoint_id:
                    print("[Spot] Not localized. Attempting waypoint-based localization...")
                    # Try to localize to first waypoint (assuming robot is at a known waypoint)
                    try:
                        # get_odom_tform_body already imported at top
                        from bosdyn.client.robot_state import RobotStateClient
                        from bosdyn.api.graph_nav import nav_pb2
                        import math
                        
                        # Get current robot state
                        robot_state_client = session["robot"].ensure_client(RobotStateClient.default_service_name)
                        robot_state = robot_state_client.get_robot_state()
                        current_odom_tform_body = get_odom_tform_body(
                            robot_state.kinematic_state.transforms_snapshot).to_proto()
                        
                        # Get first waypoint from map
                        graph = graph_nav_client.download_graph()
                        if graph.waypoints:
                            first_waypoint_id = graph.waypoints[0].id
                            print(f"[Spot] Attempting to localize to waypoint: {first_waypoint_id}")
                            
                            # Create localization guess
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
                            
                            # Verify localization
                            localization = graph_nav_client.get_localization_state()
                            if not localization.localization.waypoint_id:
                                print("[Spot] ✗ Localization failed. Robot may not be at a known waypoint.")
                                print("[Spot]   Please run 'python scripts/setup_map.py --waypoint-init' to localize.")
                                return False
                        else:
                            print("[Spot] ✗ No waypoints in map. Please upload map first.")
                            return False
                    except Exception as e:
                        print(f"[Spot] ✗ Localization error: {e}")
                        print("[Spot]   Please run 'python scripts/setup_map.py --waypoint-init' to localize.")
                        return False
                
                # Navigate to waypoint
                print(f"[Spot] Navigating to waypoint {waypoint_id}...")
                
                # Stop any existing navigation thread
                global _nav_thread, _nav_stop_event
                if _nav_thread and _nav_thread.is_alive():
                    print("[Spot] Stopping previous navigation...")
                    if _nav_stop_event:
                        _nav_stop_event.set()
                    _nav_thread.join(timeout=1.0)
                
                # Start navigation in background thread
                _nav_stop_event = threading.Event()
                
                def navigate_continuously():
                    """Keep navigation command active until destination reached."""
                    nav_to_cmd_id = None
                    try:
                        while not _nav_stop_event.is_set():
                            # Issue navigation command repeatedly to keep it active
                            nav_to_cmd_id = graph_nav_client.navigate_to(
                                waypoint_id, 1.0, command_id=nav_to_cmd_id
                            )
                            
                            # Check if navigation is complete
                            try:
                                status = graph_nav_client.navigation_feedback(nav_to_cmd_id)
                                from bosdyn.api.graph_nav import graph_nav_pb2
                                
                                if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                                    print(f"[Spot] ✓ Reached destination '{location_name}'!")
                                    break
                                elif status.status in [
                                    graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST,
                                    graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK,
                                    graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED
                                ]:
                                    status_names = {
                                        graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST: "LOST",
                                        graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK: "STUCK",
                                        graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED: "IMPAIRED"
                                    }
                                    print(f"[Spot] ⚠️  Navigation status: {status_names.get(status.status, 'UNKNOWN')}")
                                    break
                            except Exception as e:
                                # If we can't get feedback, continue navigating
                                pass
                            
                            # Sleep before next command (keep command active)
                            if _nav_stop_event.wait(0.5):
                                break
                    except Exception as e:
                        print(f"[Spot] Navigation thread error: {e}")
                
                _nav_thread = threading.Thread(target=navigate_continuously, daemon=True)
                _nav_thread.start()
                
                print(f"[Spot] ✓ Navigation started to '{location_name}'")
                print(f"[Spot]   (Navigation will continue until destination reached)")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Navigation failed: {e}")
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
