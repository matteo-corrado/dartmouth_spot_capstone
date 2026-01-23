"""Spot command dispatcher for voice intents.

This module executes Spot commands based on parsed voice intents.
Uses a persistent session connection that stays open for multiple commands.
"""
import sys
import pathlib
import time

# Add project root to path to enable imports
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, blocking_sit
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
from bosdyn.client.math_helpers import SE2Pose
from src.session import spot_session
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD

# Global session handle (initialized on first use)
_spot_session = None
_session_context = None


def ensure_spot_session():
    """Ensure Spot session is initialized and return session dict."""
    global _spot_session, _session_context
    
    if _spot_session is None:
        # Start a persistent session (don't sit/power off on exit for voice control)
        _session_context = spot_session(
            stand_on_enter=True,
            sit_on_exit=False  # Keep robot standing for voice commands
        )
        _spot_session = _session_context.__enter__()
        print("[Spot] Connected and standing ready for voice commands.")
    
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
                    print("[Spot] Not localized. Attempting to localize to nearest fiducial...")
                    graph_nav_client.set_localization(fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL)
                    localization = graph_nav_client.get_localization_state()
                    if not localization.localization.waypoint_id:
                        print("[Spot] ✗ Localization failed. Please localize manually first.")
                        print("[Spot]   Use GraphNav tools to localize, or ensure robot sees a fiducial.")
                        return False
                
                # Navigate to waypoint
                print(f"[Spot] Navigating to waypoint {waypoint_id}...")
                nav_feedback = graph_nav_client.navigate_to(waypoint_id=waypoint_id)
                print(f"[Spot] ✓ Navigation command sent to '{location_name}'")
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
