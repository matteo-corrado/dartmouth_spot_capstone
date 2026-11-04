"""walk / turn / strafe / body_height / set_speed: the branch bodies moved
from spot_dispatch._dispatch_core (Task 13), unchanged.

Called through dispatcher._dispatch_core -> _dispatch_feature (the point the
legacy branch was reached from), so the tuck before, the arm policy, the
abort checks and the restore watch in dispatcher._dispatch_locomotion wrap
these exactly as before. A raise is caught by _dispatch_feature's catch-all
(same print/traceback/False as the legacy chain's).

Everything is imported at call time: src.spot.features is imported (and the
registry loaded) from places that run during src.spot.robot's own import, so
a module-level import of the robot core here could form a cycle, and
Registry.load would then silently skip the feature.
"""


def _session():
    """The legacy chain's `session = _r.CONNECTION.ensure()` (these intents
    never get a session passed in)."""
    import src.spot.robot as _r
    return _r.CONNECTION.ensure()


def turn_rad(deg, direction):
    """Yaw change in radians. Spot's yaw is counter-clockwise positive
    (z up), so left is positive and right is negative."""
    import math
    rad = math.radians(abs(float(deg)))
    return rad if direction == "left" else -rad


def handle_turn(params, ctx):
    import time
    import src.spot.robot as _r
    from bosdyn.client.robot_command import RobotCommandBuilder
    from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
    from bosdyn.client.math_helpers import SE2Pose
    from src.spot.nav.graph_nav_utils import capped_mobility_params
    session = _session()
    cmd_client = session["cmd"]

    # Rotate in place
    deg = params.get("deg", 0)
    direction = params.get("dir", "left")

    rad = turn_rad(deg, direction)

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


def handle_walk(params, ctx):
    import time
    import src.spot.robot as _r
    from bosdyn.client.robot_command import RobotCommandBuilder
    from src.spot.nav.graph_nav_utils import capped_mobility_params, walk_speed_cap
    session = _session()
    cmd_client = session["cmd"]

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


def handle_strafe(params, ctx):
    import time
    import src.spot.robot as _r
    from bosdyn.client.robot_command import RobotCommandBuilder
    from src.spot.nav.graph_nav_utils import capped_mobility_params, walk_speed_cap
    session = _session()
    cmd_client = session["cmd"]

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


def handle_body_height(params, ctx):
    from bosdyn.client.robot_command import RobotCommandBuilder
    session = _session()
    cmd_client = session["cmd"]

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


def handle_set_speed(params, ctx):
    _session()["cmd"]  # the legacy chain brought up a session (and read "cmd") before every branch

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
