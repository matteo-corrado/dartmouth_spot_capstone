"""Visual-nav feature handlers (moved from spot_dispatch._dispatch_core):
go_to_object, check_obstacles, plus the describe declaration.

The owner singletons are looked up at call time (import inside the
functions: src.spot.robot must not be imported when the feature loads).
Errors that escape a handler are caught by dispatcher._dispatch_feature,
the same catch-all the legacy branches had.
"""
import time

from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.math_helpers import SE2Pose
from src.spot.nav.location_manager import list_locations as _list_saved_locations
from src.spot.nav.graph_nav_utils import capped_mobility_params


def _say(speak, text: str):
    from src.spot.robot.dispatcher import _say as say
    say(speak, text)


def _session():
    """(session, cmd_client), as the legacy branches got them before the branch."""
    import src.spot.robot as _r
    session = _r.CONNECTION.ensure()
    return session, session["cmd"]


def handle_go_to_object(params, ctx):
    import src.spot.robot as _r
    from src.spot.robot import dispatcher as _dispatcher
    from src.spot.robot import motion as _motion
    session, cmd_client = _session()
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
        return _dispatcher._dispatch_core({"intent": "go_to", "params": {"location": desc_normalized}})

    # (2) world-object shortcut — fiducials and docks
    try:
        from src.spot.perception.world_objects import find_navigation_target
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
            from src.spot.features.visual_nav.navigator import navigate_to_object
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


def handle_check_obstacles(params, ctx):
    speak = ctx.speak
    session, cmd_client = _session()
    # Query Spot's native LocalGrid obstacle field via the
    # perception module. Read-only — never commands motion.
    # Returns structured (distance, direction) so the LLM can
    # verbalize it. On a base Spot the grid may not be available
    # at all (probe script reports this) — in that case we log
    # and return True since "no obstacles reported" is a valid
    # answer, not a failure.
    try:
        from src.spot.perception.obstacle_query import (
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


def handle_describe(params, ctx):
    """describe is run by client_mic's VLM branch and never dispatched there.
    spot_dispatch had no describe branch, so a dispatched describe got a
    session and then fell to the unknown-intent line. Same here."""
    _session()
    print("[Spot] Unknown intent: describe")
    return False
