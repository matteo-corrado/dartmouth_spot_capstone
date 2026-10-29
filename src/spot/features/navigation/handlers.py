"""Navigation feature handlers (moved from spot_dispatch._dispatch_core):
walk_to, go_to, tour, patrol, come_back, load_map, save_location,
list_maps, list_locations.

GraphNav plumbing (navigate_sequence, start_thread, ensure_localized,
save_home_waypoint) stays in src/spot/nav/runner.py; handlers reach it via
_r.NAV. The owner singletons are looked up at call time (import inside the
functions: src.spot.robot must not be imported when the feature loads).

Each handler gets (params, ctx). Errors that escape a handler are caught by
dispatcher._dispatch_feature, the same catch-all the legacy branches had.
"""
import pathlib
import re
import time

from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.math_helpers import SE2Pose
from src.spot.nav.location_manager import (
    list_locations as _list_saved_locations,
    normalize_location_name as _normalize_location_name,
)
from src.spot.nav.map_loader import list_available_maps, read_last_used, resolve_map_path, write_last_used
from src.spot.nav.graph_nav_utils import (
    upload_graph_and_snapshots,
    initialize_localization,
    capped_mobility_params,
)

project_root = pathlib.Path(__file__).resolve().parents[4]


def _say(speak, text: str):
    from src.spot.robot.dispatcher import _say as say
    say(speak, text)


def _session():
    """(session, cmd_client), as the legacy branches got them before the branch."""
    import src.spot.robot as _r
    session = _r.CONNECTION.ensure()
    return session, session["cmd"]


def _is_named_location(name: str) -> bool:
    """Return True if the location has a meaningful human-assigned name.

    Filters out auto-generated names like 'waypoint_6' or 'localize_-_1'.
    """
    return not re.match(r"^(waypoint_\d+|localize_)", name)


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


# ---------------------------------------------------------------- registry handlers
def handle_list_maps(params, ctx):
    return _handle_list_maps(ctx.speak)


def handle_list_locations(params, ctx):
    return _handle_list_locations(ctx.speak)


def handle_walk_to(params, ctx):
    import src.spot.robot as _r
    session, cmd_client = _session()
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


def handle_load_map(params, ctx):
    import src.spot.robot as _r
    session, cmd_client = _session()
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


def handle_save_location(params, ctx):
    session, cmd_client = _session()
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


def handle_go_to(params, ctx):
    import src.spot.robot as _r
    session, cmd_client = _session()
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


def handle_tour(params, ctx):
    import src.spot.robot as _r
    session, cmd_client = _session()
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


def handle_patrol(params, ctx):
    import src.spot.robot as _r
    session, cmd_client = _session()
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


def handle_come_back(params, ctx):
    import src.spot.robot as _r
    session, cmd_client = _session()
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
