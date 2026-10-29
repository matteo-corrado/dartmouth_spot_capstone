"""Read-only robot state: the LLM state dict, camera frames, standing checks."""
import pathlib

from bosdyn.client.image import ImageClient

import src.spot.robot as _robot
from src.spot.nav.location_manager import list_locations as _list_saved_locations
from src.spot.nav.map_loader import list_available_maps, read_last_used

# src/spot/robot/state.py -> repo root (same root spot_dispatch uses)
project_root = pathlib.Path(__file__).resolve().parents[3]

# Spot's e-stop hardware exposes a small enum on robot_state.estop_states[].state.
# Used both by get_robot_state_dict() (for the LLM context) and the "status"
# action handler (for the user-facing report) — keep the mapping in one place
# so they can't drift.
_ESTOP_STATE_NAMES = {0: "unknown", 1: "cut", 2: "not_cut", 3: "soft_stop"}

# Camera source names for Spot's fisheye cameras
CAMERA_SOURCES = {
    "front": "frontleft_fisheye_image",
    "front_left": "frontleft_fisheye_image",
    "front_right": "frontright_fisheye_image",
    "left": "left_fisheye_image",
    "right": "right_fisheye_image",
    "back": "back_fisheye_image",
}


def _current_tts_volume_percent():
    """Read current TTS volume as a 1-100 percent integer (best-effort)."""
    try:
        from src.spot.tts.speaker import _tts_instance
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
    # Disk-backed state (locations.json + maps/.last_used). Snapshot once
    # so the localization reverse-lookup below reuses the same dict instead
    # of re-reading locations.json a second time per LLM round.
    saved_locs = _list_saved_locations()

    state = {
        "battery_percent": "unknown",
        "estimated_runtime_minutes": "unknown",
        "is_powered": False,
        "is_standing": "unknown",
        "current_location": "unknown",
        "saved_locations": ", ".join(saved_locs.keys()) or "none",
        "estop_status": "unknown",
        "tts_volume_percent": _current_tts_volume_percent(),
        "available_maps": "unknown",
        "current_map": "unknown",
        "estop_holder": "unknown",
        "num_saved_locations": len(saved_locs),
    }

    try:
        maps = list_available_maps(project_root)
        state["available_maps"] = ", ".join(maps) if maps else "none"
        state["current_map"] = read_last_used(project_root) or "unknown"
    except OSError as e:
        print(f"[Spot] WARN: map state unavailable: {e}")

    if _robot.CONNECTION._session is None:
        state["session"] = "not connected"
        return state

    try:
        state_client = _robot.CONNECTION._session["state"]
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
        if robot_state.estop_states:
            state["estop_status"] = _ESTOP_STATE_NAMES.get(
                robot_state.estop_states[0].state, "unknown"
            )

        # estop_holder: names of endpoints currently asserting an e-stop cut.
        # An empty list means no endpoint is cutting power — robot is free to move.
        try:
            cutting = []
            for entry in robot_state.estop_states:
                if entry.state != entry.STATE_NOT_ESTOPPED:
                    cutting.append(entry.name)
            state["estop_holder"] = ", ".join(cutting) if cutting else "none"
        except Exception:
            state["estop_holder"] = "unknown"

    except Exception as e:
        state["state_error"] = str(e)

    # Current localization (GraphNav)
    try:
        from bosdyn.client.graph_nav import GraphNavClient

        gn = _robot.CONNECTION._session["robot"].ensure_client(
            GraphNavClient.default_service_name
        )
        loc = gn.get_localization_state()
        wp = loc.localization.waypoint_id
        if wp:
            # Reverse-lookup friendly name from the snapshot taken above.
            friendly = next(
                (name for name, wid in saved_locs.items() if wid == wp), None
            )
            state["current_location"] = friendly or wp
        else:
            state["current_location"] = "not localized"
    except Exception as e:
        # Surface localization failures via state_error so the LLM and
        # whoever is reading the logs can see why current_location is unknown.
        state["state_error"] = f"localization: {e}"

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
        session = _robot.CONNECTION.ensure()
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


def is_standing(session) -> bool:
    """True only if the robot reports STANDING/STEPPING (False on any doubt)."""
    try:
        return session["state"].get_robot_state().behavior_state.state in (3, 4)
    except Exception:
        return False


def standing_and_powered(session) -> bool:
    try:
        rs = session["state"].get_robot_state()
        powered = rs.power_state.motor_power_state == rs.power_state.STATE_ON
        standing = rs.behavior_state.state in (3, 4)  # STANDING, STEPPING
        return bool(powered and standing)
    except Exception as e:
        print(f"[Spot] emote: robot state unavailable: {e}")
        return False
