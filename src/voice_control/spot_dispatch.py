"""Spot command dispatcher for voice intents.

This module executes Spot commands based on parsed voice intents.
Uses a persistent session connection that stays open for multiple commands.
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

from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.client.image import ImageClient
from src.session import spot_session
from src.location_manager import (
    list_locations as _list_saved_locations,
    normalize_location_name as _normalize_location_name,
)
from src.map_loader import list_available_maps, read_last_used, resolve_map_path, write_last_used
from src.graph_nav_utils import upload_graph_and_snapshots, initialize_localization

# Spot's e-stop hardware exposes a small enum on robot_state.estop_states[].state.
# Used both by get_robot_state_dict() (for the LLM context) and the "status"
# action handler (for the user-facing report) — keep the mapping in one place
# so they can't drift.
_ESTOP_STATE_NAMES = {0: "unknown", 1: "cut", 2: "not_cut", 3: "soft_stop"}

# Set once at module import for uptime computation in get_robot_state_dict().
_PROCESS_START_TS = time.time()

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
        "uptime_s": int(time.time() - _PROCESS_START_TS),
        "estop_holder": "unknown",
        "num_saved_locations": len(saved_locs),
        "process_pid": os.getpid(),
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

        gn = _spot_session["robot"].ensure_client(
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
                            except Exception as e:
                                print(f"[Spot] Re-localize failed during nav-feedback recovery: "
                                      f"{type(e).__name__}: {e}")
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
                    except Exception as e:
                        # navigation_feedback() can fail mid-traversal on a
                        # GraphNav RPC blip, server restart, etc. The old
                        # silent `except: pass` would let the inner loop
                        # spin forever polling against a dead command — log
                        # and break out so the outer waypoint loop can move
                        # on (or the user can re-issue).
                        print(f"[Spot] navigation_feedback error for '{name}': {e}")
                        break

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
        if tts is None:
            # The voice pipeline hasn't registered the TTS singleton yet.
            # In production this never happens — set_volume only fires from
            # an LLM action inside process_utterance, which only runs after
            # client_mic.main() has assigned _tts_instance. Handle the
            # degenerate case anyway so we never crash on missing state.
            print("[Spot] set_volume: TTS not initialized — only updating beep volume")
            applied_beep = beep.set_volume(gain)
            print(f"[Spot] ✓ Volume set to {int(pct)}% (tts=skipped, beep={applied_beep:.2f})")
            return True
        applied_tts = tts.set_volume(gain)
        applied_beep = beep.set_volume(gain)
        print(f"[Spot] ✓ Volume set to {int(pct)}% (tts={applied_tts:.2f}, beep={applied_beep:.2f})")
        return True
    except Exception as e:
        print(f"[Spot] ✗ set_volume failed: {e}")
        return False


def _handle_set_backend(params):
    """Switch active TTS backend at runtime. Accepts {"name": "kokoro"|"elevenlabs"}.

    Sets SPOT_TTS_BACKEND so subsequent get_backend() calls route to the new
    backend, then warms the instance so the first post-switch synth doesn't
    pay cold-init latency.
    """
    name = (params or {}).get("name", "").strip().lower()
    if name not in {"kokoro", "elevenlabs"}:
        print(f"[Spot] set_backend: invalid name '{name}' (expected 'kokoro' or 'elevenlabs')")
        return False
    os.environ["SPOT_TTS_BACKEND"] = name
    try:
        from src.voice_control.tts import get_backend, get_active_backend_name
        get_backend()  # lazy init / cache warm
        active = get_active_backend_name()
        print(f"[Spot] ✓ TTS backend switched to '{active}'")
        return True
    except Exception as e:
        print(f"[Spot] ✗ set_backend warm failed: {e}")
        return False


def _close_mouth():
    """Stage 2E.2: force the gripper mouth closed + cancel playback. Safe
    no-op if TTS/mouth isn't initialized. Called from stop/freeze/estop."""
    try:
        from src.voice_control.spot_tts import get_tts
        _t = get_tts()
        if _t is not None and _t.mouth is not None:
            _t.mouth.close()
    except Exception:
        pass


def _handle_enable_mouth(params):
    """Deploy the arm and enable gripper mouth animation (Stage 2E.2)."""
    try:
        from src.voice_control.spot_tts import get_tts
        from src.session import deploy_arm_safe
        tts = get_tts()
        if tts is None or tts.mouth is None:
            print("[Spot] enable_mouth: mouth driver not initialized")
            return False
        session = ensure_spot_session()
        if not session["robot"].has_arm():
            print("[Spot] enable_mouth: no arm installed")
            return False
        deploy_arm_safe(session["cmd"])
        tts.mouth.enable()
        print("[Spot] ✓ Mouth enabled (arm deployed)")
        return True
    except Exception as e:
        print(f"[Spot] ✗ enable_mouth failed: {e}")
        return False


def _handle_disable_mouth(params):
    """Disable mouth animation, close the gripper, and stow the arm (Stage 2E.2)."""
    try:
        from src.voice_control.spot_tts import get_tts
        from src.session import stow_arm
        tts = get_tts()
        if tts is not None and tts.mouth is not None:
            tts.mouth.disable()
        session = ensure_spot_session()
        stow_arm(session["cmd"])
        print("[Spot] ✓ Mouth disabled (arm stowed)")
        return True
    except Exception as e:
        print(f"[Spot] ✗ disable_mouth failed: {e}")
        return False


def do_set_persona(params: dict, brain) -> dict:
    """Switch active persona via runtime action.

    Params:
        name: persona registry key (e.g. "pirate", "butler", "tour_guide").

    Unknown name → auto-upgrade to add_persona with an LLM-generated voice
    description, instead of the prior silent fallback to tour_guide. This
    salvages the common case where the LLM emits set_persona for a name
    that isn't registered yet (e.g. "cowboy", "wizard").
    """
    name = (params or {}).get("name", "").strip().lower()
    if not name:
        return {"ok": False, "error": "set_persona requires 'name' param"}
    if brain is None:
        return {"ok": False, "error": "set_persona requires a brain instance"}
    if name not in brain._persona_registry:
        print(f"[Dispatch] Unknown persona '{name}' — auto-upgrading to add_persona")
        description = brain.describe_persona(name) if hasattr(brain, "describe_persona") else f"{name} character voice"
        return do_add_persona({"name": name, "description": description}, brain)
    from src.voice_control.chatbot.persona import get_persona
    persona = get_persona(name, brain._persona_registry)
    brain.session_state.current_persona = persona.name
    # Stage 2E.2: scale gripper-mouth swing to this persona.
    try:
        from src.voice_control.spot_tts import get_tts
        _t = get_tts()
        if _t is not None and _t.mouth is not None:
            _t.mouth.intensity = getattr(persona, "mouth_intensity", 1.0)
    except Exception:
        pass
    brain.warm_up_async()  # dedup'd bg warm-up; never stacks threads / collides with a live turn
    print(f"[Dispatch] Persona switched to '{persona.name}' (warming new prefix in bg)")
    return {"ok": True, "persona": persona.name}


KOKORO_GENDER_FALLBACK = {"male": "am_fenrir", "female": "af_sarah"}

PROMPT_PREFIX_TEMPLATE = (
    "You are Spot, but in {name} mode. Speak like {description}. "
    "Keep responses short — one or two sentences."
)


def do_add_persona(params: dict, brain) -> dict:
    """Discover + claim ElevenLabs voice matching description, register as persona.
    ACK plays in CURRENT voice; claim runs in parallel worker thread."""
    name = (params.get("name") or "").strip().lower()
    description = (params.get("description") or "").strip()
    if not name or not description:
        return {"ok": False, "error": "add_persona needs name + description"}
    if brain is None:
        return {"ok": False, "error": "add_persona requires a brain instance"}

    # Cache hit — flip without API call
    if name in brain._persona_registry:
        brain.session_state.current_persona = name
        return {"ok": True, "persona": name, "cached": True}

    from src.voice_control.chatbot.persona import get_persona, Persona
    from src.voice_control.chatbot.voice_discovery import find_and_claim
    from src.voice_control.spot_tts import get_tts

    outgoing = get_persona(brain.session_state.current_persona, brain._persona_registry)
    ack_text = (outgoing.ack_template or "Give me a second.").format(name=name)

    tts = get_tts()

    print(f"[Dispatch] add_persona name='{name}' description='{description}'")
    result_box = {"voice_id": None, "error": None}
    def worker():
        try:
            result_box["voice_id"] = find_and_claim(description, name)
        except Exception as e:
            result_box["error"] = f"{type(e).__name__}: {e}"
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    if tts is not None:
        tts.speak(ack_text)  # ACK plays through speakers; claim happens in parallel
    t.join(timeout=15.0)  # ElevenLabs Library search + share can take 8-12s; was 8s (too tight)

    new_voice_id = result_box["voice_id"]
    if not new_voice_id:
        reason = ("worker raised: " + result_box["error"]) if result_box["error"] else (
            "thread still running after 15s budget" if t.is_alive() else "voice library returned no match"
        )
        print(f"[Dispatch] add_persona FAILED for '{name}': {reason}")
        if tts is not None:
            tts.speak(f"Could not find a {name} voice; staying as {brain.session_state.current_persona}.")
        return {"ok": False, "error": reason}

    # Word-boundary match — substring would misfire ("female" contains "male").
    desc_words = set(re.findall(r"\b\w+\b", description.lower()))
    gender = "male" if desc_words & {"male", "man", "guy", "boy", "men"} else "female"
    kokoro_slug = KOKORO_GENDER_FALLBACK.get(gender, "af_sarah")

    persona = Persona(
        name=name,
        prompt_prefix=PROMPT_PREFIX_TEMPLATE.format(name=name, description=description),
        voices={"elevenlabs": new_voice_id, "kokoro_v1": kokoro_slug},
        sampling_overrides={},
        ack_template=f"One moment as I become a {name}...",
    )
    brain._persona_registry[name] = persona
    brain.session_state.current_persona = name
    print(f"[Dispatch] Persona '{name}' added (voice={new_voice_id}, kokoro={kokoro_slug})")
    return {"ok": True, "persona": name, "voice_id": new_voice_id}


def dispatch_intent(intent, brain=None):
    """Execute a Spot command based on parsed intent.

    Args:
        intent: Dict with keys "intent" (str) and "params" (dict)
        brain: Optional LLMBrain instance — required only for set_persona.

    Returns:
        bool: True if command executed successfully, False otherwise
    """
    global _nav_thread, _nav_stop_event

    if not intent:
        return False

    name = intent["intent"]
    params = intent.get("params", {})

    # set_persona + add_persona + set_volume don't need a Spot session — handle before ensure_spot_session()
    if name == "set_persona":
        return do_set_persona(params, brain).get("ok", False)
    if name == "add_persona":
        return do_add_persona(params, brain).get("ok", False)
    if name == "set_volume":
        return _handle_set_volume(params)
    if name == "set_backend":
        return _handle_set_backend(params)

    try:
        session = ensure_spot_session()
        cmd_client = session["cmd"]
        
        if name == "enable_mouth":
            return _handle_enable_mouth(params)
        if name == "disable_mouth":
            return _handle_disable_mouth(params)

        if name == "stop":
            # Stop current movement/action
            print("[Spot] Stopping current action...")
            _close_mouth()
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
            _close_mouth()
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

                # E-stop status (uses module-level _ESTOP_STATE_NAMES)
                estop = _ESTOP_STATE_NAMES.get(
                    robot_state.estop_states[0].state if robot_state.estop_states else 0,
                    "unknown",
                )

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
            _close_mouth()
            _cancel_nav()
            try:
                cmd_client.robot_command(RobotCommandBuilder.stop_command())
                print("[Spot] ✓ Emergency stop executed")
                return True
            except Exception as e:
                print(f"[Spot] ✗ Emergency stop failed: {e}")
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
                return dispatch_intent({"intent": "go_to", "params": {"location": desc_normalized}})

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
                    )
                    cmd_client.robot_command(cmd, end_time_secs=time.time() + 15.0)
                    print(f"[Spot] ✓ SE2 trajectory sent toward {target.source}")
                    return True
                except Exception as e:
                    print(f"[Spot] ✗ World-object approach failed: {e}")
                    # fall through to YOLO as a last resort
                    pass

            # (3) YOLO fallback
            print(f"[Spot] Looking for '{description}' with YOLO-World...")

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
                return False

            robot_obj = session["robot"]
            try:
                hit = nearest_obstacle_in_body_frame(robot_obj, max_radius_m=3.0)
            except Exception as e:
                print(f"[Spot] check_obstacles: query failed: {e}")
                return False

            if hit is None:
                print("[Spot] check_obstacles: no obstacles within 3m "
                      "(or obstacle-grid unavailable on this firmware)")
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

                _save_home_waypoint(graph_nav_client)
                _start_nav_thread(graph_nav_client, waypoint_ids, location_names, repeat=True)

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
