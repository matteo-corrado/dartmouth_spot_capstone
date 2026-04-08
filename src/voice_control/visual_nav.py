"""Visual navigation for Spot — YOLO object seeking + person following.

Uses ultralytics YOLO models:
- YOLO-World (open-vocabulary) for "go to the red chair" style commands
- YOLOv8n for person following (class 0 = person)

Both run on CPU to avoid GPU/VRAM competition with the LLM.

Install: pip install ultralytics
"""

import os
import time
import math
import io
import pathlib
import threading
import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

try:
    from PIL import Image
except ImportError:
    Image = None

from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FRONT_CAMERA = "frontleft_fisheye_image"
DEVICE = "cpu"          # Keep GPU free for LLM/VLM
CONF_THRESH = 0.25

# All cameras checked during initial object search (front first, then sides, then back)
# Each entry: (source_name, yaw_to_turn_radians) — yaw to face that direction
ALL_CAMERAS = [
    ("frontleft_fisheye_image",  0.0),           # front — already facing
    ("frontright_fisheye_image", 0.0),           # front right — same general direction
    ("left_fisheye_image",       math.radians(90)),   # left — turn 90° left
    ("right_fisheye_image",      math.radians(-90)),  # right — turn 90° right
    ("back_fisheye_image",       math.radians(180)),  # back — turn 180°
]

# Approach mode (go_to_object) — SDK manipulation API path (Tier 2a)
APPROACH_OFFSET_M = 0.8         # how close to stand from the target (WalkToObjectInImage)
APPROACH_TIMEOUT_S = 30.0       # max wall-clock for the SDK to complete the walk
APPROACH_POLL_S = 0.25          # feedback poll cadence

# Legacy visual-servo constants — only used if WalkToObjectInImage submission
# fails BEFORE the SDK takes over (e.g. ImageSource not pinhole). Kept here as
# a documented fallback path; removed entirely once the SDK path is proven
# stable on every camera.
APPROACH_STEP = 0.5             # meters forward per loop
APPROACH_DONE_FRAC = 0.50       # bbox height / image height → close enough
APPROACH_YAW_GAIN = 1.0         # steering proportional gain
APPROACH_MAX_YAW = 0.35         # max yaw correction per step (rad, ~20°)

# Follow mode — all cameras, front priority with depth
FOLLOW_FRONT_CAMS = [
    ("frontleft_fisheye_image",  "frontleft_depth_in_visual_frame"),
    ("frontright_fisheye_image", "frontright_depth_in_visual_frame"),
]
FOLLOW_SCAN_CAMS = [
    ("left_fisheye_image",       math.radians(90)),    # left — turn 90° left
    ("right_fisheye_image",      math.radians(-90)),   # right — turn 90° right
    ("back_fisheye_image",       math.radians(180)),   # back — turn 180°
]
FOLLOW_DIST_MARGIN = 1.5        # meters — desired distance from person
FOLLOW_CLOSE_BBOX = 0.55        # bbox height fraction — "close enough" (depth fallback)
FOLLOW_MAX_VEL = 0.8            # max forward velocity (m/s) — Spot walks up to ~1.6 m/s
FOLLOW_FAR_VEL = 0.4            # forward velocity when using bbox only (no depth)
FOLLOW_HOLD_EXIT = 80           # consecutive hold cycles → person reached, exit follow (~8s at 10Hz)
FOLLOW_LOCK_DIST = 0.3          # max normalized distance to match locked person (0-1 scale)
FOLLOW_CONTROL_HZ = 10          # velocity command rate (Hz)
FOLLOW_VEL_SMOOTH = 0.6         # velocity EMA smoothing (0=instant, 1=no change)

# Timing
MOVE_SETTLE = 1.5               # seconds to let a step complete
LOST_TIMEOUT = 10.0             # seconds without detection → give up

# ---------------------------------------------------------------------------
# Collision pre-check (Tier 4c of perception overhaul — feature flag)
# ---------------------------------------------------------------------------
# Enabled via env var ``SPOT_COLLISION_PRECHECK=1``. Off by default because
# the obstacle_distance grid semantics on this Spot have not yet been
# empirically verified — see plan Tier 4 risks. Failure mode is deliberately
# **fail-open**: if the check raises, times out, or returns None (no grid
# available), forward motion proceeds normally. Never brick the robot on a
# query failure.
COLLISION_PRECHECK_ENABLED = os.environ.get("SPOT_COLLISION_PRECHECK", "").lower() in (
    "1", "true", "yes", "on",
)
COLLISION_MIN_DISTANCE_M = 0.6   # forward obstacle closer than this blocks vx
COLLISION_QUERY_RADIUS_M = 2.0   # how far the obstacle query scans

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_world_model = None
_person_model = None


def _get_world_model():
    """Load YOLO-World for open-vocabulary detection (first call downloads ~28MB)."""
    global _world_model
    if _world_model is None:
        from ultralytics import YOLOWorld
        print("[VisualNav] Loading YOLO-World model...")
        _world_model = YOLOWorld(str(_PROJECT_ROOT / "yolov8s-worldv2.pt"))
        print("[VisualNav] YOLO-World ready")
    return _world_model


def _get_person_model():
    """Load YOLOv8n for person detection (first call downloads ~7MB)."""
    global _person_model
    if _person_model is None:
        from ultralytics import YOLO
        print("[VisualNav] Loading YOLOv8n model...")
        _person_model = YOLO(str(_PROJECT_ROOT / "yolov8n.pt"))
        print("[VisualNav] YOLOv8n ready")
    return _person_model


def preload_models():
    """Pre-load YOLO models in a background thread to avoid first-use latency.

    Called during startup. Both models run on CPU so they don't compete
    with the LLM for VRAM. Safe to call multiple times (lazy loaders
    short-circuit if already loaded).
    """
    import threading

    def _load():
        try:
            _get_person_model()
            # YOLO-World is heavy (~28MB, slow CPU init) and only needed for
            # "go to <object>" commands — lazy-load on first use instead of
            # blocking startup and starving the audio thread.
        except Exception as e:
            print(f"[VisualNav] Pre-load failed (will retry on first use): {e}")

    t = threading.Thread(target=_load, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Image capture + detection helpers
# ---------------------------------------------------------------------------
def _capture_rotated(session, source=None):
    """Capture fisheye image and rotate 90° CW for upright orientation."""
    if Image is None:
        print("[VisualNav] Pillow not installed")
        return None
    if source is None:
        source = FRONT_CAMERA
    try:
        robot = session["robot"]
        img_client = robot.ensure_client(ImageClient.default_service_name)
        resps = img_client.get_image_from_sources([source])
        if not resps:
            return None
        data = resps[0].shot.image.data
        img = Image.open(io.BytesIO(data))
        return img.transpose(Image.Transpose.ROTATE_270)  # 90° CW
    except Exception as e:
        print(f"[VisualNav] Capture failed: {e}")
        return None


def _find_object(img, description):
    """Detect described object with YOLO-World. Returns [x1,y1,x2,y2] or None."""
    model = _get_world_model()
    model.set_classes([description])
    results = model.predict(img, device=DEVICE, conf=CONF_THRESH, verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        best = boxes.conf.argmax().item()
        return boxes.xyxy[best].cpu().numpy()
    return None


def _find_person(img):
    """Detect nearest person with YOLOv8n. Returns [x1,y1,x2,y2] or None."""
    model = _get_person_model()
    results = model.predict(img, device=DEVICE, classes=[0], conf=CONF_THRESH, verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        # Pick largest bbox (closest person)
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best = areas.argmax().item()
        return boxes.xyxy[best].cpu().numpy()
    return None


def _capture_multi(session, sources):
    """Capture multiple image sources with explicit format control.

    Depth sources (name contains 'depth') get FORMAT_RAW + DEPTH_U16.
    Visual sources get FORMAT_JPEG for smaller transfer.
    """
    try:
        robot = session["robot"]
        img_client = robot.ensure_client(ImageClient.default_service_name)
        requests = []
        for src in sources:
            if "depth" in src:
                requests.append(image_pb2.ImageRequest(
                    image_source_name=src,
                    image_format=image_pb2.Image.FORMAT_RAW,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                ))
            else:
                requests.append(image_pb2.ImageRequest(
                    image_source_name=src,
                    quality_percent=75,
                    image_format=image_pb2.Image.FORMAT_JPEG,
                ))
        resps = img_client.get_image(requests)
        return {r.source.name: r for r in resps}
    except Exception as e:
        print(f"[VisualNav] Multi-capture failed: {e}")
        return {}


# One-time warnings keyed by source name so a quirky camera can't spam the log.
_depth_scale_warnings_emitted: set[str] = set()
_depth_decode_failures_logged: set[str] = set()


def _decode_depth(resp):
    """Decode depth image response into numpy array of meters with NaN for invalid pixels.

    Uses ``resp.source.depth_scale`` when > 0 (SDK convention: ``meters = raw / depth_scale``;
    typically depth_scale == 1000.0, i.e. raw values are millimeters). Falls back to 1000.0
    with a one-time warning per source if the scale is missing/zero. Invalid pixels (raw 0
    and raw 65535, per the Spot SDK convention) are mapped to NaN. Returns None only on hard
    errors (malformed response, wrong format, decode exception).
    """
    source_name = getattr(getattr(resp, "source", None), "name", "<unknown>")
    try:
        img = resp.shot.image
        if img.rows == 0 or img.cols == 0:
            return None
        if img.format != image_pb2.Image.FORMAT_RAW:
            print(f"[VisualNav] Depth not RAW (format={img.format}), skipping")
            return None
        raw = np.frombuffer(img.data, dtype=np.uint16).reshape(img.rows, img.cols)

        depth_scale = getattr(resp.source, "depth_scale", 0.0)
        if not depth_scale or depth_scale <= 0:
            if source_name not in _depth_scale_warnings_emitted:
                _depth_scale_warnings_emitted.add(source_name)
                print(f"[VisualNav] WARNING: source '{source_name}' has no valid depth_scale "
                      f"(got {depth_scale!r}); falling back to 1000.0 (mm → m)")
            depth_scale = 1000.0

        # SDK convention: meters = raw / depth_scale. Raw 0 and raw 65535 are invalid sentinels.
        invalid_mask = (raw == 0) | (raw == 65535)
        meters = raw.astype(np.float32) / float(depth_scale)
        meters[invalid_mask] = np.nan
        return meters
    except Exception as e:
        if source_name not in _depth_decode_failures_logged:
            _depth_decode_failures_logged.add(source_name)
            print(f"[VisualNav] Depth decode failed for source '{source_name}': "
                  f"{type(e).__name__}: {e} (further failures for this source will be silent)")
        return None


def _depth_at_bbox(depth_arr, bbox):
    """Get depth in meters at bbox center, un-rotating coords for depth image.

    YOLO bbox is in rotated (90° CW) image coords.
    Depth image is in original camera frame.
    Un-rotate: orig_x = rot_y, orig_y = H_orig - 1 - rot_x

    TODO(perception-overhaul-1b): replace this body with inner-60% bbox sampling
    + optional mask path. BLOCKED on linked-hopping-anchor Stage 0 (rotation
    table) and Stage 4 (seg model). The replacement signature must be
    ``_depth_at_bbox(depth_arr, bbox, mask=None) -> float | None`` per the
    sibling-plan contract in humming-stargazing-shell.md "INSTRUCTIONS FOR
    THE linked-hopping-anchor.md AGENT" §3. Until then this function uses an
    11×11 center patch and is forward-compat with Tier 1a NaN sentinels.
    """
    x1, y1, x2, y2 = bbox
    cx_rot = (x1 + x2) / 2.0
    cy_rot = (y1 + y2) / 2.0

    orig_rows, orig_cols = depth_arr.shape
    orig_x = int(cy_rot)
    orig_y = int(orig_rows - 1 - cx_rot)
    orig_x = max(0, min(orig_cols - 1, orig_x))
    orig_y = max(0, min(orig_rows - 1, orig_y))

    # Average a small patch for robustness
    pad = 5
    patch = depth_arr[max(0, orig_y - pad):min(orig_rows, orig_y + pad + 1),
                      max(0, orig_x - pad):min(orig_cols, orig_x + pad + 1)]
    # NaN-aware until Tier 1b ships its proper inner-60%/mask replacement.
    valid = patch[~np.isnan(patch) & (patch > 0.1)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def detect_in_image(image_bytes: bytes, description: str) -> dict:
    """Run YOLO-World on JPEG bytes to check if an object is present.

    Args:
        image_bytes: Raw JPEG data (from capture_frame).
        description: What to look for (e.g. "blue chair").

    Returns:
        dict with "found" (bool), "confidence" (float), "position" (str).
    """
    if Image is None:
        return {"found": False}
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.transpose(Image.Transpose.ROTATE_270)  # 90° CW for fisheye

        model = _get_world_model()
        model.set_classes([description])
        results = model.predict(img, device=DEVICE, conf=CONF_THRESH, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best = boxes.conf.argmax().item()
            bbox = boxes.xyxy[best].cpu().numpy()
            conf = float(boxes.conf[best].cpu())
            w, h = img.size
            cx = (bbox[0] + bbox[2]) / 2.0

            if cx < w * 0.33:
                position = "on the left side"
            elif cx > w * 0.67:
                position = "on the right side"
            else:
                position = "in the center"

            return {"found": True, "confidence": round(conf, 2), "position": position}
        return {"found": False}
    except Exception as e:
        print(f"[VisualNav] detect_in_image error: {e}")
        return {"found": False}


# ---------------------------------------------------------------------------
# Body-frame bearing + pixel un-rotation (Tiers 2b/2c of perception overhaul)
# ---------------------------------------------------------------------------
def _capture_with_response(session, source: str):
    """Capture from one camera, returning ``(rotated_PIL, image_response)``.

    The companion to ``_capture_rotated`` that ALSO returns the raw
    ``ImageResponse`` proto. ``WalkToObjectInImage`` (Tier 2a) needs the
    ``shot.transforms_snapshot`` and ``shot.frame_name_image_sensor``
    fields, so we can't throw the response away after decoding.

    Returns ``(None, None)`` on any failure.
    """
    if Image is None:
        return None, None
    try:
        robot = session["robot"]
        img_client = robot.ensure_client(ImageClient.default_service_name)
        resps = img_client.get_image_from_sources([source])
        if not resps:
            return None, None
        resp = resps[0]
        data = resp.shot.image.data
        pil = Image.open(io.BytesIO(data))
        pil = pil.transpose(Image.Transpose.ROTATE_270)  # 90° CW
        return pil, resp
    except Exception as e:
        print(f"[VisualNav] Capture (with response) failed for {source}: {e}")
        return None, None


def _unrotate_pixel(rot_x: float, rot_y: float,
                    rot_w: int, rot_h: int) -> tuple[float, float]:
    """Map a pixel from rotated (model) coords to native sensor coords.

    All callers currently rotate captured frames by ``ROTATE_270``
    (90° CW) before handing them to YOLO, so YOLO bboxes live in the
    rotated frame. The SDK's ``pixel_to_camera_space`` and
    ``WalkToObjectInImage`` need pixels in the **native** unrotated
    sensor frame.

    Inverse of 90° CW: ``native_x = rot_y``,
    ``native_y = rot_w - 1 - rot_x``. Numerically verified against the
    inverse rotation in ``_depth_at_bbox``.

    TODO(perception-overhaul-2b/linked-hopping-anchor-stage0): when
    linked-hopping-anchor's per-camera rotation table lands, replace
    this with the shared helper that knows each source's actual angle.
    """
    native_x = float(rot_y)
    native_y = float(rot_w - 1 - rot_x)
    return native_x, native_y


def _bearing_in_body_frame(image_response, native_x: float, native_y: float):
    """Return the body-frame yaw (radians) of a pixel in a captured image.

    Uses ``bosdyn.client.image.pixel_to_camera_space`` (pinhole only)
    plus the captured frame snapshot to project a unit ray from the
    pixel and compute its body-frame bearing — properly accounting for
    the camera mount offset by transforming both the camera origin and
    the depth-1 endpoint into the body frame and taking the difference.

    Returns ``None`` if the source isn't pinhole, the frame chain is
    missing, or anything else goes wrong. Pinhole-everywhere on this
    Spot is empirically confirmed by the perception probe; see
    ``docs/project/perception_probe_results.md``.
    """
    if image_response.source.WhichOneof("camera_models") != "pinhole":
        return None
    try:
        from bosdyn.client.image import pixel_to_camera_space
        from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME

        snap = image_response.shot.transforms_snapshot
        sensor_frame = image_response.shot.frame_name_image_sensor
        if not sensor_frame:
            return None
        body_T_cam = get_a_tform_b(snap, BODY_FRAME_NAME, sensor_frame)
        if body_T_cam is None:
            return None

        endpoint_cam = pixel_to_camera_space(
            image_response.source, native_x, native_y, depth=1.0
        )
        origin_body = body_T_cam.transform_point(0.0, 0.0, 0.0)
        endpoint_body = body_T_cam.transform_point(
            endpoint_cam[0], endpoint_cam[1], endpoint_cam[2]
        )
        dx = endpoint_body[0] - origin_body[0]
        dy = endpoint_body[1] - origin_body[1]
        return math.atan2(dy, dx)
    except Exception as e:
        print(f"[VisualNav] bearing_in_body_frame failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Steering + movement
# ---------------------------------------------------------------------------
def _steer(bbox, img_w, img_h, gain, max_yaw):
    """Compute (yaw_rad, bbox_height_fraction) from bbox center position.

    Spot body frame: +yaw = CCW = turn left.
    Object to the right of image center → negative yaw → turn right.

    TODO(perception-overhaul-2b): replace pixel-offset → yaw with a
    body-frame ray bearing (atan2 of the deprojected ray). Pixel-offset
    is wrong for fisheye distortion. BLOCKED on Tier 2a (we get the body-
    frame ray for free once navigate_to_object switches to
    WalkToObjectInImage / WalkToObjectRayInWorld). Reference technique:
    /tmp/perception/geometry.py:pixel_to_ptz_angles_transform Steps 1–4.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    offset = (cx - img_w / 2.0) / (img_w / 2.0)  # -1..+1
    yaw = -gain * offset
    yaw = max(-max_yaw, min(max_yaw, yaw))
    frac = (y2 - y1) / img_h
    return yaw, frac


def _move(session, x, y, yaw, timeout=5.0):
    """Send body-frame trajectory command (obstacle avoidance is on by default)."""
    try:
        state = session["state"].get_robot_state()
        tree = state.kinematic_state.transforms_snapshot
        cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            goal_x_rt_body=x, goal_y_rt_body=y,
            goal_heading_rt_body=yaw, frame_tree_snapshot=tree
        )
        session["cmd"].robot_command(cmd, end_time_secs=time.time() + timeout)
        return True
    except Exception as e:
        print(f"[VisualNav] Move failed: {e}")
        return False


def _send_velocity(session, vx, vy, v_rot, duration=0.5):
    """Send velocity command for smooth continuous motion.

    Args:
        vx: Forward speed (m/s, positive = forward)
        vy: Lateral speed (m/s, positive = left)
        v_rot: Rotational speed (rad/s, positive = CCW/left)
        duration: Command duration — robot auto-stops if no new command
    """
    try:
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=vx, v_y=vy, v_rot=v_rot
        )
        session["cmd"].robot_command(cmd, end_time_secs=time.time() + duration)
        return True
    except Exception as e:
        print(f"[VisualNav] Velocity cmd failed: {e}")
        return False


def _forward_collision_blocks(session, min_distance_m: float = COLLISION_MIN_DISTANCE_M) -> tuple[bool, str]:
    """Query Spot's native LocalGrid obstacle field for a close forward obstacle.

    Returns ``(blocks, reason)`` where ``blocks=True`` means "don't move
    forward right now". **Fail-open**: if the feature flag is off, the
    perception helpers are unavailable, the RPC fails, no grid is
    exposed, or no obstacle is found, this returns ``(False, <reason>)``
    so callers never get stuck on a query failure. Only explicit hits
    within ``min_distance_m`` and within ±90° of the forward direction
    cause a block.

    Enable with ``SPOT_COLLISION_PRECHECK=1`` in the environment.
    """
    if not COLLISION_PRECHECK_ENABLED:
        return False, "disabled"
    try:
        from src.voice_control.perception.obstacle_query import (
            nearest_obstacle_in_body_frame,
        )
        robot = session["robot"]
        hit = nearest_obstacle_in_body_frame(
            robot, max_radius_m=COLLISION_QUERY_RADIUS_M
        )
    except Exception as e:  # fail-open on any unexpected error
        return False, f"precheck_error:{type(e).__name__}"

    if hit is None:
        return False, "no_obstacle"
    # Bearing convention matches obstacle_query: 0 = forward, ±π/2 = sides.
    # Only block if the obstacle is in front (within ±90°).
    if abs(hit.bearing_rad) > math.pi / 2:
        return False, (
            f"behind_{math.degrees(hit.bearing_rad):.0f}deg "
            f"({hit.grid_name})"
        )
    if hit.distance_m < min_distance_m:
        return True, (
            f"obstacle_{hit.distance_m:.2f}m@"
            f"{math.degrees(hit.bearing_rad):.0f}deg "
            f"({hit.grid_name})"
        )
    return False, (
        f"clear_{hit.distance_m:.2f}m@"
        f"{math.degrees(hit.bearing_rad):.0f}deg"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def navigate_to_object(description: str, session, stop_event: threading.Event) -> bool:
    """Walk Spot to a described object using the SDK manipulation API.

    Phase 1 (SCAN): check all 5 body cameras for the target. The SCAN
    phase is still a serial loop today; ``linked-hopping-anchor`` Stage 2
    will turn it into a single batched call.

    Phase 2 (APPROACH): once the target is located on one camera, capture
    a fresh frame from that camera, run the detector once more to get a
    crisp bbox, and submit a ``WalkToObjectInImage`` request. The SDK
    handles frame transforms, depth fusion, planning, and the actual
    walk — we just poll until it reports ``MANIP_STATE_DONE``.

    The probe at ``docs/project/perception_probe_results.md`` confirmed
    every body camera is pinhole, so ``WalkToObjectInImage`` is safe.
    On any submission failure (camera not pinhole, manipulation client
    unavailable, etc.) the function logs and returns False — there is
    no fallback to the old visual-servo loop, on purpose.

    Args:
        description: What to look for (e.g. "red chair", "backpack").
        session: Spot session dict with "robot", "cmd", "state" keys.
        stop_event: Threading event — set to cancel.

    Returns:
        True if the SDK reports a successful walk-to-object, False if
        the target is not visible, the SDK rejects the request, or the
        approach times out.
    """
    print(f"[VisualNav] Looking for: '{description}'")

    # ---- SCAN phase ------------------------------------------------------
    found_cam = None
    found_image = None
    found_bbox = None
    turn_yaw = 0.0
    for cam_source, yaw in ALL_CAMERAS:
        if stop_event.is_set():
            return False
        cam_label = cam_source.replace("_fisheye_image", "")
        img, resp = _capture_with_response(session, cam_source)
        if img is None:
            continue
        bbox = _find_object(img, description)
        if bbox is not None:
            print(f"[VisualNav] Found '{description}' on {cam_label} camera")
            found_cam = cam_source
            found_image = (img, resp)
            found_bbox = bbox
            turn_yaw = yaw
            break

    if found_cam is None:
        print(f"[VisualNav] '{description}' not visible on any camera")
        return False

    # ---- Turn-to-face (only if target was found on a side/back cam) ------
    # WalkToObjectInImage understands per-camera transforms, so technically
    # the turn isn't required. But the manipulation API plans much shorter
    # paths when the target is already in front, so we keep the existing
    # turn-then-approach pattern. After turning, recapture from the FRONT
    # camera so the manipulation request uses the same camera Spot is now
    # facing.
    if abs(turn_yaw) > 0.1:
        cam_label = found_cam.replace("_fisheye_image", "")
        print(f"[VisualNav] Turning to face {cam_label} direction...")
        _move(session, 0, 0, turn_yaw)
        time.sleep(MOVE_SETTLE + 0.5)

        if stop_event.is_set():
            return False

        # Re-acquire from the front camera after the body has settled.
        front_img, front_resp = _capture_with_response(session, FRONT_CAMERA)
        if front_img is None:
            print(f"[VisualNav] Re-capture failed after turn")
            return False
        front_bbox = _find_object(front_img, description)
        if front_bbox is None:
            print(f"[VisualNav] Lost '{description}' after turn")
            return False
        found_cam = FRONT_CAMERA
        found_image = (front_img, front_resp)
        found_bbox = front_bbox

    pil_img, image_response = found_image
    bbox = found_bbox

    # ---- APPROACH phase: WalkToObjectInImage -----------------------------
    # The bbox is in rotated (model) coordinates because YOLO sees the
    # rotated frame. Un-rotate the center to get the pixel in the native
    # sensor frame, which is what pixel_to_camera_space + the manipulation
    # API expect. The intrinsics in image_response.source.pinhole are
    # also for the native frame.
    cx_rot = (bbox[0] + bbox[2]) / 2.0
    cy_rot = (bbox[1] + bbox[3]) / 2.0
    rot_w, rot_h = pil_img.size
    px_native, py_native = _unrotate_pixel(cx_rot, cy_rot, rot_w, rot_h)

    if image_response.source.WhichOneof("camera_models") != "pinhole":
        print(f"[VisualNav] {found_cam} is not pinhole — cannot use "
              f"WalkToObjectInImage")
        return False

    try:
        from bosdyn.client.manipulation_api_client import ManipulationApiClient
        from bosdyn.api import manipulation_api_pb2, geometry_pb2
        from google.protobuf import wrappers_pb2
    except ImportError as e:
        print(f"[VisualNav] manipulation API import failed: {e}")
        return False

    try:
        manipulation_client = session["robot"].ensure_client(
            ManipulationApiClient.default_service_name
        )
    except Exception as e:
        print(f"[VisualNav] manipulation client unavailable: {e}")
        return False

    # Optional pre-flight bearing log so we can confirm the body-frame
    # ray makes sense before handing off to the SDK. Useful for catching
    # frame-tree mistakes during initial deployment.
    bearing = _bearing_in_body_frame(image_response, px_native, py_native)
    if bearing is not None:
        print(f"[VisualNav] target bearing in body frame: "
              f"{math.degrees(bearing):.0f}°")

    walk_to = manipulation_api_pb2.WalkToObjectInImage(
        pixel_xy=geometry_pb2.Vec2(x=px_native, y=py_native),
        transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
        frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
        camera_model=image_response.source.pinhole,
        offset_distance=wrappers_pb2.FloatValue(value=APPROACH_OFFSET_M),
    )
    request = manipulation_api_pb2.ManipulationApiRequest(
        walk_to_object_in_image=walk_to
    )

    try:
        cmd_response = manipulation_client.manipulation_api_command(
            manipulation_api_request=request
        )
    except Exception as e:
        print(f"[VisualNav] WalkToObjectInImage submit failed: {e}")
        return False

    cmd_id = cmd_response.manipulation_cmd_id
    print(f"[VisualNav] Walking to '{description}' "
          f"(cmd_id={cmd_id}, offset={APPROACH_OFFSET_M:.2f}m)")

    # Poll until done, failed, or timeout. There is no fallback path
    # here on failure — the SDK either succeeds or it doesn't, and we
    # surface the result honestly to the caller.
    poll_start = time.time()
    last_state = None
    failure_states = {
        manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
        manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
        manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
        manipulation_api_pb2.MANIP_STATE_PLACE_FAILED,
        manipulation_api_pb2.MANIP_STATE_PLACE_FAILED_TO_RAYCAST_INTO_MAP,
    }

    while not stop_event.is_set():
        if time.time() - poll_start > APPROACH_TIMEOUT_S:
            print(f"[VisualNav] Approach timeout after {APPROACH_TIMEOUT_S}s")
            return False
        try:
            feedback = manipulation_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=
                manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_id
                )
            )
        except Exception as e:
            print(f"[VisualNav] feedback poll failed: {e}")
            return False

        state = feedback.current_state
        if state != last_state:
            try:
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
            except Exception:
                state_name = f"STATE_{state}"
            print(f"[VisualNav] approach state: {state_name}")
            last_state = state

        if state == manipulation_api_pb2.MANIP_STATE_DONE:
            print(f"[VisualNav] ✓ Reached '{description}'")
            return True
        if state in failure_states:
            try:
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
            except Exception:
                state_name = f"STATE_{state}"
            print(f"[VisualNav] ✗ Approach failed: {state_name}")
            return False

        time.sleep(APPROACH_POLL_S)

    return False


def follow_person(session, stop_event: threading.Event) -> bool:
    """Follow the nearest person using continuous velocity control.

    Architecture: two threads for smooth, responsive tracking.
    - Detector thread: captures images + runs YOLO continuously (~2-3 Hz)
    - Control thread (main): reads latest detection, sends velocity at ~10Hz

    Both front cameras contribute detections with per-camera person locking
    to prevent target switching when multiple people are visible.

    Args:
        session: Spot session dict.
        stop_event: Threading event — set to cancel.

    Returns:
        True if cleanly stopped, False if person lost.
    """
    print("[VisualNav] Follow mode — continuous tracking...")

    # Discover available image sources
    robot = session["robot"]
    img_client = robot.ensure_client(ImageClient.default_service_name)
    available = {s.name for s in img_client.list_image_sources()}

    front_pairs = []
    for vis, dep in FOLLOW_FRONT_CAMS:
        if vis in available:
            front_pairs.append((vis, dep if dep in available else None))

    scan_cams = []
    for vis, yaw in FOLLOW_SCAN_CAMS:
        if vis in available:
            scan_cams.append((vis, yaw))

    all_sources = []
    for vis, dep in front_pairs:
        all_sources.append(vis)
        if dep:
            all_sources.append(dep)
    for vis, _ in scan_cams:
        all_sources.append(vis)

    has_depth = any(d is not None for _, d in front_pairs)
    print(f"[VisualNav] {len(front_pairs)} front + {len(scan_cams)} scan cameras, "
          f"depth={'yes' if has_depth else 'no'}")

    if not front_pairs and not scan_cams:
        print("[VisualNav] No cameras available!")
        return False

    model = _get_person_model()

    # --- Shared state (detector thread → control thread) ---
    det_lock = threading.Lock()
    detection = {
        'bbox': None,           # [x1,y1,x2,y2] or None
        'cam': None,            # camera name
        'depth': 0.0,           # meters (0 = unavailable)
        'img_size': (1, 1),     # (w, h) of rotated image
        'timestamp': 0.0,       # when this detection was made
        'scan_yaw': None,       # set when person found on scan camera
        'scan_time': 0.0,
    }
    det_stop = threading.Event()

    # Per-camera person lock (detector thread only — no sharing needed)
    locked_centers = {}  # {cam_name: (cx_norm, cy_norm)}

    def _detector_loop():
        """Background: capture + YOLO on all cameras, store latest result."""
        while not stop_event.is_set() and not det_stop.is_set():
            try:
                resps = _capture_multi(session, all_sources)
            except Exception as e:
                print(f"[Follow/det] Capture error: {e}")
                time.sleep(0.1)
                continue
            if not resps:
                time.sleep(0.05)
                continue

            # --- Front cameras (per-camera locking) ---
            best_bbox = None
            best_conf = 0.0
            best_vis = None
            best_img = None
            best_dep = None

            for vis_name, dep_name in front_pairs:
                if vis_name not in resps:
                    continue
                try:
                    data = resps[vis_name].shot.image.data
                    img = Image.open(io.BytesIO(data))
                    img = img.transpose(Image.Transpose.ROTATE_270)
                except Exception:
                    continue

                try:
                    results = model.predict(img, device=DEVICE, classes=[0],
                                            conf=CONF_THRESH, verbose=False)
                except Exception:
                    continue
                if not results or len(results[0].boxes) == 0:
                    locked_centers.pop(vis_name, None)
                    continue

                boxes = results[0].boxes
                w_img, h_img = img.size

                if vis_name in locked_centers:
                    # Match closest to this camera's locked position
                    best_dist_lock = float('inf')
                    match_idx = None
                    for i in range(len(boxes)):
                        b = boxes.xyxy[i].cpu().numpy()
                        cx_n = ((b[0] + b[2]) / 2.0) / w_img
                        cy_n = ((b[1] + b[3]) / 2.0) / h_img
                        dist = math.sqrt(
                            (cx_n - locked_centers[vis_name][0])**2 +
                            (cy_n - locked_centers[vis_name][1])**2)
                        if dist < best_dist_lock:
                            best_dist_lock = dist
                            match_idx = i

                    if match_idx is not None and best_dist_lock < FOLLOW_LOCK_DIST:
                        bbox = boxes.xyxy[match_idx].cpu().numpy()
                        conf = float(boxes.conf[match_idx].cpu())
                        locked_centers[vis_name] = (
                            ((bbox[0] + bbox[2]) / 2.0) / w_img,
                            ((bbox[1] + bbox[3]) / 2.0) / h_img)
                    else:
                        # Lock lost on this camera
                        locked_centers.pop(vis_name, None)
                        continue
                else:
                    # No lock — pick largest (closest person)
                    areas = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) *
                             (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]))
                    idx = areas.argmax().item()
                    conf = float(boxes.conf[idx].cpu())
                    bbox = boxes.xyxy[idx].cpu().numpy()
                    locked_centers[vis_name] = (
                        ((bbox[0] + bbox[2]) / 2.0) / w_img,
                        ((bbox[1] + bbox[3]) / 2.0) / h_img)
                    cam_label = vis_name.replace("_fisheye_image", "")
                    print(f"[Follow] Locked onto person on {cam_label}")

                if conf > best_conf:
                    best_conf = conf
                    best_bbox = bbox
                    best_vis = vis_name
                    best_img = img
                    best_dep = dep_name

            if best_bbox is not None:
                # Compute depth
                depth = 0.0
                if best_dep and best_dep in resps:
                    depth_arr = _decode_depth(resps[best_dep])
                    if depth_arr is not None:
                        depth = _depth_at_bbox(depth_arr, best_bbox)

                w, h = best_img.size
                with det_lock:
                    detection['bbox'] = best_bbox.copy()
                    detection['cam'] = best_vis
                    detection['depth'] = depth
                    detection['img_size'] = (w, h)
                    detection['timestamp'] = time.time()
                    detection['scan_yaw'] = None
                continue

            # --- Scan cameras (side/back) ---
            for vis_name, turn_yaw in scan_cams:
                if vis_name not in resps:
                    continue
                try:
                    data = resps[vis_name].shot.image.data
                    img = Image.open(io.BytesIO(data))
                    img = img.transpose(Image.Transpose.ROTATE_270)
                except Exception:
                    continue

                try:
                    results = model.predict(img, device=DEVICE, classes=[0],
                                            conf=CONF_THRESH, verbose=False)
                except Exception:
                    continue
                if results and len(results[0].boxes) > 0:
                    cam = vis_name.replace("_fisheye_image", "")
                    print(f"[Follow] Person on {cam} camera — turn needed")
                    locked_centers.clear()
                    with det_lock:
                        detection['bbox'] = None
                        detection['scan_yaw'] = turn_yaw
                        detection['scan_time'] = time.time()
                    break

            # Nothing found — clear front detection (keep scan_yaw if just set)
            with det_lock:
                if detection.get('scan_yaw') is None:
                    detection['bbox'] = None

    # Start detector thread
    det_thread = threading.Thread(target=_detector_loop, daemon=True)
    det_thread.start()

    # --- Control loop (main thread, ~10Hz) ---
    last_seen = time.time()
    last_vx = 0.0
    last_v_rot = 0.0
    hold_count = 0
    log_counter = 0
    last_turn_time = 0.0  # cooldown: ignore scan turns for 3s after a turn

    try:
        while not stop_event.is_set():
            with det_lock:
                det = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                       for k, v in detection.items()}

            now = time.time()
            log_counter += 1
            should_log = (log_counter % (FOLLOW_CONTROL_HZ * 2) == 0)  # ~every 2s

            # --- Handle scan camera turn (with cooldown) ---
            if (det.get('scan_yaw') is not None and
                    det['scan_time'] > last_turn_time + 3.0):
                cam_dir = "left" if det['scan_yaw'] > 0 else "right"
                if abs(det['scan_yaw']) > 2.0:
                    cam_dir = "back"
                print(f"[Follow] Turning {cam_dir} toward person...")
                _move(session, 0, 0, det['scan_yaw'])
                last_turn_time = time.time()
                last_seen = time.time()  # fresh timestamp so we don't trigger lost
                last_vx = 0.0
                last_v_rot = 0.0
                time.sleep(1.5)
                with det_lock:
                    detection['scan_yaw'] = None
                continue

            # --- Front camera person tracking ---
            det_age = now - det['timestamp']
            if det['bbox'] is not None and det_age < 2.0:
                last_seen = det['timestamp']
                w, h = det['img_size']
                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2.0
                offset = (cx - w / 2.0) / (w / 2.0)
                target_v_rot = max(-0.8, min(0.8, -1.0 * offset))

                # Forward speed from depth or bbox size
                holding = False
                if det['depth'] > 0.3:
                    gap = det['depth'] - FOLLOW_DIST_MARGIN
                    if gap < 0.2:
                        target_vx = 0.0
                        holding = True
                    else:
                        target_vx = min(gap * 0.6, FOLLOW_MAX_VEL)
                else:
                    frac = (y2 - y1) / h
                    if frac >= FOLLOW_CLOSE_BBOX:
                        target_vx = 0.0
                        holding = True
                    else:
                        target_vx = FOLLOW_FAR_VEL

                # Smooth velocity (EMA)
                vx = FOLLOW_VEL_SMOOTH * last_vx + (1 - FOLLOW_VEL_SMOOTH) * target_vx
                v_rot = FOLLOW_VEL_SMOOTH * last_v_rot + (1 - FOLLOW_VEL_SMOOTH) * target_v_rot

                # Collision pre-check (feature-flagged, fail-open). Only
                # clamp vx — keep v_rot so the robot can still turn to
                # keep the person in frame even when stopped.
                if vx > 0.05:  # ignore near-zero "already holding" case
                    blocks, reason = _forward_collision_blocks(session)
                    if blocks:
                        if should_log:
                            print(f"[Follow] collision_precheck: STOP vx ({reason})")
                        vx = 0.0

                last_vx = vx
                last_v_rot = v_rot

                _send_velocity(session, vx, 0, v_rot, duration=0.3)

                if should_log:
                    cam = det['cam'].replace("_fisheye_image", "") if det['cam'] else "?"
                    if det['depth'] > 0.3:
                        print(f"[Follow] {cam} {det['depth']:.1f}m — "
                              f"vx={vx:.2f} rot={v_rot:.2f}")
                    else:
                        frac = (y2 - y1) / h
                        print(f"[Follow] {cam} bbox={frac:.0%} — "
                              f"vx={vx:.2f} rot={v_rot:.2f}")

                if holding:
                    hold_count += 1
                    if hold_count >= FOLLOW_HOLD_EXIT:
                        print(f"[VisualNav] Reached person ({hold_count} holds)")
                        _send_velocity(session, 0, 0, 0)
                        return True
                else:
                    hold_count = 0

            else:
                # No recent detection — decelerate and search
                hold_count = 0
                if now - last_seen > LOST_TIMEOUT:
                    print("[VisualNav] Lost person — stopping")
                    _send_velocity(session, 0, 0, 0)
                    return False

                # Decelerate forward, slow search turn
                last_vx *= 0.85
                search_dir = 0.3 if last_v_rot >= 0 else -0.3
                _send_velocity(session, last_vx, 0, search_dir, duration=0.3)

            time.sleep(1.0 / FOLLOW_CONTROL_HZ)

    finally:
        _send_velocity(session, 0, 0, 0)
        det_stop.set()
        det_thread.join(timeout=2.0)
        print("[VisualNav] Follow stopped")

    return True
