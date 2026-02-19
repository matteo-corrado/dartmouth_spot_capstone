"""Visual navigation for Spot — YOLO object seeking + person following.

Uses ultralytics YOLO models:
- YOLO-World (open-vocabulary) for "go to the red chair" style commands
- YOLOv8n for person following (class 0 = person)

Both run on CPU to avoid GPU/VRAM competition with the LLM.

Install: pip install ultralytics
"""

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

# Approach mode (go_to_object)
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


def _decode_depth(resp):
    """Decode depth image response into numpy array (meters).

    Forces scale = 0.001 (mm → m) for DEPTH_U16 regardless of API depth_scale,
    which can return incorrect values on some firmware.
    """
    try:
        img = resp.shot.image
        if img.rows == 0 or img.cols == 0:
            return None
        if img.format != image_pb2.Image.FORMAT_RAW:
            print(f"[VisualNav] Depth not RAW (format={img.format}), skipping")
            return None
        arr = np.frombuffer(img.data, dtype=np.uint16).reshape(img.rows, img.cols)
        # Always use 0.001 (mm → m) for uint16 depth — API depth_scale is unreliable
        return arr.astype(np.float32) * 0.001
    except Exception as e:
        print(f"[VisualNav] Depth decode: {e}")
        return None


def _depth_at_bbox(depth_arr, bbox):
    """Get depth in meters at bbox center, un-rotating coords for depth image.

    YOLO bbox is in rotated (90° CW) image coords.
    Depth image is in original camera frame.
    Un-rotate: orig_x = rot_y, orig_y = H_orig - 1 - rot_x
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
    valid = patch[patch > 0.1]  # filter invalid readings
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
# Steering + movement
# ---------------------------------------------------------------------------
def _steer(bbox, img_w, img_h, gain, max_yaw):
    """Compute (yaw_rad, bbox_height_fraction) from bbox center position.

    Spot body frame: +yaw = CCW = turn left.
    Object to the right of image center → negative yaw → turn right.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def navigate_to_object(description: str, session, stop_event: threading.Event) -> bool:
    """Visual servo to approach a described object.

    Phase 1: Scan 360° to find the object.
    Phase 2: Walk toward it, correcting heading each step.
    Stops when bbox is large enough (close) or target is lost.

    Args:
        description: What to look for (e.g. "red chair", "backpack").
        session: Spot session dict with "robot", "cmd", "state" keys.
        stop_event: Threading event — set to cancel.

    Returns:
        True if arrived near the object, False if lost/cancelled.
    """
    print(f"[VisualNav] Looking for: '{description}'")

    # Check all cameras (front, sides, back) without moving
    found_cam = None
    turn_yaw = 0.0
    for cam_source, yaw in ALL_CAMERAS:
        if stop_event.is_set():
            return False
        cam_label = cam_source.replace("_fisheye_image", "")
        img = _capture_rotated(session, source=cam_source)
        if img is not None and _find_object(img, description) is not None:
            print(f"[VisualNav] Found '{description}' on {cam_label} camera")
            found_cam = cam_source
            turn_yaw = yaw
            break

    if found_cam is None:
        print(f"[VisualNav] '{description}' not visible on any camera")
        return False

    # Turn to face the object if found on a side/back camera
    if abs(turn_yaw) > 0.1:
        cam_label = found_cam.replace("_fisheye_image", "")
        print(f"[VisualNav] Turning to face {cam_label} direction...")
        _move(session, 0, 0, turn_yaw)
        time.sleep(MOVE_SETTLE + 0.5)

    print(f"[VisualNav] Approaching '{description}'...")

    # Approach (always uses front camera from here)
    last_seen = time.time()

    while not stop_event.is_set():
        img = _capture_rotated(session)
        if img is None:
            time.sleep(0.5)
            continue

        bbox = _find_object(img, description)

        if bbox is None:
            if time.time() - last_seen > LOST_TIMEOUT:
                print(f"[VisualNav] Lost '{description}'")
                return False
            time.sleep(0.5)
            continue

        last_seen = time.time()
        w, h = img.size
        yaw, frac = _steer(bbox, w, h, APPROACH_YAW_GAIN, APPROACH_MAX_YAW)

        if frac >= APPROACH_DONE_FRAC:
            print(f"[VisualNav] Reached '{description}' (bbox {frac:.0%} of frame)")
            return True

        _move(session, APPROACH_STEP, 0, yaw)
        time.sleep(MOVE_SETTLE)

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
