# visual_nav.py

`src/voice_control/visual_nav.py` -- YOLO detection and visual navigation.

Provides camera-based object seeking (YOLO-World open-vocabulary) and person
following (YOLOv8n). Both models run on CPU to avoid GPU/VRAM competition
with the LLM and VLM.

## Architecture

### Object approach (`navigate_to_object`)

1. **Scan phase:** Check all cameras (front, front_right, left, right, back)
   without moving. If found on a side/back camera, turn to face it.
2. **Approach phase:** Repeatedly capture from front camera, detect object,
   compute steering correction, step forward. Stop when bbox height exceeds
   `APPROACH_DONE_FRAC` of frame height.

### Person following (`follow_person`)

Two-thread architecture for smooth, responsive tracking:

- **Detector thread (~2-3 Hz):** Captures from all cameras, runs YOLOv8n,
  stores latest detection (bbox, camera, depth, timestamp) in shared state.
  Per-camera person locking prevents target switching when multiple people
  are visible.
- **Control thread (10 Hz):** Reads latest detection, computes forward
  velocity and yaw rate, sends velocity commands. EMA smoothing prevents
  jerky motion.

Depth from stereo cameras is used when available for precise distance control.
Falls back to bbox size heuristic when depth is unavailable.

### Fisheye handling

Spot's fisheye cameras are mounted sideways. Images are rotated 90 degrees CW
for YOLO detection. Depth coordinate un-rotation:
`orig_x = rot_y`, `orig_y = H_orig - 1 - rot_x` (uses `raw_rows` not `raw_cols`).

## Key Functions

### `preload_models()`

Background-load YOLO-World and YOLOv8n models in a daemon thread. Called
during startup to avoid first-use latency. Safe to call multiple times.

### `detect_in_image(image_bytes, description) -> dict`

Run YOLO-World on raw JPEG bytes.

```python
def detect_in_image(image_bytes: bytes, description: str) -> dict
```

**Returns:**
```python
{"found": True, "confidence": 0.72, "position": "in the center"}
# or
{"found": False}
```

Position is one of: `"on the left side"`, `"in the center"`, `"on the right side"`.

### `navigate_to_object(description, session, stop_event) -> bool`

Visual servo to approach a described object.

```python
def navigate_to_object(
    description: str,                    # e.g. "red chair"
    session: dict,                       # Spot session
    stop_event: threading.Event          # Set to cancel
) -> bool
```

### `follow_person(session, stop_event) -> bool`

Follow the nearest person using continuous velocity control.

```python
def follow_person(
    session: dict,
    stop_event: threading.Event
) -> bool
```

Returns True if cleanly stopped, False if person lost.

## Configuration Constants

| Name | Default | Description |
|------|---------|-------------|
| `CONF_THRESH` | 0.25 | YOLO confidence threshold |
| `DEVICE` | `"cpu"` | Inference device |
| `APPROACH_STEP` | 0.5 | Meters forward per approach loop |
| `APPROACH_DONE_FRAC` | 0.50 | Bbox height fraction to consider "arrived" |
| `APPROACH_YAW_GAIN` | 1.0 | Steering proportional gain |
| `APPROACH_MAX_YAW` | 0.35 | Max yaw correction per step (rad, ~20 deg) |
| `FOLLOW_DIST_MARGIN` | 1.5 | Desired following distance (meters) |
| `FOLLOW_MAX_VEL` | 0.8 | Max forward velocity (m/s) |
| `FOLLOW_FAR_VEL` | 0.4 | Forward velocity when using bbox only (no depth) |
| `FOLLOW_CLOSE_BBOX` | 0.55 | Bbox height fraction = "close enough" |
| `FOLLOW_CONTROL_HZ` | 10 | Velocity command rate (Hz) |
| `FOLLOW_VEL_SMOOTH` | 0.6 | Velocity EMA smoothing (0=instant, 1=frozen) |
| `FOLLOW_HOLD_EXIT` | 80 | Consecutive hold cycles before exiting (~8s) |
| `FOLLOW_LOCK_DIST` | 0.3 | Max normalized distance to match locked person |
| `MOVE_SETTLE` | 1.5 | Seconds to let a step complete |
| `LOST_TIMEOUT` | 10.0 | Seconds without detection before giving up |

## Usage Example

```python
from visual_nav import detect_in_image, navigate_to_object, preload_models
import threading

preload_models()

# Check if object is visible
result = detect_in_image(jpeg_bytes, "blue backpack")
if result["found"]:
    print(f"Found at {result['position']}, conf={result['confidence']}")

# Approach the object (runs in calling thread until done or cancelled)
stop = threading.Event()
success = navigate_to_object("blue backpack", spot_session, stop)
```

## Dependencies

**Project modules:** none (receives session dict from caller)

**External packages:** `ultralytics` (YOLO, YOLOWorld), `PIL` (Pillow),
`numpy`, `bosdyn.client` (robot_command, image), `bosdyn.api` (image_pb2)
