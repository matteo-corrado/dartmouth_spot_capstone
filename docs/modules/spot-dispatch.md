# spot_dispatch.py

`src/voice_control/spot_dispatch.py` -- Command execution and navigation.

Central dispatcher that translates parsed intents into Boston Dynamics SDK
calls. Manages a persistent Spot session (lazy singleton), background
navigation threads, and camera capture.

## Key Functions

### `dispatch_intent(intent) -> bool`

Main executor for all commands. Receives an intent dict from the LLM brain
or regex parser and calls the appropriate BD SDK methods.

```python
def dispatch_intent(intent: dict) -> bool
```

**Args:**
- `intent` -- Dict with `"intent"` (str) and `"params"` (dict).

**Returns:** True if command executed successfully.

**Supported intents:** `stop`, `freeze`, `estop`, `stand`, `sit`, `selfright`,
`walk`, `strafe`, `turn`, `body_height`, `set_speed`, `go_to`, `tour`,
`patrol`, `come_back`, `save_location`, `list_locations`, `open_door`,
`go_to_object`, `follow_me`, `describe`, `battery_status`, `status`,
`power_off`.

### `ensure_spot_session() -> dict`

Lazy singleton pattern. On first call, opens a `spot_session()` context
manager with `stand_on_enter=True`, `sit_on_exit=False` (keeps robot standing
for voice commands). Subsequent calls return the cached session.

### `close_spot_session()`

Graceful shutdown: sit, power off, exit context manager. Called from
`client_mic.py` on Ctrl+C.

### `get_robot_state_dict() -> dict`

Collect current robot state for LLM context. Returns:

```python
{
    "battery_percent": int,
    "estimated_runtime_minutes": int,
    "is_powered": bool,
    "is_standing": str,
    "current_location": str,      # Friendly name or waypoint ID
    "saved_locations": str,       # Comma-separated names
    "estop_status": str,          # "not_cut", "cut", "soft_stop"
}
```

### `capture_frame(camera) -> bytes`

Capture a JPEG frame from any of Spot's fisheye cameras.

```python
def capture_frame(camera: str = "front") -> bytes
```

### `is_navigating() -> bool`

Returns True if a navigation thread is alive (motor noise expected).

### `is_follow_mode() -> bool`

Returns True if the active navigation mode is "follow".

## Navigation Architecture

Navigation commands (`go_to`, `tour`, `patrol`, `come_back`, `go_to_object`,
`follow_me`) run in **background threads** so the main voice loop stays
responsive to safety commands.

- `_nav_thread` -- The active navigation Thread.
- `_nav_stop_event` -- Threading Event; set to cancel navigation.
- `_nav_mode` -- One of `"graphnav"`, `"visual_nav"`, `"follow"`, or None.
- `_home_waypoint` -- Saved departure waypoint for `come_back`.

New navigation commands cancel any in-progress navigation via `_cancel_nav()`
before starting.

### GraphNav Navigation

`go_to`, `tour`, and `patrol` use BD's GraphNav API. The sequence handler
`_navigate_waypoint_sequence()` supports:

- Multi-waypoint sequences with per-waypoint retry on STATUS_STUCK
- Continuous patrol looping (`repeat=True`)
- Re-localization on STATUS_LOST
- Skipping unreachable waypoints in multi-waypoint tours

### Visual Navigation

`go_to_object` delegates to `visual_nav.navigate_to_object()` (YOLO-World).
`follow_me` delegates to `visual_nav.follow_person()` (YOLOv8n).

## Camera Sources

```python
CAMERA_SOURCES = {
    "front":       "frontleft_fisheye_image",
    "front_left":  "frontleft_fisheye_image",
    "front_right": "frontright_fisheye_image",
    "left":        "left_fisheye_image",
    "right":       "right_fisheye_image",
    "back":        "back_fisheye_image",
}
```

## Speed Multipliers

| Speed | Multiplier |
|-------|-----------|
| slow | 0.5 |
| normal | 1.0 |
| fast | 1.5 |

## Usage Example

```python
from spot_dispatch import dispatch_intent, get_robot_state_dict, close_spot_session

# Execute a command (session auto-initializes on first call)
dispatch_intent({"intent": "walk", "params": {"direction": "forward", "distance": 2.0}})

# Get robot state for LLM context
state = get_robot_state_dict()

# Graceful shutdown
close_spot_session()
```

## Dependencies

**Project modules:** `src.session` (`spot_session`), `src.location_manager`
(`list_locations`), `visual_nav` (`navigate_to_object`, `follow_person`)

**External packages:** `bosdyn.client` (robot_command, frame_helpers,
math_helpers, image, graph_nav)
