# Dispatcher (formerly spot_dispatch.py)

> **spot_dispatch.py is deleted.** Its code was split across the `src/spot/` packages (see [package-map.md](../architecture/package-map.md)). Core command dispatch is in `src/spot/robot/dispatcher.py`; optional behaviours live in per-feature handlers at `src/spot/features/*/handlers.py`. This page keeps its name so existing links work.

The dispatcher translates parsed intents into Boston Dynamics SDK calls. It manages a persistent Spot session (lazy singleton), background navigation threads, and camera capture.

## Where the old spot_dispatch code went

| Old job | Now in |
|---|---|
| `dispatch_intent`, safety dispatch, core branches (stop, freeze, estop, stand, sit, selfright, power_off, stow, battery_status, status) | `src/spot/robot/dispatcher.py` (`dispatch_intent`, `dispatch_safety`, `_dispatch_core`, `_dispatch_locomotion`, `_dispatch_feature`) |
| `ensure_spot_session`, `close_spot_session`, the lazy session | `src/spot/robot/connection.py` (`Connection` class; the `CONNECTION` owner is created in `src/spot/robot/__init__.py`) |
| `get_robot_state_dict`, `capture_frame`, `CAMERA_SOURCES` | `src/spot/robot/state.py` |
| Motion helpers (send stop, trajectory wait, restore) | `src/spot/robot/motion.py` |
| Arm pose, stow, gaze hold | `src/spot/robot/arm.py` (`ARM`) |
| Navigation thread, stop event, mode, home waypoint, `is_navigating`, `is_follow_mode`, `_navigate_waypoint_sequence` | `src/spot/nav/runner.py` (`NavRunner` class; the `NAV` owner is created in `src/spot/robot/__init__.py`) |
| Safety generation counter | `src/spot/safety/gen.py` (`SafetyGen` class; the `SAFETY_GEN` owner is created in `src/spot/robot/__init__.py`) |
| walk, strafe, turn, body_height, set_speed | `src/spot/features/walk/handlers.py` |
| go_to, tour, patrol, come_back, save_location, list_locations, walk_to, load_map, list_maps | `src/spot/features/navigation/handlers.py` |
| go_to_object, describe, check_obstacles | `src/spot/features/visual_nav/handlers.py` (logic in `navigator.py`) |
| follow_me | `src/spot/features/follow/handlers.py` |
| emote / dance | `src/spot/features/dance/handlers.py` |
| gaze (`gaze_at_bearing`) | `src/spot/features/gaze/handlers.py` |
| enable_mouth / disable_mouth | `src/spot/features/mouth/handlers.py` |
| set_persona, add_persona, set_volume, set_backend | `src/spot/features/settings/handlers.py` |

Core intents are declared only in `src/spot/core/core_intents.py`; their branches stay in `_dispatch_core`. Feature handlers have the signature `(params, ctx) -> bool` and are reached through `registry_dispatch`.

## Key Functions

### `dispatch_intent(intent, brain=None, speak=None) -> bool`

Main executor. Receives an intent dict from the LLM brain or regex parser.

- `intent` -- dict with `"intent"` (str) and `"params"` (dict).
- `speak` -- optional callable used to say result or refusal lines.
- Returns True if the command executed successfully.

Routing: stop/freeze/estop go to `dispatch_safety` without taking the dispatch lock (SDK stop first, then the original branch, then a no-wait arm stow; with no session they do nothing and never power on a parked robot). Intents flagged `unlocked` go straight to `_dispatch_core`. Everything else runs under the dispatch lock: locomotion intents through `_dispatch_locomotion` (which tucks a deployed arm first and restores it only after a natural finish), core intents through `_dispatch_core`, and feature actions through `_dispatch_feature` to the registered handler. stop/freeze/estop/sit/power_off/stow bump the safety generation, which aborts locomotion still being set up on another thread.

Supported intents are the core intents plus whatever the enabled features declare (`src/spot/features/enabled.py`); there is no fixed list in the dispatcher.

### `dispatch_safety(intent, brain=None, speak=None) -> bool`

The halt path. Never goes through the event bus.

### `ensure_spot_session(map_path=None)` / `close_spot_session()` -- `robot/connection.py`

Lazy singleton. On first call opens a `spot_session()` (`src/spot/robot/session.py`) with `stand_on_enter=True`, `sit_on_exit=False` (keeps the robot standing for voice commands); later calls return the cached session. `close_spot_session()` does the graceful shutdown (sit, power off, exit the context manager); the app calls it once from its shutdown path.

### `get_robot_state_dict() -> dict` -- `robot/state.py`

Collects robot state for LLM context: `battery_percent`, `estimated_runtime_minutes`, `is_powered`, `is_standing`, `current_location`, `saved_locations`, `estop_status` (`not_cut`, `cut`, `soft_stop`). Read the function for the full current key set.

### `capture_frame(camera="front") -> bytes` -- `robot/state.py`

Capture a JPEG frame from one of Spot's fisheye cameras.

### `is_navigating()` / `is_follow_mode()` -- `nav/runner.py`

Module-level helpers in `nav/runner.py` over the `NavRunner`. True while a navigation thread is alive (motor noise expected) or while the mode is "follow".

## Navigation Architecture

Navigation commands (`go_to`, `tour`, `patrol`, `come_back`, `go_to_object`, `follow_me`) run in **background threads** so the main voice loop stays responsive to safety commands. The state lives on `NavRunner` (`src/spot/nav/runner.py`): the thread, a stop event, the mode (`"graphnav"`, `"visual_nav"`, `"follow"`, `"emote"`, or None) and the saved home waypoint for `come_back`. `NavRunner.spawn(fn, mode, gen)` cancels any in-progress navigation (`cancel()`) before starting a new one.

### GraphNav Navigation

`go_to`, `tour` and `patrol` use BD's GraphNav API. `NavRunner.navigate_sequence()` supports:

- Multi-waypoint sequences with per-waypoint retry on STATUS_STUCK
- Continuous patrol looping (`repeat=True`)
- Re-localization on STATUS_LOST
- Skipping unreachable waypoints in multi-waypoint tours

### Visual Navigation

`go_to_object` delegates to `navigate_to_object()` and `follow_me` to `follow_person()`, both in `src/spot/features/visual_nav/navigator.py`.

## Camera Sources

`CAMERA_SOURCES` in `src/spot/robot/state.py`:

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

## Speed Limits

Applied by `set_speed` in `src/spot/features/walk/handlers.py` (approximate m/s velocity limits; the old doc called these multipliers).

| Speed | Limit (m/s) |
|-------|-----------|
| slow | 0.5 |
| normal | 1.0 |
| fast | 1.6 |

## Usage Example

```python
from src.spot.robot.dispatcher import dispatch_intent
from src.spot.robot.state import get_robot_state_dict
from src.spot.robot.connection import close_spot_session

# Execute a command (session auto-initializes on first call)
dispatch_intent({"intent": "walk", "params": {"direction": "forward", "distance": 2.0}})

# Get robot state for LLM context
state = get_robot_state_dict()

# Graceful shutdown
close_spot_session()
```

Do not run this on a live system; it connects to the robot.

## Dependencies

**Project modules:** `src.spot.robot.{connection,session,state,arm,motion}`, `src.spot.nav.{runner,location_manager}`, `src.spot.safety.gen`, `src.spot.core.core_intents`, the registry and `src.spot.features.*.handlers`

**External packages:** `bosdyn.client` (robot_command, frame_helpers, math_helpers, image, graph_nav)
