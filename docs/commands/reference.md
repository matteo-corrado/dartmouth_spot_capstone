# Voice Command Reference

Complete reference for all voice commands supported by the Spot voice control
system. Commands are processed by the LLM Brain (`qwen2.5:7b`) which interprets
natural language and maps it to structured actions.

## Wake Word

Say **"Hey Spot"** before giving a command. The wake word detector uses
sherpa-onnx keyword spotting and runs on every audio frame with near-zero
latency.

Two usage patterns:

1. **Two-step:** Say "Hey Spot", wait for the chime, then say your command.
2. **One-breath:** Say "Hey Spot stand up" in a single utterance. The wake
   phrase is stripped and the command ("stand up") is processed immediately.

After the wake word triggers, the system listens for 5 minutes before
requiring "Hey Spot" again.

Safety commands (`stop`, `freeze`, `estop`) always work regardless of wake
word state.

## Command Table

### Safety

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| stop | "stop", "halt" | Stop all movement immediately | None |
| freeze | "freeze" | Hold current position | None |
| estop | "emergency stop", "e-stop" | Emergency stop (cuts motors) | None |

Safety commands bypass the LLM entirely -- they are matched by regex in
under 5ms and dispatched directly to the robot.

### Posture

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| stand | "stand up", "get up" | Stand up from sitting | None |
| sit | "sit down", "lay down" | Sit down | None |
| selfright | "self right", "recover", "get up from fall" | Recover from a fall | None |

### Body

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| crouch | "crouch", "get low", "duck" | Lower body to minimum height | `height: -0.15` |
| stand tall | "stand tall", "max height" | Raise body to maximum height | `height: 0.1` |
| normal height | "normal height" | Reset body to default height | `height: 0.0` |

Body height ranges from -0.15 (crouch) to 0.1 (tall). The LLM can also set
intermediate heights when asked (e.g., "lower yourself a bit").

### Movement

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| walk forward | "walk forward", "go forward 2 meters" | Walk forward | `direction: forward, distance: <m>` |
| walk backward | "walk backward", "back up" | Walk backward | `direction: backward, distance: <m>` |
| strafe left | "strafe left", "move left" | Sideways movement left | `direction: left, distance: <m>` |
| strafe right | "strafe right", "move right" | Sideways movement right | `direction: right, distance: <m>` |
| turn left | "turn left", "turn left 45 degrees" | Rotate left in place | `dir: left, deg: <degrees>` |
| turn right | "turn right" | Rotate right in place | `dir: right, deg: <degrees>` |
| turn around | "turn around" | Rotate 180 degrees | `deg: 180` |

**Default values:**
- Walk distance: 1.0 meters
- Strafe distance: 0.5 meters
- Turn angle: 90 degrees

### Speed

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| slow | "go slow", "slow down" | Set speed to slow | `speed: slow` |
| normal | "normal speed" | Set speed to normal | `speed: normal` |
| fast | "go fast", "speed up" | Set speed to fast | `speed: fast` |

### Navigation

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| go to | "go to kitchen", "navigate to lab" | Navigate to a saved location | `location: <name>` |
| tour | "tour kitchen then lab", "visit everywhere" | Visit locations in sequence (one pass) | `locations: [<names>]` or `locations: "all"` |
| patrol | "patrol the hallway and lab", "keep patrolling" | Loop through locations continuously | `locations: [<names>]` or `locations: "all"` |
| come back | "come back", "go home", "return" | Return to position before last navigation | None |
| save location | "save this location as kitchen" | Save current position with a name | `location: <name>` |
| list locations | "list locations", "what locations do you know?" | List all saved locations | None |

Navigation uses GraphNav (BD SDK). Locations are stored in `locations.json`
as human-readable names mapped to GraphNav waypoint IDs.

- **Tour** visits each location once in order, then stops.
- **Patrol** loops continuously until you say "stop".
- **Come back** returns to the waypoint the robot was at before the last
  `go_to`, `tour`, or `patrol` command.

Location names are automatically normalized to `lowercase_with_underscores`.

### Vision

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| describe | "what do you see?", "look around" | Capture photo and describe scene via VLM | `camera: <side>` |
| describe with query | "do you see a chair?", "is there a person?" | Describe scene with YOLO hint for specific object | `camera: <side>, query: <text>` |
| go_to_object | "go to the red chair", "find the backpack" | Walk toward a visible object using camera | `description: <text>` |
| follow_me | "follow me", "come with me" | Follow the nearest person | None |

**Describe** uses the VLM (`qwen2.5vl:7b`). When a specific query is given,
YOLO-World runs first to provide a detection hint to the VLM.

**go_to_object** uses YOLO-World (open-vocabulary) to detect the target
object across all cameras, then steers toward it. This is for visible
objects, not saved map locations. If a saved location matches the name,
`go_to` (GraphNav) is used instead.

**follow_me** uses YOLOv8n for person detection on CPU. The robot follows
the nearest detected person, maintaining distance.

Camera options: `front` (default), `left`, `right`, `back`.

### Status

| Voice Command | Example Phrases | Action | Parameters |
|---|---|---|---|
| battery | "battery level", "how much battery?" | Report battery percentage and runtime | None |
| status | "status report", "how are you doing?" | Full status: battery, posture, location, faults | None |
| power off | "power off", "shut down" | Safely power off the robot | None |

## Command Chaining

The LLM supports chaining multiple actions in a single utterance. Actions
execute sequentially with synchronization waits between them.

**Examples:**

| Utterance | Actions |
|---|---|
| "go to kitchen then sit down" | `go_to(kitchen)` -> wait for nav -> `sit` |
| "stand up and walk forward 3 meters" | `stand` -> wait 4s -> `walk(forward, 3)` |
| "go to lab then come back" | `go_to(lab)` -> wait for nav -> `come_back` |
| "turn left and walk forward" | `turn(left, 90)` -> wait 3s -> `walk(forward, 1)` |

If any action in the chain fails, the remaining actions are skipped.

## Multi-Location Navigation

When visiting multiple locations, prefer naming them in a single command.
The LLM will use a single `tour` action instead of chaining multiple `go_to`
actions, which is more efficient.

| Utterance | Parsed As |
|---|---|
| "go to A then B then C" | `tour(locations: [A, B, C])` (single action) |
| "visit everywhere" | `tour(locations: "all")` |
| "keep patrolling A and B" | `patrol(locations: [A, B])` |

## Conversation

The LLM also handles conversational queries that do not require physical
actions. For example:

- "How are you doing?" -- responds conversationally (no action)
- "What's your name?" -- responds conversationally (no action)
- "Tell me a joke" -- responds conversationally (no action)

The LLM maintains a conversation history of the last 12 messages (6 turns)
for context.
