# intent.py

`src/voice_control/intent.py` -- Regex-based intent parser.

Maps free-text voice commands to structured intent dicts using precompiled
regex patterns. Used as the primary parser in `--no-brain` mode and as the
safety command fast-path in LLM mode.

## Key Function

### `parse_intent(text) -> dict | None`

```python
def parse_intent(text: str) -> dict | None
```

Cleans Whisper punctuation artifacts (removes periods), then searches 80+
precompiled patterns in priority order. Returns on first match.

**Returns:**
```python
{"intent": "walk", "params": {"direction": "forward", "distance": 2.0}, "raw": "walk forward 2 meters"}
# or
None  # no match
```

## Pattern Categories

| Category | Example Intents |
|----------|----------------|
| Safety | `stop`, `freeze`, `estop` |
| Posture | `stand`, `sit`, `selfright` |
| Body height | `body_height` (crouch/tall/normal) |
| Walking | `walk` (forward/backward with optional distance) |
| Strafing | `strafe` (left/right with optional distance) |
| Turning | `turn` (left/right with optional degrees, "turn around" = 180) |
| Speed | `set_speed` (slow/normal/fast) |
| Visual nav | `go_to_object` (find/look for/search), `follow_me` |
| Navigation | `go_to`, `tour`, `patrol`, `come_back`, `save_location` |
| Status | `battery_status`, `status`, `power_off` |
| Door | `open_door` |
| Other | `list_locations` |

## Pattern Features

- All patterns are case-insensitive (`re.IGNORECASE`).
- Named capture groups extract parameters (e.g. `(?P<dist>\d+(?:\.\d+)?)`).
- Template values use `$name` syntax to substitute captured groups.
- Numeric parameters are auto-converted: `"2"` -> `int(2)`, `"1.5"` -> `float(1.5)`.
- Location names are normalized to `lowercase_with_underscores`.
- Degree values are clamped to [0, 360].
- Patterns are ordered by specificity (distance variants before bare commands).

## Internal Helpers

### `_convert_numeric(s) -> int | float | str`

Convert string numbers to int or float. Returns original value if not numeric.

### `_resolve_param(value, match) -> Any`

Resolve a param template against a regex match. Supports `$name` group
substitution, callable values, and recursive list/dict resolution.

## Usage Example

```python
from intent import parse_intent

result = parse_intent("turn left 45 degrees")
# {"intent": "turn", "params": {"deg": 45, "dir": "left"}, "raw": "turn left 45 degrees"}

result = parse_intent("go to the kitchen")
# {"intent": "go_to", "params": {"location": "kitchen"}, "raw": "go to the kitchen"}

result = parse_intent("find the red chair")
# {"intent": "go_to_object", "params": {"description": "red chair"}, "raw": "find the red chair"}

result = parse_intent("hello there")
# None
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `re` (stdlib)
