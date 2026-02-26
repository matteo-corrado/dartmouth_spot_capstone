# Adding New Voice Commands

How to add new voice commands to the system. There are two layers: the LLM brain (primary) and the regex parser (fallback).

---

## Architecture Overview

When a user speaks a command, it flows through:

1. **Safety regex** (`client_mic.py`) — instant match for stop/freeze/estop
2. **LLM Brain** (`llm_brain.py`) — structured JSON output with actions + response
3. **Regex fallback** (`intent.py`) — used in `--no-brain` mode or as LLM backup

To add a new command, you typically modify all three layers.

---

## Step 1: Add to the LLM System Prompt

Edit `src/voice_control/llm_brain.py`, find the `SYSTEM_PROMPT` string.

### Add the action definition

In the `AVAILABLE ACTIONS:` section, add a line:

```
- my_action: Description of what it does. Params: {"param1": "<type>", "param2": <type>}.
```

### Add a non-obvious example (optional)

Only add an example if the mapping is ambiguous. Simple commands like "do X" → `my_action` don't need examples.

Add examples for:
- Commands where multiple actions could match
- Commands with unusual parameter extraction
- Commands that differ from similar-sounding ones

### Add rules (if needed)

If there's disambiguation logic (like go_to vs go_to_object), add a RULES entry.

---

## Step 2: Add to the Regex Parser

Edit `src/voice_control/intent.py`, add a pattern to the `COMMANDS` list:

```python
COMMANDS = [
    # ... existing patterns ...

    # === MY NEW COMMAND ===
    (r"\b(?:do something|activate widget)\s+(?P<target>.+?)(?:\s*\.|$)",
     "my_action", {"target": "$target"}),
]
```

Pattern syntax:
- Use `\b` word boundaries
- Use `(?:...)` non-capturing groups for alternatives
- Use `(?P<name>...)` named groups for parameter extraction
- Use `$name` in params to reference captured groups

The parser returns `{"intent": "my_action", "params": {"target": "..."}, "raw": "original text"}`.

---

## Step 3: Add the Executor

Edit `src/voice_control/spot_dispatch.py`, find the `dispatch_intent()` function.

Add a new case:

```python
def dispatch_intent(intent):
    cmd = intent["intent"]
    params = intent.get("params", {})

    # ... existing cases ...

    elif cmd == "my_action":
        target = params.get("target", "default")
        return _execute_my_action(target)
```

Then implement the handler function:

```python
def _execute_my_action(target):
    """Execute my custom action."""
    session = ensure_spot_session()
    if not session:
        return False

    # Use BD SDK to control Spot
    robot = session["robot"]
    cmd_client = session["cmd"]

    # ... your implementation ...
    return True
```

---

## Step 4: Test

### Test LLM understanding (no robot needed)

```bash
python src/voice_control/llm_brain.py
```

Type your command and verify the JSON output contains the correct action and params.

### Test regex parsing

```python
from src.voice_control.intent import parse_intent
result = parse_intent("do something with the widget")
print(result)
# Should output: {"intent": "my_action", "params": {"target": "widget"}, "raw": "..."}
```

### Test full pipeline

```bash
python scripts/run_voice_control.py
```

Say the command and verify it executes correctly.

---

## Example: Adding a "spin" Command

### 1. LLM Prompt (llm_brain.py)

```
- spin: Spin in a circle. Params: {"speed": "slow"|"fast"}. Default "fast".
```

### 2. Regex (intent.py)

```python
(r"\b(?:spin|do a spin|pirouette)\b", "spin", {"speed": "fast"}),
(r"\bslow spin\b", "spin", {"speed": "slow"}),
```

### 3. Executor (spot_dispatch.py)

```python
elif cmd == "spin":
    speed = params.get("speed", "fast")
    deg_per_sec = 180 if speed == "fast" else 60
    # Turn 360 degrees using existing turn logic
    from bosdyn.client.robot_command import RobotCommandBuilder
    cmd_client = session["cmd"]
    # ... build and send rotation command ...
    return True
```

---

## Adding a Background (Threaded) Command

Some commands run for an extended time — navigation, following, patrols. These run in a background thread so the voice pipeline can keep listening for "stop" or other commands.

The pattern used by `go_to`, `follow_me`, and `tour`:

### 1. Thread + stop event in spot_dispatch.py

```python
import threading

# Module-level globals (already exist in spot_dispatch.py)
_nav_thread = None
_nav_stop_event = threading.Event()

def _execute_my_long_action(target):
    """Background action that can be cancelled."""
    global _nav_thread, _nav_stop_event

    # Cancel any existing background action
    _cancel_nav()

    session = ensure_spot_session()
    if not session:
        return False

    _nav_stop_event.clear()

    def _worker():
        # Your long-running logic here
        while not _nav_stop_event.is_set():
            # Do work in small increments
            # Check _nav_stop_event frequently so "stop" is responsive
            time.sleep(0.5)

    _nav_thread = threading.Thread(target=_worker, daemon=True)
    _nav_thread.start()
    return True
```

### 2. Cancellation support

The `_cancel_nav()` function (already in `spot_dispatch.py`) sets the stop event and waits for the thread:

```python
def _cancel_nav():
    global _nav_thread
    if _nav_thread and _nav_thread.is_alive():
        _nav_stop_event.set()
        _nav_thread.join(timeout=5)
    _nav_thread = None
```

When the user says "stop", `dispatch_intent` calls `_cancel_nav()` which signals the background thread to exit.

### 3. Key rules for background commands

- Always check `_nav_stop_event.is_set()` in your worker loop (every 0.5-1s)
- Call `_cancel_nav()` at the start to cancel any prior background action
- Use `daemon=True` threads so they don't block program exit
- Store the thread in `_nav_thread` so the stop system knows about it

---

## Tips

- Keep action names lowercase with underscores
- Match the intent name across all three files (prompt, regex, dispatch)
- The LLM is the primary path — regex is fallback for `--no-brain` mode
- Test the LLM interactively before adding regex patterns
- For reading robot state in your handler, use `session["state"].get_robot_state()`
- For capturing camera images, see `capture_frame()` in `spot_dispatch.py`
- To test without a physical robot, use `--no-spot` in test scripts that support it
