# Adding New Voice Commands

How to add a new voice command. After the 2026-10 restructure a command is a **feature**: one folder under `src/spot/features/` plus one line in `src/spot/features/enabled.py`. The LLM prompt line, the GBNF grammar, the regex table and the dispatcher routing are all generated from the feature declaration, so you no longer edit `llm_brain.py`, `intent.py` or the dispatcher by hand.

The full checklist is in [package-map.md](../architecture/package-map.md) (How to add a feature). This page is the short version with a worked example.

---

## How a command flows

1. **Safety** (`src/spot/safety/`): stop/freeze/estop are matched at frame level by the safety keyword spotter and again on the transcript. Core intents are declared only in `src/spot/core/core_intents.py`; do not add safety behaviour in a feature.
2. **LLM brain** (`src/spot/brain/llm_brain.py`): structured JSON with actions and a response. The prompt action list and the grammar come from the enabled features.
3. **Regex fallback** (`src/spot/brain/intent.py`): used with `--no-brain` and as backup. Rows come from each feature's `intent_rules`.
4. **Dispatcher** (`src/spot/robot/dispatcher.py`): core actions are handled there; any other action is routed to the feature's handler.

---

## Step 1: Declare the feature

Create `src/spot/features/spin/__init__.py`:

```python
from src.spot.core.registry import ActionSpec, Feature, IntentRule
from src.spot.features.spin import handlers as _h


def _spin_rule(m):
    return {"intent": "spin", "params": {"speed": "fast"}}


FEATURE = Feature(
    name="spin",
    actions=(
        ActionSpec(
            name="spin", handler=_h.handle_spin, grammar_params=None,
            prompt_line='- spin: Spin in a circle. Params: {"speed": "slow"|"fast"}. Default "fast".',
            locomotion=True, grammar_rank=300, prompt_rank=300),
    ),
    intent_rules=(
        IntentRule(priority=300, pattern=r"\b(?:spin|do a spin|pirouette)\b", build=_spin_rule),
    ),
)
```

- `prompt_line` is the line the LLM sees. Leave it `None` to keep the action regex-only.
- The `grammar_rank` / `prompt_rank` values order the generated lists. Copy `features/walk/__init__.py` or `features/dance/__init__.py` as a starting point; they show the full set of flags (`locomotion`, `info`, `short_path_blocked`, ...).
- `Registry.add` rejects a bad declaration at load (an action named like a core intent, or contradictory flags). Check the registry source for the rules.

## Step 2: Write the handler

Create `src/spot/features/spin/handlers.py`. A handler is `(params, ctx) -> bool`:

```python
def handle_spin(params, ctx):
    # Import robot owners at call time, not module level (avoids an import
    # cycle while the registry loads).
    import src.spot.robot as _r
    session = _r.CONNECTION.ensure()
    cmd_client = session["cmd"]
    speed = params.get("speed", "fast")
    # ... build and send the rotation command with the BD SDK ...
    return True
```

See `features/walk/handlers.py` for a real locomotion handler. Handlers marked `locomotion` are wrapped by the dispatcher (arm tuck before, restore after a natural finish, abort checks on the safety generation).

## Step 3: Enable it

Add one line to `ENABLED` in `src/spot/features/enabled.py`:

```python
    "spin",
```

Nothing else to edit. To turn it off again, delete that line.

---

## Adding a Background (Threaded) Command

Some commands run for an extended time (navigation, following, patrols). They run in a background thread so the voice pipeline keeps listening for "stop". Use the navigation runner (`NavRunner` in `src/spot/nav/runner.py`, owner `NAV` in `src/spot/robot/__init__.py`) instead of creating your own thread:

```python
def handle_my_long_action(params, ctx):
    import src.spot.robot as _r
    from src.spot.robot import motion as _motion

    def _worker(stop_event):
        while not stop_event.is_set():
            # work in small increments; check stop_event often so "stop" is responsive
            stop_event.wait(0.5)

    _r.NAV.cancel()                      # cancel any prior background action
    return _r.NAV.spawn(
        lambda ev: _motion.thread_then_restore(lambda: _worker(ev), ev),
        mode="my_mode", gen=_r.SAFETY_GEN.dispatch_current())
```

This is the pattern used by `features/follow/handlers.py`. Key rules:

- Check the stop event in your worker loop every 0.5-1 s.
- Call `NAV.cancel()` first so an earlier background action is stopped.
- Pass the safety generation (`gen`) so a stop issued before the thread starts wins.
- A stop from the safety path cancels the runner; you do not wire that yourself.

---

## Step 4: Test (no robot, nothing that opens the mic)

Check the generated outputs against the committed copies in the repo tests, and add a declaration test next to the existing ones, `tests/spot/features/test_<name>_declared.py`. See `tests/spot/test_registry_identity.py` for how the generated grammar, prompt and intent table are compared with golden copies. `src/spot/brain/grammar/spot_action.gbnf` is a committed snapshot of the generated grammar; it is produced from `spot_action.gbnf.tmpl` by `build_grammar` in `src/spot/brain/grammar/generate.py`, so do not edit it by hand.

For regex parsing:

```python
from src.spot.brain.intent import parse_intent
print(parse_intent("do a spin"))
# {"intent": "spin", "params": {"speed": "fast"}, "raw": "do a spin"}
```

For the full pipeline, run `python scripts/run_voice_control.py` (on the robot machine, when the demo pipeline is not already running) and say the command.

---

## Tips

- Keep action names lowercase with underscores, and never reuse a core intent name.
- The LLM is the primary path; regex is the fallback for `--no-brain` mode.
- To read robot state, use `get_robot_state_dict()` in `src/spot/robot/state.py`.
- To capture camera images, see `capture_frame()` in `src/spot/robot/state.py`.
- To test without a physical robot, run with `--dry-run`.
