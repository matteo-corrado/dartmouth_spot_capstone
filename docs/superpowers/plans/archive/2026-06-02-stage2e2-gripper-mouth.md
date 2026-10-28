# Stage 2E.2 Gripper Puppet Mouth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spot's gripper opens and closes in sync with TTS audio, persona-scaled, with a runtime enable/disable and arm-safety gating.

**Architecture:** Each TTS chunk's PCM is fully materialized before playback (`AudioPlayer._materialize`). A pure `envelope()` function turns those samples into a per-frame `gripper_open_fraction` array at `MOUTH_FPS`. A `MouthDriver` thread paces those fractions against the `on_play_start` monotonic clock and commands the gripper; `on_play_end` closes it. No parallel PCM tap.

**Tech Stack:** Python 3.10, numpy, scipy.signal (`butter`/`lfilter`), bosdyn-client (`RobotCommandBuilder.claw_gripper_open_fraction_command`, `arm_ready_command`, `arm_stow_command`), pytest.

**Safety deviation from spec §6 (surfaced):** startup default is **mouth disabled / arm stowed**. `enable_mouth` deploys the arm and enables; `disable_mouth` disables and stows. Auto-deploying an arm near people at startup is unsafe, so we do not default-on as §6 sketched.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `scripts/spike_gripper_rate.py` | Hardware spike: measure max gripper command rate (Task 0 gate) |
| Create | `src/voice_control/animation/__init__.py` | Module marker |
| Create | `src/voice_control/animation/mouth.py` | `MOUTH_FPS`, `envelope()` pure fn, `MouthDriver` class |
| Create | `tests/voice_control/animation/__init__.py` | Test package marker |
| Create | `tests/voice_control/animation/test_mouth.py` | `envelope()` + `MouthDriver` unit tests |
| Modify | `src/session.py` | `deploy_arm_safe()` / `stow_arm()` helpers |
| Modify | `src/voice_control/chatbot/session_state.py` | Add `arm_deployed`, `mouth_enabled` fields |
| Modify | `src/voice_control/chatbot/persona.py` | Add `mouth_intensity` to `Persona` + `load_registry` |
| Modify | `config/personas.yaml` | Add `mouth_intensity` per persona |
| Modify | `src/voice_control/spot_tts.py` | Attach `mouth`; compute envelope in `_render`; hook `on_play_start`/`on_play_end` |
| Modify | `src/voice_control/spot_dispatch.py` | `enable_mouth`/`disable_mouth`; close+stow on stop/freeze/estop |
| Modify | `src/voice_control/grammar/spot_action.gbnf` | Add `enable_mouth`/`disable_mouth` to `action-name` |
| Modify | `src/voice_control/client_mic.py` | Construct `MouthDriver`, attach to TTS singleton |
| Modify | `docs/project/stage2-rollback.md` | Append 2E.2 rollback layer |

---

## Task 0: Feasibility spike (HARD GATE — no other task starts until this passes)

**Files:**
- Create: `scripts/spike_gripper_rate.py`

This task is hardware measurement, not TDD. Spot must be powered with the arm installed.

- [ ] **Step 1: Write the spike script**

```python
"""Stage 2E.2 feasibility spike — measure max gripper command rate on Spot.

Deploys the arm to the ready pose, then modulates gripper_open_fraction by a
sine wave at several target rates. Prints the ACHIEVED rate and jitter so we
know the practical ceiling before building the mouth animator.

Run on a powered Spot with arm installed, operator holding e-stop:
    python scripts/spike_gripper_rate.py --hostname $SPOT_IP
"""
import argparse, math, time
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_command import (
    RobotCommandClient, RobotCommandBuilder, blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive


def run_rate(cmd_client, hz: float, secs: float = 30.0) -> dict:
    period = 1.0 / hz
    n = int(secs * hz)
    start = time.monotonic()
    sent = 0
    gaps = []
    last = start
    for i in range(n):
        target = start + i * period
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        frac = 0.5 * (1.0 + math.sin(2 * math.pi * 1.0 * (time.monotonic() - start)))
        cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(frac))
        sent += 1
        t = time.monotonic()
        gaps.append(t - last)
        last = t
    elapsed = time.monotonic() - start
    gaps = gaps[1:]  # drop first
    mean_gap = sum(gaps) / len(gaps)
    jitter = (max(gaps) - min(gaps))
    return {"target_hz": hz, "achieved_hz": sent / elapsed,
            "mean_gap_ms": mean_gap * 1000, "jitter_ms": jitter * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hostname", required=True)
    args = ap.parse_args()
    sdk = bosdyn.client.create_standard_sdk("mouth-spike")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "Spot has no arm — mouth animation is impossible"
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        blocking_stand(cmd_client, timeout_sec=10)
        cmd_id = cmd_client.robot_command(RobotCommandBuilder.arm_ready_command())
        block_until_arm_arrives(cmd_client, cmd_id, 5.0)
        for hz in (5, 10, 20, 30):
            print(run_rate(cmd_client, hz))
        cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(0.0))
        cmd_id = cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
        block_until_arm_arrives(cmd_client, cmd_id, 5.0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on Spot and record results**

Run: `python scripts/spike_gripper_rate.py --hostname $SPOT_IP`
Expected: four dicts printed, one per target rate. Record `achieved_hz` and `jitter_ms` for each.

- [ ] **Step 3: Decide the fork and set MOUTH_FPS**

- If achieved_hz ≥ 10 at some rate with smooth visible motion → set `MOUTH_FPS` (Task 1) to the highest smooth rate (cap at 20). Proceed with envelope-streaming as planned.
- If capped below ~10 Hz → STOP. The envelope-streaming approach is not viable; revisit the design's syllable-trigger fork before continuing. Do not proceed to Task 1.

- [ ] **Step 4: Commit**

```bash
git add scripts/spike_gripper_rate.py
git commit -m "stage2e2: gripper rate feasibility spike script"
```

- [ ] **Step 5: REVIEW GATE 1** — review the recorded rate/jitter against the ≥10 Hz threshold with the user before writing animation code.

---

## Task 1: Envelope function

**Files:**
- Create: `src/voice_control/animation/__init__.py` (empty)
- Create: `src/voice_control/animation/mouth.py`
- Create: `tests/voice_control/animation/__init__.py` (empty)
- Create: `tests/voice_control/animation/test_mouth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/voice_control/animation/test_mouth.py
import numpy as np
from src.voice_control.animation.mouth import envelope, MOUTH_FPS


def _tone(freq, secs, rate=24000, amp=0.8):
    t = np.arange(int(secs * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_silence_keeps_gripper_closed():
    env = envelope(np.zeros(24000, dtype=np.float32), rate=24000, fps=20)
    assert env.shape[0] == 20  # 1s at 20 fps
    assert float(env.max()) == 0.0


def test_tone_opens_gripper():
    env = envelope(_tone(200, 1.0), rate=24000, fps=20)
    assert float(env.max()) > 0.9          # normalized to peak
    assert float(env.max()) <= 1.0


def test_gate_cuts_inter_word_silence():
    voiced = _tone(200, 0.5)
    silent = np.zeros(int(0.5 * 24000), dtype=np.float32)
    env = envelope(np.concatenate([voiced, silent]), rate=24000, fps=20)
    assert float(env[:10].max()) > 0.5     # first 0.5s voiced
    assert float(env[10:].max()) == 0.0    # last 0.5s gated to closed


def test_intensity_scales_and_clamps():
    base = envelope(_tone(200, 1.0), rate=24000, fps=20, intensity=1.0)
    hot = envelope(_tone(200, 1.0), rate=24000, fps=20, intensity=2.0)
    assert float(hot.max()) <= 1.0         # clamped
    # a mid-range frame should scale up under higher intensity
    mid_i = int(np.argmin(np.abs(base - 0.4)))
    assert hot[mid_i] >= base[mid_i]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/voice_control/animation/test_mouth.py -v`
Expected: FAIL — `ModuleNotFoundError: src.voice_control.animation.mouth`

- [ ] **Step 3: Write minimal implementation**

```python
# src/voice_control/animation/mouth.py
"""Stage 2E.2 — gripper puppet mouth.

`envelope()` turns a TTS chunk's PCM into a per-frame gripper_open_fraction
array. `MouthDriver` paces those fractions to the arm in sync with playback.

MOUTH_FPS is set from the Task 0 feasibility spike — the proven smooth
gripper command rate on our Spot. Default 20; lower it if the spike found a
lower ceiling.
"""
import numpy as np
from scipy.signal import butter, lfilter

MOUTH_FPS = 20  # set from spike_gripper_rate.py measurement


def envelope(samples: np.ndarray, rate: int, fps: int = MOUTH_FPS,
             gate: float = 0.06, intensity: float = 1.0,
             cutoff_hz: float = 50.0) -> np.ndarray:
    """PCM float32 in [-1,1] -> per-frame gripper_open_fraction in [0,1].

    rectify -> lowpass -> window-max downsample to fps -> normalize by peak
    -> noise gate -> intensity scale + clamp.
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    rect = np.abs(samples.astype(np.float32))
    b, a = butter(2, cutoff_hz / (rate / 2.0), btype="low")
    smooth = lfilter(b, a, rect)
    hop = max(1, int(round(rate / fps)))
    n_frames = int(np.ceil(smooth.size / hop))
    frames = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frames[i] = smooth[i * hop:(i + 1) * hop].max()
    peak = float(frames.max())
    if peak > 0:
        frames = frames / peak
    frames[frames < gate] = 0.0
    return np.clip(frames * intensity, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/voice_control/animation/test_mouth.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/animation/__init__.py src/voice_control/animation/mouth.py tests/voice_control/animation/
git commit -m "stage2e2: PCM envelope -> gripper fraction (pure fn + tests)"
```

---

## Task 2: MouthDriver (paced gripper thread, dry-run testable)

**Files:**
- Modify: `src/voice_control/animation/mouth.py`
- Modify: `tests/voice_control/animation/test_mouth.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/voice_control/animation/test_mouth.py
import time
from src.voice_control.animation.mouth import MouthDriver


def test_driver_dry_run_emits_all_frames_in_order():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    fractions = [0.0, 0.5, 1.0, 0.5, 0.0]
    d.play(fractions, t0=time.monotonic())
    d.wait()
    assert [round(f, 3) for _, f in d.log] == fractions


def test_driver_dry_run_paces_to_clock():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    t0 = time.monotonic()
    d.play([0.0, 0.5, 1.0], t0=t0)
    d.wait()
    for i, (t_emit, _) in enumerate(d.log):
        assert abs((t_emit - t0) - i / 10.0) < 0.05


def test_disabled_driver_is_noop():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)  # starts disabled
    d.play([1.0, 1.0, 1.0], t0=time.monotonic())
    d.wait()
    assert d.log == []


def test_close_forces_zero():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    d.close()
    assert d.log[-1][1] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/voice_control/animation/test_mouth.py -k driver -v`
Expected: FAIL — `cannot import name 'MouthDriver'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/voice_control/animation/mouth.py
import threading
import time as _time

try:
    from bosdyn.client.robot_command import RobotCommandBuilder
except Exception:  # bosdyn not importable in unit-test env
    RobotCommandBuilder = None


class MouthDriver:
    """Paces gripper_open_fraction commands to the arm in sync with playback.

    Disabled by default (arm-safety). `enable()` is called by the
    enable_mouth dispatch handler AFTER the arm is deployed. `dry_run`
    records fractions to `self.log` instead of commanding the robot.
    """

    def __init__(self, cmd_client=None, fps: int = MOUTH_FPS,
                 dry_run: bool = False):
        self._cmd = cmd_client
        self.fps = fps
        self.dry_run = dry_run
        self.enabled = False
        self.intensity = 1.0
        self.log = []  # list[(monotonic_ts, fraction)] — dry_run only
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.close()

    def _send(self, frac: float):
        frac = float(max(0.0, min(1.0, frac)))
        if self.dry_run:
            self.log.append((_time.monotonic(), frac))
            return
        if self._cmd is None or RobotCommandBuilder is None:
            return
        self._cmd.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(frac))

    def play(self, fractions, t0: float):
        if not self.enabled:
            return
        self._cancel()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(list(fractions), t0, self._stop),
            daemon=True)
        self._thread.start()

    def _run(self, fractions, t0, stop):
        for i, frac in enumerate(fractions):
            if stop.is_set():
                return
            target = t0 + i / self.fps
            now = _time.monotonic()
            if target > now:
                stop.wait(target - now)
            if stop.is_set():
                return
            self._send(frac)

    def _cancel(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._stop.set()
                self._thread.join(timeout=1.0)

    def wait(self):
        if self._thread:
            self._thread.join(timeout=5.0)

    def close(self):
        """Force gripper closed and cancel any in-flight playback."""
        self._cancel()
        self._send(0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/voice_control/animation/test_mouth.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/animation/mouth.py tests/voice_control/animation/test_mouth.py
git commit -m "stage2e2: MouthDriver paced gripper thread + dry-run tests"
```

---

## Task 3: Arm deploy/stow helpers

**Files:**
- Modify: `src/session.py`

No unit test — these are thin bosdyn wrappers exercised live in Task 10. Keep them minimal.

- [ ] **Step 1: Add helpers at the end of `src/session.py`**

```python
def deploy_arm_safe(cmd_client, timeout_sec: float = 5.0):
    """Deploy the arm to the ready pose (chest-height carry). Caller must
    have motors powered and the robot standing. Returns True on arrival."""
    from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
    cmd_id = cmd_client.robot_command(RobotCommandBuilder.arm_ready_command())
    return block_until_arm_arrives(cmd_client, cmd_id, timeout_sec)


def stow_arm(cmd_client, timeout_sec: float = 5.0):
    """Close the gripper then stow the arm. Safe to call when already stowed."""
    from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
    cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(0.0))
    cmd_id = cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
    return block_until_arm_arrives(cmd_client, cmd_id, timeout_sec)
```

- [ ] **Step 2: Verify import resolves**

Run: `python -c "import src.session as s; assert hasattr(s,'deploy_arm_safe') and hasattr(s,'stow_arm')"`
Expected: no output, exit 0

- [ ] **Step 3: Commit**

```bash
git add src/session.py
git commit -m "stage2e2: arm deploy_arm_safe / stow_arm helpers"
```

---

## Task 4: SessionState fields

**Files:**
- Modify: `src/voice_control/chatbot/session_state.py`
- Modify: `tests/voice_control/chatbot/test_session_state.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/voice_control/chatbot/test_session_state.py
from src.voice_control.chatbot.session_state import SessionState


def test_mouth_and_arm_defaults_safe():
    s = SessionState()
    assert s.arm_deployed is False
    assert s.mouth_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/voice_control/chatbot/test_session_state.py::test_mouth_and_arm_defaults_safe -v`
Expected: FAIL — `AttributeError: 'SessionState' object has no attribute 'arm_deployed'`

- [ ] **Step 3: Add the fields**

In `src/voice_control/chatbot/session_state.py`, add to the dataclass body after `last_comment_ts`:

```python
    arm_deployed: bool = False
    mouth_enabled: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/voice_control/chatbot/test_session_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/chatbot/session_state.py tests/voice_control/chatbot/test_session_state.py
git commit -m "stage2e2: SessionState arm_deployed + mouth_enabled (default off)"
```

---

## Task 5: Per-persona mouth_intensity

**Files:**
- Modify: `src/voice_control/chatbot/persona.py`
- Modify: `config/personas.yaml`
- Modify: `tests/voice_control/chatbot/test_persona.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/voice_control/chatbot/test_persona.py
from src.voice_control.chatbot.persona import load_registry, get_persona


def test_persona_mouth_intensity_loaded_and_defaulted(tmp_path):
    reg = load_registry()  # real config/personas.yaml
    assert get_persona("pirate", reg).mouth_intensity == 1.3
    assert get_persona("tour_guide", reg).mouth_intensity == 1.0  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/voice_control/chatbot/test_persona.py::test_persona_mouth_intensity_loaded_and_defaulted -v`
Expected: FAIL — `AttributeError: 'Persona' object has no attribute 'mouth_intensity'`

- [ ] **Step 3: Add the field + loader read**

In `src/voice_control/chatbot/persona.py`, add to the `Persona` dataclass (after `ack_template`):

```python
    mouth_intensity: float = 1.0
```

In `load_registry`, where the `Persona(...)` is constructed (around line 73), add the kwarg:

```python
            mouth_intensity=float(entry.get("mouth_intensity", 1.0)),
```

Also add `mouth_intensity=1.0` to the `DEFAULT_PERSONA` fallback `Persona(...)` near line 83 so it stays valid.

- [ ] **Step 4: Add intensity to `config/personas.yaml`**

Add a `mouth_intensity:` line to the `pirate` block (value `1.3`). Leave others to default 1.0; optionally add `0.7` to a `butler` block if present.

```yaml
pirate:
  # ... existing keys ...
  mouth_intensity: 1.3
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/voice_control/chatbot/test_persona.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/chatbot/persona.py config/personas.yaml tests/voice_control/chatbot/test_persona.py
git commit -m "stage2e2: per-persona mouth_intensity (pirate 1.3, default 1.0)"
```

---

## Task 6: Wire envelope into SpotTTS playback

**Files:**
- Modify: `src/voice_control/spot_tts.py`

The `_render` closure already produces `(samples, 24000)`. We compute the envelope there into a per-call holder, then drive the mouth from the play callbacks. `on_play_start`/`on_play_end` already carry latency-trace lambdas — compose, don't overwrite.

- [ ] **Step 1: Add `mouth` attribute to `SpotTTS.__init__`**

In `src/voice_control/spot_tts.py`, at the end of `__init__` (after `self._load_model()` or before it, but after `self._player`):

```python
        self.mouth = None  # set by client_mic after the Spot session is up
```

- [ ] **Step 2: Compute envelope in `_render` and drive from callbacks**

In `speak()`, inside `_render`, change the return to also stash the envelope. Replace the final lines of `_render`:

```python
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if gain != 1.0:
                samples = samples * gain
            return samples, 24000
```

with:

```python
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if gain != 1.0:
                samples = samples * gain
            if self.mouth is not None and self.mouth.enabled:
                from src.voice_control.animation.mouth import envelope
                frame_holder["frames"] = envelope(
                    samples, 24000, fps=self.mouth.fps,
                    intensity=self.mouth.intensity)
            return samples, 24000
```

Immediately before `def _render():`, add the holder:

```python
        frame_holder = {}
        import time as _t
```

- [ ] **Step 3: Compose the play callbacks**

In `speak()`, the block that sets `cb_play_start`/`cb_play_end` (currently trace-only or None). Replace the assignment of those two so they also drive the mouth. After the existing `if trace is not None: ... else: cb_render_start = cb_render_end = cb_play_start = cb_play_end = None`, add:

```python
        _trace_play_start, _trace_play_end = cb_play_start, cb_play_end

        def cb_play_start():
            if _trace_play_start:
                _trace_play_start()
            frames = frame_holder.get("frames")
            if self.mouth is not None and self.mouth.enabled and frames is not None and len(frames):
                self.mouth.play(frames, _t.monotonic())

        def cb_play_end():
            if _trace_play_end:
                _trace_play_end()
            if self.mouth is not None:
                self.mouth.close()
```

(These shadow the prior `cb_play_start`/`cb_play_end` names, so the existing `enqueue_render(..., on_play_start=cb_play_start, on_play_end=cb_play_end)` call needs no change.)

- [ ] **Step 4: Verify import + no syntax error**

Run: `python -c "import src.voice_control.spot_tts"`
Expected: no error (module imports; backend smoke-check may warn but must not raise ImportError)

- [ ] **Step 5: Verify existing TTS tests still pass**

Run: `pytest tests/voice_control/tts/ -v`
Expected: PASS (no regressions — mouth is None in tests, so all hooks are no-ops)

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/spot_tts.py
git commit -m "stage2e2: drive MouthDriver from TTS render + play callbacks"
```

---

## Task 7: Dispatch actions + safety wiring

**Files:**
- Modify: `src/voice_control/spot_dispatch.py`

- [ ] **Step 1: Add the two handlers** (place near `_handle_set_backend`)

```python
def _handle_enable_mouth(params):
    """Deploy the arm and enable gripper mouth animation."""
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
    """Disable mouth animation, close the gripper, and stow the arm."""
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
```

- [ ] **Step 2: Route the actions in `dispatch_intent`**

`enable_mouth` needs a session (arm deploy); route it inside the `try:` block alongside `stop`/`freeze`. Add these branches at the top of the `if name == "stop":` chain (after `cmd_client = session["cmd"]`):

```python
        if name == "enable_mouth":
            return _handle_enable_mouth(params)
        if name == "disable_mouth":
            return _handle_disable_mouth(params)
```

- [ ] **Step 3: Close the mouth on stop/freeze/estop**

In the `stop` and `freeze` handler bodies (and `estop` if present), add as the FIRST line inside each handler, before sending the robot command:

```python
            try:
                from src.voice_control.spot_tts import get_tts
                _t = get_tts()
                if _t is not None and _t.mouth is not None:
                    _t.mouth.close()
            except Exception:
                pass
```

- [ ] **Step 4: Verify import**

Run: `python -c "import src.voice_control.spot_dispatch"`
Expected: no error

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/spot_dispatch.py
git commit -m "stage2e2: enable_mouth/disable_mouth actions + close gripper on stop/freeze/estop"
```

---

## Task 8: Grammar allowlist

**Files:**
- Modify: `src/voice_control/grammar/spot_action.gbnf`

Omitting this makes the LLM unable to emit the new actions (known misfire gap).

- [ ] **Step 1: Add the actions to `action-name`**

Append `| "\"enable_mouth\"" | "\"disable_mouth\""` to the end of the single-line `action-name ::=` rule (keep it on one line — the parser fails on multi-line alternation).

- [ ] **Step 2: Verify the grammar still loads**

Run: `python -c "from pathlib import Path; g=Path('src/voice_control/grammar/spot_action.gbnf').read_text(); assert 'enable_mouth' in g and 'disable_mouth' in g and g.count(chr(10)+'action-name')<=1"`
Expected: no error (actions present, rule still single-line)

- [ ] **Step 3: Commit**

```bash
git add src/voice_control/grammar/spot_action.gbnf
git commit -m "stage2e2: add enable_mouth/disable_mouth to GBNF action allowlist"
```

---

## Task 9: Construct MouthDriver and attach to TTS

**Files:**
- Modify: `src/voice_control/client_mic.py`
- Modify: `src/voice_control/spot_dispatch.py` (persona intensity sync)

- [ ] **Step 1: Attach a MouthDriver to the TTS singleton**

In `src/voice_control/client_mic.py`, after the Spot session is ensured (after the `ensure_spot_session(map_path=args.map)` call near line 780) and `tts` exists, add:

```python
                try:
                    from src.voice_control.animation.mouth import MouthDriver
                    from src.voice_control.spot_dispatch import ensure_spot_session as _ess
                    _sess = _ess()
                    tts.mouth = MouthDriver(cmd_client=_sess["cmd"])
                    print("[client_mic] MouthDriver attached (disabled until enable_mouth)")
                except Exception as _e:
                    print(f"[client_mic] MouthDriver not attached: {_e}")
```

(If the pipeline runs headless with no robot, `ensure_spot_session` raises and the mouth simply stays unattached — TTS is unaffected.)

- [ ] **Step 2: Sync persona intensity on persona switch**

In `src/voice_control/spot_dispatch.py`, inside `do_set_persona`, after the persona switch succeeds (where the new `Persona` object is in scope), add:

```python
    try:
        from src.voice_control.spot_tts import get_tts
        _t = get_tts()
        if _t is not None and _t.mouth is not None:
            _t.mouth.intensity = getattr(persona, "mouth_intensity", 1.0)
    except Exception:
        pass
```

(Use the local variable name that `do_set_persona` already binds for the resolved persona; if it is not named `persona`, adapt to the actual name.)

- [ ] **Step 3: Verify imports**

Run: `python -c "import src.voice_control.client_mic"`
Expected: no error (may print device warnings; must not raise)

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/client_mic.py src/voice_control/spot_dispatch.py
git commit -m "stage2e2: attach MouthDriver in client_mic + sync persona mouth_intensity"
```

---

## Task 10: Review gates + live validation

**Files:** none (process task)

- [ ] **Step 1: Full unit suite green**

Run: `pytest tests/voice_control/animation/ tests/voice_control/chatbot/ tests/voice_control/tts/ -v`
Expected: all PASS

- [ ] **Step 2: REVIEW GATE 2 — safety-reviewer agent**

Dispatch the `safety-reviewer` agent on the diff touching `src/session.py`, `src/voice_control/animation/mouth.py`, and `src/voice_control/spot_dispatch.py`. It must confirm: gripper forced closed on stop/freeze/estop and on `on_play_end`; arm deploy gated behind `has_arm()` + powered; no stuck-open path; stow on disable. Fix any finding before live test.

- [ ] **Step 3: REVIEW GATE 3 — code-reviewer agent**

Dispatch the `feature-dev:code-reviewer` agent on the `spot_tts.py` + `client_mic.py` integration diff. Confirm the play-callback composition does not drop the latency trace marks and the envelope holder has no cross-utterance race.

- [ ] **Step 4: REVIEW GATE 4 — live validation on Spot**

Power Spot (arm installed), operator holding e-stop. Run the voice pipeline. Say "enable your mouth" → arm deploys. Speak any utterance → gripper visibly tracks speech. Say "stop moving your mouth" → arm stows. Trigger e-stop mid-utterance → gripper closes immediately.
Expected: synced motion, clean stop, no stuck-open.

- [ ] **Step 5: Rollback note + tag**

Append a 2E.2 layer to `docs/project/stage2-rollback.md` (kill switch: `disable_mouth`, or leave `tts.mouth` unattached). Then:

```bash
git add docs/project/stage2-rollback.md
git commit -m "stage2e2: rollback layer for gripper mouth"
git tag stage2e2-complete
```

---

## Self-Review

- **Spec coverage:** envelope/MouthDriver (§Components), spike-first gate (Task 0), safety close-on-stop + arm gating (§Safety), persona intensity (Task 5), enable/disable + GBNF (Tasks 7–8), 4 review gates (§Review gates), dry-run + unit tests (§Testing), rollback (Task 10). Gaze explicitly out of scope — no task, matches design.
- **Placeholder scan:** none — every code step shows code; spike default `MOUTH_FPS=20` is a real value adjusted by Task 0.
- **Type consistency:** `MouthDriver(cmd_client, fps, dry_run)`, `.enable()/.disable()/.play(fractions, t0)/.close()/.wait()`, `.enabled/.intensity/.fps/.log` used consistently across Tasks 2/6/7/9. `envelope(samples, rate, fps, gate, intensity, cutoff_hz)` consistent Tasks 1/6. `deploy_arm_safe(cmd_client)`/`stow_arm(cmd_client)` consistent Tasks 3/7. Session dict key `"cmd"`/`"robot"` matches dispatch.
