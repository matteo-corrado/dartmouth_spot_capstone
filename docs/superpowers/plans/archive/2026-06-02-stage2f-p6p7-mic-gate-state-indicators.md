# Stage 2F P6+P7 (minimal) — Mic Gate + State Indicators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully gate the mic from the moment Spot commits to a response until 0.5s after it finishes speaking (no late-processed speech, no barge-in), let Spot re-listen for follow-ups without re-waking, and signal its state with chimes + onboard RGB LEDs.

**Architecture:** Extend the existing `VoiceState` enum (`WAKE_WORD, LISTENING, THINKING, RESPONDING`) and drive three new behaviors off state transitions in the `client_mic.py` mic loop: (1) a `_mic_gated` flag the audio callback honors so no audio is captured during THINKING/RESPONDING + a 0.5s reopen settle; (2) a per-transition indicator (chime via `audio_feedback`, LED via a new `spot_leds.StateLeds` wrapper over Boston Dynamics' `AudioVisualClient`, pulse behaviors, auto-degrades when the robot has no AV system); (3) a post-response **full** queue drain (keep nothing) — the current keep-~1s drain is the late-speech leak. All non-trivial logic lands in small pure helper modules (mirroring `barge_in.py`) so it is unit-testable; the mic loop only wires them together.

**Tech Stack:** Python 3.10 (spot-env venv), pytest 8, numpy, `bosdyn-client` 5.x (`bosdyn.client.audio_visual.AudioVisualClient`, `bosdyn.api.audio_visual_pb2`), sherpa-onnx VAD/wake (unchanged).

**Branch:** `tour_guide_upgrade_matteo` (current). All work commits here.

**Scope (explicit):** This is a *minimal* slice of spec P6 + P7 (`docs/superpowers/specs/2026-05-28-stage2f-wake-asr-overhaul-design.md`). IN: enum migration, hard mic gate (no barge-in), re-listen-without-rewake, state chimes, state LEDs. OUT (deferred): `SPEAKER_ID` state + speaker-ID logic (P4), `CONFIRM` sub-state + session object (P4), `SPOT_LISTEN_MODE` alexa/continuous flag (not requested — continuous-style re-listen is the only behavior), web badge indicator, reject chime (P4), Personal-VAD gate (P8).

---

## Decisions (locked with user 2026-06-02)

- **No barge-in.** Mic is fully gated while Spot thinks/speaks; the existing `barge_in.py` post-TTS *reset* stays (it is cleanup, not interruption).
- **Reopen delay = 0.5s** after playback goes idle, then re-listen.
- **Listen window = 15s** (unchanged `LISTENING_TIMEOUT`); after 15s idle → back to `WAKE_WORD`.
- **One plan, LEDs last + auto-degrade.** Tasks 1–7 are pure on-device software, testable with no robot. Task 8 is the on-robot LED verification (V5). If the robot has no AV system, LEDs silently no-op and chimes still work.

---

## File Structure

**New files:**
- `src/voice_control/response_gate.py` — pure predicate for when to reopen the mic after a response (mirrors `barge_in.py`). No audio, no threads.
- `src/voice_control/state_feedback.py` — pure mapping: `VoiceState` → chime method name, `VoiceState` → LED color. No I/O.
- `src/voice_control/spot_leds.py` — `StateLeds` wrapper over `AudioVisualClient`: registers pulse behaviors, runs one per state, auto-degrades to no-op when no robot / no AV system / any RPC error. Owns one daemon refresh thread.
- `scripts/diag_av_leds.py` — standalone on-robot diagnostic: checks `has_audio_visual_system`, lists behaviors, drives each state color for 2s. Used for Task 8 (V5).
- `tests/voice_control/test_response_gate.py`
- `tests/voice_control/test_state_feedback.py`
- `tests/voice_control/test_spot_leds.py`
- `tests/voice_control/test_audio_feedback.py` (new — covers the new `thinking()` tone + that existing tones still build)
- `tests/voice_control/test_drain_queue.py`

**Modified files:**
- `src/voice_control/client_mic.py` — enum (`:122`), `audio_callback` (`:206`), `_drain_audio_queue` (`~:1118`), the two `process_utterance` call sites (`~:1037`, `~:1078`), the busy→idle reset block (`~:863`), `main()` wiring (`~:704`).
- `src/voice_control/audio_feedback.py` — add `thinking()` method (`~:127`).

---

## Task 1: Enum migration — drop dead `RECORDING`, add `THINKING` + `RESPONDING`

**Files:**
- Modify: `src/voice_control/client_mic.py:122-125`
- Test: `tests/voice_control/test_state_feedback.py` (created in Task 4 imports the enum; this task just changes it)

Context: `RECORDING` has zero references repo-wide (verified: only the definition line matches `grep -rn RECORDING src/`). `THINKING` = ASR+LLM running. `RESPONDING` = TTS playing.

- [ ] **Step 1: Edit the enum**

In `src/voice_control/client_mic.py`, replace:

```python
class VoiceState(Enum):
    WAKE_WORD = auto()   # Waiting for "hey spot" (detected via ASR, not a separate model)
    LISTENING = auto()   # Wake word heard, waiting for speech
    RECORDING = auto()   # Speech detected, accumulating audio
```

with:

```python
class VoiceState(Enum):
    WAKE_WORD = auto()   # Waiting for "hey spot" (detected via ASR, not a separate model)
    LISTENING = auto()   # Wake word heard, waiting for speech
    THINKING = auto()    # Utterance captured — ASR + LLM running (mic gated)
    RESPONDING = auto()  # TTS playing the response (mic gated)
```

- [ ] **Step 2: Verify nothing referenced `RECORDING`**

Run: `grep -rn 'VoiceState.RECORDING\|\.RECORDING' src/ tests/`
Expected: no matches (empty output).

- [ ] **Step 3: Import-smoke the module**

Run: `spot-env/bin/python -c "from src.voice_control.client_mic import VoiceState; print([s.name for s in VoiceState])"`
Expected: `['WAKE_WORD', 'LISTENING', 'THINKING', 'RESPONDING']`

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/client_mic.py
git commit -m "stage2f p6: VoiceState drop dead RECORDING, add THINKING+RESPONDING"
```

---

## Task 2: Pure drain helper — full drain vs keep-tail

**Files:**
- Create: `tests/voice_control/test_drain_queue.py`
- Modify: `src/voice_control/client_mic.py` (`_drain_audio_queue`, `~:1118`)

Context: today `_drain_audio_queue()` always keeps the last ~1s of frames. After a *response* that kept tail is exactly the THINKING-window speech that gets replayed as a phantom command. We add a `keep_tail` switch and extract the partition decision into a pure, testable function.

- [ ] **Step 1: Write the failing test**

Create `tests/voice_control/test_drain_queue.py`:

```python
# tests/voice_control/test_drain_queue.py
"""Unit tests for the audio-queue drain partition (Stage 2F P6)."""
from src.voice_control.client_mic import partition_drain


def test_keep_tail_keeps_last_n_and_reports_dropped():
    frames = list(range(100))
    kept, dropped = partition_drain(frames, keep_count=33)
    assert kept == list(range(67, 100))   # last 33
    assert dropped == 67


def test_keep_tail_when_fewer_than_keep_count_keeps_all():
    frames = [1, 2, 3]
    kept, dropped = partition_drain(frames, keep_count=33)
    assert kept == [1, 2, 3]
    assert dropped == 0


def test_full_drain_keeps_nothing():
    frames = list(range(50))
    kept, dropped = partition_drain(frames, keep_count=0)
    assert kept == []
    assert dropped == 50


def test_empty_input():
    assert partition_drain([], keep_count=33) == ([], 0)
```

- [ ] **Step 2: Run it, watch it fail**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_drain_queue.py -v`
Expected: FAIL — `ImportError: cannot import name 'partition_drain'`.

- [ ] **Step 3: Add the pure helper + wire it into `_drain_audio_queue`**

In `src/voice_control/client_mic.py`, add this module-level function directly **above** `def _drain_audio_queue`:

```python
def partition_drain(frames, keep_count):
    """Split drained frames into (kept, dropped_count).

    keep_count > 0 keeps the most recent `keep_count` frames (used after a
    wake fire so an in-breath command isn't lost); keep_count == 0 keeps
    nothing (used after a response so post-decision speech can't replay).
    """
    if not frames:
        return [], 0
    if keep_count <= 0:
        return [], len(frames)
    dropped = max(0, len(frames) - keep_count)
    return frames[dropped:], dropped
```

Then replace the body of `_drain_audio_queue` with a `keep_tail` parameter:

```python
def _drain_audio_queue(keep_tail: bool = True):
    """Discard stale audio frames accumulated during processing.

    keep_tail=True keeps the most recent ~1s (a wake fire's in-breath command
    could be in there). keep_tail=False drains everything — used after a
    response so speech uttered while Spot was thinking/speaking is never
    replayed as a phantom command.
    """
    frames = []
    while not audio_queue.empty():
        try:
            frames.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    keep_count = max(1000 // FRAME_MS, 1) if keep_tail else 0  # ~33 frames at 30ms
    kept, dropped = partition_drain(frames, keep_count)

    for frame in kept:
        audio_queue.put(frame)

    if dropped:
        print(f"[Drained {dropped} stale audio chunks, kept {len(kept)}]")
```

- [ ] **Step 4: Run the test, watch it pass**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_drain_queue.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/client_mic.py tests/voice_control/test_drain_queue.py
git commit -m "stage2f p6: _drain_audio_queue gains keep_tail; pure partition_drain + tests"
```

---

## Task 3: New `thinking()` chime

**Files:**
- Modify: `src/voice_control/audio_feedback.py` (`~:127`, after `chain_next`)
- Create: `tests/voice_control/test_audio_feedback.py`

Context: there is no "processing" tone. Add a low→mid two-note "working" blip, distinct from `listening` (single high blip) and `wake_detected` (rising triad).

- [ ] **Step 1: Write the failing test**

Create `tests/voice_control/test_audio_feedback.py`:

```python
# tests/voice_control/test_audio_feedback.py
"""Unit tests for audio feedback tone generation (Stage 2F P6)."""
import numpy as np
from src.voice_control.audio_feedback import AudioFeedback, _tone


class RecordingPlayer:
    """Stand-in AudioPlayer that records enqueued samples."""
    def __init__(self):
        self.calls = []
    def is_available(self):
        return True
    def enqueue_raw(self, samples, rate, label=""):
        self.calls.append((samples, rate, label))


def test_thinking_tone_enqueues_nonempty_float32():
    fb = AudioFeedback(player=RecordingPlayer(), volume=1.0)
    fb.thinking()
    samples, rate, label = fb._player.calls[0]
    assert label == "beep:thinking"
    assert rate == 48000
    assert samples.dtype == np.float32
    assert len(samples) > 0
    assert np.max(np.abs(samples)) > 0.0


def test_existing_tones_still_build():
    fb = AudioFeedback(player=RecordingPlayer(), volume=1.0)
    for name in ("wake_detected", "listening", "command_ok", "error",
                 "chain_next", "ready", "thinking"):
        getattr(fb, name)()
    assert len(fb._player.calls) == 7


def test_noop_without_player():
    fb = AudioFeedback(player=None)
    fb.thinking()  # must not raise
```

- [ ] **Step 2: Run it, watch it fail**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_audio_feedback.py -v`
Expected: FAIL — `AttributeError: 'AudioFeedback' object has no attribute 'thinking'`.

- [ ] **Step 3: Add the `thinking()` method**

In `src/voice_control/audio_feedback.py`, add immediately after the `chain_next` method (before `ready`):

```python
    def thinking(self):
        """Working blip — utterance captured, ASR+LLM running (A4 -> D5, 130ms)."""
        a4 = _tone(440, 60, 0.18)
        gap = np.zeros(int(SAMPLE_RATE * 0.015), dtype=np.float32)
        d5 = _tone(587, 60, 0.18)
        samples = np.concatenate([a4[:-int(SAMPLE_RATE*TAIL_MS/1000)], gap, d5])
        self._play(samples, "thinking")
```

- [ ] **Step 4: Run the tests, watch them pass**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_audio_feedback.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/audio_feedback.py tests/voice_control/test_audio_feedback.py
git commit -m "stage2f p6: add thinking() processing chime + audio_feedback tests"
```

---

## Task 4: Pure state→indicator mapping (`state_feedback.py`)

**Files:**
- Create: `src/voice_control/state_feedback.py`
- Create: `tests/voice_control/test_state_feedback.py`

Context: one place that says "entering state X → play chime Y, set LED color Z". Pure data + functions so the loop just calls it. Colors are RGB 0-255 with vector magnitude ≤ 255 (BD warranty cap for the AV LEDs).

- [ ] **Step 1: Write the failing test**

Create `tests/voice_control/test_state_feedback.py`:

```python
# tests/voice_control/test_state_feedback.py
"""Unit tests for the state -> chime/LED indicator mapping (Stage 2F P6/P7)."""
import math
from src.voice_control.client_mic import VoiceState
from src.voice_control.state_feedback import chime_for, led_color_for, LED_PERIOD_S


def test_listening_plays_ready_chime():
    assert chime_for(VoiceState.LISTENING) == "ready"


def test_thinking_plays_thinking_chime():
    assert chime_for(VoiceState.THINKING) == "thinking"


def test_responding_has_no_chime():
    # TTS itself is the audio cue; a chime over speech would be noise.
    assert chime_for(VoiceState.RESPONDING) is None


def test_wake_word_has_no_chime():
    assert chime_for(VoiceState.WAKE_WORD) is None


def test_every_state_has_an_led_color():
    for st in VoiceState:
        r, g, b = led_color_for(st)
        assert all(0 <= c <= 255 for c in (r, g, b))


def test_led_magnitudes_within_warranty_cap():
    for st in VoiceState:
        r, g, b = led_color_for(st)
        assert math.sqrt(r * r + g * g + b * b) <= 255.0


def test_states_are_visually_distinct():
    colors = {st: led_color_for(st) for st in VoiceState}
    assert len(set(colors.values())) == len(VoiceState)


def test_period_defined_for_every_state():
    for st in VoiceState:
        assert LED_PERIOD_S[st] > 0
```

- [ ] **Step 2: Run it, watch it fail**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_state_feedback.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.voice_control.state_feedback'`.

- [ ] **Step 3: Write the module**

Create `src/voice_control/state_feedback.py`:

```python
# src/voice_control/state_feedback.py
"""Pure state -> indicator mapping (Stage 2F P6/P7).

No audio device, no robot, no threads. The live loop (client_mic.py) calls
`chime_for(new_state)` to pick a beep method and `led_color_for(state)` /
`LED_PERIOD_S[state]` to drive the onboard LEDs. Kept pure so the policy is
unit-testable and the loop stays thin.
"""
from __future__ import annotations

from src.voice_control.client_mic import VoiceState

# Chime fired when ENTERING a state. None = silent transition.
# Names map to methods on the audio_feedback `beep` singleton.
_CHIME = {
    VoiceState.WAKE_WORD: None,    # dropping back to idle is silent
    VoiceState.LISTENING: "ready",   # "go ahead" cue (also = response-done)
    VoiceState.THINKING: "thinking",  # "got it, working"
    VoiceState.RESPONDING: None,    # the speech itself is the cue
}

# RGB 0-255 per state. Vector magnitude kept <= 255 (BD AV-LED warranty cap).
# Distinct hues: idle=dim blue, listening=green, thinking=amber, speaking=cyan.
_LED_COLOR = {
    VoiceState.WAKE_WORD: (0, 0, 60),
    VoiceState.LISTENING: (0, 120, 0),
    VoiceState.THINKING: (150, 80, 0),
    VoiceState.RESPONDING: (0, 90, 120),
}

# Pulse period (seconds) per state. Slow = calm/idle, fast = active.
LED_PERIOD_S = {
    VoiceState.WAKE_WORD: 3.0,
    VoiceState.LISTENING: 1.5,
    VoiceState.THINKING: 0.7,
    VoiceState.RESPONDING: 1.0,
}


def chime_for(state: VoiceState):
    """Name of the beep method to play when entering `state`, or None."""
    return _CHIME.get(state)


def led_color_for(state: VoiceState):
    """(r, g, b) 0-255 for the state's LED pulse."""
    return _LED_COLOR.get(state, (0, 0, 0))
```

- [ ] **Step 4: Run the tests, watch them pass**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_state_feedback.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/state_feedback.py tests/voice_control/test_state_feedback.py
git commit -m "stage2f p6: pure state->chime/LED mapping (state_feedback) + tests"
```

---

## Task 5: Mic-gate reopen predicate (`response_gate.py`)

**Files:**
- Create: `src/voice_control/response_gate.py`
- Create: `tests/voice_control/test_response_gate.py`

Context: after playback goes idle we keep the mic gated for `RESPONSE_REOPEN_DELAY_S` (0.5s) so the speaker tail/echo can't bleed into the first reopened frames. This predicate decides "is the 0.5s settle over?" — pure, mirrors `barge_in.should_reset_after_player`.

- [ ] **Step 1: Write the failing test**

Create `tests/voice_control/test_response_gate.py`:

```python
# tests/voice_control/test_response_gate.py
"""Unit tests for the post-response mic-reopen predicate (Stage 2F P6)."""
from src.voice_control.response_gate import should_reopen_mic, RESPONSE_REOPEN_DELAY_S


def test_reopens_once_settle_elapsed():
    assert should_reopen_mic(gated=True, reopen_at=10.0, now=10.0) is True
    assert should_reopen_mic(gated=True, reopen_at=10.0, now=10.6) is True


def test_stays_gated_before_settle():
    assert should_reopen_mic(gated=True, reopen_at=10.0, now=9.7) is False


def test_not_gated_never_reopens():
    assert should_reopen_mic(gated=False, reopen_at=10.0, now=99.0) is False


def test_no_pending_reopen_time():
    assert should_reopen_mic(gated=True, reopen_at=None, now=99.0) is False


def test_delay_is_half_second():
    assert RESPONSE_REOPEN_DELAY_S == 0.5
```

- [ ] **Step 2: Run it, watch it fail**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_response_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.voice_control.response_gate'`.

- [ ] **Step 3: Write the module**

Create `src/voice_control/response_gate.py`:

```python
# src/voice_control/response_gate.py
"""Pure predicate for reopening the mic after a response (Stage 2F P6).

The mic is hard-gated from the moment Spot commits to a response (THINKING)
through the end of TTS plus a short settle, so no speech uttered during that
window is ever processed (no barge-in, no late phantom commands). This module
only answers "has the settle elapsed?" — the live loop owns the flag and the
actual drain/reset. No audio, no threads, no VoiceState.
"""
from __future__ import annotations

# Seconds the mic stays gated after playback goes idle, before re-listening.
# Covers the OS audio-buffer drain + room echo of Spot's own voice.
RESPONSE_REOPEN_DELAY_S = 0.5


def should_reopen_mic(gated: bool, reopen_at, now: float) -> bool:
    """True once the post-response settle has elapsed and the mic may reopen.

    gated:     is the mic currently gated for a response?
    reopen_at: monotonic time the settle ends (None until playback goes idle).
    now:       current monotonic time.
    """
    return gated and reopen_at is not None and now >= reopen_at
```

- [ ] **Step 4: Run the tests, watch them pass**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_response_gate.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/response_gate.py tests/voice_control/test_response_gate.py
git commit -m "stage2f p6: pure mic-reopen settle predicate (response_gate) + tests"
```

---

## Task 6: `StateLeds` wrapper over `AudioVisualClient`

**Files:**
- Create: `src/voice_control/spot_leds.py`
- Create: `tests/voice_control/test_spot_leds.py`

Context: Spot's 5 onboard RGB LED groups are driven via `AudioVisualClient`: register named **behaviors** (`add_or_modify_behavior`) then `run_behavior(name, end_time_secs)`. We register one pulse behavior per state (pulse, not solid — solid-on can overheat the AV board) and refresh the active one from a daemon thread (so the LED auto-stops if the process dies and end_time expires). Everything degrades to a no-op when there is no robot, no AV system, or any RPC error — chimes remain the fallback indicator.

Proto shape (verified against installed `bosdyn.api.audio_visual_pb2`):
`AudioVisualBehavior{enabled, priority, led_sequence_group}`;
`LedSequenceGroup{front_center, front_left, front_right, hind_left, hind_right: LedSequence}`;
`LedSequence{pulse_sequence: PulseSequence}`; `PulseSequence{color: Color, period: Duration}`;
`Color{rgb: RGB{r,g,b}}`. Capability gate: `robot.get_cached_hardware_hardware_configuration().has_audio_visual_system`.

- [ ] **Step 1: Write the failing test**

Create `tests/voice_control/test_spot_leds.py`:

```python
# tests/voice_control/test_spot_leds.py
"""Unit tests for the StateLeds AV wrapper degrade + dispatch (Stage 2F P7)."""
import pytest
from src.voice_control.client_mic import VoiceState
from src.voice_control.spot_leds import StateLeds


class FakeAvClient:
    def __init__(self):
        self.added = []
        self.ran = []
        self.stopped = []
    def add_or_modify_behavior(self, name, behavior):
        self.added.append((name, behavior))
    def run_behavior(self, name, end_time_secs, **kw):
        self.ran.append(name)
    def stop_behavior(self, name, **kw):
        self.stopped.append(name)


class FakeHw:
    def __init__(self, has_av):
        self.has_audio_visual_system = has_av


class FakeRobot:
    def __init__(self, has_av=True, av_client=None, raise_on_client=False):
        self._hw = FakeHw(has_av)
        self._av = av_client
        self._raise = raise_on_client
    def get_cached_hardware_hardware_configuration(self):
        return self._hw
    def ensure_client(self, name):
        if self._raise:
            raise RuntimeError("no AV service")
        return self._av


def test_no_robot_is_noop():
    leds = StateLeds(get_robot=lambda: None, start_thread=False)
    leds.set_state(VoiceState.LISTENING)  # must not raise
    assert leds.enabled is False


def test_robot_without_av_system_degrades():
    robot = FakeRobot(has_av=False, av_client=FakeAvClient())
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)
    assert leds.enabled is False


def test_client_init_failure_degrades_permanently():
    robot = FakeRobot(has_av=True, raise_on_client=True)
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)
    assert leds.enabled is False
    # a later call must not retry/raise
    leds.set_state(VoiceState.THINKING)


def test_registers_one_behavior_per_state_and_runs_active():
    av = FakeAvClient()
    robot = FakeRobot(has_av=True, av_client=av)
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.THINKING)
    assert leds.enabled is True
    # one behavior registered per VoiceState
    assert {n for n, _ in av.added} == {f"spotvoice_{s.name.lower()}" for s in VoiceState}
    # the active state's behavior was run
    assert av.ran[-1] == "spotvoice_thinking"


def test_set_state_runs_new_behavior():
    av = FakeAvClient()
    robot = FakeRobot(has_av=True, av_client=av)
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)
    leds.set_state(VoiceState.RESPONDING)
    assert av.ran[-1] == "spotvoice_responding"


def test_rpc_error_during_run_degrades_not_raises():
    class BoomClient(FakeAvClient):
        def run_behavior(self, name, end_time_secs, **kw):
            raise RuntimeError("rpc down")
    robot = FakeRobot(has_av=True, av_client=BoomClient())
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)  # must swallow
    assert leds.enabled is False
```

- [ ] **Step 2: Run it, watch it fail**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_spot_leds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.voice_control.spot_leds'`.

- [ ] **Step 3: Write the module**

Create `src/voice_control/spot_leds.py`:

```python
# src/voice_control/spot_leds.py
"""Onboard RGB LED state indicator for the voice pipeline (Stage 2F P7).

Drives Spot's 5 body LED groups via AudioVisualClient. One pulse behavior is
registered per VoiceState; set_state() runs the matching one. A daemon thread
re-runs the active behavior every REFRESH_S so the LED auto-extinguishes if
this process dies (run_behavior's end_time expires). Pulse (not solid) avoids
the AV-board overheat the SDK warns about.

DEGRADES TO NO-OP, never raises into the caller, on any of: no robot, robot
without an AV system, AV client init failure, or any RPC error. In those cases
chimes remain the only state indicator. This module is import-safe without a
robot connection.
"""
from __future__ import annotations

import threading
import time

from src.voice_control.client_mic import VoiceState
from src.voice_control.state_feedback import led_color_for, LED_PERIOD_S

REFRESH_S = 2.0          # re-run cadence; end_time = now + REFRESH_S + margin
_END_MARGIN_S = 0.5
_PRIORITY = 10


def _behavior_name(state: VoiceState) -> str:
    return f"spotvoice_{state.name.lower()}"


def _build_pulse_behavior(r: int, g: int, b: int, period_s: float):
    """Build an AudioVisualBehavior that pulses all 5 LED groups one color."""
    from bosdyn.api import audio_visual_pb2 as av  # local import: optional dep
    grp = av.LedSequenceGroup()
    for led in (grp.front_center, grp.front_left, grp.front_right,
                grp.hind_left, grp.hind_right):
        led.pulse_sequence.color.rgb.r = int(r)
        led.pulse_sequence.color.rgb.g = int(g)
        led.pulse_sequence.color.rgb.b = int(b)
        led.pulse_sequence.period.seconds = int(period_s)
        led.pulse_sequence.period.nanos = int((period_s % 1) * 1e9)
    return av.AudioVisualBehavior(enabled=True, priority=_PRIORITY,
                                  led_sequence_group=grp)


class StateLeds:
    """Maps VoiceState -> onboard LED pulse. No-op when unavailable."""

    def __init__(self, get_robot, start_thread: bool = True):
        self._get_robot = get_robot
        self.enabled = False
        self._av = None
        self._initialized = False
        self._failed = False          # once True, never retry (permanent degrade)
        self._active = None           # active behavior name
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    # -- public ----------------------------------------------------------
    def set_state(self, state: VoiceState) -> None:
        """Show `state` on the LEDs. Swallows all errors (degrade)."""
        if self._failed:
            return
        if not self._ensure():
            return
        name = _behavior_name(state)
        with self._lock:
            self._active = name
        self._run(name)

    def shutdown(self) -> None:
        self._stop.set()
        if self._av is not None and self._active is not None:
            try:
                self._av.stop_behavior(self._active)
            except Exception:
                pass

    # -- internal --------------------------------------------------------
    def _ensure(self) -> bool:
        """Lazily init the AV client + register behaviors. Degrades on failure."""
        if self._initialized:
            return self.enabled
        self._initialized = True
        try:
            robot = self._get_robot()
            if robot is None:
                self._failed = True
                return False
            hw = robot.get_cached_hardware_hardware_configuration()
            if not getattr(hw, "has_audio_visual_system", False):
                print("[LEDs] Robot has no audio-visual system — LED indicator off.")
                self._failed = True
                return False
            from bosdyn.client.audio_visual import AudioVisualClient
            self._av = robot.ensure_client(AudioVisualClient.default_service_name)
            for st in VoiceState:
                r, g, b = led_color_for(st)
                beh = _build_pulse_behavior(r, g, b, LED_PERIOD_S[st])
                self._av.add_or_modify_behavior(_behavior_name(st), beh)
            self.enabled = True
            print("[LEDs] Onboard RGB state indicator active.")
            return True
        except Exception as e:
            print(f"[LEDs] Disabled (init failed: {type(e).__name__}: {e}).")
            self._failed = True
            self.enabled = False
            return False

    def _run(self, name: str) -> None:
        try:
            from bosdyn.util import now_sec
            self._av.run_behavior(name, end_time_secs=now_sec() + REFRESH_S + _END_MARGIN_S)
        except Exception as e:
            print(f"[LEDs] Disabled (run failed: {type(e).__name__}: {e}).")
            self._failed = True
            self.enabled = False

    def _refresh_loop(self) -> None:
        while not self._stop.wait(REFRESH_S):
            if not self.enabled or self._failed:
                continue
            with self._lock:
                name = self._active
            if name is not None:
                self._run(name)
```

Note: `now_sec` and the proto imports are local so importing `spot_leds` never requires `bosdyn` at module load (keeps unit tests robot-free — the FakeAvClient path never imports `bosdyn.util` because `run_behavior` is faked... but `_build_pulse_behavior` does import `bosdyn.api`). `bosdyn` **is** installed in spot-env, so the tests import it fine; the local imports only protect against import-time crashes in non-spot tooling.

- [ ] **Step 4: Run the tests, watch them pass**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_spot_leds.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/spot_leds.py tests/voice_control/test_spot_leds.py
git commit -m "stage2f p7: StateLeds AV-LED wrapper (pulse per state, auto-degrade) + tests"
```

---

## Task 7: Wire the gate + indicators into the mic loop

**Files:**
- Modify: `src/voice_control/client_mic.py` — globals (`~:190`), `audio_callback` (`:218`), `main()` (`~:704`), the busy→idle reset block (`~:863`), the two `process_utterance` call sites (`~:1037` and `~:1078`).

Context: this is the integration. The pure helpers are tested; here we connect them. The mic loop is not unit-tested (it owns the audio device), so verification is an import smoke + dry-run + the Task 9 on-robot run.

**Transition policy implemented here:**
- Enter `THINKING` right before a committed (non-`safety_only`) `process_utterance` → play `thinking` chime, set LED, set `_mic_gated=True`. (The chime plays before the mic is needed again; it queues behind nothing.)
- Enter `RESPONDING` is implicit during TTS (player busy) — we set the LED to RESPONDING when arming `response_pending`, no chime.
- On the player busy→idle edge (response done): keep gated, set `_mic_reopen_at = now + RESPONSE_REOPEN_DELAY_S`, **full-drain** (`keep_tail=False`), reset detectors.
- Each loop top: if `should_reopen_mic(...)` → clear gate, `_mic_reopen_at=None`, go to `LISTENING`, play `ready` chime, set LED, reset the 15s timer.

- [ ] **Step 1: Add imports + gate globals**

Near the top of `client_mic.py`, with the other `from src.voice_control...` imports, add:

```python
from src.voice_control.response_gate import should_reopen_mic, RESPONSE_REOPEN_DELAY_S
from src.voice_control.state_feedback import chime_for
```

Next to the `_audio_player` global (around `:190`), add:

```python
# Stage 2F P6: hard mic gate during THINKING/RESPONDING + the 0.5s reopen
# settle. The audio callback drops frames while gated, so no speech uttered
# from the moment Spot commits to a response until 0.5s after it stops
# speaking is ever captured (no barge-in, no late phantom commands).
_mic_gated = False
_mic_reopen_at = None  # monotonic time the post-response settle ends
_state_leds = None     # StateLeds instance (set in main); None = LEDs off
```

- [ ] **Step 2: Honor the gate in `audio_callback`**

In `audio_callback` (`:218`), change the busy check from:

```python
    if _audio_player is not None and _audio_player.is_busy():
        _player_last_busy_at = now
        return  # discard frame — speaker is active
```

to:

```python
    if _mic_gated or (_audio_player is not None and _audio_player.is_busy()):
        _player_last_busy_at = now
        return  # discard frame — gated for a response, or speaker active
```

(`_mic_gated` is a module global read here; the loop writes it. No lock needed — single bool, GIL-atomic, and a one-frame race is harmless.)

- [ ] **Step 3: Add a small `_set_voice_state` helper for chime+LED**

Add this module-level helper above the mic-loop function (near `_drain_audio_queue`):

```python
def _enter_state(new_state):
    """Play the entry chime + set LEDs for a state transition. Best-effort."""
    name = chime_for(new_state)
    if name is not None:
        getattr(beep, name)()
    if _state_leds is not None:
        _state_leds.set_state(new_state)
```

- [ ] **Step 4: Construct `StateLeds` in `main()`**

In `main()`, right after `beep.set_player(_audio_player)` / `beep.set_volume(args.volume)` (`~:705`), add:

```python
    # Stage 2F P7: onboard RGB LED state indicator. Lazily binds to the Spot
    # AV service on first state change; no-op in dry-run or if the robot has
    # no AV system. Chimes are the fallback indicator either way.
    global _state_leds
    if os.environ.get("SPOT_DRY_RUN") == "1":
        _state_leds = None
    else:
        from src.voice_control.spot_leds import StateLeds
        from src.voice_control.spot_dispatch import ensure_spot_session

        def _led_robot():
            try:
                return ensure_spot_session()["robot"]
            except Exception:
                return None

        _state_leds = StateLeds(get_robot=_led_robot)
```

- [ ] **Step 5: Gate + THINKING at the two `process_utterance` call sites**

There are two call sites (max-duration `~:1037`, silence-timeout `~:1078`). Both currently look like:

```python
                            safety_only = (state == VoiceState.WAKE_WORD)
                            result = process_utterance(stub, speech_buffer, speech_float_buffer,
                                                       brain, safety_only=safety_only, tts=tts,
                                                       has_wake_detector=bool(wake_detector))
                            is_speaking = False
                            speech_buffer.clear()
                            speech_float_buffer.clear()
                            _drain_audio_queue()
                            if result != "wake_detected" and not safety_only:
                                response_pending = True
```

For **each** of the two call sites, change them to gate before the call and arm RESPONDING after a real response. Replace each block with:

```python
                            safety_only = (state == VoiceState.WAKE_WORD)
                            if not safety_only:
                                # Commit to processing: gate the mic and show THINKING
                                # before ASR+LLM so nothing said meanwhile is captured.
                                global _mic_gated
                                _mic_gated = True
                                _enter_state(VoiceState.THINKING)
                                state = VoiceState.THINKING
                            result = process_utterance(stub, speech_buffer, speech_float_buffer,
                                                       brain, safety_only=safety_only, tts=tts,
                                                       has_wake_detector=bool(wake_detector))
                            is_speaking = False
                            speech_buffer.clear()
                            speech_float_buffer.clear()
                            if result != "wake_detected" and not safety_only:
                                response_pending = True
                                _enter_state(VoiceState.RESPONDING)
                                state = VoiceState.RESPONDING
                            elif result == "wake_detected" and state == VoiceState.WAKE_WORD:
                                print(">>> Now listening for commands...")
                                beep.wake_detected()
                                state = VoiceState.LISTENING
                                listening_start_time = time.time()
                                _drain_audio_queue()
                            elif use_wake_word:
                                # safety-only / ignored utterance: keep prior behavior
                                _drain_audio_queue()
```

Notes:
- The old unconditional `_drain_audio_queue()` is removed from the committed-response path — the **full** drain now happens on the busy→idle edge (Step 6). Wake/safety paths still drain (keep tail) as before.
- `global _mic_gated` must appear once per function; if the linter complains about the second occurrence inside the same function, hoist a single `global _mic_gated, _mic_reopen_at` to the top of the mic-loop function instead of inline. Verify with Step 8.

- [ ] **Step 6: Full-drain + arm reopen on the busy→idle edge**

The reset block at the top of the loop (`~:863`) currently calls the barge-in reset which drains keep-tail. Replace the `if should_reset_after_player(...)` block's body so that, after the existing detector reset, it switches to a **full** drain and arms the reopen settle. Change:

```python
            from src.voice_control.barge_in import should_reset_after_player
            busy = _audio_player.is_busy() if _audio_player is not None else False
            if should_reset_after_player(response_pending, player_was_busy, busy):
                vad.reset()
                if wake_detector:
                    wake_detector.reset()
                preroll_buffer.clear()
                pending_speech_frames.clear()
                consecutive_speech = 0
                is_speaking = False
                speech_buffer.clear()
                speech_float_buffer.clear()
                _drain_audio_queue()
                response_pending = False
                print("[Barge-in] detector state reset after TTS response")
            player_was_busy = busy
```

to:

```python
            from src.voice_control.barge_in import should_reset_after_player
            busy = _audio_player.is_busy() if _audio_player is not None else False
            if should_reset_after_player(response_pending, player_was_busy, busy):
                vad.reset()
                if wake_detector:
                    wake_detector.reset()
                preroll_buffer.clear()
                pending_speech_frames.clear()
                consecutive_speech = 0
                is_speaking = False
                speech_buffer.clear()
                speech_float_buffer.clear()
                _drain_audio_queue(keep_tail=False)  # full drain: no late phantom commands
                response_pending = False
                _mic_reopen_at = time.monotonic() + RESPONSE_REOPEN_DELAY_S
                print("[Mic] response done — gated, reopening in "
                      f"{RESPONSE_REOPEN_DELAY_S:.1f}s")
            player_was_busy = busy

            # Stage 2F P6: reopen the mic once the post-response settle elapses.
            if should_reopen_mic(_mic_gated, _mic_reopen_at, time.monotonic()):
                _mic_gated = False
                _mic_reopen_at = None
                _drain_audio_queue(keep_tail=False)  # drop anything from the settle window
                state = VoiceState.LISTENING
                listening_start_time = time.time()
                _enter_state(VoiceState.LISTENING)
                print(">>> Listening for follow-up (no wake needed)...")
```

Add `global _mic_gated, _mic_reopen_at` (and `_state_leds` is read-only here) at the **top** of the mic-loop function so both the reset block and Step 5 can assign them. Locate the function that contains this loop and add the `global` line just after its `try:`-preceding setup (next to `response_pending = False`).

- [ ] **Step 7: Reduce the queue-get timeout so reopen fires near 0.5s**

While gated, the callback drops frames, so `audio_queue.get(timeout=1.0)` would block a full second between loop cycles and the 0.5s reopen could slip to ~1s. Change the get timeout (`~:888`) from `1.0` to `0.2`:

```python
            try:
                pcm = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
```

This only changes idle loop cadence (more responsive to SIGTERM too); it does not change audio handling.

- [ ] **Step 8: Import-smoke + lint the loop**

Run: `spot-env/bin/python -c "import src.voice_control.client_mic"`
Expected: no `SyntaxError`, no `SyntaxError: name '_mic_gated' is assigned to before global declaration`. If the global-declaration error appears, move the single `global _mic_gated, _mic_reopen_at` to the top of the loop function and remove the inline `global` from Step 5.

- [ ] **Step 9: Full unit-test sweep**

Run: `spot-env/bin/python -m pytest tests/voice_control/test_drain_queue.py tests/voice_control/test_audio_feedback.py tests/voice_control/test_state_feedback.py tests/voice_control/test_response_gate.py tests/voice_control/test_spot_leds.py tests/voice_control/test_barge_in.py -v`
Expected: all PASS (barge_in regression intact).

- [ ] **Step 10: Dry-run smoke (no robot, no audio device needed)**

Run: `SPOT_DRY_RUN=1 spot-env/bin/python -c "import src.voice_control.client_mic as m; print('THINKING' in [s.name for s in m.VoiceState]); print(m.partition_drain(list(range(50)), 0))"`
Expected: `True` then `([], 50)`.

- [ ] **Step 11: Commit**

```bash
git add src/voice_control/client_mic.py
git commit -m "stage2f p6: wire mic gate + THINKING/RESPONDING + full post-response drain + reopen settle"
```

---

## Task 8: On-robot LED verification (V5) + diagnostic script

**Files:**
- Create: `scripts/diag_av_leds.py`

Context: confirms the robot actually exposes the AV LED service and our pulse behaviors render. This is the only robot-gated task; LEDs auto-degrade if it fails, so the rest of the feature ships regardless.

- [ ] **Step 1: Write the diagnostic script**

Create `scripts/diag_av_leds.py`:

```python
#!/usr/bin/env python3
"""Stage 2F P7 / V5 — verify Spot onboard AV LED control.

Connects, reports whether the robot has an AV system, lists existing
behaviors, then drives each voice state's pulse color for 2s. Run on the
Jetson with the robot powered (motors NOT required — AV needs no lease).

    spot-env/bin/python scripts/diag_av_leds.py --hostname <ROBOT_IP>
"""
import argparse
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.audio_visual import AudioVisualClient
from bosdyn.util import now_sec

from src.voice_control.client_mic import VoiceState
from src.voice_control.spot_leds import _build_pulse_behavior, _behavior_name
from src.voice_control.state_feedback import led_color_for, LED_PERIOD_S


def main():
    p = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(p)
    args = p.parse_args()

    sdk = bosdyn.client.create_standard_sdk("av-led-diag")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    hw = robot.get_cached_hardware_hardware_configuration()
    print(f"has_audio_visual_system = {getattr(hw, 'has_audio_visual_system', None)}")

    av = robot.ensure_client(AudioVisualClient.default_service_name)
    try:
        print("existing behaviors:", av.list_behaviors())
    except Exception as e:
        print(f"list_behaviors failed: {e}")

    for st in VoiceState:
        r, g, b = led_color_for(st)
        name = _behavior_name(st)
        av.add_or_modify_behavior(name, _build_pulse_behavior(r, g, b, LED_PERIOD_S[st]))
        print(f"running {name} rgb=({r},{g},{b}) for 2s...")
        av.run_behavior(name, end_time_secs=now_sec() + 2.5)
        time.sleep(2.0)
        av.stop_behavior(name)

    print("done — if you saw blue/green/amber/cyan pulses, V5 PASSES.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the robot (manual)**

Run: `spot-env/bin/python scripts/diag_av_leds.py --hostname <ROBOT_IP>`
Expected outcomes:
- `has_audio_visual_system = True` and four visible pulse colors → **V5 PASS**, LEDs are live.
- `has_audio_visual_system = False` or `DoesNotExistError` / RPC error → **V5 FAIL**: LEDs degrade to no-op automatically (chimes still indicate state). Record the result and move on — no code change needed.

- [ ] **Step 3: Record the V5 result + commit the script**

Append one line to `docs/project/status.md` under a "Stage 2F P7 V5" note: PASS/FAIL + robot firmware version.

```bash
git add scripts/diag_av_leds.py docs/project/status.md
git commit -m "stage2f p7: AV-LED diagnostic script + record V5 result"
```

---

## Task 9: End-to-end on-robot verification

**Files:** none (manual checklist).

Run the full pipeline and confirm each requirement. Use `scripts/run_voice_control.py` as usual.

- [ ] **Step 1: Late-speech gate**

Say "Hey Spot, tell me about Dartmouth." While Spot is *thinking and speaking*, keep talking ("blah blah stop talking about random things"). 
Verify: Spot finishes its response and does **not** then execute/answer anything you said during the response. Log shows `[Mic] response done — gated` and `[Drained N stale audio chunks, kept 0]`.

- [ ] **Step 2: Re-listen without re-wake**

After Spot finishes and you hear the `ready` chime (~0.5s later), say "stand up" **without** "Hey Spot". 
Verify: Spot stands. Log shows `>>> Listening for follow-up (no wake needed)...`.

- [ ] **Step 3: Window timeout**

After a response, stay silent 15s. 
Verify: log shows the listening timeout and Spot requires "Hey Spot" again.

- [ ] **Step 4: Chimes**

Verify distinct chimes: `thinking` blip when you finish a command, `ready` chime when it reopens for follow-up. No chime over speech.

- [ ] **Step 5: LEDs (only if Task 8 V5 PASSED)**

Verify LED colors track state: dim blue idle → green listening → amber thinking → cyan speaking → green again. If V5 failed, confirm LEDs are simply off and everything else still works.

- [ ] **Step 6: Safety still cold-works**

From `WAKE_WORD` (no wake said), say "stop". 
Verify: the always-on C1 safety KWS still halts Spot immediately (this path is untouched by the gate — confirm no regression).

- [ ] **Step 7: Commit any tuning**

If you adjusted `RESPONSE_REOPEN_DELAY_S`, a chime, or an LED color during testing:

```bash
git add -A
git commit -m "stage2f p6/p7: tune reopen delay / chimes / LED colors from on-robot test"
```

---

## Self-Review

**Spec coverage (user requirements 2026-06-02):**
- "ASR/VAD + mic completely stop once robot decides to respond; no delayed processing of speech said during the response" → Task 7 Step 2 (gate in callback) + Step 5 (gate set at commit, before ASR) + Step 6 (full drain on busy→idle and on reopen). ✔
- "no barge-in, fully gate the mic during response" → `_mic_gated` honored in `audio_callback`; no interrupt path added; existing barge-in module is reset-only. ✔
- "accept more input without re-waking after a small delay" → Task 7 Step 6 reopen to `LISTENING` after `RESPONSE_REOPEN_DELAY_S=0.5`; 15s window kept. ✔
- "noises to indicate ready/processing/state" → Task 3 `thinking()` + Task 4 mapping + Task 7 `_enter_state`. ✔
- "LEDs reflect state" (P7) → Task 6 `StateLeds` + Task 8 verify + Task 7 wiring. ✔
- "web badge not needed" → out of scope, not built. ✔

**Placeholder scan:** No TBD/TODO. Every code step shows full code. Manual robot steps (Tasks 8–9) are inherently checklist-based but each names exact commands/observations.

**Type/name consistency:**
- `partition_drain(frames, keep_count)` — defined Task 2, used by `_drain_audio_queue` (Task 2) and tested (Task 2). ✔
- `_drain_audio_queue(keep_tail=...)` — defined Task 2, called Task 7 Steps 5/6. ✔
- `chime_for` / `led_color_for` / `LED_PERIOD_S` — defined Task 4, used by `_enter_state` (Task 7), `StateLeds` (Task 6), diag (Task 8). ✔
- `should_reopen_mic` / `RESPONSE_REOPEN_DELAY_S` — defined Task 5, used Task 7 Step 6. ✔
- `_behavior_name` / `_build_pulse_behavior` — defined Task 6, reused by diag (Task 8). ✔
- `_enter_state`, `_mic_gated`, `_mic_reopen_at`, `_state_leds` — all defined in Task 7 before use. ✔
- `VoiceState.{THINKING,RESPONDING}` — added Task 1, used everywhere after. ✔

**Known sequencing note:** Tasks 4, 5, 6 import from `client_mic` (for `VoiceState`), and Task 6's test imports `state_feedback`. Task order (1→4→5→6→7) satisfies all import dependencies. Run the full sweep at Task 7 Step 9 to catch any ordering mistake.

**Risk:** Task 8 (V5) is the only uncertain gate. The whole LED channel degrades to no-op on failure, so a V5 FAIL does not block Tasks 1–7 or 9. No other hardware dependency.
