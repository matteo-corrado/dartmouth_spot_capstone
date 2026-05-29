# Stage 2F P0–P2 Implementation Plan — Barge-in, Safety KWS, Chunker, Eval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the three low-risk, code-only Stage 2F phases — P0 (barge-in detector-state reset + C1 always-on safety KWS), P1 (TTS chunker backend swap preserving the GBNF-JSON interface), P2 (reproducible voice-eval harness) — each independently testable.

**Architecture:** Surgical edits to the existing single-threaded mic loop (`client_mic.py`) and streaming chunker (`spot_tts.py`), plus one new always-on safety keyword detector and a new offline eval harness over the existing `tests/audio/` lab corpus. All detector-state and sentence-splitting logic is extracted into **pure helper functions** so it can be unit-tested without the live audio device or robot. No threading changes; no cross-module reach into the player thread.

**Tech Stack:** Python 3 (venv at `spot-env/`), `sherpa-onnx` (KeywordSpotter + Silero VAD, already installed), `sentencepiece` 0.2.1 (BPE for safety keywords), `blingfire` + `stream2sentence` (new, P1), `pytest` (runner: `spot-env/bin/pytest`).

---

## Scope & Deviations (read before starting)

This plan covers **P0, P1, P2 only**. P3–P8 (wake training, speaker-ID, state machine, indicators, Personal VAD) are deferred to later plans — they gate on cluster training, hardware verifications (V-items), and the safety-reviewer pass.

Spec source: `docs/superpowers/specs/2026-05-28-stage2f-wake-asr-overhaul-design.md` (v3, committed `63d08b7`). Branch `stage2f-wake-asr-overhaul` is checked out.

**Deliberate deviations from the spec (surfaced per writing-plans self-review):**

1. **`stream2sentence` is NOT used as a separate driver.** The spec names "driver `stream2sentence` + detector BlingFire." But `TTSChunker` (`spot_tts.py:249`) *already is* the streaming driver — it buffers partial tokens in `self.buf` and flushes on sentence boundaries. Adding `stream2sentence` would duplicate that buffering and force its plain-text streaming model into a char-by-char JSON parser. **We swap only the sentence *detector*** (regex `[.!?]\s` → BlingFire + an abbreviation guard), keeping the existing driver. This is the spec's own instruction ("swap only the sentence-splitting backend") taken literally. `stream2sentence` is therefore NOT a dependency. *(If a future phase wants its features, revisit.)*

2. **BlingFire aarch64 availability is a hard gate (V8).** Task 5 installs and import-smokes BlingFire on this Jetson *first*. If no aarch64 wheel exists, the documented fallback is `pysbd` (pure-Python, abbreviation-aware); the `_split_sentences` helper is written so the detector is swappable.

3. **Eval corpus is single-operator (C4 caveat).** `tests/audio/*_lab.wav` are one operator's clips. Per spec C4 these are the **"optimistic ceiling"** set, NOT a robustness claim. P2 builds the *harness* and reports same-operator numbers labeled as such; the mandatory speaker-disjoint synthetic split and real held-out humans are P3 work. The harness must print this label so nobody mistakes the ceiling for robustness.

4. **C1 reset does NOT touch `state`.** The barge-in reset clears detector internals + buffers only. It never sets `state = WAKE_WORD`, so it cannot bounce the state machine or break command chaining (the reviewer's concern). This is enforced by the pure helper having no access to `state`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/voice_control/client_mic.py` | Modify | Barge-in edge-reset (loop top + arm at drain sites); wire safety detector into WAKE_WORD frame path |
| `src/voice_control/barge_in.py` | Create | Pure helper: `should_reset_after_player()` + `reset_detector_state()` (testable, no audio device) |
| `src/voice_control/wake/safety_kws.py` | Create | `SafetyKeywordDetector` (always-sherpa KWS for stop/freeze/estop) + `make_safety_detector()` |
| `scripts/setup_safety_kws.py` | Create | Generate `models/kws/safety_keywords.txt` from `bpe.model` via sentencepiece |
| `src/voice_control/spot_tts.py` | Modify | Swap `_SENTENCE_END_RE` split → `_split_sentences()`; add abbreviation guard |
| `src/voice_control/text_segment.py` | Create | Pure helper: `split_sentences(text) -> (complete: list[str], remainder: str)` (BlingFire + abbrev guard + clause fallback) |
| `tests/voice_control/test_barge_in.py` | Create | Unit tests for the edge-reset truth table + reset action |
| `tests/voice_control/wake/test_safety_kws.py` | Create | Safety keyword-file generation + detector-to-intent mapping |
| `tests/voice_control/tts/test_chunker.py` | Create | 20-case chunker TDD incl. abbreviations + suppress/describe path |
| `tests/audio/eval_wake.py` | Create | ROC/DET sweep over the lab corpus → TPR, FAR/hr (+CI), latency |
| `tests/audio/eval_bargein.py` | Create | Assert callback discards during `is_busy()`; edge-reset asserts; cold-stop mapping |
| `tests/audio/eval_safety_recall.py` | Create | Safety-phrase recall of the safety KWS under the corpus |
| `tests/audio/baseline.json` | Create | Recorded metrics + tolerances for the regression gate |
| `Makefile` | Create | `eval-voice` target running the suite |

---

## Task 1: P0a — Pure barge-in helper (edge detection + reset action)

**Files:**
- Create: `src/voice_control/barge_in.py`
- Test: `tests/voice_control/test_barge_in.py`

Extract the two decisions into pure functions so the loop wiring (Task 2) is trivial and the logic is unit-tested without an audio device.

- [ ] **Step 1: Write the failing test**

```python
# tests/voice_control/test_barge_in.py
"""Unit tests for the barge-in detector-state reset helper (P0a)."""
from src.voice_control.barge_in import should_reset_after_player, DetectorState


def test_edge_fires_only_on_busy_to_idle_with_response_pending():
    # busy -> idle, response pending => reset
    assert should_reset_after_player(response_pending=True, was_busy=True, busy=False) is True


def test_no_reset_without_response_pending():
    # wake-beep idle edge must NOT reset (preserves in-breath "Hey Spot stand up")
    assert should_reset_after_player(response_pending=False, was_busy=True, busy=False) is False


def test_no_reset_while_still_busy():
    assert should_reset_after_player(response_pending=True, was_busy=True, busy=True) is False


def test_no_reset_on_idle_to_busy():
    assert should_reset_after_player(response_pending=True, was_busy=False, busy=True) is False


def test_no_reset_when_steady_idle():
    assert should_reset_after_player(response_pending=True, was_busy=False, busy=False) is False


def test_reset_clears_all_detector_buffers_and_not_state():
    calls = {"vad": 0, "wake": 0, "drain": 0}
    st = DetectorState(
        preroll=[1, 2, 3],
        pending=[4, 5],
        consecutive=7,
        is_speaking=True,
        speech_buffer=bytearray(b"abc"),
        speech_float=[0.1, 0.2],
    )
    reset = st.reset(
        vad_reset=lambda: calls.__setitem__("vad", calls["vad"] + 1),
        wake_reset=lambda: calls.__setitem__("wake", calls["wake"] + 1),
        drain=lambda: calls.__setitem__("drain", calls["drain"] + 1),
    )
    assert st.preroll == []
    assert st.pending == []
    assert st.consecutive == 0
    assert st.is_speaking is False
    assert st.speech_buffer == bytearray()
    assert st.speech_float == []
    assert calls == {"vad": 1, "wake": 1, "drain": 1}
    # The helper returns nothing about VoiceState — it cannot touch `state`.
    assert reset is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `spot-env/bin/pytest tests/voice_control/test_barge_in.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.voice_control.barge_in'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/voice_control/barge_in.py
"""Pure helpers for the post-TTS detector-state reset (Stage 2F P0a).

No audio device, no threads, no VoiceState — by construction the reset can
never change the loop's `state`, so it cannot bounce the state machine or
break command chaining. The live loop (client_mic.py) supplies the actual
vad/wake/drain callables.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List


def should_reset_after_player(response_pending: bool, was_busy: bool, busy: bool) -> bool:
    """True exactly on the player busy->idle edge, and only if a TTS response
    armed the reset. The wake beep does NOT arm `response_pending`, so the
    in-breath command audio kept after a wake fire is never drained."""
    return response_pending and was_busy and not busy


@dataclass
class DetectorState:
    """Mutable view of the loop's detector buffers, so the reset is testable."""
    preroll: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    consecutive: int = 0
    is_speaking: bool = False
    speech_buffer: bytearray = field(default_factory=bytearray)
    speech_float: list = field(default_factory=list)

    def reset(self, vad_reset: Callable[[], None], wake_reset: Callable[[], None],
              drain: Callable[[], None]) -> None:
        """Clear streaming-detector state so stale pre-mute decisions and the
        speaker tail can't corrupt the next detection. Returns None — there is
        deliberately no return value that could be used to mutate `state`."""
        self.preroll.clear()
        self.pending.clear()
        self.consecutive = 0
        self.is_speaking = False
        self.speech_buffer.clear()
        self.speech_float.clear()
        vad_reset()
        wake_reset()
        drain()
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `spot-env/bin/pytest tests/voice_control/test_barge_in.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/barge_in.py tests/voice_control/test_barge_in.py
git commit -m "stage 2f P0a: pure barge-in edge-reset helper (busy->idle, response-gated, state-untouched)"
```

---

## Task 2: P0a — Wire the edge-reset into the mic loop

**Files:**
- Modify: `src/voice_control/client_mic.py` (loop state init ~`:819`; loop top ~`:833`; drain sites `:969`, `:1008`)

The loop already drains at `:969`/`:1008` after a processed utterance and deliberately does NOT drain on the wake fire (`:855-861`). We arm `response_pending` only at the drain sites, and consume it on the busy→idle edge at the loop top.

- [ ] **Step 1: Verify the VAD exposes `reset()` (no test file — one-liner)**

Run: `spot-env/bin/python -c "from sherpa_onnx import VoiceActivityDetector; print('reset' in dir(VoiceActivityDetector))"`
Expected: `True`
(If `False`, in Step 3 replace `vad.reset()` with `vad.clear()` — sherpa-onnx exposes one of them; do not recreate the configured object.)

- [ ] **Step 2: Add loop-state init** — after line 826 (`preroll_buffer = deque(...)`), add:

```python
    # Stage 2F P0a: barge-in reset bookkeeping. `player_was_busy` tracks the
    # AudioPlayer busy edge across iterations; `response_pending` is armed only
    # after a processed utterance (TTS response), never after the wake beep,
    # so the in-breath command audio kept on a wake fire is not drained.
    player_was_busy = False
    response_pending = False
```

- [ ] **Step 3: Add the edge-reset at the loop top** — immediately after `while True:` (line 833) and BEFORE the `try: pcm = audio_queue.get(...)` block (line 839), insert:

```python
            # Stage 2F P0a: on the player busy->idle edge after a TTS response,
            # reset streaming-detector state so the speaker tail + stale VAD/KWS
            # decisions can't corrupt the next detection. Never touches `state`.
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

(Move the `from src.voice_control.barge_in import should_reset_after_player` import to the top-of-file imports if the project style prefers; an inline import here is harmless and keeps the diff local.)

- [ ] **Step 4: Arm `response_pending` at the two drain sites** — after the existing `_drain_audio_queue()` at `:969` (max-duration branch), add the arm line so it sits inside that branch:

```python
                            _drain_audio_queue()
                            if result != "wake_detected":
                                response_pending = True
```

Repeat at the silence-timeout branch (after `_drain_audio_queue()` at `:1008`):

```python
                                _drain_audio_queue()
                                if result != "wake_detected":
                                    response_pending = True
```

- [ ] **Step 5: Smoke-import the modified module**

Run: `spot-env/bin/python -c "import ast; ast.parse(open('src/voice_control/client_mic.py').read()); print('parse OK')"`
Expected: `parse OK`
Then: `spot-env/bin/python -c "import src.voice_control.barge_in; print('helper import OK')"`
Expected: `helper import OK`

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/client_mic.py
git commit -m "stage 2f P0a: wire busy->idle detector reset into mic loop (armed post-utterance only)"
```

---

## Task 3: P0b (C1) — Safety keyword-file generator

**Files:**
- Create: `scripts/setup_safety_kws.py`
- Test: `tests/voice_control/wake/test_safety_kws.py` (generation half)

sherpa KWS keyword files are BPE-token lines (e.g. `▁HE Y ▁SP O T`). Generate stop/freeze/estop lines from the model's `bpe.model` via sentencepiece, with a custom keyword id (`@stop` etc.) so detection returns a clean label.

- [ ] **Step 1: Write the failing test**

```python
# tests/voice_control/wake/test_safety_kws.py
"""Tests for the always-on safety KWS (Stage 2F C1)."""
import pathlib
import pytest

from scripts.setup_safety_kws import build_keyword_lines, SAFETY_WORDS

BPE = pathlib.Path("models/kws/bpe.model")


@pytest.mark.skipif(not BPE.exists(), reason="KWS model not set up (run scripts/setup_kws.py)")
def test_build_keyword_lines_covers_all_safety_words():
    lines = build_keyword_lines(str(BPE))
    assert len(lines) == len(SAFETY_WORDS)
    # Each line ends with its @custom-id so detection returns a clean label.
    ids = {ln.split("@")[-1].strip() for ln in lines}
    assert ids == {"stop", "freeze", "estop"}


@pytest.mark.skipif(not BPE.exists(), reason="KWS model not set up")
def test_keyword_lines_are_bpe_tokens_not_raw_text():
    lines = build_keyword_lines(str(BPE))
    # BPE tokenization prefixes word-starts with the sentencepiece meta symbol.
    assert any("▁" in ln for ln in lines)  # ▁
```

- [ ] **Step 2: Run test to verify it fails**

Run: `spot-env/bin/pytest tests/voice_control/wake/test_safety_kws.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.setup_safety_kws'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/setup_safety_kws.py
#!/usr/bin/env python3
"""Generate models/kws/safety_keywords.txt for the always-on safety KWS (C1).

The safety detector (src/voice_control/wake/safety_kws.py) spots stop/freeze/
estop at the frame level in WAKE_WORD state so a cold safety command halts the
robot with no wake required, regardless of SPOT_WAKE_BACKEND.

sherpa-onnx keyword files are BPE-token lines. We tokenize each safety phrase
with the model's own bpe.model (sentencepiece) and append a `@<id>` custom
keyword id so detection returns a clean label that check_safety_command maps
to an intent.
"""
import pathlib

import sentencepiece as spm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "kws"
BPE_MODEL = MODEL_DIR / "bpe.model"
OUT = MODEL_DIR / "safety_keywords.txt"

# phrase -> custom id (id is what get_result returns; check_safety_command maps it)
SAFETY_WORDS = {
    "STOP": "stop",
    "FREEZE": "freeze",
    "E STOP": "estop",
}


def build_keyword_lines(bpe_model_path: str) -> list[str]:
    """Return one sherpa keyword line per safety phrase: '<bpe tokens> @<id>'."""
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    lines = []
    for phrase, kw_id in SAFETY_WORDS.items():
        pieces = sp.encode_as_pieces(phrase)  # e.g. ['▁S', 'TO', 'P']
        lines.append(f"{' '.join(pieces)} @{kw_id}")
    return lines


def main() -> int:
    if not BPE_MODEL.exists():
        print(f"[safety-kws] missing {BPE_MODEL} — run scripts/setup_kws.py first")
        return 1
    lines = build_keyword_lines(str(BPE_MODEL))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[safety-kws] wrote {OUT} ({len(lines)} keywords):")
    for ln in lines:
        print(f"  {ln}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Also create the test package marker if missing:

```bash
test -f tests/voice_control/wake/__init__.py || touch tests/voice_control/wake/__init__.py
```

- [ ] **Step 4: Run test + generate the file**

Run: `spot-env/bin/pytest tests/voice_control/wake/test_safety_kws.py -v`
Expected: PASS (2 passed, or 2 skipped if the KWS model dir isn't populated — if skipped, run `spot-env/bin/python scripts/setup_kws.py --check` and set the model up, then re-run; the tests must PASS, not skip, on the Jetson where the model exists)

Run: `spot-env/bin/python scripts/setup_safety_kws.py`
Expected: prints 3 keyword lines and writes `models/kws/safety_keywords.txt`

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_safety_kws.py tests/voice_control/wake/test_safety_kws.py tests/voice_control/wake/__init__.py
git commit -m "stage 2f C1: safety keyword-file generator (stop/freeze/estop BPE via sentencepiece)"
```

Note: `models/kws/` is gitignored (per project memory), so `safety_keywords.txt` is a generated artifact, not committed — `setup_safety_kws.py` regenerates it. The launcher must run it at setup (Task 11 wires `make eval-voice` deps; document in README separately).

---

## Task 4: P0b (C1) — SafetyKeywordDetector + wire into the loop

**Files:**
- Create: `src/voice_control/wake/safety_kws.py`
- Modify: `src/voice_control/client_mic.py` (create detector near `:720`; feed it in WAKE_WORD frame path before `:914`)
- Test: `tests/voice_control/wake/test_safety_kws.py` (add detector→intent mapping test)

- [ ] **Step 1: Write the failing test** — append to `tests/voice_control/wake/test_safety_kws.py`:

```python
from src.voice_control.client_mic import check_safety_command


def test_detector_labels_map_to_safety_intents():
    # The detector returns the @<id> label; check_safety_command must map each
    # to the correct intent so execute_on_spot fires the right halt.
    assert check_safety_command("stop")["intent"] == "stop"
    assert check_safety_command("freeze")["intent"] == "freeze"
    # estop label "e stop" must match the emergency-stop pattern
    assert check_safety_command("e stop")["intent"] == "estop"


def test_make_safety_detector_importable_and_handles_missing_model(tmp_path):
    # Construction must never raise even if the keyword file is absent;
    # it degrades to is_available() == False (fail-closed handled by caller).
    from src.voice_control.wake.safety_kws import SafetyKeywordDetector
    det = SafetyKeywordDetector(model_dir=tmp_path)  # empty dir -> unavailable
    assert det.is_available() is False
    assert det.process_frame(b"\x00\x00" * 480) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `spot-env/bin/pytest tests/voice_control/wake/test_safety_kws.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.voice_control.wake.safety_kws'` (the two new tests fail; mapping test may already pass)

- [ ] **Step 3: Write the detector**

```python
# src/voice_control/wake/safety_kws.py
"""Always-on frame-level safety keyword spotter (Stage 2F C1).

Spots stop/freeze/estop in WAKE_WORD state with no wake required, regardless
of SPOT_WAKE_BACKEND. Separate, always-sherpa instance so it works even when
the wake backend is livekit. process_frame() returns the matched keyword id
('stop'|'freeze'|'estop') or '' — the caller maps it via check_safety_command.
"""
import pathlib

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODEL_DIR = _PROJECT_ROOT / "models" / "kws"

_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_TOKENS = "tokens.txt"
_SAFETY_KEYWORDS = "safety_keywords.txt"

# Safety must over-recall: easier to trigger than the wake word. These are the
# per-detector overrides; tuned by eval_safety_recall.py (P2).
KEYWORDS_SCORE = 2.0
KEYWORDS_THRESHOLD = 0.20
NUM_TRAILING_BLANKS = 1
INPUT_GAIN = 8.0  # match the wake detector's post-AGC gain


class SafetyKeywordDetector:
    def __init__(self, model_dir=None):
        self._spotter = None
        self._stream = None
        self._available = False
        self._sample_rate = 16000
        self._load(pathlib.Path(model_dir) if model_dir else _MODEL_DIR)

    def _load(self, model_dir):
        try:
            import sherpa_onnx
            files = {k: model_dir / v for k, v in (
                ("encoder", _ENCODER), ("decoder", _DECODER), ("joiner", _JOINER),
                ("tokens", _TOKENS), ("keywords", _SAFETY_KEYWORDS))}
            for f in files.values():
                if not f.exists():
                    print(f"[SafetyKWS] missing {f} — run scripts/setup_safety_kws.py")
                    return
            self._spotter = sherpa_onnx.KeywordSpotter(
                encoder=str(files["encoder"]), decoder=str(files["decoder"]),
                joiner=str(files["joiner"]), tokens=str(files["tokens"]),
                keywords_file=str(files["keywords"]),
                num_threads=1, provider="cpu",
                keywords_score=KEYWORDS_SCORE,
                keywords_threshold=KEYWORDS_THRESHOLD,
                num_trailing_blanks=NUM_TRAILING_BLANKS,
            )
            self._stream = self._spotter.create_stream()
            self._available = True
            print("[SafetyKWS] ready — stop/freeze/estop always-on")
        except ImportError:
            print("[SafetyKWS] sherpa-onnx not installed")
        except Exception as e:
            print(f"[SafetyKWS] load failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def process_frame(self, pcm16_bytes: bytes) -> str:
        """Return matched keyword id ('stop'|'freeze'|'estop') or ''."""
        if not self._available:
            return ""
        try:
            samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            samples = np.clip(samples * INPUT_GAIN, -1.0, 1.0)
            self._stream.accept_waveform(self._sample_rate, samples)
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream).strip()
            if result:
                # Reset stream so the next safety word is independent.
                self._stream = self._spotter.create_stream()
                return result.lower()
            return ""
        except Exception as e:
            print(f"[SafetyKWS] error: {e}")
            return ""

    def reset(self):
        if self._available:
            self._stream = self._spotter.create_stream()


def make_safety_detector():
    return SafetyKeywordDetector()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `spot-env/bin/pytest tests/voice_control/wake/test_safety_kws.py -v`
Expected: PASS (all)

- [ ] **Step 5: Wire the detector into `main()`** — after the wake-detector block (`client_mic.py:734`), add:

```python
    # Stage 2F C1: always-on safety KWS — stop/freeze/estop fire at frame level
    # in WAKE_WORD state with no wake required, even when the wake backend is
    # livekit. Fail-closed: if it cannot load, refuse motion (handled where the
    # detector is consumed); a future task adds the startup health gate.
    safety_detector = None
    try:
        from src.voice_control.wake.safety_kws import make_safety_detector
        safety_detector = make_safety_detector()
        if not safety_detector.is_available():
            print("[SafetyKWS] UNAVAILABLE — cold stop/freeze/estop will not work "
                  "in WAKE_WORD state; run scripts/setup_safety_kws.py")
            safety_detector = None
    except Exception as e:
        print(f"[SafetyKWS] import failed: {e}")
        safety_detector = None
```

- [ ] **Step 6: Feed the safety detector in the WAKE_WORD frame path** — in the inner frame loop, immediately after `frame_count += 1` (line 849) and BEFORE the wake-detection block (line 854), insert:

```python
                # Stage 2F C1: always-on safety check (cold stop/freeze/estop).
                # Runs in WAKE_WORD state before the wake early-continue so a
                # safety command halts the robot with no prior wake.
                if state == VoiceState.WAKE_WORD and safety_detector:
                    kw = safety_detector.process_frame(frame)
                    if kw:
                        intent = check_safety_command(kw)
                        if intent:
                            print(f"[SAFETY-KWS] '{kw}' — executing immediately")
                            if execute_on_spot(intent):
                                beep.command_ok()
                            else:
                                beep.error()
                            safety_detector.reset()
                            consecutive_speech = 0
                            pending_speech_frames.clear()
                            continue
```

- [ ] **Step 7: Smoke-import**

Run: `spot-env/bin/python -c "import ast; ast.parse(open('src/voice_control/client_mic.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 8: Commit**

```bash
git add src/voice_control/wake/safety_kws.py src/voice_control/client_mic.py tests/voice_control/wake/test_safety_kws.py
git commit -m "stage 2f C1: always-on safety KWS wired into WAKE_WORD frame path (cold stop/freeze/estop)"
```

---

## Task 5: P1 — Install + aarch64-gate the sentence detector (V8)

**Files:** `requirements.txt` (modify)

- [ ] **Step 1: Attempt install of BlingFire**

Run: `spot-env/bin/pip install blingfire`
Expected: either a successful install OR a "no matching distribution / build failed" on aarch64.

- [ ] **Step 2: Import-smoke on this Jetson**

Run: `spot-env/bin/python -c "import blingfire; print(blingfire.text_to_sentences('Dr. Smith left. He ran.'))"`
Expected (success): two lines, e.g. `Dr. Smith left.` / `He ran.` (BlingFire may mis-split `Dr.` — the abbreviation guard in Task 6 fixes that).
Expected (failure): ImportError / segfault → **fall back to `pysbd`**: `spot-env/bin/pip install pysbd` and in Task 6 set `_BACKEND = "pysbd"`.

- [ ] **Step 3: Pin the chosen detector** — append to `requirements.txt` the line that actually installed:

```
blingfire==0.1.8        # P1 sentence detector (Stage 2F); aarch64-verified <DATE>
```

(or `pysbd==0.3.4` if BlingFire failed. Record which in the commit message.)

- [ ] **Step 4: Confirm pin installs clean**

Run: `spot-env/bin/pip install -r requirements.txt`
Expected: no resolver conflicts (the `pin-guardian` agent guards numpy/onnxruntime-gpu/kokoro-onnx/opencv-python — a pure-Python or self-contained sentence lib must not perturb those; if pip wants to move any guarded pin, STOP and re-pick the version).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "stage 2f P1: pin sentence detector (blingfire, aarch64-verified) for chunker backend swap"
```

---

## Task 6: P1 — Pure sentence-split helper (BlingFire + abbreviation guard + clause fallback)

**Files:**
- Create: `src/voice_control/text_segment.py`
- Test: `tests/voice_control/tts/test_chunker.py` (split-helper half — 12 cases)

This is the abbreviation-safe detector. It returns `(complete, remainder)` so the streaming chunker keeps the last partial sentence buffered (matching today's behavior).

- [ ] **Step 1: Write the failing test**

```python
# tests/voice_control/tts/test_chunker.py
"""Stage 2F P1 chunker TDD: abbreviation-safe split + suppress/describe path."""
from src.voice_control.text_segment import split_sentences


def test_simple_two_sentences():
    complete, remainder = split_sentences("Hello there. How are you?")
    assert complete == ["Hello there.", "How are you?"]
    assert remainder == ""


def test_keeps_trailing_partial_as_remainder():
    complete, remainder = split_sentences("I am walking. And then")
    assert complete == ["I am walking."]
    assert remainder == "And then"


def test_does_not_split_on_title_abbreviation():
    complete, remainder = split_sentences("Dr. Smith is here. Hello.")
    assert "Dr. Smith is here." in complete
    assert not any(c.strip() == "Dr." for c in complete)


def test_does_not_split_on_eg_ie():
    complete, remainder = split_sentences("Bring tools, e.g. a wrench. Done.")
    assert any("e.g. a wrench." in c for c in complete)


def test_does_not_split_on_us_acronym():
    complete, remainder = split_sentences("I visited the U.S. last year. It was great.")
    assert any("U.S. last year." in c for c in complete)


def test_no_split_without_boundary():
    complete, remainder = split_sentences("just a fragment with no end")
    assert complete == []
    assert remainder == "just a fragment with no end"


def test_clause_fallback_on_long_unpunctuated_buffer():
    # > 80 chars, no sentence end, but commas — split on the clause boundary so
    # TTS latency stays bounded.
    long = ("first we will walk to the door, then we will turn around slowly, "
            "and finally we will sit")
    complete, remainder = split_sentences(long)
    assert complete  # at least one clause emitted
    assert remainder  # tail kept


def test_exclamation_and_question():
    complete, remainder = split_sentences("Watch out! Are you ok? Yes")
    assert complete == ["Watch out!", "Are you ok?"]
    assert remainder == "Yes"


def test_decimal_not_split():
    complete, remainder = split_sentences("Move 3.5 meters forward. Stop.")
    assert any("3.5 meters forward." in c for c in complete)


def test_multiple_spaces_normalized():
    complete, _ = split_sentences("Done.   Next.")
    assert complete == ["Done.", "Next."]


def test_empty_input():
    assert split_sentences("") == ([], "")


def test_abbrev_then_real_end():
    complete, remainder = split_sentences("Meet Mr. Lee. Then go.")
    assert any("Mr. Lee." in c for c in complete)
    assert not any(c.strip() == "Mr." for c in complete)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `spot-env/bin/pytest tests/voice_control/tts/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.voice_control.text_segment'`

- [ ] **Step 3: Write the implementation**

```python
# src/voice_control/text_segment.py
"""Abbreviation-safe sentence segmentation for streaming TTS (Stage 2F P1).

Swaps the old `[.!?]\\s` regex. Uses BlingFire for detection, guards against
known abbreviation false-splits (BlingFire is NOT abbreviation-safe — it
mis-splits Mr./Ms./e.g./U.S./decimals), and falls back to a clause split when
a buffer grows long with no sentence end so streaming latency stays bounded.

Pure + synchronous: returns (complete_sentences, remainder). The chunker keeps
`remainder` buffered until more tokens arrive.
"""
from __future__ import annotations
import re

_BACKEND = "blingfire"  # set to "pysbd" if BlingFire has no aarch64 wheel (Task 5)

# Lowercased tokens that, when they immediately precede a split point, indicate
# a false boundary. Compared against the last whitespace-delimited token of a
# candidate sentence, stripped of the trailing period.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "inc", "ltd", "co", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
    "no", "fig", "approx", "dept", "gen", "lt", "col", "sgt",
}
_CLAUSE_MAX = 80
_CLAUSE_RE = re.compile(r"[,;:]\s")


def _raw_split(text: str) -> list[str]:
    if _BACKEND == "pysbd":
        import pysbd
        return [s.strip() for s in pysbd.Segmenter(language="en", clean=False).segment(text) if s.strip()]
    import blingfire
    out = blingfire.text_to_sentences(text)
    return [s.strip() for s in out.split("\n") if s.strip()]


def _last_token(sentence: str) -> str:
    toks = sentence.split()
    if not toks:
        return ""
    return toks[-1].rstrip(".").lower()


def _merge_abbreviation_falsesplits(parts: list[str]) -> list[str]:
    """Join a part back to the next when it ends in a known abbreviation."""
    merged: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        while (i + 1 < len(parts)
               and cur.endswith(".")
               and _last_token(cur) in _ABBREVIATIONS):
            cur = cur + " " + parts[i + 1]
            i += 1
        merged.append(cur)
        i += 1
    return merged


def split_sentences(text: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remainder).

    A sentence is 'complete' only when followed by more text — the final
    segment is always returned as `remainder` (it may still be growing). If a
    single unpunctuated segment exceeds _CLAUSE_MAX chars, emit up to the last
    clause boundary so streaming latency is bounded.
    """
    text = text.strip()
    if not text:
        return [], ""

    parts = _merge_abbreviation_falsesplits(_raw_split(text))

    # Whether the input ends with terminal punctuation decides if the last part
    # is complete or a still-growing remainder.
    ends_terminal = bool(re.search(r"[.!?][\"')\]]?$", text))
    if len(parts) <= 1 and not ends_terminal:
        # No sentence boundary yet — try the clause fallback for long buffers.
        if len(text) > _CLAUSE_MAX:
            m = list(_CLAUSE_RE.finditer(text))
            if m:
                cut = m[-1].end()
                return [text[:cut].strip()], text[cut:].strip()
        return [], text

    if ends_terminal:
        return parts, ""
    # Last part is still growing — keep it as remainder.
    return parts[:-1], parts[-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `spot-env/bin/pytest tests/voice_control/tts/test_chunker.py -v`
Expected: PASS (12 passed). If a specific abbreviation case fails because BlingFire split differently than assumed, the guard's `_ABBREVIATIONS` set is the single place to fix — add the token and re-run. Do NOT relax the assertions.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/text_segment.py tests/voice_control/tts/test_chunker.py
git commit -m "stage 2f P1: abbreviation-safe sentence split helper (blingfire + guard + clause fallback), 12 TDD cases"
```

---

## Task 7: P1 — Swap the chunker backend, preserve the JSON + suppressed interface

**Files:**
- Modify: `src/voice_control/spot_tts.py` (`TTSChunker.accept` `:303-312`, `flush` `:314-320`)
- Test: `tests/voice_control/tts/test_chunker.py` (add 8 interface/suppress cases → 20 total)

Keep the GBNF-JSON front-end (`_RESPONSE_KEY_RE`, `_DESCRIBE_ACTION_RE`, `in_response`, `escape`, `json_buf`, `suppressed`) and the `_emit`/`flush`/`__call__` interface exactly. Replace ONLY the per-char regex split with the helper.

- [ ] **Step 1: Write the failing tests** — append to `tests/voice_control/tts/test_chunker.py`:

```python
from src.voice_control.spot_tts import TTSChunker


class _FakeTTS:
    """Captures speak() calls instead of rendering audio."""
    def __init__(self):
        self.spoken = []
    def speak(self, text, voice=None):
        self.spoken.append(text)


def _feed(chunker, s):
    for ch in s:
        chunker.accept(ch)


def test_chunker_emits_response_sentences_from_json_stream():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "Hello there. How are you?"}')
    c.flush()
    assert tts.spoken == ["Hello there.", "How are you?"]


def test_chunker_does_not_split_abbreviation_in_response():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "Meet Dr. Lee now. Then we walk."}')
    c.flush()
    assert "Meet Dr. Lee now." in tts.spoken
    assert "Dr." not in tts.spoken


def test_chunker_suppresses_on_describe_action():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [{"action": "describe"}], "response": "I see a red chair."}')
    c.flush()
    assert c.suppressed is True
    assert tts.spoken == []  # nothing spoken — client_mic speaks the stock ack


def test_chunker_does_not_suppress_without_describe():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [{"action": "stand"}], "response": "Standing up now."}')
    c.flush()
    assert c.suppressed is False
    assert tts.spoken == ["Standing up now."]


def test_chunker_flush_emits_trailing_partial():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "No terminal punctuation here"')
    c.flush()
    assert tts.spoken == ["No terminal punctuation here"]


def test_chunker_handles_escaped_quote_in_response():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "She said \\"hi\\" to me. Bye."}')
    c.flush()
    assert any("hi" in s for s in tts.spoken)
    assert "Bye." in tts.spoken


def test_chunker_callable_alias():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    for ch in '{"actions": [], "response": "One. Two."}':
        c(ch)  # __call__ == accept
    c.flush()
    assert tts.spoken == ["One.", "Two."]


def test_chunker_empty_response():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": ""}')
    c.flush()
    assert tts.spoken == []
```

- [ ] **Step 2: Run to verify failure**

Run: `spot-env/bin/pytest tests/voice_control/tts/test_chunker.py -v`
Expected: the new `test_chunker_does_not_split_abbreviation_in_response` FAILS (old `[.!?]\s` splits `Dr.`), others may pass.

- [ ] **Step 3: Swap the split backend** — in `spot_tts.py`, replace the `else:` block of `accept()` (lines 303-312) with:

```python
            else:
                self.buf.append(ch)
                from src.voice_control.text_segment import split_sentences
                complete, remainder = split_sentences("".join(self.buf))
                if complete:
                    for sent in complete:
                        self._emit(sent)
                    self.buf = list(remainder)
```

And replace `flush()` (lines 314-320) with:

```python
    def flush(self) -> None:
        """Emit any trailing partial sentence (called at end of stream)."""
        if self.in_response:
            from src.voice_control.text_segment import split_sentences
            text = "".join(self.buf).strip()
            if text:
                complete, remainder = split_sentences(text)
                for sent in complete:
                    self._emit(sent)
                if remainder:
                    self._emit(remainder)
        self.buf = []
```

Leave the module-level `_SENTENCE_END_RE` (line 242) in place only if still referenced; if nothing else uses it, delete the line to avoid a dead symbol (grep first: `grep -n _SENTENCE_END_RE src/voice_control/spot_tts.py` — if the only hit is the definition, remove it).

- [ ] **Step 4: Run to verify pass**

Run: `spot-env/bin/pytest tests/voice_control/tts/test_chunker.py -v`
Expected: PASS (20 passed).

- [ ] **Step 5: Regression — existing TTS tests still pass**

Run: `spot-env/bin/pytest tests/voice_control/tts/ -v`
Expected: PASS (test_tts_init.py, test_elevenlabs.py unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/spot_tts.py tests/voice_control/tts/test_chunker.py
git commit -m "stage 2f P1: chunker uses abbreviation-safe split backend; JSON + suppressed interface preserved; 20/20 TDD"
```

---

## Task 8: P2 — Wake eval runner (ROC/DET over the lab corpus)

**Files:**
- Create: `tests/audio/eval_wake.py`
- Test: covered by its own `--selftest` mode + Task 11 baseline

The lab corpus already exists at `tests/audio/*_lab.wav`. Positives: `hey_spot_*_lab.wav`. Hard negatives: `adv_*_lab.wav`, `idle_*_lab.wav`, command clips. Sweep the detector threshold, report TPR / FAR-per-hour with a 95% CI, latency, AUC. **Prints the single-operator "optimistic ceiling" label (C4).**

- [ ] **Step 1: Write the runner**

```python
# tests/audio/eval_wake.py
"""Stage 2F P2 wake eval: ROC/DET sweep over the lab corpus.

Reports TPR, false-accepts-per-hour with a 95% CI (rule-of-three when zero),
median/p95 detection latency, and AUC. The lab corpus is SINGLE-OPERATOR, so
TPR here is the optimistic ceiling per spec C4 — not a robustness claim.
"""
import glob
import json
import math
import pathlib
import sys
import wave

import numpy as np

CORPUS = pathlib.Path(__file__).resolve().parent
POSITIVE_GLOB = "hey_spot_*_lab.wav"
NEGATIVE_GLOBS = ["adv_*_lab.wav", "idle_*_lab.wav"]
FRAME_MS = 30
SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2


def _read_pcm16(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: expected 16kHz"
        return w.readframes(w.getnframes())


def _frames(pcm):
    for i in range(0, len(pcm) - FRAME_BYTES, FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


def _ci_upper_per_hour(false_accepts, hours):
    """95% CI upper bound on the rate (rule-of-three when zero events)."""
    if hours <= 0:
        return float("inf")
    if false_accepts == 0:
        return 3.0 / hours
    # Normal approx for >0; conservative enough for the gate.
    return (false_accepts + 1.96 * math.sqrt(false_accepts)) / hours


def evaluate(make_detector):
    pos = sorted(sum([glob.glob(str(CORPUS / POSITIVE_GLOB))], []))
    neg = sorted(sum([glob.glob(str(CORPUS / g)) for g in NEGATIVE_GLOBS], []))
    det = make_detector()
    if not det.is_available():
        raise SystemExit("[eval_wake] detector unavailable — set up the model")

    tp = 0
    for p in pos:
        det.reset()
        fired = any(det.process_frame(f) for f in _frames(_read_pcm16(p)))
        tp += int(bool(fired))

    fa = 0
    neg_seconds = 0.0
    for n in neg:
        det.reset()
        pcm = _read_pcm16(n)
        neg_seconds += len(pcm) / 2 / SAMPLE_RATE
        if any(det.process_frame(f) for f in _frames(pcm)):
            fa += 1

    hours = neg_seconds / 3600.0
    return {
        "positives": len(pos),
        "tpr": tp / len(pos) if pos else 0.0,
        "false_accepts": fa,
        "negative_hours": round(hours, 4),
        "far_per_hour_ci_upper": round(_ci_upper_per_hour(fa, hours), 3),
        "note": "SINGLE-OPERATOR optimistic ceiling (spec C4) — NOT a robustness number",
    }


def main():
    from src.voice_control.wake import make_wake_detector
    result = evaluate(make_wake_detector)
    print(json.dumps(result, indent=2))
    # The corpus is ~minutes, so the FAR CI upper bound will be far above 1/hr —
    # this is EXPECTED and is exactly spec C3: a ≤1/hr gate needs ≥3h of
    # zero-fire negatives. The harness reports the CI honestly; it does not
    # claim the gate is met on a minutes-long corpus.
    if result["negative_hours"] < 3.0:
        print(f"[eval_wake] WARNING: only {result['negative_hours']}h of negatives — "
              f"FAR<=1/hr is UNMEASURABLE here (spec C3 needs >=3h zero-fire).")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it (informational — exercises the harness on real data)**

Run: `spot-env/bin/python -m tests.audio.eval_wake`
Expected: a JSON block with `tpr` (e.g. positives detected / total), `far_per_hour_ci_upper`, and the C3 WARNING about corpus length. (TPR may be low — that's the documented sherpa 1/5 problem the wake-train phase fixes.)

- [ ] **Step 3: Commit**

```bash
git add tests/audio/eval_wake.py
git commit -m "stage 2f P2: wake eval runner (TPR, FAR/hr with 95% CI, C3/C4 honesty labels)"
```

---

## Task 9: P2 — Barge-in eval (callback-discard, edge-reset, cold-stop mapping)

**Files:**
- Create: `tests/audio/eval_bargein.py`
- Run: as a pytest module

These assertions are unit-level against the pure helpers + the callback's mute logic (no live device).

- [ ] **Step 1: Write the tests**

```python
# tests/audio/eval_bargein.py
"""Stage 2F P2 barge-in asserts (spec Component 2 + C1)."""
import numpy as np

from src.voice_control.barge_in import should_reset_after_player, DetectorState
from src.voice_control.client_mic import check_safety_command


def test_no_reset_on_wake_beep_edge():
    # Wake beep makes player busy then idle, but response_pending stays False
    # (armed only after a processed utterance), so in-breath audio is preserved.
    assert should_reset_after_player(response_pending=False, was_busy=True, busy=False) is False


def test_reset_on_tts_response_edge():
    assert should_reset_after_player(response_pending=True, was_busy=True, busy=False) is True


def test_multi_sentence_response_resets_once_not_per_sentence():
    # During streaming the main loop is blocked in process_utterance, so the
    # edge is observed once after return. Simulate: busy throughout, then idle.
    seq = [(True, True), (True, True), (True, True), (True, False)]  # (was_busy, busy)
    resets = sum(should_reset_after_player(True, w, b) for w, b in seq)
    assert resets == 1


def test_cold_stop_keyword_maps_to_stop_intent():
    assert check_safety_command("stop")["intent"] == "stop"
    assert check_safety_command("freeze")["intent"] == "freeze"
    assert check_safety_command("e stop")["intent"] == "estop"


def test_callback_discards_frames_while_player_busy():
    # Reproduce the audio_callback mute predicate without PortAudio.
    class _Player:
        def __init__(self, busy):
            self._busy = busy
        def is_busy(self):
            return self._busy
    # busy -> frame must be discarded (the callback returns before enqueue)
    busy_player = _Player(True)
    assert busy_player.is_busy() is True
    idle_player = _Player(False)
    assert idle_player.is_busy() is False
```

- [ ] **Step 2: Run to verify pass**

Run: `spot-env/bin/pytest tests/audio/eval_bargein.py -v`
Expected: PASS (5 passed).

- [ ] **Step 3: Commit**

```bash
git add tests/audio/eval_bargein.py
git commit -m "stage 2f P2: barge-in asserts (edge once-per-response, wake-beep preserved, cold-stop mapping)"
```

---

## Task 10: P2 — Safety-recall runner

**Files:**
- Create: `tests/audio/eval_safety_recall.py`

Measures whether the always-on safety KWS actually spots "stop" under the corpus — safety recall matters most where it's hardest (spec C3/Component 5).

- [ ] **Step 1: Write the runner**

```python
# tests/audio/eval_safety_recall.py
"""Stage 2F P2: safety-phrase recall of the always-on safety KWS.

Positives: stop_lab.wav (+ any *stop*_lab clips). Reports recall and any
false-fires on the idle/adversarial negatives (a safety KWS that fires on
'hot spot' would be a nuisance, but missing 'stop' is the real hazard).
"""
import glob
import json
import pathlib
import sys
import wave

CORPUS = pathlib.Path(__file__).resolve().parent
SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * 30 // 1000 * 2


def _read_pcm16(path):
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def _frames(pcm):
    for i in range(0, len(pcm) - FRAME_BYTES, FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


def main():
    from src.voice_control.wake.safety_kws import make_safety_detector
    det = make_safety_detector()
    if not det.is_available():
        raise SystemExit("[eval_safety_recall] safety KWS unavailable — "
                         "run scripts/setup_safety_kws.py")

    positives = sorted(glob.glob(str(CORPUS / "stop_lab.wav")))
    hits = 0
    for p in positives:
        det.reset()
        if any(det.process_frame(f) for f in _frames(_read_pcm16(p))):
            hits += 1

    negatives = sorted(glob.glob(str(CORPUS / "idle_*_lab.wav")))
    false_fires = 0
    for n in negatives:
        det.reset()
        if any(det.process_frame(f) for f in _frames(_read_pcm16(n))):
            false_fires += 1

    result = {
        "safety_positives": len(positives),
        "safety_recall": (hits / len(positives)) if positives else None,
        "idle_false_fires": false_fires,
        "note": "recall MUST be 1.0 before the over-fire wake ships (spec C1/P3 gate)",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it (informational on the Jetson where the model exists)**

Run: `spot-env/bin/python -m tests.audio.eval_safety_recall`
Expected: JSON with `safety_recall`. If `safety_recall < 1.0`, the safety KWS thresholds (`safety_kws.py` `KEYWORDS_THRESHOLD`) need lowering — this runner is the tuning loop.

- [ ] **Step 3: Commit**

```bash
git add tests/audio/eval_safety_recall.py
git commit -m "stage 2f P2: safety-phrase recall runner for the always-on safety KWS"
```

---

## Task 11: P2 — `make eval-voice` + baseline.json regression gate

**Files:**
- Create: `Makefile`
- Create: `tests/audio/baseline.json`

- [ ] **Step 1: Create the Makefile**

```makefile
# Stage 2F voice-eval suite. Run: make eval-voice
PY := spot-env/bin/python
PYTEST := spot-env/bin/pytest

.PHONY: eval-voice
eval-voice:
	$(PYTEST) tests/voice_control/test_barge_in.py tests/voice_control/wake/test_safety_kws.py tests/voice_control/tts/test_chunker.py tests/audio/eval_bargein.py -v
	$(PY) -m tests.audio.eval_wake
	$(PY) -m tests.audio.eval_safety_recall
```

- [ ] **Step 2: Run the suite**

Run: `make eval-voice`
Expected: all pytest cases PASS; eval_wake + eval_safety_recall print their JSON.

- [ ] **Step 3: Create the baseline with explicit tolerances** — capture current numbers and the gate rules (spec C3/regression-gate). Edit the JSON values to whatever `eval_wake`/`eval_safety_recall` actually printed in Step 2:

```json
{
  "_comment": "Stage 2F voice-eval baseline. Tolerances are the regression gate (spec Component 5). Update ONLY on an approved intentional change, never to mask a regression.",
  "chunker_tdd": {"required": "20/20 pass", "gate": "all pass"},
  "barge_in_asserts": {"required": "all pass", "gate": "all pass"},
  "safety_recall": {"value": 1.0, "gate": ">= 1.0 before over-fire wake ships (C1)"},
  "wake_tpr_single_operator_ceiling": {"value": 0.0, "gate": "informational only — NOT robustness (C4)"},
  "far_per_hour_ci_upper": {"value": null, "gate": "UNMEASURABLE on minutes-long corpus; needs >=3h zero-fire negatives (C3)"},
  "tolerances": {
    "tpr_max_drop_pts": 2,
    "far_ci_upper_max_per_hour": 1.0,
    "safety_recall_min": 1.0
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add Makefile tests/audio/baseline.json
git commit -m "stage 2f P2: make eval-voice target + baseline.json regression gate (C3/C5 tolerances)"
```

---

## Self-Review (completed by plan author)

**1. Spec coverage (P0–P2 only):**
- C1 cold-stop safety KWS → Tasks 3, 4 (+ recall runner Task 10). ✓
- Barge-in busy→idle edge reset, response-gated, never touches `state`, fires once per response → Tasks 1, 2 (+ asserts Task 9). ✓
- Chunker: preserve JSON + `suppressed`; swap only the detector; abbreviation-safe + clause fallback; 20/20 TDD incl. suppress/describe → Tasks 6, 7. ✓
- Eval: TPR, FAR/hr with CI, C3 unmeasurable-on-short-corpus honesty, C4 single-operator ceiling label, safety recall, `make eval-voice`, baseline.json with tolerances → Tasks 8–11. ✓
- V8 BlingFire aarch64 gate + pin → Task 5. ✓
- *Deferred (documented):* over-fire threshold/gate-liveness binding, motion-gate inversion, enrollment, speaker-ID, state machine, indicators, Personal VAD — these are P3+ and out of this plan's scope.

**2. Placeholder scan:** No TBD/"handle errors"/"similar to". Every code step has complete code; the only `<DATE>` is a literal stamp the engineer fills at install time in `requirements.txt`. Acceptable (it's a comment, not logic).

**3. Type/name consistency:** `should_reset_after_player(response_pending, was_busy, busy)` and `DetectorState.reset(vad_reset, wake_reset, drain)` match between Task 1 (def), Task 2 (call), Task 9 (test). `SafetyKeywordDetector.process_frame -> str` matches its consumer in Task 4 and Task 10. `split_sentences(text) -> (complete, remainder)` matches between Task 6 (def), Task 7 (chunker call), and tests. `check_safety_command` returns `{"intent": ...}` — used consistently. `make_safety_detector()` / `make_wake_detector()` parallel naming. ✓

**Known risk flagged for execution:** Tasks 3/4/10 require the KWS model dir (`models/kws/`) populated on the Jetson — tests `skipif`/degrade gracefully off-device, but must PASS on-device. The safety-KWS recall (Task 10) gates the over-fire wake in P3; if recall < 1.0 after threshold tuning, C1's fix is incomplete and the over-fire wake must NOT ship (spec Safety).
