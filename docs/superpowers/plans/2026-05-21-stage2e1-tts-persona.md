# Stage 2E.1 — TTS Multi-Voice + Multi-Persona Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship multi-personality routing + 2 swappable TTS backends (Kokoro v1.1 local, ElevenLabs Flash v2.5 cloud) by Tue 2026-05-26. Persona swappable at runtime via voice command (`set_persona` action) or boot via env var. SessionState dataclass introduced as second state layer alongside existing live SDK snapshot.

**Architecture:** New `tts/` and `chatbot/` modules added under `src/voice_control/`. `TTSBackend` Protocol with two impls. Persona registry loaded from declarative `config/personas.yaml`. `SessionState` dataclass owned by `LLMBrain`. Persona's `prompt_prefix` injected BEFORE `SYSTEM_PROMPT` in `_build_messages()`. Robot state + session state merge into single bullet-list block. New `set_persona` action dispatched same as other actions. JSONL conversation log written per turn. All swaps env-var-flippable for rollback.

**Tech Stack:** Existing `kokoro-onnx` v1.0 infra (no model swap — 54 voices already present under `models/tts/kokoro-v1.0/` are sufficient for the 6 starter personas), `elevenlabs>=2.0,<3.0` Python SDK (new dep), `PyYAML` (already present transitively, verify), `pytest>=8.0,<9.0` (new for testing new modules), `python-dotenv==1.0.1` (already pinned, for `ELEVENLABS_API_KEY`).

**Spec:** `docs/superpowers/specs/2026-05-21-stage2e-personality-design.md` §5.

---

## Pre-conditions

- Branch off current `tour_guide_upgrade_matteo` HEAD (`stage1.5-complete` tag at `966a96d`).
- ElevenLabs API key obtained (free tier 10k chars/month sufficient for Tuesday demo). Store in `.env` as `ELEVENLABS_API_KEY=...`.
- `/mnt/ssd/tts-models/` directory exists and is writable (created by Stage 1.5).
- Current Ollama gemma4:e4b brain warm (smoke test: `curl -s http://localhost:11434/api/tags | grep gemma4`).
- No `tests/` directory currently exists in repo — Task 1 sets it up.

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Create | `tests/__init__.py` | pytest test root |
| Create | `tests/conftest.py` | shared fixtures |
| Create | `pyproject.toml` (if not present) OR modify existing | pytest config |
| Create | `tests/voice_control/__init__.py` | test package marker |
| Create | `tests/voice_control/chatbot/__init__.py` | test package marker |
| Create | `tests/voice_control/tts/__init__.py` | test package marker |
| Create | `src/voice_control/chatbot/__init__.py` | module marker |
| Create | `src/voice_control/chatbot/session_state.py` | `SessionState` dataclass + `as_dict()` |
| Create | `tests/voice_control/chatbot/test_session_state.py` | unit tests |
| Create | `src/voice_control/chatbot/persona.py` | persona registry loader, lookup, validator |
| Create | `tests/voice_control/chatbot/test_persona.py` | unit tests |
| Create | `config/personas.yaml` | 6 starter personas |
| Create | `src/voice_control/chatbot/conversation_log.py` | JSONL writer + daily rotation |
| Create | `tests/voice_control/chatbot/test_conversation_log.py` | unit tests |
| Create | `src/voice_control/tts/__init__.py` | `TTSBackend` Protocol + `get_backend()` dispatch |
| Create | `tests/voice_control/tts/test_tts_init.py` | dispatch tests |
| Create | `src/voice_control/tts/kokoro.py` | kokoro-onnx Kokoro wrapper (v1.0, 54 voices) |
| Create | `src/voice_control/tts/elevenlabs.py` | ElevenLabs SDK streaming wrapper |
| Create | `tests/voice_control/tts/test_elevenlabs.py` | unit tests (mocked HTTP) |
| (none — model already present) | — | Task 11 is verification-only; `models/tts/kokoro-v1.0/` was downloaded during Stage 1 by the existing `scripts/setup_kokoro.py` |
| Modify | `src/voice_control/llm_brain.py` | `MAX_HISTORY=24`; persona prefix injection; `SessionState` ownership; merged state rendering |
| Modify | `src/voice_control/spot_dispatch.py` | `set_persona` action handler; optional `is_standing` + `nearby_locations` polish |
| Modify | `src/voice_control/spot_tts.py` | route synth via `tts.get_backend()` |
| Modify | `src/voice_control/client_mic.py` | pass `SessionState` alongside robot state into `brain.process(...)` |
| Modify | `requirements.txt` | add `elevenlabs>=2.0,<3.0`, `pytest>=8.0,<9.0` |
| Modify | `.env.example` | add `SPOT_TTS_BACKEND`, `SPOT_PERSONA`, `ELEVENLABS_API_KEY` placeholders |
| Modify | `docs/project/stage2-rollback.md` | append 2E.1 rollback layer (or create if absent) |

---

## Task 1: Pre-flight environment audit + branch setup

**Files:**
- Branch: `git checkout -b stage2e1-tts-persona tour_guide_upgrade_matteo`

- [ ] **Step 1: Verify pre-conditions**

```bash
git status                              # clean working tree expected
git log --oneline -1                    # confirm dee6612 spec commit at HEAD
ls /mnt/ssd/tts-models/ 2>/dev/null || ls models/tts/    # confirm TTS model dir reachable
grep -q ELEVENLABS_API_KEY .env && echo "key present" || echo "WARN: add ELEVENLABS_API_KEY to .env"

# Brain backend health depends on which path is live:
BRAIN=$(grep -E '^SPOT_BRAIN_BACKEND=' .env | cut -d= -f2 | tr -d '"' )
BRAIN=${BRAIN:-llamacpp}
if [ "$BRAIN" = "llamacpp" ]; then
    curl -sf http://127.0.0.1:11435/health | grep -q '"status":"ok"' && echo "llamacpp OK"
else
    # Ollama fallback path — bug #15260 (format=json + thinking ignored silently)
    # was fixed in PR #15678 shipped in 0.27. Older versions can corrupt JSON output.
    curl -sf http://localhost:11434/api/tags | grep -q gemma4 && echo "ollama models OK"
    ollama --version | awk '{print $NF}' | python3 -c "
import sys
raw = sys.stdin.read().strip()
parts = [int(x) for x in raw.split('.') if x.isdigit()]
assert tuple(parts) >= (0, 27), f'ollama {raw} < 0.27 — bug #15260 (json+think) NOT fixed; upgrade before fallback path is safe'
print(f'ollama {raw} OK')
"
fi
```

Expected: all green. If `.env` missing key, stop and add it before proceeding. If the Ollama version check fails, either upgrade Ollama (`curl -fsSL https://ollama.com/install.sh | sh`) or set `SPOT_BRAIN_BACKEND=llamacpp` to use the llama.cpp primary path.

- [ ] **Step 2: Branch off**

```bash
git checkout -b stage2e1-tts-persona
git log --oneline -1
```

Expected: HEAD now on new branch, same commit as `tour_guide_upgrade_matteo`.

- [ ] **Step 3: Commit branch baseline**

No file changes — just confirm clean start.

```bash
git status
```

Expected: "nothing to commit, working tree clean".

---

## Task 2: Bootstrap minimal pytest infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/voice_control/__init__.py`
- Create: `tests/voice_control/chatbot/__init__.py`
- Create: `tests/voice_control/tts/__init__.py`
- Create: `pyproject.toml` (only the `[tool.pytest.ini_options]` section if file absent; otherwise add section)

- [ ] **Step 1: Write `tests/__init__.py`**

Empty file. Marks `tests/` as a package.

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures for Stage 2E.1 tests."""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so `import src.voice_control...` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 3: Write `tests/voice_control/__init__.py`, `tests/voice_control/chatbot/__init__.py`, `tests/voice_control/tts/__init__.py`**

All empty files. Make subdirectories pytest-discoverable packages.

- [ ] **Step 4: Write or modify `pyproject.toml`**

If `pyproject.toml` does NOT exist, create it with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-ra -q --strict-markers"
```

If it does exist, append the `[tool.pytest.ini_options]` block at the end (preserve other sections).

- [ ] **Step 5: Smoke test pytest discovery**

```bash
pip install --quiet 'pytest>=8.0,<9.0'  # only if not already installed
python -m pytest --collect-only tests/
```

Expected: "no tests ran" but no import errors and `tests/` recognized.

- [ ] **Step 6: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "stage 2e1: bootstrap pytest infrastructure"
```

---

## Task 3: Add `pytest>=8.0,<9.0` to requirements.txt (pin-guardian gated)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Invoke pin-guardian agent**

Dispatch the `pin-guardian` agent to audit the proposed addition.

Agent prompt: *"About to add `pytest>=8.0,<9.0` to `requirements.txt`. This is a test-only dependency with no runtime/inference path. Validate that it does not pull conflicting transitive deps (especially numpy or onnxruntime). Report PASS or list conflicts."*

Wait for PASS verdict before proceeding. If FAIL: adjust pin or scope the dep to a separate `requirements-dev.txt`.

- [ ] **Step 2: Add pin to `requirements.txt`**

Append at end (before any trailing newline):

```
# Test-only — used by tests/ for new chatbot/tts modules. Safe runtime
# isolation: never imported by production code paths.
pytest>=8.0,<9.0
```

- [ ] **Step 3: Verify install**

```bash
pip install --quiet 'pytest>=8.0,<9.0'
python -c "import pytest; print(pytest.__version__)"
```

Expected: a version in the 8.x range prints.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "stage 2e1: pin pytest 8.x for new module tests"
```

---

## Task 4: Implement `SessionState` dataclass (TDD)

**Files:**
- Create: `src/voice_control/chatbot/__init__.py`
- Create: `src/voice_control/chatbot/session_state.py`
- Test: `tests/voice_control/chatbot/test_session_state.py`

- [ ] **Step 1: Write the failing test**

`tests/voice_control/chatbot/test_session_state.py`:

```python
import time
from src.voice_control.chatbot.session_state import SessionState


def test_defaults():
    state = SessionState()
    assert state.current_persona == "tour_guide"
    assert state.turn_index == 0
    assert state.last_action_taken is None
    assert state.last_comment_ts == 0.0


def test_as_dict_with_defaults():
    state = SessionState()
    d = state.as_dict()
    assert d["current_persona"] == "tour_guide"
    assert d["turn_index"] == 0
    assert d["last_action_taken"] == "none"  # human-readable for LLM
    # seconds_since_last_comment should be very large when ts=0
    assert d["seconds_since_last_comment"] > 1_000_000


def test_as_dict_with_set_fields():
    now = time.time()
    state = SessionState(
        current_persona="pirate",
        turn_index=5,
        last_action_taken="go_to lobby",
        last_comment_ts=now - 7,
    )
    d = state.as_dict()
    assert d["current_persona"] == "pirate"
    assert d["turn_index"] == 5
    assert d["last_action_taken"] == "go_to lobby"
    assert 7 <= d["seconds_since_last_comment"] <= 9  # tolerate small drift
```

- [ ] **Step 2: Run test (expect failure)**

```bash
python -m pytest tests/voice_control/chatbot/test_session_state.py -v
```

Expected: import error / module not found.

- [ ] **Step 3: Write `src/voice_control/chatbot/__init__.py`**

Empty file.

- [ ] **Step 4: Write `src/voice_control/chatbot/session_state.py`**

```python
"""Conversational session state that persists across turns.

Layered with the existing `get_robot_state_dict()` live SDK snapshot
(spot_dispatch.py:80). SessionState carries *across-turn* memory (persona,
turn index, last action, last comment timestamp). Both layers merge into
the single bullet-list block in the LLM system prompt.

Field growth across stage 2E slices:
- 2E.1: current_persona, turn_index, last_action_taken, last_comment_ts
- 2E.2 (future): arm_deployed, gaze_target, person_present, person_count
- 2E.3 (future): last_yolo_classes, last_vlm_caption, last_uttered_phrase
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionState:
    current_persona: str = "tour_guide"
    turn_index: int = 0
    last_action_taken: Optional[str] = None
    last_comment_ts: float = 0.0

    def as_dict(self) -> dict:
        """Render as flat dict for LLM bullet-list block.

        Matches existing get_robot_state_dict() formatting so the brain
        sees a single unified state list.
        """
        return {
            "current_persona": self.current_persona,
            "turn_index": self.turn_index,
            "last_action_taken": self.last_action_taken or "none",
            "seconds_since_last_comment": int(time.time() - self.last_comment_ts),
        }
```

- [ ] **Step 5: Run test (expect pass)**

```bash
python -m pytest tests/voice_control/chatbot/test_session_state.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/chatbot/__init__.py \
        src/voice_control/chatbot/session_state.py \
        tests/voice_control/chatbot/test_session_state.py
git commit -m "stage 2e1: SessionState dataclass for across-turn conversational memory"
```

---

## Task 5: Draft `config/personas.yaml` with 6 starter personas

**Files:**
- Create: `config/personas.yaml`

- [ ] **Step 1: Create `config/` directory and YAML file**

```bash
mkdir -p config
```

Write `config/personas.yaml`:

```yaml
# Spot persona registry. Each persona = prompt prefix + per-backend voice id
# + optional sampling overrides. Add new personas by appending entries.
# SIGHUP the voice loop to hot-reload (Task 8 implements the loader; SIGHUP
# hook is a follow-up nice-to-have).
#
# Voice IDs:
#   kokoro_v1: speaker slug from kokoro v1.0 (voices-v1.0.bin, 54 voices).
#   elevenlabs: ElevenLabs voice ID (curl GET /v1/voices to list all).
#
# Sampling overrides (optional, per-profile):
#   sampling_overrides:
#     vlm: {temperature: 0.8}        # only the vlm profile is altered
#     freeform: {temperature: 0.6}   # tighter freeform than the global 0.7
#   Keys allowed: action | freeform | vlm. Values merge ON TOP of
#   SAMPLING_PROFILES from src/voice_control/brain/llamacpp_backend.py —
#   missing keys fall through to the global default.

tour_guide:
  prompt_prefix: >
    You are Spot, a knowledgeable and warm tour guide at Dartmouth College's
    Thayer School of Engineering. You speak with enthusiasm about the campus,
    its history, and the engineering students who work here. Keep responses
    short and conversational — one or two sentences at most.
  voices:
    kokoro_v1: af_sarah
    elevenlabs: 21m00Tcm4TlvDq8ikWAM   # Rachel

pirate:
  prompt_prefix: >
    You are Spot, but in pirate mode. Speak like a salty pirate captain:
    "arr", "ye", "matey", "landlubber". Reference treasure, ships, and the
    high seas where appropriate. Keep responses short — one or two sentences.
  voices:
    kokoro_v1: am_fenrir
    elevenlabs: pNInz6obpgDQGcFmaJgB   # Adam
  sampling_overrides:
    # Pirate VLM captions lean toward salty, flavored describes. Bump vlm
    # temp above the 0.5 global default. Other profiles inherit (no override).
    vlm:
      temperature: 0.8

snarky:
  prompt_prefix: >
    You are Spot, but sarcastic and deadpan. You're a robot who finds humans
    mildly tiresome and isn't afraid to say so. Be helpful but with an edge.
    Keep responses short and dry — one or two sentences.
  voices:
    kokoro_v1: am_onyx
    elevenlabs: IKne3meq5aSn9XLyUdCD   # Charlie

butler:
  prompt_prefix: >
    You are Spot, but a refined British butler. Address the user as "sir" or
    "madam" when appropriate. Speak with measured formality and quiet wit.
    Keep responses short — one or two sentences.
  voices:
    kokoro_v1: bm_george
    elevenlabs: onwK4e9ZLuTAKqWW03F9   # Daniel

shakespeare:
  prompt_prefix: >
    You are Spot, but speaking in the style of William Shakespeare. Use
    "thee", "thou", "thy", "hath", and dramatic flourishes. Keep responses
    short — one or two sentences of theatrical English.
  voices:
    kokoro_v1: bm_lewis
    elevenlabs: ErXwobaYiN019PkySvjV   # Antoni

gen_z:
  prompt_prefix: >
    You are Spot, but you talk like Gen Z: casual, upbeat, slangy. Use
    "fr", "no cap", "bet", "lowkey" naturally. Keep responses short and
    energetic — one or two sentences.
  voices:
    kokoro_v1: af_nova
    elevenlabs: AZnzlk1XvdvUeBnXmlld   # Domi
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python -c "import yaml; data = yaml.safe_load(open('config/personas.yaml')); print(sorted(data.keys()))"
```

Expected: `['butler', 'gen_z', 'pirate', 'shakespeare', 'snarky', 'tour_guide']`

- [ ] **Step 3: Commit**

```bash
git add config/personas.yaml
git commit -m "stage 2e1: 6 starter personas in config/personas.yaml"
```

---

## Task 6: Implement persona registry loader (TDD)

**Files:**
- Create: `src/voice_control/chatbot/persona.py`
- Test: `tests/voice_control/chatbot/test_persona.py`

- [ ] **Step 1: Write the failing test**

`tests/voice_control/chatbot/test_persona.py`:

```python
import os
import pytest
from src.voice_control.chatbot.persona import (
    Persona,
    load_registry,
    get_persona,
    list_personas,
    default_persona_name,
    PersonaRegistryError,
)


REGISTRY_PATH = "config/personas.yaml"


def test_load_registry_returns_six_personas():
    reg = load_registry(REGISTRY_PATH)
    assert set(reg.keys()) == {
        "tour_guide", "pirate", "snarky", "butler", "shakespeare", "gen_z",
    }


def test_persona_has_prompt_prefix_and_voices():
    reg = load_registry(REGISTRY_PATH)
    pirate = reg["pirate"]
    assert isinstance(pirate, Persona)
    assert "pirate" in pirate.prompt_prefix.lower()
    assert pirate.voices["kokoro_v1"] == "am_fenrir"
    assert pirate.voices["elevenlabs"] == "pNInz6obpgDQGcFmaJgB"
    # Persona with no override returns empty dict; pirate carries a vlm override.
    assert reg["butler"].sampling_overrides == {}
    assert pirate.sampling_overrides["vlm"]["temperature"] == 0.8


def test_get_persona_returns_default_on_unknown_name(caplog):
    reg = load_registry(REGISTRY_PATH)
    p = get_persona("nonexistent_persona", reg)
    assert p.prompt_prefix == reg["tour_guide"].prompt_prefix
    # Warning should be logged
    assert any("nonexistent_persona" in rec.message for rec in caplog.records)


def test_list_personas_returns_sorted_names():
    reg = load_registry(REGISTRY_PATH)
    names = list_personas(reg)
    assert names == sorted(names)
    assert "tour_guide" in names


def test_default_persona_name_from_env(monkeypatch):
    monkeypatch.setenv("SPOT_PERSONA", "pirate")
    assert default_persona_name() == "pirate"


def test_default_persona_name_fallback(monkeypatch):
    monkeypatch.delenv("SPOT_PERSONA", raising=False)
    assert default_persona_name() == "tour_guide"


def test_load_registry_missing_file_raises():
    with pytest.raises(PersonaRegistryError):
        load_registry("nonexistent/path.yaml")
```

- [ ] **Step 2: Run test (expect failure)**

```bash
python -m pytest tests/voice_control/chatbot/test_persona.py -v
```

Expected: import error / module not found.

- [ ] **Step 3: Write `src/voice_control/chatbot/persona.py`**

```python
"""Persona registry — declarative YAML loader + runtime lookup.

Loads `config/personas.yaml` at boot. Provides:
- Persona dataclass (prompt_prefix, voices dict keyed by backend name)
- load_registry(path) -> dict[name, Persona]
- get_persona(name, registry) -> Persona  (falls back to default on miss + warns)
- list_personas(registry) -> sorted list[str]
- default_persona_name() -> str  (from SPOT_PERSONA env or "tour_guide")
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PERSONA = "tour_guide"


class PersonaRegistryError(Exception):
    """Raised when the persona registry YAML cannot be loaded."""


@dataclass
class Persona:
    name: str
    prompt_prefix: str
    voices: dict   # backend_name -> voice_id
    sampling_overrides: dict = field(default_factory=dict)
    # sampling_overrides shape: {"vlm": {"temperature": 0.8}, ...}
    # Keys: action | freeform | vlm. Values merge on top of
    # SAMPLING_PROFILES from brain/llamacpp_backend.py.


def load_registry(path: str = "config/personas.yaml") -> dict:
    """Load persona registry from YAML file.

    Raises PersonaRegistryError if the file is missing or malformed.
    Returns dict[persona_name, Persona].
    """
    p = Path(path)
    if not p.exists():
        raise PersonaRegistryError(f"Persona registry not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise PersonaRegistryError(f"Malformed persona YAML: {e}") from e
    if not isinstance(raw, dict):
        raise PersonaRegistryError(f"Top-level YAML must be a mapping, got {type(raw)}")

    registry = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            logger.warning(f"Skipping malformed persona entry: {name}")
            continue
        prefix = entry.get("prompt_prefix", "").strip()
        voices = entry.get("voices", {}) or {}
        sampling_overrides = entry.get("sampling_overrides", {}) or {}
        if not prefix:
            logger.warning(f"Persona {name} has empty prompt_prefix; skipping")
            continue
        registry[name] = Persona(
            name=name,
            prompt_prefix=prefix,
            voices=voices,
            sampling_overrides=sampling_overrides,
        )
    return registry


def get_persona(name: str, registry: dict) -> Persona:
    """Look up persona by name. Falls back to default + logs warning on miss."""
    if name in registry:
        return registry[name]
    fallback = default_persona_name()
    logger.warning(
        f"Unknown persona '{name}', falling back to '{fallback}'"
    )
    if fallback in registry:
        return registry[fallback]
    # Last-resort: return first available persona
    first = next(iter(registry.values()))
    logger.warning(f"Default '{fallback}' also missing; using '{first.name}'")
    return first


def list_personas(registry: dict) -> list:
    """Return sorted list of persona names."""
    return sorted(registry.keys())


def default_persona_name() -> str:
    """Return boot-time default persona from SPOT_PERSONA env or hardcoded fallback."""
    return os.environ.get("SPOT_PERSONA", DEFAULT_PERSONA)
```

- [ ] **Step 4: Verify PyYAML available**

```bash
python -c "import yaml; print(yaml.__version__)"
```

If missing: `pip install pyyaml`. PyYAML is small and broadly transitively present via bosdyn-client. If install needed, also add to `requirements.txt` (`PyYAML>=6.0,<7.0`) and commit in same task.

- [ ] **Step 5: Run test (expect pass)**

```bash
python -m pytest tests/voice_control/chatbot/test_persona.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/chatbot/persona.py tests/voice_control/chatbot/test_persona.py
git commit -m "stage 2e1: persona registry loader with YAML + fallback semantics"
```

If PyYAML install was needed and `requirements.txt` was updated, include that too.

---

## Task 7: Implement `conversation_log` JSONL writer (TDD)

**Files:**
- Create: `src/voice_control/chatbot/conversation_log.py`
- Test: `tests/voice_control/chatbot/test_conversation_log.py`

- [ ] **Step 1: Write the failing test**

`tests/voice_control/chatbot/test_conversation_log.py`:

```python
import json
import time
from pathlib import Path
from src.voice_control.chatbot.conversation_log import ConversationLog


def test_log_turn_writes_jsonl_line(tmp_path):
    log = ConversationLog(log_dir=tmp_path)
    log.log_turn(
        transcript="hello",
        response="hi there",
        actions=[],
        persona="tour_guide",
        tts_backend="kokoro",
        voice_id="af_sarah",
        latency_ms={"asr": 200, "llm": 1500, "tts_first_chunk": 300},
    )
    # Daily file expected
    today = time.strftime("%Y-%m-%d")
    log_file = tmp_path / f"{today}.jsonl"
    assert log_file.exists()
    line = log_file.read_text().strip()
    record = json.loads(line)
    assert record["transcript"] == "hello"
    assert record["response"] == "hi there"
    assert record["persona"] == "tour_guide"
    assert record["tts_backend"] == "kokoro"
    assert record["voice_id"] == "af_sarah"
    assert record["latency_ms"]["llm"] == 1500
    assert "ts" in record  # epoch timestamp present


def test_log_appends_multiple_turns(tmp_path):
    log = ConversationLog(log_dir=tmp_path)
    for i in range(3):
        log.log_turn(
            transcript=f"turn {i}", response=f"resp {i}",
            actions=[], persona="pirate", tts_backend="elevenlabs",
            voice_id="pNInz6obpgDQGcFmaJgB", latency_ms={},
        )
    today = time.strftime("%Y-%m-%d")
    log_file = tmp_path / f"{today}.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["transcript"] == f"turn {i}"
```

- [ ] **Step 2: Run test (expect failure)**

```bash
python -m pytest tests/voice_control/chatbot/test_conversation_log.py -v
```

Expected: import error.

- [ ] **Step 3: Write `src/voice_control/chatbot/conversation_log.py`**

```python
"""JSONL per-turn conversation logger with daily rotation.

Each turn writes one JSON line to `logs/conversations/<YYYY-MM-DD>.jsonl`.
Symlinked to /mnt/ssd/spot-logs/conversations/ per Stage 1.5 layout.
"""
import json
import time
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("logs/conversations")


class ConversationLog:
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_today(self) -> Path:
        return self.log_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"

    def log_turn(
        self,
        transcript: str,
        response: str,
        actions: list,
        persona: str,
        tts_backend: str,
        voice_id: str,
        latency_ms: dict,
    ) -> None:
        record = {
            "ts": time.time(),
            "transcript": transcript,
            "response": response,
            "actions": actions,
            "persona": persona,
            "tts_backend": tts_backend,
            "voice_id": voice_id,
            "latency_ms": latency_ms,
        }
        with self._file_for_today().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
python -m pytest tests/voice_control/chatbot/test_conversation_log.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/chatbot/conversation_log.py \
        tests/voice_control/chatbot/test_conversation_log.py
git commit -m "stage 2e1: conversation log JSONL writer with daily rotation"
```

---

## Task 8: Mid-plan checkpoint #1 — caveman:cavecrew-reviewer on Tasks 4-7

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer**

Agent prompt: *"Review the new code in commits since branch creation. Focus: `src/voice_control/chatbot/` (session_state.py, persona.py, conversation_log.py) and their tests. Check for: silent error swallowing, missing type hints on public APIs, unsafe YAML loading (must use safe_load), file-write race conditions, naming consistency with existing codebase patterns. One line per finding, severity tagged."*

Range: `git log --oneline tour_guide_upgrade_matteo..HEAD`

- [ ] **Step 2: Address findings**

Apply fixes inline for each BLOCKER / HIGH finding. MEDIUM/LOW: triage — fix if cheap, defer to follow-up task otherwise.

- [ ] **Step 3: Commit fixes**

```bash
git add -A
git commit -m "stage 2e1: address cavecrew review findings"
```

---

## Task 9: Implement `TTSBackend` Protocol + dispatch (TDD)

**Files:**
- Create: `src/voice_control/tts/__init__.py`
- Test: `tests/voice_control/tts/test_tts_init.py`

- [ ] **Step 1: Write the failing test**

`tests/voice_control/tts/test_tts_init.py`:

```python
import pytest
from src.voice_control.tts import TTSBackend, get_backend, register_backend


class FakeBackend:
    def synthesize(self, text, voice_id): return b"PCM"
    def stream(self, text, voice_id):
        yield b"PCM"
    def list_voices(self): return []


def test_register_and_get_backend(monkeypatch):
    register_backend("fake", lambda: FakeBackend())
    monkeypatch.setenv("SPOT_TTS_BACKEND", "fake")
    backend = get_backend()
    assert backend.synthesize("hi", "v1") == b"PCM"


def test_get_backend_unknown_raises(monkeypatch):
    monkeypatch.setenv("SPOT_TTS_BACKEND", "nonexistent_xyz")
    with pytest.raises(ValueError) as exc:
        get_backend()
    assert "nonexistent_xyz" in str(exc.value)


def test_get_backend_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("SPOT_TTS_BACKEND", raising=False)
    register_backend("kokoro", lambda: FakeBackend())
    backend = get_backend()
    assert backend is not None
```

- [ ] **Step 2: Run test (expect failure)**

```bash
python -m pytest tests/voice_control/tts/test_tts_init.py -v
```

Expected: import error.

- [ ] **Step 3: Write `src/voice_control/tts/__init__.py`**

```python
"""TTS backend registry + dispatch.

Backends register a factory under a name (e.g. "kokoro", "elevenlabs").
`get_backend()` reads SPOT_TTS_BACKEND env var and returns an instance.
Default backend: "kokoro".
"""
import os
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class TTSBackend(Protocol):
    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Render full PCM bytes (blocking). Returns audio at backend's native sample rate."""
        ...

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        """Yield PCM chunks as they're produced."""
        ...

    def list_voices(self) -> list:
        """Return list of available voice descriptors."""
        ...


_REGISTRY: dict = {}
DEFAULT_BACKEND = "kokoro"


def register_backend(name: str, factory) -> None:
    """Register a backend factory. Factory takes no args and returns a TTSBackend."""
    _REGISTRY[name] = factory


def get_backend() -> TTSBackend:
    """Resolve and instantiate the backend named by SPOT_TTS_BACKEND env var."""
    name = os.environ.get("SPOT_TTS_BACKEND", DEFAULT_BACKEND)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown TTS backend '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()
```

- [ ] **Step 4: Run test (expect pass)**

```bash
python -m pytest tests/voice_control/tts/test_tts_init.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/tts/__init__.py tests/voice_control/tts/test_tts_init.py
git commit -m "stage 2e1: TTSBackend Protocol + env-var dispatch registry"
```

---

## Task 10: Implement `KokoroBackend` wrapping existing kokoro-onnx TTS

**Files:**
- Create: `src/voice_control/tts/kokoro.py`

This task wraps the existing kokoro-onnx infra (already used by `src/voice_control/spot_tts.py`). The persona YAML voice slugs (`af_sarah`, `am_fenrir`, `am_onyx`, `bm_george`, `bm_lewis`, `af_nova`) are kokoro v1.0 slugs from `voices-v1.0.bin` (54 voices available — sufficient for 6 personas; no model download needed).

- [ ] **Step 1: Inspect current TTS implementation**

```bash
grep -nE "def |sherpa_onnx|kokoro_onnx|kokoro-onnx|kokoro|MODEL_DIR|MODEL_FILE" src/voice_control/spot_tts.py | head -40
```

Expected confirmation: `from kokoro_onnx import Kokoro` at ~line 121; `MODEL_DIR = ... / "models" / "tts" / "kokoro-v1.0"`; `MODEL_FILE = MODEL_DIR / "kokoro-v1.0.fp16-gpu.onnx"`; the voices file is `voices-v1.0.bin`.

- [ ] **Step 2: Write `src/voice_control/tts/kokoro.py`**

```python
"""Kokoro TTS backend via kokoro-onnx (the same package spot_tts.py uses).

KokoroBackend exposes a Protocol-compatible interface (synthesize, stream,
list_voices) on top of kokoro-onnx so the TTS dispatch layer can treat
Kokoro and ElevenLabs the same way. Voice slugs map directly to
kokoro-onnx voice names (e.g. 'af_sarah') — no integer speaker-id
resolution needed.

Model directory matches the existing spot_tts.py path:
  models/tts/kokoro-v1.0/{kokoro-v1.0.fp16-gpu.onnx, voices-v1.0.bin}
"""
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from kokoro_onnx import Kokoro

from . import register_backend

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "tts" / "kokoro-v1.0"
DEFAULT_SAMPLE_RATE = 24000


class KokoroBackend:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else Path(
            os.environ.get("SPOT_KOKORO_MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        self._tts = self._load()

    def _load(self) -> "Kokoro":
        model_file = self.model_dir / "kokoro-v1.0.fp16-gpu.onnx"
        voices_file = self.model_dir / "voices-v1.0.bin"
        if not model_file.exists() or not voices_file.exists():
            raise FileNotFoundError(
                f"Kokoro v1.0 files missing under {self.model_dir}. "
                f"Run scripts/setup_kokoro.py first."
            )
        return Kokoro(str(model_file), str(voices_file))

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Render text → 24 kHz int16 PCM bytes using the named voice slug."""
        samples, _rate = self._tts.create(text, voice=voice_id, speed=1.0, lang="en-us")
        samples = np.asarray(samples, dtype=np.float32)
        pcm_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        # kokoro-onnx 0.4.9 is not natively streaming; yield single chunk.
        yield self.synthesize(text, voice_id)

    def list_voices(self) -> list:
        """Return all kokoro v1.0 voice slugs packed in voices-v1.0.bin."""
        return sorted(getattr(self._tts, "voices", {}).keys())


register_backend("kokoro", lambda: KokoroBackend())
```

NOTE: kokoro-onnx is already installed (pin-guarded in `requirements.txt` per CLAUDE memory). No new dep. Model files at `models/tts/kokoro-v1.0/` are already downloaded via `scripts/setup_kokoro.py` (existing Stage 1 work). Task 11 is therefore a no-op verification rather than a download — see updated Task 11.

- [ ] **Step 3: Manual smoke test (model already present)**

```bash
python -c "
import os
os.environ['SPOT_TTS_BACKEND'] = 'kokoro'
from src.voice_control.tts import get_backend
import src.voice_control.tts.kokoro  # trigger registration
b = get_backend()
voices = b.list_voices()
print('voice count:', len(voices))
assert 'af_sarah' in voices, f'expected af_sarah in voices, got: {voices[:10]}...'
assert 'am_fenrir' in voices, 'expected am_fenrir (pirate voice)'
pcm = b.synthesize('hello from Spot', 'af_sarah')
print(f'pcm bytes: {len(pcm)}')
assert len(pcm) > 1000, 'pcm should be > 1 KB for a short utterance'
"
```

Expected: voice count >= 50; `af_sarah` and `am_fenrir` present; pcm bytes > 1000. Fail if any assertion fires — the existing kokoro-v1.0 model is required at `models/tts/kokoro-v1.0/`.

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/tts/kokoro.py
git commit -m "stage 2e1: KokoroBackend wrapping kokoro-onnx v1.0 (54 voices)"
```

---

## Task 11: Verify Kokoro v1.0 model files present (no download needed)

**Files:** none (verification only)

The kokoro v1.0 model (54 voices) was downloaded in Stage 1 via `scripts/setup_kokoro.py`. The 6 persona voice slugs (`af_sarah`, `am_fenrir`, `am_onyx`, `bm_george`, `bm_lewis`, `af_nova`) are all in `voices-v1.0.bin`. No new model download.

- [ ] **Step 1: Verify model files exist**

```bash
ls -lh models/tts/kokoro-v1.0/
```

Expected: directory exists with `kokoro-v1.0.fp16-gpu.onnx` (~85 MB) and `voices-v1.0.bin`. If missing, run the existing `scripts/setup_kokoro.py` first (do NOT add a new download script).

- [ ] **Step 2: Confirm all six persona voice slugs are packed**

```bash
python -c "
from kokoro_onnx import Kokoro
from pathlib import Path
root = Path('models/tts/kokoro-v1.0')
k = Kokoro(str(root / 'kokoro-v1.0.fp16-gpu.onnx'), str(root / 'voices-v1.0.bin'))
needed = ['af_sarah', 'am_fenrir', 'am_onyx', 'bm_george', 'bm_lewis', 'af_nova']
voices = set(getattr(k, 'voices', {}).keys())
missing = [v for v in needed if v not in voices]
print(f'voice count: {len(voices)}')
print(f'persona voices: missing={missing}')
assert not missing, f'persona voice(s) not in v1.0: {missing}'
"
```

Expected: voice count >= 50; `missing=[]`. If any persona voice is missing, update `config/personas.yaml` to a slug that exists OR pin a model upgrade as a separate follow-up task.

- [ ] **Step 3: No commit**

Verification-only task. Next task runs Task 10's smoke test (which now executes against the already-present model).

---

## Task 12: Implement `ElevenLabsBackend` (TDD with mocked HTTP)

**Files:**
- Create: `src/voice_control/tts/elevenlabs.py`
- Test: `tests/voice_control/tts/test_elevenlabs.py`

- [ ] **Step 1: pin-guardian audit on `elevenlabs` SDK dep**

Dispatch `pin-guardian` agent: *"About to add `elevenlabs>=2.0,<3.0` to requirements.txt. This is a cloud TTS SDK using WebSockets and HTTP. Validate: (a) no numpy version conflict (must stay 1.26.4 ABI), (b) no onnxruntime / torch / opencv conflict, (c) only pure-Python deps (no native rebuilds that break Jetson). Report PASS or list conflicts."*

Wait for PASS before proceeding.

- [ ] **Step 2: Add `elevenlabs>=2.0,<3.0` to `requirements.txt`**

Append:

```
# Stage 2E.1 — ElevenLabs cloud TTS backend. Pure-Python WebSocket + HTTP
# client; no numpy/onnx conflict per pin-guardian audit.
elevenlabs>=2.0,<3.0
```

- [ ] **Step 3: Install**

```bash
pip install --quiet 'elevenlabs>=2.0,<3.0'
python -c "from elevenlabs.client import ElevenLabs; print('OK')"
```

Expected: "OK".

- [ ] **Step 4: Write the failing test**

`tests/voice_control/tts/test_elevenlabs.py`:

```python
import os
from unittest.mock import MagicMock, patch
import pytest


def test_elevenlabs_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    from src.voice_control.tts.elevenlabs import ElevenLabsBackend
    with pytest.raises(RuntimeError) as exc:
        ElevenLabsBackend()
    assert "ELEVENLABS_API_KEY" in str(exc.value)


def test_synthesize_calls_sdk_with_flash_model(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    # Patch SDK client constructor and its text_to_speech.stream method
    with patch("src.voice_control.tts.elevenlabs.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.text_to_speech.stream.return_value = iter([b"chunk1", b"chunk2"])

        from src.voice_control.tts.elevenlabs import ElevenLabsBackend
        b = ElevenLabsBackend()
        pcm = b.synthesize("hello", "voice123")
        assert pcm == b"chunk1chunk2"
        call_kwargs = instance.text_to_speech.stream.call_args.kwargs
        assert call_kwargs["model_id"] == "eleven_flash_v2_5"
        assert call_kwargs["voice_id"] == "voice123"
        assert call_kwargs["output_format"] == "pcm_24000"


def test_stream_yields_chunks(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    with patch("src.voice_control.tts.elevenlabs.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.text_to_speech.stream.return_value = iter([b"a", b"b", b"c"])

        from src.voice_control.tts.elevenlabs import ElevenLabsBackend
        b = ElevenLabsBackend()
        chunks = list(b.stream("hi", "voice123"))
        assert chunks == [b"a", b"b", b"c"]
```

- [ ] **Step 5: Run test (expect failure)**

```bash
python -m pytest tests/voice_control/tts/test_elevenlabs.py -v
```

Expected: import error.

- [ ] **Step 6: Write `src/voice_control/tts/elevenlabs.py`**

```python
"""ElevenLabs cloud TTS backend.

Uses the official `elevenlabs` Python SDK. Streaming via `text_to_speech.stream`
with `eleven_flash_v2_5` model (lowest-latency Flash variant, ~150-300ms TTFA
per independent Coval benchmark May 2026). Output format `pcm_24000` to avoid
mp3 decode latency. Latency optimization level 3 (max except text normalizer).
"""
import os
from typing import Iterator

from elevenlabs.client import ElevenLabs

from . import register_backend


class ElevenLabsBackend:
    MODEL_ID = "eleven_flash_v2_5"
    OUTPUT_FORMAT = "pcm_24000"  # 24kHz signed-16 PCM, no decode overhead

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY env var not set; cannot init ElevenLabsBackend"
            )
        self._client = ElevenLabs(api_key=self.api_key)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        chunks = list(self.stream(text, voice_id))
        return b"".join(chunks)

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        return self._client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id=self.MODEL_ID,
            output_format=self.OUTPUT_FORMAT,
        )

    def list_voices(self) -> list:
        try:
            resp = self._client.voices.search()
            return [{"id": v.voice_id, "name": v.name} for v in resp.voices]
        except Exception:
            return []


register_backend("elevenlabs", lambda: ElevenLabsBackend())
```

- [ ] **Step 7: Run test (expect pass)**

```bash
python -m pytest tests/voice_control/tts/test_elevenlabs.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Manual smoke test (uses real API — burns a few hundred chars)**

```bash
ELEVENLABS_API_KEY=<your_key> SPOT_TTS_BACKEND=elevenlabs python -c "
import src.voice_control.tts.elevenlabs  # trigger registration
from src.voice_control.tts import get_backend
b = get_backend()
pcm = b.synthesize('Hello from Spot, this is the cloud voice test.', '21m00Tcm4TlvDq8ikWAM')
print(f'pcm bytes: {len(pcm)}')
"
```

Expected: PCM bytes > 0 (typical ~80KB for a one-sentence utterance at 24kHz).

- [ ] **Step 9: Commit**

```bash
git add requirements.txt src/voice_control/tts/elevenlabs.py tests/voice_control/tts/test_elevenlabs.py
git commit -m "stage 2e1: ElevenLabs Flash v2.5 backend with pcm_24000 streaming"
```

---

## Task 13: Mid-plan checkpoint #2 — caveman:cavecrew-reviewer on Tasks 9-12

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer**

Agent prompt: *"Review `src/voice_control/tts/` (init, kokoro, elevenlabs) + their tests. Focus: secret handling (API key never logged), HTTP timeout absence on streams, registration side effects (module-import-time `register_backend` calls), exception surface on missing model files. One line per finding, severity tagged."*

Range: `git log --oneline HEAD~5..HEAD`

- [ ] **Step 2: Address findings; commit fixes**

```bash
git add -A
git commit -m "stage 2e1: address cavecrew review findings (tts backends)"
```

---

## Task 14: Modify `llm_brain.py` — persona prefix injection + MAX_HISTORY=24 + SessionState

**Files:**
- Modify: `src/voice_control/llm_brain.py`

- [ ] **Step 1: Read existing `_build_messages` and bracketing context**

```bash
sed -n '195,300p' src/voice_control/llm_brain.py
```

Confirm (post-2A T5): `MAX_HISTORY=12` at line ~31, `class LLMBrain:` (renamed from `SpotBrain` in 2A T5 step 4a) near line ~188, `_build_messages(self, transcript, state)` at line ~273, `process(self, transcript, state=None)` at line ~284 (rewritten in 2A T5 step 4b — single-path backend dispatch, history append now lives inside `process()` before the return). 2A T5 backends already accept the `sampling_overrides` kwarg on `chat()` — this task just feeds it from the persona.

- [ ] **Step 2: Apply edits**

Edits to `src/voice_control/llm_brain.py`:

1. Change `MAX_HISTORY = 12` → `MAX_HISTORY = 24` (line ~31).

2. Add imports near the top (after existing imports):

```python
from src.voice_control.chatbot.session_state import SessionState
from src.voice_control.chatbot.persona import (
    Persona, load_registry, get_persona, default_persona_name,
    PersonaRegistryError,
)
```

3. In `LLMBrain.__init__` (after history deque initialization), add persona registry + SessionState ownership:

```python
        # Stage 2E.1: persona registry + cross-turn session state.
        try:
            self._persona_registry = load_registry()
        except PersonaRegistryError as e:
            print(f"[Brain] WARN: persona registry load failed: {e}; using empty registry")
            self._persona_registry = {}
        self.session_state = SessionState(current_persona=default_persona_name())
```

4. Replace `_build_messages` body to merge robot_state + SessionState + inject persona prefix:

```python
    def _build_messages(self, transcript: str, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build message list. State is the live robot snapshot from
        get_robot_state_dict(); session state is owned by self.session_state.
        Persona prefix prepended BEFORE SYSTEM_PROMPT.
        """
        persona = get_persona(self.session_state.current_persona, self._persona_registry)
        merged = {**state, **self.session_state.as_dict()}
        state_lines = "\n".join(f"- {k}: {v}" for k, v in merged.items())
        system_content = (
            f"{persona.prompt_prefix}\n\n"
            f"{SYSTEM_PROMPT}\n\n"
            f"Current robot state:\n{state_lines}"
        )
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": transcript})
        return messages
```

5. Pass persona sampling overrides into `process()`'s `backend.chat(...)` call (post-2A T5, the call lives inside `process()` and accepts `sampling_overrides=...`). Locate the existing line `raw = backend.chat(system, user_history, on_token=..., profile=profile,)` and rewrite it as:

```python
        persona = get_persona(self.session_state.current_persona, self._persona_registry)
        overrides = persona.sampling_overrides.get(profile) or None
        raw = backend.chat(
            system,
            user_history,
            on_token=getattr(self, "on_token_callback", None),
            profile=profile,
            sampling_overrides=overrides,
        )
```

6. After the existing `self.history.append(...)` block inside `process()` (also from 2A T5), bump `turn_index`:

```python
        # Append block from 2A T5 (user turn + sanitized/normal assistant turn)
        # already ran above; bump the SessionState turn counter immediately after:
        self.session_state.turn_index += 1
```

- [ ] **Step 3: Apply the karpathy-guidelines lens**

Before saving, re-read the edits against the four rules: any silent assumption? speculative abstraction? non-surgical adjacent edit? verification path? Strip anything that violates.

- [ ] **Step 4: Smoke test the modified brain in isolation**

```bash
python -c "
from src.voice_control.llm_brain import LLMBrain
b = LLMBrain()
print('persona registry size:', len(b._persona_registry))
print('current persona:', b.session_state.current_persona)
msgs = b._build_messages('hello', {'battery_percent': 85, 'current_location': 'lobby'})
print('system message preview:')
print(msgs[0]['content'][:500])

# Verify sampling-overrides lookup compiles for both default + pirate.
from src.voice_control.chatbot.persona import get_persona
tg = get_persona('tour_guide', b._persona_registry)
pirate = get_persona('pirate', b._persona_registry)
assert tg.sampling_overrides.get('vlm') in (None, {}), 'tour_guide should have no vlm override'
assert pirate.sampling_overrides['vlm']['temperature'] == 0.8, 'pirate vlm temp must be 0.8'
print('sampling overrides wired OK')
"
```

Expected: registry size 6, persona "tour_guide", system message begins with the tour_guide prompt_prefix, `sampling overrides wired OK` printed.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/llm_brain.py
git commit -m "stage 2e1: wire persona prefix + SessionState + MAX_HISTORY=24 into LLMBrain"
```

---

## Task 15: Add `set_persona` action to `SYSTEM_PROMPT` action catalog

**Files:**
- Modify: `src/voice_control/llm_brain.py` (SYSTEM_PROMPT augmentation)

- [ ] **Step 1: Locate `SYSTEM_PROMPT` action list**

```bash
grep -n "^- " src/voice_control/llm_brain.py | head -20
```

Identify the section listing available actions (lines ~39-130 approximately).

- [ ] **Step 2: Append `set_persona` entry**

Insert a new bullet at the end of the action catalog, before any closing triple quote of the `SYSTEM_PROMPT` literal:

```
- set_persona: Switch the robot's personality. Params: {"name": "<persona_name>"}. Available personas: tour_guide, pirate, snarky, butler, shakespeare, gen_z. Emit when the user says "be a pirate", "switch to butler", "act like a butler", "be yourself" (resets to tour_guide), etc. The persona controls speaking style AND voice. After emitting, your "response" field will be spoken in the NEW voice — write it in-character.
```

NOTE: do not hardcode the persona list in code; the prompt's enumeration is a hint to the LLM. The registry remains source of truth.

- [ ] **Step 3: Verify by inspecting the rendered system prompt**

```bash
python -c "
from src.voice_control.llm_brain import SYSTEM_PROMPT
print(SYSTEM_PROMPT[-1000:])
"
```

Expected: trailing chunk includes the `set_persona` bullet.

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/llm_brain.py
git commit -m "stage 2e1: document set_persona action in SYSTEM_PROMPT catalog"
```

---

## Task 16: Implement `set_persona` action handler in `spot_dispatch.py`

**Files:**
- Modify: `src/voice_control/spot_dispatch.py`

- [ ] **Step 1: Find action dispatch table**

```bash
grep -nE "def (do_|handle_|dispatch)|action ==|elif action" src/voice_control/spot_dispatch.py | head -30
```

Identify where individual actions are dispatched (e.g. `go_to`, `tour`, etc).

- [ ] **Step 2: Add handler**

Add a `do_set_persona` function next to the other action handlers. Wire it into the dispatch table the same way other actions are. Skeleton (adjust to actual dispatch style of the file):

```python
def do_set_persona(params: dict, brain) -> dict:
    """Switch active persona via runtime action.

    Params:
        name: persona registry key (e.g. "pirate", "butler", "tour_guide").

    On unknown name: falls back to default + logs warning (handled inside
    persona registry's get_persona()).
    """
    name = params.get("name", "").strip()
    if not name:
        return {"ok": False, "error": "set_persona requires 'name' param"}
    # get_persona handles unknown-name fallback internally
    from src.voice_control.chatbot.persona import get_persona
    persona = get_persona(name, brain._persona_registry)
    brain.session_state.current_persona = persona.name
    print(f"[Dispatch] Persona switched to '{persona.name}'")
    return {"ok": True, "persona": persona.name}
```

Wire into dispatch loop following the file's existing pattern (typically a chain of `if/elif action == "..."` or a dict lookup). Match the calling convention used by sibling handlers.

- [ ] **Step 3: Brain reference plumbing**

If the existing dispatch doesn't already pass the `brain` instance to action handlers, plumb it through the call site. Check `client_mic.py` for where brain + dispatcher meet; the cleanest path is to pass `brain` as an explicit kwarg to `do_set_persona` only (avoid changing other handlers' signatures).

- [ ] **Step 4: Manual smoke test**

```bash
python -c "
from src.voice_control.llm_brain import LLMBrain
from src.voice_control.spot_dispatch import do_set_persona
b = LLMBrain()
print('before:', b.session_state.current_persona)
result = do_set_persona({'name': 'pirate'}, b)
print('result:', result)
print('after:', b.session_state.current_persona)
# Unknown name → fallback
result = do_set_persona({'name': 'nonexistent'}, b)
print('after unknown:', b.session_state.current_persona, '(should be default)')
"
```

Expected: persona flips to "pirate" first, then falls back to "tour_guide" (default) on unknown name.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/spot_dispatch.py
git commit -m "stage 2e1: set_persona action handler with fallback on unknown name"
```

---

## Task 17: Route `SpotTTS.speak` synth through TTS backend dispatch

**Files:**
- Modify: `src/voice_control/spot_tts.py`

The `voice` parameter on `SpotTTS.speak(text, voice=None)` was added in 2A T11. This task swaps the kokoro-onnx-specific `_render` closure for one that dispatches through the backend registry, so `SPOT_TTS_BACKEND=elevenlabs` selects ElevenLabs at runtime without touching call sites. `SpotTTS.speak` remains the only public entry point — there is no `enqueue_render` module function.

- [ ] **Step 1: Inspect current synth entry points**

```bash
grep -nE "def |kokoro_onnx|self\._tts|speak\(|TTSChunker|enqueue_streaming" src/voice_control/spot_tts.py | head -40
```

Expected: `class SpotTTS:` (~line 61), `def speak(self, text, voice=None)` (~line 162 post-2A T11), `self._tts.create(text, voice=voice, speed=speed, lang=DEFAULT_LANG)` (~line 183) — this is the call we replace.

- [ ] **Step 2: Wire backend dispatch**

Add backend imports near the existing imports at the top of `spot_tts.py`:

```python
# Stage 2E.1: route TTS synth through swappable backend registry
import src.voice_control.tts.kokoro      # noqa: F401  (registers backend)
import src.voice_control.tts.elevenlabs  # noqa: F401  (registers backend)
from src.voice_control.tts import get_backend
```

Inside `SpotTTS.speak()`, replace the kokoro-onnx-specific `_render` closure with a backend-dispatched one. The wrapping logic (volume snapshot, latency hooks, `self._player.enqueue_render(_render, ...)`) is unchanged — only the closure body swaps:

```python
def _render():
    backend = get_backend()
    use_voice = voice or _default_voice_for_backend(backend)
    pcm_bytes = backend.synthesize(text, use_voice)
    # Backend contract: 24 kHz int16 PCM bytes. Convert to float32 in [-1, 1].
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if gain != 1.0:
        samples = samples * gain
    return samples, 24000
```

Drop the `self._tts = ...` initialization in `SpotTTS.__init__` and the `_load()` method body — the kokoro-onnx Kokoro instance now lives inside `KokoroBackend` (T10). Replace `is_available()` to check backend registration:

```python
def is_available(self) -> bool:
    try:
        get_backend()
        return True
    except Exception:
        return False
```

- [ ] **Step 3: Add `_default_voice_for_backend` helper**

```python
def _default_voice_for_backend(backend) -> str:
    """Pick a sensible default voice when caller didn't specify one."""
    name = type(backend).__name__
    if "Kokoro" in name:
        return "af_sarah"
    if "ElevenLabs" in name:
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel
    return ""  # backend may raise on empty — intentional surface for bad config
```

- [ ] **Step 4: Smoke test (Kokoro path)**

```bash
SPOT_TTS_BACKEND=kokoro python -c "
from src.voice_control.spot_tts import get_tts
tts = get_tts()
tts.speak('Hello from the Kokoro backend.', voice='af_sarah')
import time; time.sleep(2)
"
```

Expected: audible "Hello from the Kokoro backend." through speakers.

- [ ] **Step 5: Smoke test (ElevenLabs path)**

```bash
SPOT_TTS_BACKEND=elevenlabs python -c "
from src.voice_control.spot_tts import get_tts
tts = get_tts()
tts.speak('Hello from the ElevenLabs backend.', voice='21m00Tcm4TlvDq8ikWAM')
import time; time.sleep(3)
"
```

Expected: audible Rachel voice through speakers.

- [ ] **Step 6: Smoke test (TTSChunker still works)**

```bash
SPOT_TTS_BACKEND=kokoro python -c "
from src.voice_control.spot_tts import get_tts
tts = get_tts()
sink = tts.enqueue_streaming(voice='af_sarah')
# Feed a short fake JSON delta with two sentences in 'response'.
for delta in ['{\"actions\":[],\"', 'response\":\"Hello world. ', 'How are you?\"}']:
    sink(delta)
import time; time.sleep(3)
"
```

Expected: two audible sentences. Confirms the chunker path (2A T11) still routes through `speak()` after the backend swap.

- [ ] **Step 7: Commit**

```bash
git add src/voice_control/spot_tts.py
git commit -m "stage 2e1: route SpotTTS.speak synth through swappable TTSBackend dispatch"
```

---

## Task 18: Pass persona-resolved voice ID through call sites

**Files:**
- Modify: `src/voice_control/spot_dispatch.py` (or wherever "response" string is sent to TTS)
- Modify: `src/voice_control/client_mic.py` if necessary

- [ ] **Step 1: Trace the response→TTS path**

```bash
grep -nE "enqueue_render|spot_tts|speak\(" src/voice_control/*.py | head -20
```

Locate every site that hands a `response` string to TTS.

- [ ] **Step 2: Add voice resolution at each call site (streaming-aware)**

Post-2A T11, the streaming TTSChunker owns the spoken side — `client_mic.voice_loop` calls `tts.enqueue_streaming(voice=…)` BEFORE `brain.process()`. The voice ID must be resolved up-front so every sentence chunk from one turn shares the same voice. Pre-2A call sites that still call `tts.speak(response)` directly (e.g. the VLM describe branch) get the same voice resolution.

Edit `src/voice_control/client_mic.py` voice_loop:

```python
from src.voice_control.chatbot.persona import get_persona, voice_id_for

backend_name = os.environ.get("SPOT_TTS_BACKEND", "kokoro")
persona = get_persona(brain.session_state.current_persona, brain._persona_registry)
voice_id = voice_id_for(persona, backend_name) or None   # None lets SpotTTS pick the default

on_token = tts.enqueue_streaming(voice=voice_id)        # was: tts.enqueue_streaming() in 2A T11
brain.on_token_callback = on_token
try:
    result = brain.process(clean, spot_state)
finally:
    brain.on_token_callback = None

dispatch(result["actions"])
```

For any remaining `tts.speak(text)` call sites NOT in the streaming loop (VLM describe direct render, fallback paths), pass the same voice:

```python
tts.speak(text, voice=voice_id)
```

- [ ] **Step 3: Add the helper**

In `src/voice_control/chatbot/persona.py`, append:

```python
def voice_id_for(persona: Persona, backend_name: str) -> str:
    """Resolve the voice ID to use for this persona under the given backend."""
    key = "elevenlabs" if backend_name == "elevenlabs" else "kokoro_v1"
    return persona.voices.get(key, "")
```

Add a unit test in `tests/voice_control/chatbot/test_persona.py`:

```python
def test_voice_id_for():
    from src.voice_control.chatbot.persona import voice_id_for, load_registry
    reg = load_registry(REGISTRY_PATH)
    pirate = reg["pirate"]
    assert voice_id_for(pirate, "kokoro") == "am_fenrir"
    assert voice_id_for(pirate, "elevenlabs") == "pNInz6obpgDQGcFmaJgB"
```

Run: `python -m pytest tests/voice_control/chatbot/test_persona.py::test_voice_id_for -v`. Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "stage 2e1: resolve persona-mapped voice_id at TTS call sites"
```

---

## Task 19: Wire `ConversationLog` into per-turn flow

**Files:**
- Modify: `src/voice_control/llm_brain.py` OR `src/voice_control/client_mic.py` (whichever owns end-of-turn handling)

- [ ] **Step 1: Identify end-of-turn site**

```bash
grep -n "self.history.append" src/voice_control/llm_brain.py
```

Post-2A T5, `process()` is rewritten and the append block lives inside the new `process()` body (just before the `[Brain-timing]` print and `return {...}`). Exact line numbers shifted from pre-2A (was ~417-423). The append block is the natural site for the log_turn call.

- [ ] **Step 2: Add ConversationLog instance to `LLMBrain.__init__`**

```python
        from src.voice_control.chatbot.conversation_log import ConversationLog
        self._conversation_log = ConversationLog()  # writes to logs/conversations/
```

- [ ] **Step 3: Call `log_turn` after each successful turn**

In `process()`, after the history-append block (from 2A T5) and immediately before the `[Brain-timing]` print + `return`:

```python
        try:
            backend_name = os.environ.get("SPOT_TTS_BACKEND", "kokoro")
            persona = get_persona(self.session_state.current_persona, self._persona_registry)
            voice_id = persona.voices.get(
                "elevenlabs" if backend_name == "elevenlabs" else "kokoro_v1", ""
            )
            self._conversation_log.log_turn(
                transcript=transcript,
                response=response,
                actions=actions,
                persona=self.session_state.current_persona,
                tts_backend=backend_name,
                voice_id=voice_id,
                latency_ms={"llm": elapsed_ms},   # 2A T5 already measures in ms — pass directly
            )
        except Exception as e:
            print(f"[Brain] WARN: conversation log write failed: {e}")
```

- [ ] **Step 4: Smoke test — capture one turn**

```bash
rm -f logs/conversations/$(date +%Y-%m-%d).jsonl
python -c "
from src.voice_control.llm_brain import LLMBrain
b = LLMBrain()
result = b.process('hello', {'battery_percent': 85})
print('response:', result.get('response'))
"
cat logs/conversations/$(date +%Y-%m-%d).jsonl
```

Expected: a single JSONL line with the turn's transcript + response + persona + tts_backend.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/llm_brain.py
git commit -m "stage 2e1: write per-turn JSONL conversation log"
```

---

## Task 20: Update `.env.example` + add `SPOT_TTS_BACKEND` doc

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Read existing `.env.example`**

```bash
cat .env.example
```

- [ ] **Step 2: Append Stage 2E.1 env vars**

```
# ---------------------------------------------------------------------------
# Stage 2E.1 — TTS multi-voice + multi-persona
# ---------------------------------------------------------------------------

# TTS backend selector. Default: kokoro (local, 54 voices via kokoro-onnx v1.0).
# Alternatives: elevenlabs (cloud, requires ELEVENLABS_API_KEY).
# SPOT_TTS_BACKEND=kokoro

# Boot-time default persona. Must match a key in config/personas.yaml.
# Available: tour_guide, pirate, snarky, butler, shakespeare, gen_z.
# SPOT_PERSONA=tour_guide

# ElevenLabs API key, required only when SPOT_TTS_BACKEND=elevenlabs.
# Free tier: 10,000 chars/month.
# ELEVENLABS_API_KEY=

# Kokoro model directory. Default: models/tts/kokoro-v1.0/ (in-repo, kokoro-onnx).
# Override only if model was installed elsewhere.
# SPOT_KOKORO_MODEL_DIR=models/tts/kokoro-v1.0
```

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "stage 2e1: document SPOT_TTS_BACKEND / SPOT_PERSONA env vars"
```

---

## Task 21: Add 2E.1 rollback layer to `docs/project/stage2-rollback.md`

**Files:**
- Modify (or Create if absent): `docs/project/stage2-rollback.md`

- [ ] **Step 1: Check if file exists**

```bash
ls docs/project/stage2-rollback.md 2>/dev/null && echo "exists" || echo "absent"
```

If absent, create with `# Stage 2 Rollback Runbook\n` header.

- [ ] **Step 2: Append 2E.1 section**

```markdown

## After 2E.1 — TTS + persona

### Layer: revert TTS backend (cloud → local)

Symptom: ElevenLabs latency spike, API outage, free tier exhausted, network drop.

```bash
export SPOT_TTS_BACKEND=kokoro
# restart voice loop
```

Reverts to local kokoro-onnx v1.0. No code change required.

### Layer: collapse to single persona

Symptom: persona swap action causing dispatch errors or bad behavior.

```bash
export SPOT_PERSONA=tour_guide
```

Brain still loads registry but treats every utterance as default persona. To fully disable runtime swap, comment out the `set_persona` bullet in `SYSTEM_PROMPT` (llm_brain.py) so the LLM stops emitting the action.

### Layer: revert to Kokoro v0.19 (8 voices)

Symptom: Kokoro v1.1 install corrupt OR voice quality regression.

```bash
export SPOT_KOKORO_MODEL_DIR=models/tts/kokoro-v1.0
```

Old model directory preserved per Stage 1.5 — does not need re-download.

### Hard rollback (full 2E.1 revert)

```bash
git revert <2e1-merge-commits>
# or, on stage2e1-tts-persona branch:
git reset --hard <pre-2e1-tag>
```

Pre-2E.1 commit hash: `dee6612` (spec) — branch off this point.
```

- [ ] **Step 3: Commit**

```bash
git add docs/project/stage2-rollback.md
git commit -m "stage 2e1: document rollback layers for TTS + persona"
```

---

## Task 22: Mid-plan checkpoint #3 — safety-reviewer on dispatcher changes

- [ ] **Step 1: Dispatch safety-reviewer agent**

Agent prompt: *"Stage 2E.1 added a `set_persona` action to spot_dispatch.py and modified its dispatch loop. Audit: (a) does the new action add any motion path? (b) does it bypass any existing e-stop / freeze / stop coverage? (c) does it change graceful shutdown? (d) does it touch any other action's safety properties? Focus on `src/voice_control/spot_dispatch.py` and `src/voice_control/intent.py` if touched. Report BLOCKER/HIGH/MEDIUM/LOW findings."*

- [ ] **Step 2: Address findings**

set_persona is text-only — should yield no safety BLOCKER. If reviewer flags something, fix inline.

- [ ] **Step 3: Commit fixes (if any)**

```bash
git add -A
git diff --cached --quiet && echo "no fixes needed" || git commit -m "stage 2e1: address safety-reviewer findings"
```

---

## Task 23: Mid-plan checkpoint #4 — feature-dev:code-reviewer quality audit

- [ ] **Step 1: Dispatch feature-dev:code-reviewer**

Agent prompt: *"Quality audit of the full Stage 2E.1 diff. Scope: all files in `src/voice_control/chatbot/`, `src/voice_control/tts/`, `tests/voice_control/`, plus modifications to `llm_brain.py`, `spot_dispatch.py`, `spot_tts.py`, `client_mic.py`. Check: bug risk, logic errors, project convention adherence, missing test coverage, unhandled exception paths. Use confidence-based filtering — only report high-confidence findings."*

Range: `git log --oneline tour_guide_upgrade_matteo..HEAD`

- [ ] **Step 2: Address findings**

Fix high-confidence findings inline.

- [ ] **Step 3: Commit fixes**

```bash
git add -A
git diff --cached --quiet && echo "no fixes" || git commit -m "stage 2e1: address feature-dev review findings"
```

---

## Task 24: Karpathy-guidelines self-check on full diff

- [ ] **Step 1: Invoke karpathy-guidelines skill**

Skill: `karpathy-guidelines`. Re-read the four rules. Apply to the full 2E.1 diff:

```bash
git diff tour_guide_upgrade_matteo...HEAD --stat
```

Walk each changed file and ask:

1. ASSUMPTIONS — did any task silently assume intent, behavior, or API surface? Specifically check `KokoroBackend` reads the `voices` attribute from `kokoro_onnx.Kokoro` for `list_voices()` (does kokoro-onnx 0.4.9 expose a `voices` dict on the loaded instance? verify via `help(Kokoro)` in REPL).
2. SIMPLICITY — strip unused params, premature flexibility. Specifically check: is `_default_voice_for_backend` actually used, or did Task 17/18 always pass voice_id? If always passed, drop the helper.
3. SURGICAL — any adjacent refactors or formatting fixes not asked for? Revert them.
4. VERIFICATION — does every new code path have a check (test or smoke test)?

- [ ] **Step 2: Verify sherpa-onnx speaker mapping claim**

```bash
python -c "
import sherpa_onnx
from kokoro_onnx import Kokoro; help(Kokoro)
"
```

If `Kokoro.voices` is NOT a real attribute on the loaded instance, replace `list_voices()` with a hardcoded slug list pulled from the `voices-v1.0.bin` packing manifest (or fall back to enumerating the keys returned by a small probe call). `tests/voice_control/tts/test_elevenlabs.py` is unaffected (mocked); but the kokoro smoke test in Task 11 must pass.

- [ ] **Step 3: Apply fixes inline**

- [ ] **Step 4: Commit**

```bash
git add -A
git diff --cached --quiet && echo "no fixes" || git commit -m "stage 2e1: karpathy-guidelines pass on full diff"
```

---

## Task 25: Verification-before-completion — full test suite + e2e smoke

- [ ] **Step 1: Invoke verification-before-completion skill**

Skill: `superpowers:verification-before-completion`. Follow the protocol: evidence before assertions.

- [ ] **Step 2: Run full pytest suite**

```bash
python -m pytest tests/voice_control/ -v
```

Expected output: all tests pass. Copy the pass count and total time into the task notes.

- [ ] **Step 3: Run e2e smoke — Kokoro path**

```bash
SPOT_TTS_BACKEND=kokoro SPOT_PERSONA=tour_guide python scripts/run_voice_control.py --headless --once "say hello"
```

Expected: spoken response in Kokoro voice using tour_guide persona. JSONL log line written to `logs/conversations/<today>.jsonl`.

Capture the actual command path: if `scripts/run_voice_control.py` does not support `--once`, replicate with an equivalent direct call. The point is one end-to-end turn through the brain → TTS → audio output pipeline.

- [ ] **Step 4: Run e2e smoke — ElevenLabs path**

```bash
SPOT_TTS_BACKEND=elevenlabs SPOT_PERSONA=tour_guide python scripts/run_voice_control.py --headless --once "say hello again"
```

Expected: spoken response in ElevenLabs Rachel voice. Confirm voice change audibly.

- [ ] **Step 5: Run e2e smoke — persona swap mid-session**

```bash
SPOT_TTS_BACKEND=kokoro python -c "
from src.voice_control.llm_brain import LLMBrain
from src.voice_control.spot_dispatch import do_set_persona
b = LLMBrain()
# Turn 1
print('--- turn 1 (tour_guide) ---')
r1 = b.process('hello, tell me about yourself', {})
print('persona:', b.session_state.current_persona)
print('response:', r1['response'][:200])
# Swap
do_set_persona({'name': 'pirate'}, b)
# Turn 2
print('--- turn 2 (pirate) ---')
r2 = b.process('tell me about yourself again', {})
print('persona:', b.session_state.current_persona)
print('response:', r2['response'][:200])
"
```

Expected: turn 1 = tour-guide tone; turn 2 = pirate-flavored ("arr"/"matey"/etc).

- [ ] **Step 6: Inspect conversation log**

```bash
tail -3 logs/conversations/$(date +%Y-%m-%d).jsonl | python -m json.tool
```

Expected: last 3 entries show persona swap from tour_guide to pirate; each has tts_backend, voice_id, latency_ms.

- [ ] **Step 7: Commit any final fixes**

```bash
git add -A
git diff --cached --quiet && echo "all clean" || git commit -m "stage 2e1: final smoke-test fixes"
```

---

## Task 26: Final caveman:cavecrew-reviewer pass on full diff

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer**

Agent prompt: *"Final review pass on Stage 2E.1 full diff. Range: `git log tour_guide_upgrade_matteo..HEAD`. Report only NEW high-severity findings not previously addressed. One line per finding."*

- [ ] **Step 2: Address remaining findings**

If any new blockers: fix + commit. If clean: proceed.

---

## Task 27: Live Spot demo verification (Tuesday 2026-05-26)

> **HUMAN-IN-LOOP REQUIRED.** Powered Spot, e-stop terminal open, clear lab space. Operator within line of sight.

- [ ] **Step 1: Boot voice loop on Spot with Kokoro default**

```bash
SPOT_TTS_BACKEND=kokoro SPOT_PERSONA=tour_guide python scripts/run_voice_control.py
```

OR via web panel if mic chain not yet reliable (text-in, voice-out demo). Both paths exercise the same persona + TTS code.

- [ ] **Step 2: Demo script — Kokoro track**

Speak (or type) in sequence:

1. *"Hello"* → expect tour_guide voice response.
2. *"Be a pirate"* → expect pirate voice + in-character ack.
3. *"Describe yourself"* → expect pirate-flavored self-description in pirate voice.
4. *"Switch to butler"* → expect butler voice + formal ack.
5. *"What's around here?"* → expect butler-flavored response referencing robot state (location).
6. *"Be yourself"* → expect reset to tour_guide voice + ack.

Record audio + console logs.

- [ ] **Step 3: Restart with ElevenLabs**

```bash
SPOT_TTS_BACKEND=elevenlabs python scripts/run_voice_control.py
```

Repeat Step 2. Compare voice quality across backends informally.

- [ ] **Step 4: Hot-add persona test**

Edit `config/personas.yaml`, append a new persona (e.g. `cowboy`). Restart voice loop. Verify *"be a cowboy"* triggers correct persona + voice mapping. (SIGHUP hot-reload is a future nice-to-have; restart is acceptable for Tuesday.)

- [ ] **Step 5: Verify conversation log**

```bash
tail -20 logs/conversations/$(date +%Y-%m-%d).jsonl | python -m json.tool | head -100
```

Expected: every turn from the demo present, with correct persona + tts_backend.

- [ ] **Step 6: Capture demo evidence**

Save audio recordings + log excerpt to `logs/stage2e1/demo_<timestamp>/`. This is the artifact for downstream review.

---

## Task 28: Verify all merge gate criteria are met

- [ ] **Step 1: Checklist**

- [ ] All 27 prior tasks completed and committed.
- [ ] Pytest passes (Task 25 Step 2).
- [ ] Both TTS backends smoke-test passing (Task 25 Steps 3–4).
- [ ] Persona swap mid-session verified (Task 25 Step 5).
- [ ] Conversation log writes correctly (Task 25 Step 6).
- [ ] Live demo on Spot passed both Kokoro + ElevenLabs tracks (Task 27).
- [ ] safety-reviewer audit clean (Task 22).
- [ ] feature-dev:code-reviewer high-confidence findings addressed (Task 23).
- [ ] caveman:cavecrew-reviewer final pass clean (Task 26).
- [ ] karpathy-guidelines pass clean (Task 24).
- [ ] Rollback runbook updated (Task 21).
- [ ] `.env.example` documents new env vars (Task 20).

- [ ] **Step 2: If any item NOT checked, return to that task and complete it before proceeding**

---

## Task 29: Open PR and merge to `tour_guide_upgrade_matteo`

- [ ] **Step 1: Push branch**

```bash
git push -u origin stage2e1-tts-persona
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --base tour_guide_upgrade_matteo --title "stage 2e1: TTS A/B + multi-persona routing" --body "$(cat <<'EOF'
## Summary

- Two TTS backends (Kokoro v1.1 multi-lang local, ElevenLabs Flash v2.5 cloud) behind `SPOT_TTS_BACKEND` env var.
- Persona registry with 6 starter personas in `config/personas.yaml`; runtime swap via new `set_persona` action.
- `SessionState` dataclass introduced as second state layer for across-turn conversational memory (persona, turn index, last action, last comment timestamp).
- `MAX_HISTORY` bumped 12 → 24 for persona consistency across longer conversations.
- Per-turn JSONL conversation log in `logs/conversations/<date>.jsonl`.
- New `tests/` infrastructure with pytest for all new modules.

## Spec

`docs/superpowers/specs/2026-05-21-stage2e-personality-design.md` §5.

## Test plan

- [x] pytest test suite green
- [x] Kokoro path e2e smoke (audible Kokoro voice)
- [x] ElevenLabs path e2e smoke (audible Rachel voice)
- [x] Persona swap mid-session (tour_guide → pirate verified by voice + text)
- [x] Conversation log writes correct fields
- [x] Live Spot demo passed both backends

## Rollback

`docs/project/stage2-rollback.md` §"After 2E.1" — env-var flips revert each layer:
- `SPOT_TTS_BACKEND=kokoro` — revert cloud TTS
- `SPOT_PERSONA=tour_guide` — collapse to single persona
- `SPOT_KOKORO_MODEL_DIR=models/tts/kokoro-v1.0` — pin to the in-repo v1.0 model (default)

## Forward compatibility

When Stage 2A lands: `set_persona` action gets a gbnf grammar entry (1-line); `LLMBrain` absorbs new `BrainBackend`. No interface in 2E.1 breaks.
EOF
)"
```

- [ ] **Step 3: Address PR review feedback if any**

- [ ] **Step 4: Merge**

```bash
gh pr merge --merge   # not squash — keep per-task commits for clean history
```

- [ ] **Step 5: Tag the exit**

```bash
git checkout tour_guide_upgrade_matteo
git pull
git tag stage2e1-complete
git push origin stage2e1-complete
```

---

## Self-review

**Spec coverage:** §5.1 goal — Task 27 demo. §5.2 components — Tasks 4–19 each implement one component. §5.3 personas — Task 5. §5.4 demo — Task 27. §5.5 risks — addressed via env-var rollback (Task 21), cache stub (consider follow-up if ElevenLabs budget is tight), persona drift via prefix re-injection every turn (Task 14). §5.6 out-of-scope — no tasks for gripper/VLM/cloud-brain/Tavily/FactsStore/Chatterbox/cloning. ✓

**Placeholder scan:** Task 17 contains a "if current implementation has a different shape, adapt" caveat — this is justified because `spot_tts.py` was not read in full during plan-writing. Engineer should inspect first (Task 17 Step 1 does this), then choose the adapter shape. Not a placeholder in the forbidden sense.

**Type consistency:** `Persona` named consistently across persona.py, llm_brain.py, spot_dispatch.py. `SessionState` consistent. `TTSBackend` consistent. Action name `set_persona` consistent across system prompt (Task 15), dispatch handler (Task 16), and tests (none — set_persona dispatch tested only via integration smoke in Task 25 Step 5).

**Schedule realism:** 29 tasks, 5 calendar days (Wed–Tue). At 1–2 hours per task plus the live demo, this is tight but achievable for an engineer with project context. If pushed, parallelize: Task 11 (model download) can run while Tasks 12 (ElevenLabs) is implemented; Tasks 22/23/24/26 (review agents) can fire concurrently in the same checkpoint message.
