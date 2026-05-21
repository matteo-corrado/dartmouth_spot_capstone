# Stage 2E.1 — TTS Multi-Voice + Multi-Persona Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship multi-personality routing + 2 swappable TTS backends (Kokoro v1.1 local, ElevenLabs Flash v2.5 cloud) by Tue 2026-05-26. Persona swappable at runtime via voice command (`set_persona` action) or boot via env var. SessionState dataclass introduced as second state layer alongside existing live SDK snapshot.

**Architecture:** New `tts/` and `chatbot/` modules added under `src/voice_control/`. `TTSBackend` Protocol with two impls. Persona registry loaded from declarative `config/personas.yaml`. `SessionState` dataclass owned by `LLMBrain`. Persona's `prompt_prefix` injected BEFORE `SYSTEM_PROMPT` in `_build_messages()`. Robot state + session state merge into single bullet-list block. New `set_persona` action dispatched same as other actions. JSONL conversation log written per turn. All swaps env-var-flippable for rollback.

**Tech Stack:** Existing sherpa-onnx Kokoro infra (model swap to `kokoro-multi-lang-v1_1`), `elevenlabs>=2.0,<3.0` Python SDK (new dep), `PyYAML` (already present transitively, verify), `pytest>=8.0,<9.0` (new for testing new modules), `python-dotenv==1.0.1` (already pinned, for `ELEVENLABS_API_KEY`).

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
| Create | `src/voice_control/tts/kokoro.py` | sherpa-onnx Kokoro wrapper |
| Create | `src/voice_control/tts/elevenlabs.py` | ElevenLabs SDK streaming wrapper |
| Create | `tests/voice_control/tts/test_elevenlabs.py` | unit tests (mocked HTTP) |
| Create | `scripts/setup_kokoro_v1_1.py` | download `kokoro-multi-lang-v1_1` to `/mnt/ssd/tts-models/` |
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
ls /mnt/ssd/tts-models/                 # confirm SSD mount + writable
curl -sf http://localhost:11434/api/tags | grep -q gemma4 && echo "ollama OK"
grep -q ELEVENLABS_API_KEY .env && echo "key present" || echo "WARN: add ELEVENLABS_API_KEY to .env"
```

Expected: all green. If `.env` missing key, stop and add it before proceeding.

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
# Spot persona registry. Each persona = prompt prefix + per-backend voice id.
# Add new personas by appending entries. SIGHUP the voice loop to hot-reload
# (Task 8 implements the loader; SIGHUP hook is a follow-up nice-to-have).
#
# Voice IDs:
#   kokoro_v1: speaker slug from kokoro-multi-lang-v1_1 (103 voices total).
#              See `python -m sherpa_onnx.kokoro list-voices` after model download.
#   elevenlabs: ElevenLabs voice ID (curl GET /v1/voices to list all).

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
from dataclasses import dataclass
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
        if not prefix:
            logger.warning(f"Persona {name} has empty prompt_prefix; skipping")
            continue
        registry[name] = Persona(name=name, prompt_prefix=prefix, voices=voices)
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

## Task 10: Implement `KokoroBackend` wrapping existing sherpa-onnx TTS

**Files:**
- Create: `src/voice_control/tts/kokoro.py`

This task wraps existing TTS code rather than reimplementing. Read `src/voice_control/spot_tts.py` first to identify the current synth call surface.

- [ ] **Step 1: Inspect current TTS implementation**

```bash
grep -nE "def |sherpa_onnx|kokoro|synthesize|model_path" src/voice_control/spot_tts.py | head -40
```

Capture: model path, voice ID parameter name, synth function name, PCM sample rate, output format.

- [ ] **Step 2: Write `src/voice_control/tts/kokoro.py`**

Adapt the import surface to mirror existing spot_tts.py constants. Skeleton (adjust constants based on Step 1 findings):

```python
"""Kokoro TTS backend via sherpa-onnx.

Wraps the existing model loader path used by spot_tts.py. Stage 2E.1
upgrades the model archive from kokoro-en-v0.19 (8 voices) to
kokoro-multi-lang-v1_1 (103 voices). Model path resolved via
SPOT_KOKORO_MODEL_DIR env var, default /mnt/ssd/tts-models/kokoro-multi-lang-v1_1/.
"""
import os
from pathlib import Path
from typing import Iterator

import sherpa_onnx

from . import register_backend

DEFAULT_MODEL_DIR = Path("/mnt/ssd/tts-models/kokoro-multi-lang-v1_1")
DEFAULT_SAMPLE_RATE = 24000


class KokoroBackend:
    def __init__(self, model_dir: Path = None):
        self.model_dir = Path(model_dir) if model_dir else Path(
            os.environ.get("SPOT_KOKORO_MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        self._tts = self._load()

    def _load(self) -> "sherpa_onnx.OfflineTts":
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(self.model_dir / "model.onnx"),
                    voices=str(self.model_dir / "voices.bin"),
                    tokens=str(self.model_dir / "tokens.txt"),
                    data_dir=str(self.model_dir / "espeak-ng-data"),
                ),
                num_threads=2,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        return sherpa_onnx.OfflineTts(config)

    def _resolve_speaker_id(self, voice_id: str) -> int:
        """Map voice slug (e.g. 'af_sarah') to integer speaker ID for sherpa-onnx."""
        # The voices.bin file is keyed by slug internally; sherpa-onnx exposes
        # them as 0..N indices in the order they were packed. Use the loader's
        # speaker_name_to_id map if available; otherwise default to 0.
        try:
            sid_map = getattr(self._tts, "speaker_name_to_id", None)
            if sid_map and voice_id in sid_map:
                return sid_map[voice_id]
        except Exception:
            pass
        return 0  # fallback to first speaker

    def synthesize(self, text: str, voice_id: str) -> bytes:
        sid = self._resolve_speaker_id(voice_id)
        audio = self._tts.generate(text, sid=sid, speed=1.0)
        # audio.samples is a list[float] in [-1, 1]; convert to int16 PCM bytes
        import numpy as np
        samples = np.asarray(audio.samples, dtype=np.float32)
        pcm_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        # sherpa-onnx Kokoro is not natively streaming; yield single chunk.
        # Future: split text into sentences and yield per sentence.
        yield self.synthesize(text, voice_id)

    def list_voices(self) -> list:
        sid_map = getattr(self._tts, "speaker_name_to_id", {}) or {}
        return sorted(sid_map.keys())


register_backend("kokoro", lambda: KokoroBackend())
```

NOTE: If Step 1 reveals the current spot_tts.py uses `kokoro-onnx` (a different package) rather than sherpa-onnx for Kokoro, adapt the import + config accordingly. The CLAUDE memory says sherpa-onnx is the TTS infra; verify before writing.

- [ ] **Step 3: Manual smoke test (requires Kokoro v1.1 model — see Task 11)**

Defer execution of this step until after Task 11 downloads the model. Stub the smoke test command here for later:

```bash
# Run after Task 11:
python -c "
import os
os.environ['SPOT_TTS_BACKEND'] = 'kokoro'
from src.voice_control.tts import get_backend
b = get_backend()
print('voices:', b.list_voices()[:5])
pcm = b.synthesize('hello from Spot', 'af_sarah')
print(f'pcm bytes: {len(pcm)}')
"
```

Expected after Task 11: voices list non-empty, PCM bytes > 0.

- [ ] **Step 4: Commit (without smoke test verification — gated on Task 11)**

```bash
git add src/voice_control/tts/kokoro.py
git commit -m "stage 2e1: KokoroBackend wrapping sherpa-onnx Kokoro multi-lang v1.1"
```

---

## Task 11: Write `scripts/setup_kokoro_v1_1.py` to download model

**Files:**
- Create: `scripts/setup_kokoro_v1_1.py`

- [ ] **Step 1: Identify download URL**

Reference: https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html (kokoro-multi-lang-v1_1 entry).

Confirm download URL via:

```bash
curl -sI 'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2'
```

Expected: HTTP 302 redirect to GitHub release asset. (URL pattern follows the v1.0 release; check the sherpa-onnx releases page if URL is stale.)

- [ ] **Step 2: Write `scripts/setup_kokoro_v1_1.py`**

```python
"""Download Kokoro multi-lang v1.1 (103 voices) model archive to SSD.

Idempotent: skips download if target directory exists with expected files.
Mirrors the pattern in scripts/setup_kokoro.py for v0.19.
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

TARGET_DIR = Path("/mnt/ssd/tts-models/kokoro-multi-lang-v1_1")
ARCHIVE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2"
ARCHIVE_NAME = "kokoro-multi-lang-v1_1.tar.bz2"

EXPECTED_FILES = [
    "model.onnx",
    "voices.bin",
    "tokens.txt",
]


def already_installed() -> bool:
    if not TARGET_DIR.exists():
        return False
    return all((TARGET_DIR / f).exists() for f in EXPECTED_FILES)


def download() -> Path:
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive_path = TARGET_DIR.parent / ARCHIVE_NAME
    if archive_path.exists():
        print(f"Archive already downloaded: {archive_path}")
        return archive_path
    print(f"Downloading {ARCHIVE_URL} -> {archive_path}")
    urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
    print(f"Downloaded {archive_path.stat().st_size / 1024 / 1024:.1f} MB")
    return archive_path


def extract(archive_path: Path) -> None:
    print(f"Extracting {archive_path} -> {TARGET_DIR.parent}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(TARGET_DIR.parent)
    print(f"Extracted to {TARGET_DIR}")


def main():
    if already_installed():
        print(f"Kokoro v1.1 already installed at {TARGET_DIR}")
        return
    archive = download()
    extract(archive)
    if not already_installed():
        print(f"ERROR: extraction did not produce expected files in {TARGET_DIR}")
        sys.exit(1)
    print("Kokoro v1.1 setup complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run setup**

```bash
python scripts/setup_kokoro_v1_1.py
```

Expected: download + extract, model files in `/mnt/ssd/tts-models/kokoro-multi-lang-v1_1/`.

- [ ] **Step 4: Run Task 10's deferred smoke test**

```bash
python -c "
import os
os.environ['SPOT_TTS_BACKEND'] = 'kokoro'
from src.voice_control.tts import get_backend
import src.voice_control.tts.kokoro  # trigger registration
b = get_backend()
print('voices:', b.list_voices()[:5])
pcm = b.synthesize('hello from Spot', 'af_sarah')
print(f'pcm bytes: {len(pcm)}')
"
```

Expected: voices list includes `af_sarah`, PCM bytes > 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_kokoro_v1_1.py
git commit -m "stage 2e1: setup script for Kokoro v1.1 multi-lang (103 voices)"
```

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

Agent prompt: *"Review `src/voice_control/tts/` (init, kokoro, elevenlabs) + scripts/setup_kokoro_v1_1.py + their tests. Focus: secret handling (API key never logged), HTTP timeout absence on streams, registration side effects (module-import-time `register_backend` calls), exception surface on missing model files. One line per finding, severity tagged."*

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

Confirm: `MAX_HISTORY=12` at line ~31, `_build_messages(self, transcript, state)` at line ~273, `process(self, transcript, state=None)` at line ~284.

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

5. In `process()` after each successful turn, bump `turn_index`:

```python
        # Existing code that appends to self.history follows; immediately after:
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
"
```

Expected: registry size 6, persona "tour_guide", system message begins with the tour_guide prompt_prefix.

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

## Task 17: Modify `spot_tts.py` — route synth through TTS backend dispatch

**Files:**
- Modify: `src/voice_control/spot_tts.py`

- [ ] **Step 1: Inspect current synth entry points**

```bash
grep -nE "def |synthesize|generate|speak|enqueue" src/voice_control/spot_tts.py | head -30
```

Identify the function called by the rest of the system to render speech (likely `enqueue_render`, `speak`, or similar).

- [ ] **Step 2: Wire backend dispatch**

Add the backend module-import + dispatch at the top of `spot_tts.py`:

```python
# Stage 2E.1: route TTS synth through swappable backend registry
import src.voice_control.tts.kokoro      # noqa: F401  (registers backend)
import src.voice_control.tts.elevenlabs  # noqa: F401  (registers backend)
from src.voice_control.tts import get_backend
```

In the synth function: instead of directly calling sherpa-onnx, resolve the backend and call `backend.synthesize(text, voice_id)`. Voice ID resolution requires knowing the current persona — accept `voice_id` as an explicit parameter (caller passes it from the persona registry lookup) OR look it up here from the brain instance's session state.

Cleanest: caller (spot_dispatch's "response" branch, or client_mic) passes both `text` and `voice_id` to `enqueue_render` or equivalent. Modify the signature additively:

```python
def enqueue_render(text: str, voice_id: Optional[str] = None) -> None:
    backend = get_backend()
    if voice_id is None:
        voice_id = _default_voice_for_backend(backend)  # e.g. "af_sarah" for kokoro
    pcm_bytes = backend.synthesize(text, voice_id)
    # ... existing playback path (queue to audio player at 24kHz)
```

If the current implementation has a different shape (e.g. streams chunks directly), adapt — use `backend.stream(...)` instead. Goal: replace the direct sherpa-onnx call with the backend abstraction without changing playback or queue semantics.

- [ ] **Step 3: Add `_default_voice_for_backend` helper**

```python
def _default_voice_for_backend(backend) -> str:
    """Pick a sensible default voice when caller didn't specify one."""
    name = type(backend).__name__
    if "Kokoro" in name:
        return "af_sarah"
    if "ElevenLabs" in name:
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel
    return ""  # backend may raise on empty — that's intended
```

- [ ] **Step 4: Smoke test (Kokoro path)**

```bash
SPOT_TTS_BACKEND=kokoro python -c "
import src.voice_control.spot_tts as tts
tts.enqueue_render('Hello from the Kokoro backend.', 'af_sarah')
import time; time.sleep(2)
"
```

Expected: audible "Hello from the Kokoro backend." through speakers.

- [ ] **Step 5: Smoke test (ElevenLabs path)**

```bash
SPOT_TTS_BACKEND=elevenlabs python -c "
import src.voice_control.spot_tts as tts
tts.enqueue_render('Hello from the ElevenLabs backend.', '21m00Tcm4TlvDq8ikWAM')
import time; time.sleep(3)
"
```

Expected: audible Rachel voice through speakers.

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/spot_tts.py
git commit -m "stage 2e1: route spot_tts synth through swappable TTSBackend dispatch"
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

- [ ] **Step 2: Add voice resolution at each call site**

At each site, before calling `enqueue_render(response)`, resolve the current voice ID:

```python
from src.voice_control.chatbot.persona import get_persona
persona = get_persona(brain.session_state.current_persona, brain._persona_registry)
backend_name = os.environ.get("SPOT_TTS_BACKEND", "kokoro")
voice_key = "elevenlabs" if backend_name == "elevenlabs" else "kokoro_v1"
voice_id = persona.voices.get(voice_key) or _default_voice_for_backend(get_backend())
enqueue_render(response, voice_id=voice_id)
```

Consider extracting this into a helper in `chatbot/persona.py` like `voice_id_for(persona, backend_name)` to avoid copy-paste.

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

The `process()` method appends to history at end of turn (~line 417-423). Best site to also write the conversation log.

- [ ] **Step 2: Add ConversationLog instance to `LLMBrain.__init__`**

```python
        from src.voice_control.chatbot.conversation_log import ConversationLog
        self._conversation_log = ConversationLog()  # writes to logs/conversations/
```

- [ ] **Step 3: Call `log_turn` after each successful turn**

In `process()`, after history appends, before return:

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
                latency_ms={"llm": int(elapsed * 1000)},
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

# TTS backend selector. Default: kokoro (local, 103 voices via sherpa-onnx).
# Alternatives: elevenlabs (cloud, requires ELEVENLABS_API_KEY).
# SPOT_TTS_BACKEND=kokoro

# Boot-time default persona. Must match a key in config/personas.yaml.
# Available: tour_guide, pirate, snarky, butler, shakespeare, gen_z.
# SPOT_PERSONA=tour_guide

# ElevenLabs API key, required only when SPOT_TTS_BACKEND=elevenlabs.
# Free tier: 10,000 chars/month.
# ELEVENLABS_API_KEY=

# Kokoro v1.1 model directory. Default: /mnt/ssd/tts-models/kokoro-multi-lang-v1_1/
# Override only if model was installed elsewhere.
# SPOT_KOKORO_MODEL_DIR=/mnt/ssd/tts-models/kokoro-multi-lang-v1_1
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

Reverts to local sherpa-onnx Kokoro v1.1. No code change required.

### Layer: collapse to single persona

Symptom: persona swap action causing dispatch errors or bad behavior.

```bash
export SPOT_PERSONA=tour_guide
```

Brain still loads registry but treats every utterance as default persona. To fully disable runtime swap, comment out the `set_persona` bullet in `SYSTEM_PROMPT` (llm_brain.py) so the LLM stops emitting the action.

### Layer: revert to Kokoro v0.19 (8 voices)

Symptom: Kokoro v1.1 install corrupt OR voice quality regression.

```bash
export SPOT_KOKORO_MODEL_DIR=/mnt/ssd/tts-models/kokoro-en-v0_19
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

1. ASSUMPTIONS — did any task silently assume intent, behavior, or API surface? Specifically check `KokoroBackend._resolve_speaker_id` (does sherpa-onnx really expose `speaker_name_to_id`? if not, this is a silent assumption).
2. SIMPLICITY — strip unused params, premature flexibility. Specifically check: is `_default_voice_for_backend` actually used, or did Task 17/18 always pass voice_id? If always passed, drop the helper.
3. SURGICAL — any adjacent refactors or formatting fixes not asked for? Revert them.
4. VERIFICATION — does every new code path have a check (test or smoke test)?

- [ ] **Step 2: Verify sherpa-onnx speaker mapping claim**

```bash
python -c "
import sherpa_onnx
help(sherpa_onnx.OfflineTts.generate)
"
```

If `speaker_name_to_id` is NOT a real attribute, replace `_resolve_speaker_id` with a deterministic slug→int map loaded from `voices.bin` metadata or from a hardcoded list shipped with the v1.1 release notes. Update `tests/voice_control/tts/test_elevenlabs.py` is unaffected (mocked); but the kokoro smoke test in Task 11 must pass.

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
- `SPOT_KOKORO_MODEL_DIR=/mnt/ssd/tts-models/kokoro-en-v0_19` — revert to v0.19

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
