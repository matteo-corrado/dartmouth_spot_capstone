# Modular Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the voice/robot code from `src/voice_control/` (two 2.5–3k-line files) into service packages + optional feature folders under `src/spot/`, with one `main`, and identical robot behaviour.

**Architecture:**
- **Services** are plain packages with one Protocol each, wired explicitly in `src/spot/app/main.py`.
- **Optional features** declare their actions in a `FEATURE` object, listed by name in `src/spot/features/enabled.py`. The GBNF, intent table, intent sets and LLM prompt section are generated from those declarations.
- **Migration is a strangler:**
  - leaf modules move with `sys.modules`-alias shims, so tests stay byte-identical;
  - the two big files are split with tests ported in the same merge;
  - shims are deleted last.

**Tech Stack:** Python 3.10, pytest, bosdyn SDK 5.2.0, sherpa-onnx, gRPC (ASR server), numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-29-modular-restructure-design.md` (read it first; this plan argues from it).

## Global Constraints

- Behaviour identical. No tuning, algorithm or bug changes. Bugs found go to `docs/project/bugs.md` only.
- The stop/freeze/e-stop path is never routed through the event bus. Its order stays exactly the same: lock-free halt first, then hush on a thread, then the safety-generation bump.
- The threading model is unchanged: the same threads, the same work on each.
- No new dependencies. `requirements.txt` untouched; if anything forces a change, run pin-guardian.
- Never call SAVE_CONFIGURATION / REBOOT / set_configuration / detach / reset on the XVF3800. Never open the mic or USB device from tests or tools while the pipeline runs.
- Voiceprints stay memory-only. Demo recordings stay out of git. Golden files hold derived numbers and transcripts only, never audio.
- Commits: one-line messages, no trailers (no Co-Authored-By, no session links).
- Tests may change only import paths and patch targets. Assertion lines and expected values stay byte-identical. No test is deleted or skipped. Test count only grows (615 at base).
- Operator commands keep identical CLI and flags: `wakespot`, `spotctl`, `web_panel`, `estop_run`, `spot-web-panel.service`, `run_voice_control.py`.
- Sensitive-file hook: Bash commands that mention KWS keyword-file names are silently rewritten.
  - Move `wake/` files with `git mv` on paths that don't name keyword files, or with the Write/Edit tools.
  - Run `git status` after every move.
- Worktree lanes start from an old main. The first step of every lane is `git reset --hard <current restructure-modular HEAD>`.
- Import root: new code is imported as `src.spot.<pkg>`, matching the existing `src.` convention. New packages get `__init__.py`.
- Freeze: while Wave 3 runs, nobody edits `client_mic.py` or `spot_dispatch.py` on other branches.

## Review Focus

1. **Running an old file as a script after it becomes a shim.** `run_voice_control.py` runs `python src/voice_control/client_mic.py` and `.../server.py`; both must still start the pipeline and the ASR server. Test: Task 15 `test_legacy_entrypoints_start`.
2. **A monkeypatch on a split shim silently checks nothing.** Expected: split shims hold no mutable module state, so patches must target the owner. Test: Task 16 `test_split_shims_hold_no_state`.
3. **A disabled feature's intent still arrives** (from regex, the LLM or spotctl). Expected: not dispatched, no crash, same "can't do that" path as an unknown intent today. Test: Task 13 `test_disabled_feature_intent_is_refused`.
4. **An event handler raises during a stop.** Expected: the halt has already happened and is unaffected, and the bus logs the error. Test: Task 2 `test_handler_exception_isolated` + Task 9 `test_halt_does_not_touch_bus`.
5. **Old and new import paths give two module objects** (patches split in half). Expected: the same object either way. Test: each Wave 2 task's `test_alias_identity`.

---

## Wave plan (lanes, ownership, reviews)

| Wave | Tasks (run in parallel within a wave) | Owns (no other task in the wave edits these) | Review |
|---|---|---|---|
| 0 | T0 (controller) | branch | — |
| 1 | T1 skeleton+bus+registry, T2 golden harness | T1: `src/spot/core/{events,registry}.py`, `src/spot/**/__init__.py`; T2: `tests/integration/**` | per-task spec+quality |
| 2 | T3 audio, T4 mic, T5 wake+safety-kws, T6 stt, T7 tts+mouth, T8 brain+chatbot, T8b robot-leaf+nav+perception+core+operator | the old files listed in each task + their new homes | per-task; **safety-reviewer** on T5, T8b; wave gate |
| 3 | T9 robot core split (from spot_dispatch), T10 client_mic split | T9: `spot_dispatch.py`, `src/spot/robot/**`, `src/spot/nav/**`, `src/spot/safety/gen.py`, dispatch tests; T10: `client_mic.py`, `src/spot/{audio/capture,mic/ring_control,safety/listener,operator,conversation,app}/**`, client_mic tests | per-task; **safety-reviewer** both; wave gate |
| 4 | T11 dance+gaze, T12 mouth+settings, T13 walk+follow, T14 navigation+visual_nav | own `src/spot/features/<name>/**` + only their own handler functions/branches in `spot_dispatch.py` (marked blocks) | per-task; **safety-reviewer** all four; wave gate |
| 5 | T15 main wiring + entry points, T16 registry cut-over (generated grammar/sets/prompt/intents) | T15: `src/spot/app/**`, `scripts/*`; T16: `src/spot/brain/{intent,grammar,llm_brain}`, `src/spot/robot/dispatcher.py` sets, `src/spot/operator/spotctl help` | per-task; **safety-reviewer** both; wave gate |
| 6 | T17 scripts port + delete 4, T18 docs → then T19 shim removal | T17 `scripts/**`; T18 `docs/**`; T19 `src/voice_control/**`, `tests/**` imports | per-task; wave gate |
| 7 | T20 final verification (controller) | — | **safety-reviewer** whole branch + `/code-review high` + on-robot checklist |

**Review protocol (every task):**
1. A fresh implementer works in a worktree.
2. A spec-compliance reviewer checks the work against this plan and the spec.
3. A code-quality reviewer checks it.
4. If any robot or safety code is touched, the safety-reviewer runs. Fixes get a scoped re-review.
5. The controller merges with `--no-ff` into `restructure-modular` and runs the full suite after each merge.
6. Before the next wave starts, a wave-gate reviewer reads the whole wave diff and runs the suite plus golden check (T2).

Ledger: `.superpowers/sdd/2026-10-29-restructure/progress.md` (gitignored) in the main checkout.

## Old → new map (source of truth for Wave 2)

| Old | New |
|---|---|
| `audio_player.py`, `audio_feedback.py`, `audio_frontend.py`, `barge_in.py`, `response_gate.py` | `src/spot/audio/{player,chimes,frontend,barge_in,response_gate}.py` |
| `mic_array.py`, `ring_animator.py`, `state_feedback.py`, `spot_leds.py` | `src/spot/mic/{xvf3800,ring_animator,state_feedback,spot_leds}.py` |
| `wake/__init__.py`, `wake/sherpa_onnx.py`, `wake/livekit.py`, `wake/phrases.py` | `src/spot/wake/{__init__,sherpa_onnx,livekit,phrases}.py` |
| `wake/safety_kws.py` | `src/spot/safety/kws.py` |
| `asr/*`, `server.py`, `protos/asr.proto`, `asr_pb2*.py` | `src/spot/stt/{backends/,server.py,asr.proto,asr_pb2*.py}` |
| `tts/*`, `spot_tts.py`, `text_segment.py` | `src/spot/tts/{__init__,kokoro,elevenlabs,speaker,text_segment}.py` (`spot_tts` → `speaker.py`) |
| `animation/mouth.py` | `src/spot/features/mouth/driver.py` |
| `intent.py`, `llm_brain.py`, `brain/*`, `chatbot/persona.py`, `chatbot/voice_discovery.py`, `grammar/`, `knowledge/` | `src/spot/brain/{intent,llm_brain,backends/,persona,voice_discovery,grammar/,knowledge/}` |
| `chatbot/conversation_log.py`, `chatbot/session_state.py`, `speaker_lock.py`, `speaker_embed.py` | `src/spot/conversation/{conversation_log,session_state,speaker_lock,voiceprint}.py` |
| `src/session.py`, `src/estop.py` | `src/spot/robot/session.py`, `src/spot/safety/estop.py` |
| `src/graph_nav_utils.py`, `src/map_loader.py`, `src/location_manager.py` | `src/spot/nav/{graph_nav_utils,map_loader,location_manager}.py` |
| `perception/*`, `world_objects.py` | `src/spot/perception/{…,world_objects}.py` |
| `visual_nav.py` | `src/spot/features/visual_nav/navigator.py` |
| `emotes.py`, `gaze.py` | `src/spot/features/dance/emotes.py`, `src/spot/features/gaze/math.py` |
| `tuning.py`, `latency.py`, `startup_status.py`, `voice_state.py` | `src/spot/core/{tuning,latency,startup_status,voice_state}.py` (`src/config.py` stays put) |
| `control_server.py` | `src/spot/operator/control_server.py` |

**Shim template (Wave 2, one-to-one moves only):**
```python
"""Moved to src/spot/<pkg>/<mod>.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.<pkg> import <mod> as _m
sys.modules[__name__] = _m
```
Exception: `server.py` and `client_mic.py` are run as scripts. Their shims don't alias. They import the new `main` and call it under `if __name__ == "__main__":`.

---

### Task 0: Base (controller, after the demo is tagged)

- [ ] **Step 1:** In `/home/spotdog/spot/dartmouth_spot_capstone-restructure`: `git merge --no-ff demo-ready-2026-10-29` (or the final demo HEAD the operator names). Expected: clean merge, or conflicts only in docs.
- [ ] **Step 2:** `pytest -q`. Record the pass count in the ledger as BASE_COUNT (≥615).
- [ ] **Step 3:** Recompute the line ranges quoted in spec 4.2 against the merged files, and note any shift in the ledger for lanes.

### Task 1: Skeleton, event bus, feature registry

**Files:**
- Create: `src/spot/__init__.py` and `__init__.py` in `app, core, audio, mic, wake, safety, stt, tts, conversation, brain, robot, nav, perception, operator, features`
- Create: `src/spot/core/events.py`, `src/spot/core/registry.py`, `src/spot/features/enabled.py` (empty list)
- Note: `tts`, `wake`, `brain` and `perception` get placeholder `__init__.py` files here. Wave 2 lanes overwrite them with the moved real `__init__.py` content (`tts` is 179 lines). That's expected, not clobbering.
- Test: `tests/spot/core/test_events.py`, `tests/spot/core/test_registry.py`

**Interfaces — Produces:**
- `EventBus.subscribe(event: str, handler: Callable[..., None]) -> None`, `EventBus.publish(event: str, **payload) -> None`.
  - Synchronous, in subscription order.
  - Each handler call is wrapped in try/except that logs `[Events] handler for <event> failed: <exc>`.
- `EVENTS: frozenset[str]` = `{"speaking_started","speaking_stopped","voice_state_changed","speaker_locked","speaker_released","robot_stood","robot_sat","safety_halt","person_seen"}`. `publish` of an unknown event raises `ValueError`.
- `ActionSpec` (frozen dataclass):
  - `name: str`
  - `handler: Callable[[dict, "Ctx"], bool]`
  - `grammar_params: str | None`
  - `prompt_line: str | None`
  - `locomotion=False`, `aborting=False`, `unlocked=False`, `short_path_blocked=False`, `info=False`, `state_only=False`, `motion_free=False`, `persona=False`
  - `arm_policy: tuple[str, str] | None = None`
  - `needs_session=True`
- `IntentRule` (frozen dataclass): `priority: int`, `pattern: str`, `build: Callable[[re.Match], dict]`.
- `Feature` (frozen dataclass):
  - `name: str`
  - `actions: tuple[ActionSpec, ...] = ()`
  - `intent_rules: tuple[IntentRule, ...] = ()`
  - `spotctl_help: tuple[str, ...] = ()`
  - `subscriptions: tuple[tuple[str, Callable], ...] = ()`
  - `start: Callable[["Ctx"], None] | None = None`, `stop: … = None`
  - `requires: tuple[str, ...] = ()`
- `Registry`:
  - `.add(feature)`: raises `ValueError` on a duplicate action name.
  - `.load(names: list[str], import_fn=importlib.import_module)`: imports `src.spot.features.<name>` and reads `FEATURE`. An `ImportError` or other exception logs `[Features] <name> skipped: <exc>` and is skipped.
  - `.action(name) -> ActionSpec | None`
  - `.names(flag: str) -> frozenset[str]`
  - `.grammar_action_names() -> list[str]` (in registration order)
  - `.intent_rules() -> list[IntentRule]` (sorted by priority, stable)
  - `.prompt_lines() -> list[str]`

- [ ] **Step 1: Write failing tests.**
  - `test_publish_calls_in_order`
  - `test_handler_exception_isolated`: the first handler raises, the second is still called, and `publish` returns normally.
  - `test_unknown_event_raises`
  - `test_duplicate_action_rejected`
  - `test_load_skips_broken_feature`: `import_fn` raises → that feature is absent, the others load, and the log contains `skipped`.
  - `test_names_by_flag`
  - `test_intent_rules_priority_stable`
- [ ] **Step 2:** `pytest tests/spot/core -v`. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement `events.py` and `registry.py` as specified.
- [ ] **Step 4:** `pytest -q`. Expected: BASE_COUNT + new tests, all pass.
- [ ] **Step 5:** Commit `spot: package skeleton, event bus, feature registry`.

### Task 2: Golden conversation harness (captured on the base)

**Files:** Create `tests/integration/conftest.py`, `tests/integration/harness.py`, `tests/integration/test_conversation_golden.py`, `tests/integration/golden/conversation.json`

**Interfaces — Produces:**
- `run_pipeline(clips: list[Path]) -> list[dict]`. It drives the real `src.voice_control.client_mic.main` with these fakes:
  - patched `sounddevice` input stream feeding the clip PCM (48 kHz → the stream's rate, via existing resample helpers) into the real `audio_callback` in real-time-scaled chunks;
  - fake ASR stub returning transcripts from `tests/integration/golden/transcripts.json` keyed by clip name;
  - fake `execute_on_spot` and `get_spot_state` (no robot);
  - fake TTS/AudioPlayer (records text, no sound);
  - XVF3800 replaced by a fake that gives a fixed DoA per clip;
  - LLM brain disabled (`_brain_ok=False` → regex path).
- It returns the ordered event records:
  - `{"kind": "utterance", ...}` from the real `build_utterance_record`;
  - `{"kind": "dispatch", "intent": ...}`;
  - `{"kind": "state", "to": ...}`;
  - `{"kind": "say", "text": ...}`.
- Clip source: env `SPOT_DEMO_AUDIO`, default `/home/spotdog/spot/dartmouth_spot_capstone/tests/audio/demo`, plus `tests/audio/*_lab.wav`. If the clips are missing, `pytest.skip("demo recordings not present")`.

**Real-time constraint:**
- The endpointer, `ChimeMask`, `RingThrottle`, filler and lock timeout read `time.monotonic()`.
- So the harness feeds audio **at real time**. Speeding it up changes decisions, and faking the clock needs a src seam.
- Budget several minutes per run. Mark the test `@pytest.mark.slow`, registered in `pyproject.toml` markers. Gates run it with `pytest tests/integration -m slow`.

**Fallback:** if `main()` cannot be driven without a src change:
- the harness drives `audio_callback` + `process_utterance` + `_finish_utterance` through the module directly, with the same fakes;
- the controller records that choice in the ledger.

**Wave 1 gate:** T2 has run twice on base with identical output before any Wave 2 lane is dispatched.

- [ ] **Step 1:** Write `test_conversation_matches_golden`.
  - It runs `run_pipeline` on the fixed clip list stored in `golden/conversation.json["clips"]`.
  - Then `assert records == golden["records"]`, comparing floats rounded to 2 dp.
- [ ] **Step 2:** Build the harness. Only test-side patches; no change under `src/`.
  - Wherever `main()` reads hardware, patch at that call site with `monkeypatch`.
  - If a seam is impossible without a src change, stop and report to the controller instead of editing src.
- [ ] **Step 3:** Generate `golden/conversation.json` at the base commit by running the harness twice.
  - Both runs must match exactly. If not, list the nondeterministic fields and exclude only those, each with a comment saying why.
  - Clip list: at least the 14 safety clips, the wake/command `_lab` clips and the idle clips.
- [ ] **Step 4:** `pytest tests/integration -v`. Expected: PASS.
- [ ] **Step 5:** Commit `tests: golden conversation harness captured on base`.

### Tasks 3–8b: Leaf moves (Wave 2, all parallel)

Every task is the same recipe, applied to its rows of the old → new map:

**Recipe:**
- [ ] **Step 1: Write failing test** `tests/spot/test_aliases_<pkg>.py::test_alias_identity`. For each moved module:
  ```python
  import importlib
  old = importlib.import_module("src.voice_control.<old>")
  new = importlib.import_module("src.spot.<pkg>.<mod>")
  assert old is new
  ```
- [ ] **Step 2:** `pytest tests/spot/test_aliases_<pkg>.py -v`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3:** Move the files.
  - `git mv` each file to its new path, then write the shim at the old path.
  - Inside the moved files, update imports of *other already-moved-in-this-task* modules to the new paths. Leave imports of modules owned by other tasks on their old paths; the shims resolve them.
  - Then `git status` to confirm every move registered.
- [ ] **Step 4:** Verify.
  - `pytest -q`: Expected: BASE_COUNT + new alias tests, all pass, with no file under `tests/voice_control/` changed (`git diff --stat tests/voice_control` empty).
  - `pytest tests/integration -q`: Expected: PASS.
- [ ] **Step 5:** Commit `spot: move <pkg> modules (aliased)`.

**Task-specific notes:**
- **T3 audio:** the rows `audio_player…response_gate`.
- **T4 mic:** the rows `mic_array…spot_leds`. Don't instantiate the XVF3800 in the alias test (import only).
- **T5 wake + safety KWS** (safety-reviewer):
  - Move `wake/*` and `wake/safety_kws.py`.
  - The `models/kws` keyword files are **not** moved; path constants stay the same.
  - Use Edit/Write for any text change in these files. Confirm with `git status`.
- **T6 stt:**
  - Moves `asr/*`, `asr_pb2*.py`, `protos/asr.proto`, `server.py`.
  - Replace the bare `import asr_pb2` / `sys.path` bootstrap in the moved `server.py` with `from src.spot.stt import asr_pb2 as pb`.
  - `src/voice_control/server.py` becomes a script shim (no alias): `from src.spot.stt.server import serve` under `__main__`. Keep a `sys.path` insert of the project root so `python src/voice_control/server.py` works.
  - `client_mic.py` keeps `import asr_pb2`. Leave `src/voice_control/asr_pb2.py` and `asr_pb2_grpc.py` as alias shims.
  - Extra test: `test_server_script_help`. Run `python src/voice_control/server.py --help` with a 20 s timeout, expect exit code 0; if `server.py` has no `--help`, assert the import of `src.spot.stt.server` succeeds instead.
- **T7 tts + mouth:** the rows `tts/*`, `spot_tts`, `text_segment`, `animation/mouth.py`.
- **T8 brain:**
  - The `intent…knowledge` rows and the conversation rows (`conversation_log`, `session_state`, `speaker_lock`, `speaker_embed`).
  - `knowledge/` and `grammar/` are data. Fix every path constant that reads them, then add a test asserting each path constant exists on disk.
  - Also grep `scripts/*.sh`, `*.service` and docker config for `knowledge/` and `grammar/` paths, and update them.
- **T8b robot leaf, nav, perception, core, operator** (safety-reviewer):
  - Rows `src/session.py`, `src/estop.py`, nav, perception, `world_objects`, `visual_nav`, `emotes`, `gaze`, core, `control_server`.
  - `src/config.py` is **not moved**. It stays where it is: 8 lines, credentials-adjacent, and the sensitive-file hook may block its shim. New code imports `from src.config import …` as today.

**Wave 2 gate:**
- A reviewer reads the merged wave diff. Every old path is a 3-line alias (or the T6 script shim), and no logic line changed: `git diff -M --stat` shows renames with ≥95% similarity.
- `pytest -q` and `pytest tests/integration` pass.
- The safety-reviewer confirms `estop.py`, `session.py` and `safety/kws.py` are byte-identical apart from imports.

### Task 9: Robot core split (from `spot_dispatch.py`), with the safety generation counter

**Files:**
- Create:
  - `src/spot/safety/gen.py`
  - `src/spot/robot/{state.py, arm.py, motion.py, dispatcher.py, connection.py}`
  - `src/spot/nav/runner.py`
- Modify: `src/voice_control/spot_dispatch.py`. It keeps only the feature handler functions and the `_dispatch_core` feature branches (Wave 4 moves them), and re-exports every moved public and private name.
- Tests: port `tests/voice_control/test_arm_policy.py`, `test_auto_arm.py` (dispatch parts), `test_fr_halt_reply.py` (dispatch parts), `test_gaze_dispatch.py` (dispatch parts), `test_spoken_info.py`, `test_emote_dispatch.py` (core parts). Patch targets only.

**Interfaces — Produces** (later tasks rely on these names):
- `SafetyGen`:
  - `.current() -> int`, `.bump() -> int`, `.aborted(gen: int) -> bool`;
  - also the `dispatch_gen` counter: `.dispatch_current()`, `.dispatch_bump()`.
- `Connection` owns the old globals `_spot_session`, `_session_context`, `_session_lock`, `_closing`:
  - `.ensure(map_path=None)`, `.close()`, `.session`, `.cmd_client()`
- `Arm` owns `_arm_pose`, `_pose_before_walk`, `_pose_during_walk`, `_arm_lock`, `_arm_settle_until`, `_last_traj`, `_mouth_resume`, `_talk_resume`:
  - `.pose() -> str`, `.set_pose(pose)`, `.send(cmd_client, pose, stop_event=None, timeout_s=5.0)`, `.stow_now(cmd_client, resend_stop=False, then_stand=False, force=False)`, `.policy_for(intent_name) -> tuple[str, str]` (= `arm_policy_for`), `.mouth_suspend()`, `.mouth_restore()`, `.restore_after_motion(stop_event=None)`
  - per spec 7.1b, the talk pose and halt-resume stay here.
- `motion` module: `send_stop(cmd_client) -> bool`, `finish_motion(natural, stop_event)`, `wait_trajectory_done(cmd_client, cmd_id, deadline, stop_event)`, `start_restore_watch(cmd_client, cmd_id, deadline, gen=None)`, `abort_motion(cmd_client, name)`
- `state` module: `get_robot_state_dict() -> dict`, `capture_frame(camera="front") -> bytes`, `is_standing(session) -> bool`, `standing_and_powered(session) -> bool`
- `NavRunner` owns `_nav_thread`, `_nav_stop_event`, `_nav_mode`, `_nav_lock`, `_home_waypoint`:
  - `.spawn(fn, mode=None, gen=None) -> bool`, `.cancel()`, `.is_navigating()`, `.is_follow_mode()`, `.ensure_localized(...)`, `.force_relocalize(...)`, `.navigate_sequence(...)`, `.start_thread(...)`
- `dispatcher` module:
  - `dispatch_intent(intent, brain=None, speak=None) -> bool`, `dispatch_safety(intent, brain=None, speak=None) -> bool`, `resume_talk_after_halt(gen)`
  - core handlers for `stop, freeze, estop, stand, sit, power_off, shutdown, stow, selfright, status, battery_status`
  - `registry_dispatch(name, params, ctx) -> bool | None`: returns `None` when the action is not registered, otherwise the handler's result.
  - `dispatch_intent` tries `registry_dispatch` first. Only on `None` does it fall through to `LEGACY_FEATURE_DISPATCH`, which calls `spot_dispatch._dispatch_core`.
  - A registered handler that returns `False` is never re-dispatched to legacy.
  - Task 16 deletes the legacy hook once Wave 4 has emptied it.
  - `REGISTRY` singleton lives in `src/spot/features/__init__.py`. `Ctx` is a frozen dataclass in `src/spot/features/ctx.py`: `connection`, `arm`, `safety`, `nav`, `speak`, `brain`, `bus`, `tuning`.
  - the constant sets keep their exact current values: `LOCOMOTION_INTENTS`, `SAFETY_INTENTS`, `SHUTDOWN_INTENTS`, `ABORTING_INTENTS`, `_UNLOCKED_INTENTS`, `INFO_INTENTS`. Task 16 replaces them.
- Module-level singletons, created once in `src/spot/robot/__init__.py`: `SAFETY_GEN`, `CONNECTION`, `ARM`, `NAV`. They are the owner objects; `main` passes them on. Tests patch attributes on these objects.
  - **Deliberate exception to spec §5** ("no singletons reached by import"). The spec allows it where a test proves identical behaviour needs it, and here about 260 monkeypatch sites do.
  - A singleton keeps each port to a mechanical patch-target change. Threading every owner through constructors would rewrite those tests' setup.
  - Retiring the singletons is a post-restructure upgrade.

- [ ] **Step 1:** Port the listed tests' patch targets to the owner objects (e.g. `monkeypatch.setattr(d, "_arm_pose", "talk")` → `monkeypatch.setattr(ARM, "_pose", "talk")`). Assertions stay byte-identical. Run them. Expected: FAIL (ImportError).
- [ ] **Step 2:** Move the code in this order: `safety/gen.py` → `connection` → `state` → `arm` → `motion` → `nav/runner` → `dispatcher`. Run `pytest -q` after each move; it must stay green except for not-yet-moved ported tests.
  - `spot_dispatch.py` re-exports moved names (`from src.spot.robot.arm import arm_policy_for` style) so `client_mic.py` and the unported tests keep working.
- [ ] **Step 3:** Write `tests/spot/robot/test_stop_order.py::test_stop_path_order_unchanged`.
  - With a fake `cmd_client` and a recording of calls, `dispatch_safety` must produce, for each of `stop`, `freeze` and `estop`, the same call sequence as the base commit.
  - Record that sequence by running the base's `spot_dispatch` once, and paste it into the test as the expected list. **The recording is the truth.** Never change code to match a sequence written anywhere else.
- [ ] **Step 4:** Verify.
  - `pytest -q`: BASE_COUNT + new tests pass.
  - `git diff <base> -- tests/voice_control | grep '^[-+] *assert'`: Expected: empty (no assertion line changed).
  - `pytest tests/integration`: PASS.
- [ ] **Step 5:** Commit `spot: robot core split out of spot_dispatch (connection, arm, motion, state, nav runner, dispatcher)`.

### Task 10: `client_mic.py` split

**Files:**
- Create:
  - `src/spot/audio/capture.py` (lines ~116–870: device find, frames/dbfs helpers, stream open/retry, `audio_callback`, `ChimeMask`, drain, noise floor)
  - `src/spot/mic/ring_control.py` (ring state, throttle, `_show_ring`/`_update_ring`/`_flush_ring`, reject flash)
  - `src/spot/safety/listener.py` (`safety_kws_loop` … `_stop_safety_thread`, `_run_safety`, `rescue_safety`, `_hush`, `_hush_player`)
  - `src/spot/operator/control.py` (`_status_snapshot`, `_on_immediate`, `apply_voiceprint_arg`, `_toggle`, `_start_control_server`)
  - `src/spot/conversation/{gate.py, utterance_log.py, filler.py, loop.py}`
  - `src/spot/app/main.py`
- Modify: `src/voice_control/client_mic.py` → a script shim plus re-exports of every moved name used by tests or scripts.
  - Keep its project-root `sys.path` bootstrap, so `python src/voice_control/client_mic.py` can import `src.spot`.
  - Keep `src/voice_control` on `sys.path` as well, for the bare `asr_pb2` alias.
- Tests: port every `tests/voice_control/test_client_mic_*.py`, plus `test_drain_queue`, `test_filler`, `test_short_path`, `test_voiceprint_gate`, `test_state_feedback`, `test_spot_leds`, `test_fr_speaking_cue` and the client_mic parts of `test_auto_arm`, `test_fr_halt_reply`, `test_gaze_dispatch` and `wake/test_safety_kws`. Patch targets only.

**Interfaces:**
- Consumes: T9 `SAFETY_GEN`, `dispatcher.dispatch_safety`, `ARM`. T10 imports them through `spot_dispatch` re-exports if T9 isn't merged yet. The controller merges T9 first, then T10 rebases.
- Produces:
  - `AudioCapture` owns `_mic_gated`, `_mic_reopen_at`, `_player_last_busy_at`, `_audio_player`, `_mic`, `_mic_dead_logged`. It has `.callback(indata, frames, time_info, status)` = the old `audio_callback`, `.open(device)`, `.drain(keep_tail=True)` and `.calibrate_floor(duration_sec, device=None)`.
  - `Ring` owns `_ring_*`, `_anim`, `_reject_until`, `_ring_throttle`. It has `.show(spec)`, `.update()`, `.flush()`, `.signal_reject()`.
  - `SafetyListener` owns `_safety_q/_stop/_active/_thread/_health` and `_last_safety_at`.
    - Methods: `.start(detector)`, `.stop()`, `.on_keyword(kw)` (= `_on_safety_kw`), `.health()`.
    - Constructed with `halt: Callable[[dict], bool]` = `dispatcher.dispatch_safety` and `hush: Callable[[], None]`. **No bus reference.**
  - `Conversation` owns `_voice`, `_status`, `_state_leds`, `_lock`, `_operator_muted`, `_crowd_mode`, `_voiceprint_mode`, `_voiceprint_threshold`, `_abort_chain`, `_brain_ok`, `_store_transcript`, `_speaking_hook`.
    - Each closure inside old `main()` (`_announce_thinking`, `_on_wake`, `_release`, `_finish_utterance`, `_on_queued`) becomes a method of the same name without the underscore prefix.
    - `process_utterance` becomes `.process_utterance(...)`, and `.run()` is the old main loop body.
  - `app.main.main(argv: list[str] | None = None) -> int`:
    - the same argparse definition as old `main()`, byte-identical options and help;
    - builds the objects above and calls `Conversation.run()`;
    - the shutdown signal handler moves here unchanged.
- [ ] **Step 1:** Port test patch targets to the owners. Run the tests. Expected: FAIL (ImportError).
- [ ] **Step 2:** Move in this order: `capture` → `ring_control` → `safety/listener` → `operator/control` → `conversation/*` → `app/main`. Run `pytest -q` after each; it stays green except for not-yet-moved ported tests.
- [ ] **Step 3:** Write `tests/spot/safety/test_listener.py::test_halt_does_not_touch_bus`. Build a `SafetyListener` with a recording `halt` and a bus whose `publish` raises. `on_keyword("stop")` must call `halt` once, before any publish, and must not raise.
- [ ] **Step 4:** Write `tests/spot/app/test_main_args.py::test_argparse_identical`. `main --help` output must equal the base's `client_mic.py --help` output, stored in `tests/spot/app/golden_help.txt`, captured on base.
- [ ] **Step 5:** Verify.
  - `pytest -q`.
  - No assertion line changed (same grep as T9).
  - `pytest tests/integration`: PASS. The harness now drives `src.spot.app.main.main`; changing its entry point is allowed, its fakes are not.
- [ ] **Step 6:** Commit `spot: client_mic split into audio capture, ring, safety listener, operator control, conversation, app main`.

**Wave 3 gate:**
- The safety-reviewer reads T9+T10 together. Check that:
  - the stop/freeze/e-stop sequence is identical;
  - the lock-free halt is still lock-free (no new lock taken on `dispatch_safety`);
  - hush still runs on its own thread;
  - the safety gen is bumped at the same point;
  - `_hold_gaze()` still runs on aborting intents.
- Golden passes. `test_stop_path_order_unchanged` passes.

### Tasks 11–14: Feature folders (Wave 4, parallel)

Each feature `X` gets `src/spot/features/X/__init__.py`, which exports `FEATURE: Feature` (T1 types), plus `handlers.py` holding the code moved from `spot_dispatch.py`.
- A lane **only deletes** its own handler functions and its own `if name == ...` branches from `spot_dispatch._dispatch_core`. It adds nothing there.
- This works because T9's `dispatch_intent` already tries the registry first, so a registered feature never reaches the legacy chain.
- Pure deletions in separate regions merge cleanly.
- `Ctx`, `REGISTRY` and `registry_dispatch` come from T9.

**Declarations:** each `ActionSpec` copies its flags from the current hand-written sets (spec 6.3 list) and its `grammar_params` / `prompt_line` text verbatim from `spot_action.gbnf` / `llm_brain.py`. `IntentRule`s copy the regex rules from `intent.py`, each with a `priority` equal to the rule's current index × 10.

| Task | Features (intents) | Extra |
|---|---|---|
| T11 | `dance` (`emote`), `gaze` (no intent; `start` runs the gaze worker; subscribes to `speaker_locked`) | clearance scan stays in `perception`, called by dance |
| T12 | `mouth` (`enable_mouth`, `disable_mouth`; `start` = auto-arm after stand), `settings` (`set_persona`, `add_persona`, `set_volume`, `set_backend`) | per spec 7.1b: close-mouth-before-stow and the talk pose stay in `Arm` |
| T13 | `walk` (`walk`, `turn`, `strafe`, `body_height`, `set_speed`), `follow` (`follow_me`) | |
| T14 | `navigation` (`walk_to`, `go_to`, `tour`, `patrol`, `come_back`, `load_map`, `save_location`, `list_maps`, `list_locations`), `visual_nav` (`go_to_object`, `check_obstacles`, `describe`) | |

Steps per task:
- [ ] **Step 1:** Write `tests/spot/features/test_<X>_declared.py::test_flags_match_legacy_sets`. For every action, the flags must equal membership in the legacy sets imported from `src.spot.robot.dispatcher` (and `intent.SHORT_PATH_BLOCKED_INTENTS`, `client_mic._STATE_ONLY_INTENTS` / `_PERSONA_INTENTS` / `_MOTION_FREE_INTENTS` via the shim). Expected: FAIL.
- [ ] **Step 2:** Port that feature's handler tests' patch targets (assertions unchanged). Expected: FAIL.
- [ ] **Step 3:** Move the handlers and write `FEATURE`. Add the name to `enabled.py`. One line per feature, in the order `dance, gaze, mouth, settings, walk, follow, navigation, visual_nav`; the controller resolves the trivial list-merge.
- [ ] **Step 4 (T13 only):** Write `tests/spot/features/test_disabled.py::test_disabled_feature_intent_is_refused`.
  - Load the registry without `walk`, after T13 has removed walk's legacy branches. `dispatch_intent({"intent": "walk", "params": {}})` returns False and raises nothing.
  - On the base, an unknown intent does the same; assert parity by calling `dispatch_intent({"intent": "no_such_intent"})` too.
- [ ] **Step 5:** `pytest -q` and `pytest tests/integration`. Both PASS.
- [ ] **Step 6:** Commit `spot: <X> feature folder(s)`.

**Wave 4 gate:**
- `_dispatch_core` contains no feature branches; core intents only, the rest go to `registry_dispatch`.
- Safety-reviewer on all four: every motion feature still checks the safety gen at request time and still goes through `Arm.policy_for`.
- Golden passes.

### Task 15: Main wiring, factories, entry points

**Files:**
- Modify: `src/spot/app/main.py`
- Create: `src/spot/{stt,tts,wake,brain,perception}/factory.py`, `src/spot/perception/vision.py`
- Modify: `scripts/run_voice_control.py`
- Test: `tests/spot/app/test_wiring.py`, `tests/spot/app/test_legacy_entrypoints.py`

**Interfaces — Produces:**
- Protocols, each in its package's `api.py`. Each one wraps the current implementation's existing public methods; no new behaviour.
  - `stt.api.SpeechToText.transcribe(pcm16: bytes, timeout: float | None) -> str`
  - `tts.api.TextToSpeech.speak(text: str, voice=None, cue: bool = True) -> None`
  - `wake.api.WakeDetector`: the current sherpa detector's public methods.
  - `brain.api.Brain`: the current `LLMBrain` public methods.
- `make_stt() -> SpeechToText` reads `SPOT_ASR_BACKEND` exactly as today. `make_tts()`, `make_wake()` and `make_brain()` read their current env/config keys unchanged.
- `Vision` Protocol: `people() -> list[dict]`, `objects() -> list[dict]`. `NullVision` returns `[]` for both.
- `main.main()` is laid out top to bottom:
  1. tuning
  2. bus
  3. robot owners (T9)
  4. `REGISTRY.load(ENABLED)`; each feature's `start(ctx)`
  5. audio capture, ring, wake, safety listener
  6. stt, tts, brain, vision
  7. conversation
  8. control server
  9. `Conversation.run()`
- Each step is one call with a short comment. No logic lives in `main` beyond construction and order.
- Events published, each from the module that owns the change:
  - `speaking_started` / `speaking_stopped` (tts speaker)
  - `voice_state_changed` (conversation)
  - `speaker_locked` / `speaker_released` (conversation)
  - `robot_stood` / `robot_sat` (dispatcher, after the RPC returns)
  - `safety_halt` (listener, after `halt` returns)
- Gaze subscribes to `speaker_locked`. Replace the current direct gaze call with that subscription only if the golden and gaze tests stay green; otherwise keep the direct call and note it in the ledger.

- [ ] **Step 1:** Write the tests.
  - `test_wiring_order`: patch every constructor to record its name, run `main` with `--dry-run` and a fake `Conversation.run`, and assert the recorded order equals the 9-step list.
  - `test_legacy_entrypoints_start`: `subprocess.run([sys.executable, "src/voice_control/client_mic.py", "--help"])` and the same for `server.py`, each with exit 0 and stdout equal to `golden_help.txt` for client_mic.
  - `test_null_vision`.
- [ ] **Step 2:** Implement. `scripts/run_voice_control.py` keeps launching the same two paths (shims), so the `pgrep -f voice_control/server.py` check still matches.
- [ ] **Step 3:** `pytest -q`, `pytest tests/integration`, and `scripts/wakespot --help` (exit 0, same text as base). All pass.
- [ ] **Step 4:** Commit `spot: one main wiring services, features, events; model factories; NullVision`.

### Task 16: Registry cut-over (generated grammar, sets, prompt, intent table)

**Files:**
- Create: `src/spot/brain/grammar/generate.py`
- Modify: `src/spot/brain/intent.py`, `src/spot/brain/llm_brain.py`, `src/spot/robot/dispatcher.py`, the `src/spot/conversation/*` set users, `src/spot/operator/control.py` (spotctl help)
- Test: `tests/spot/test_registry_identity.py`

**Interfaces — Produces:**
- `build_grammar(registry) -> str`. It replaces `action-name ::=` and the params rules in a template `spot_action.gbnf.tmpl`. The template is the current file with those two regions marked `{{ACTION_NAMES}}` and `{{PARAMS}}`.
- The current `spot_action.gbnf` is kept in git as `tests/spot/golden/spot_action.gbnf` (byte copy).

- [ ] **Step 1:** Write the identity tests, all with every feature enabled.
  - `test_grammar_identical`: `build_grammar(REGISTRY)` equals the golden GBNF after whitespace normalisation (collapse runs of spaces, strip trailing whitespace).
  - `test_sets_identical`: each generated set equals the golden literal values copied from the base into the test.
  - `test_prompt_section_identical`: byte-identical to the base prompt section (golden captured on base).
  - `test_intent_table_identical`: same rules, same order.
  - `test_disabling_dance_removes_emote_everywhere`: grammar, prompt, sets and intent rules.
- [ ] **Step 2:** Run. Expected: the identity tests FAIL until generation exists.
- [ ] **Step 3:** Before deleting anything, check how the grammar reaches llama-server.
  - Run: `grep -rn "spot_action.gbnf\|gbnf" src scripts config *.service scripts/*.service scripts/*.sh 2>/dev/null`
  - If any non-Python consumer reads the file by path (service unit, shell script, docker config), keep a generated `spot_action.gbnf` written at startup to the same path, instead of deleting it.
- [ ] **Step 3b:** Implement generation, then switch every consumer from the hand-written constant to `REGISTRY.names(flag)` / generated text. Delete the hand-written constants, `LEGACY_FEATURE_DISPATCH`, and the now-empty `spot_dispatch._dispatch_core` feature chain. Delete `spot_action.gbnf` from `src` only if Step 3 found no path consumer (the golden copy stays in tests).
- [ ] **Step 4:** `pytest -q` and `pytest tests/integration`. PASS.
- [ ] **Step 5:** Commit `spot: grammar, intent table, intent sets and prompt generated from features`.

**Wave 5 gate:**
- Safety-reviewer:
  - the generated `SAFETY_INTENTS`, `ABORTING_INTENTS`, `_UNLOCKED_INTENTS` and `SHORT_PATH_BLOCKED_INTENTS` are identical;
  - core safety intents can't be removed by disabling a feature (`stop`/`freeze`/`estop` are declared in core, not in any feature).
- Golden passes.

### Task 17: Scripts port; delete four

**Files:** Modify every `scripts/*.py` that imports `src.voice_control` or `src.session|estop|graph_nav_utils|map_loader|location_manager|config` (27 scripts per the builder check), plus `scripts/training/**`. Delete `scripts/preflight_2a.py`, `audition_personas.py`, `audition_kokoro_voices.py`, `save_self_intro.py`. Test: `tests/spot/test_scripts_import.py`.

- [ ] **Step 1:** Write `test_every_script_imports`. For each remaining `scripts/*.py`, run `python -c "import runpy; runpy.run_path(p, run_name='not_main')"` in a subprocess with a 30 s timeout; it must exit 0.
  - Scripts that touch hardware at import time are listed in the test with a reason, and get `--help` instead.
  - Also add `test_tests_record_demo_clips_still_pass` (existing test file unchanged).
- [ ] **Step 2:** Rewrite the imports to `src.spot.*`.
- [ ] **Step 3:** `pytest -q` passes. Write the list of robot/mic scripts needing an on-robot check into the ledger: `spike_gripper_rate`, `try_walk_pose`, `test_visual_nav`, `preflight_2c_capture`, `xvf_probe`, `mic_verify`, `diag_*`, `estop_run`, `record_demo_clips`.
- [ ] **Step 4:** Commit `scripts: import from src.spot; drop four desk-only one-offs`.

### Task 18: Docs

**Files:** `docs/architecture/pipeline-explainer.md`, `docs/getting-started/daily-operations.md`, `docs/scripts/reference.md`, `docs/project/status.md`; create `docs/architecture/package-map.md`.

- [ ] **Step 1:** Write `package-map.md`:
  - one table, package → what it owns → its interface → how to swap or add;
  - a "how to add a feature" checklist: folder, `FEATURE`, one line in `enabled.py`, done;
  - a "how to remove a feature" checklist.
- [ ] **Step 2:** Update the paths in the other docs. `grep -rn "voice_control/" docs/architecture docs/getting-started docs/scripts`: Expected: only historical mentions marked as such.
- [ ] **Step 3:** Commit `docs: package map and updated paths`.

### Task 19: Remove shims (after T17 and T18 are merged)

**Files:** Delete `src/voice_control/**` except the two script shims `client_mic.py` and `server.py` (kept for `run_voice_control.py` and `pgrep` compatibility), plus `src/session.py`, `src/estop.py`, `src/graph_nav_utils.py`, `src/map_loader.py`, `src/location_manager.py`, `src/config.py`. Modify all `tests/**` imports to `src.spot.*`. Delete the `tests/spot/test_aliases_*.py`.

- [ ] **Step 1:** Write `tests/spot/test_split_shims.py::test_split_shims_hold_no_state`. The two remaining script shims define no module-level names other than imports and `__main__` handling (AST check: only `Import`, `ImportFrom`, `If __name__`, `Expr` docstring).
- [ ] **Step 2:** Rewrite test imports. Assertion lines stay unchanged (grep check as in T9). Delete the shims.
- [ ] **Step 3:** Verify.
  - `pytest -q`: count ≥ BASE_COUNT + all new tests − alias tests.
  - `grep -rn "src.voice_control\|from src import session" src scripts tests`: Expected: only the two script shims and their docstrings.
- [ ] **Step 4:** Commit `spot: remove compatibility shims; src/voice_control reduced to two launch stubs`.

### Task 20: Final verification (controller)

- [ ] **Step 1:** `pytest -q` (all pass), then `pytest tests/integration` (golden PASS), then `python tests/audio/replay_demo.py`. Its numbers must equal `docs/project/demo-tuning-results.md`: wake hits, gate pass/fail, voiceprint sims, safety-KWS hits.
- [ ] **Step 2:** Safety-reviewer on the whole branch diff against base, using the Wave 3 gate checklist plus T16's.
- [ ] **Step 3:** `/code-review high` on the branch. Fix confirmed findings through a scoped lane plus re-review.
- [ ] **Step 4:** `scripts/wakespot --dry-run` starts and shuts down cleanly. Its startup status lines match the base (diff of the two logs, ignoring timestamps).
- [ ] **Step 5:** Operator runs the on-robot checklist, then the Task 17 script list:
  - startup: no overflow; `safety_kws_ok`, `mic_ok`;
  - stops: normal, quiet, 3 m;
  - arm fold, gaze log line, walk with mouth, stop stows;
  - quiet one-word commands;
  - noise check, ~20 noises → `[Short] dropped`;
  - dances at 1 m;
  - voiceprint sims;
  - mic reconnect.
- [ ] **Step 6:** Tag `restructure-verified-<date>`. **Ask the operator** before merging into `tour_guide_upgrade_matteo`.
