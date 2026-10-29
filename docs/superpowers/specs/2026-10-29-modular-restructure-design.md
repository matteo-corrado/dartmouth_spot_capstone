# Modular restructure — design

Date: 2026-10-29. Branch: `restructure-modular` (cut from `demo-poster-2026-10-29` @ 81fedfe).
Status: draft for operator approval.

## 1. Intent

The code works, but it is hard to read and hard to change. Two files hold a third of
the code: `src/voice_control/client_mic.py` (2,936 lines) and
`src/voice_control/spot_dispatch.py` (2,576 lines). Each mixes several services
(mic capture, voice gate, speaker lock, safety listener, LED ring, conversation loop;
every robot action plus arm, gaze and mouth). Finding which service owns a function
means reading the whole file.

The operator wants (their words, paraphrased):

- **Modular packages per service** (vision, mic, STT, TTS, ...) so each piece can be
  changed without breaking the pipeline.
- **One `main`** that brings the pipeline together.
- **Easy to expand or contract**: adding or removing a model or feature should break as
  little as possible and need as few reference edits as possible.
- **Identical behaviour first**; upgrades (vision etc.) come after the restructure.
- **Separate branch** from the demo, so the demo upgrades survive if this fails.

Why now: the operator is about to start a vision upgrade (not implemented yet) and
wants it to land in a clean structure instead of growing the two big files further.

## 2. Goals, non-goals, anti-goals

Goals
1. Every function lives in a package named for the service that owns it.
2. Core services are plain packages with one small interface each, wired explicitly
   in one composition file (`main`). Reading `main` top to bottom shows the pipeline.
3. Swapping a model (STT engine, TTS engine, wake engine, LLM backend, vision model) =
   add one implementation module + change one config value.
4. Optional features (dance/emotes, mouth, gaze, walking, navigation, follow, visual
   nav, persona/voice settings, info queries) are self-contained folders. Adding one =
   new folder + one line in the enabled list. Removing one = delete the folder + that
   line. No other file names it.
5. Everything that today is a hand-maintained cross-cutting list is generated from the
   loaded features (see section 6).
6. A clear place for vision to plug in (interface + null implementation), with no
   vision behaviour added.

Non-goals
- ROS 2 (considered and rejected 2026-10-29: SDK frame helpers already cover
  transforms; latency and migration cost not justified for a solo project).
- New features, tuning changes, algorithm rewrites, bug fixes. Bugs found during the
  move are logged in `docs/project/bugs.md`, not fixed in this branch.
- Splitting into multiple processes (beyond the ASR server, which is already its own
  process). Safety-as-a-process is a possible later step, not this one.
- Changing dependencies or pins.

Anti-goals (failure even if it "works")
- Any change in robot behaviour, especially the stop / freeze / e-stop paths, their
  ordering, or their latency.
- A plug-in system so indirect that "where is this called from?" gets harder.
  No auto-discovery, no import-time magic: features are listed explicitly by name.
- Breaking the operator's commands: `wakespot`, `spotctl`, the web panel, `estop_run`,
  the systemd units keep identical CLIs and flags.
- Losing the demo state: `demo-poster-2026-10-29` and `tour_guide_upgrade_matteo` are
  never modified by this work.

## 3. Constraints

- Hardware/runtime: Jetson AGX Orin, Ubuntu 22.04, Python 3.10, `spot-env` venv,
  bosdyn SDK 5.2.0, robot OS 5.2.1. Single process for the voice pipeline; ASR gRPC
  server (port 50055) stays a separate process.
- Threading model is preserved exactly: audio callback thread, safety KWS thread,
  mouth worker, gaze worker, nav/follow threads, auto-arm thread, control server
  thread, TTS/AudioPlayer worker. Nothing moves between threads.
- Project rules that stay in force: never SAVE_CONFIGURATION / REBOOT /
  set_configuration / detach / reset the XVF3800; voiceprints memory-only; demo
  recordings stay local and gitignored; one-line commit messages, no trailers;
  safety-reviewer on any robot-control or safety code; pin-guardian on requirements
  changes; don't query the mic while the pipeline runs.
- Sensitive-file hook: Bash commands that mention the KWS keyword-file names (and
  some other words) are rewritten. File moves touching `wake/` must be done with tools
  that are not rewritten and confirmed with `git status`.
- Timing: the demo is 2026-10-29. Implementation starts only after the demo is tagged
  (`demo-ready-2026-10-29` or the final demo HEAD). Before starting, merge that tag into
  `restructure-modular` so no demo-day fix is stranded.

## 4. Decided structure

### 4.1 Two kinds of module (operator agreed 2026-10-29)

- **Services** (core, always present): plain packages. One interface (a `typing.Protocol`)
  per swappable service, a small factory that picks the implementation from config,
  explicit wiring in `main`.
- **Features** (optional): one folder each, registered through an explicit list.
  The registry exists only for things that genuinely multiply (robot actions/intents and
  their side tables). Everything else is direct calls through interfaces.
- **Events**: a small in-process bus only for cross-cutting signals (see 4.4).

### 4.2 Package layout (my call; names may be adjusted by the builder if a clash appears)

New root package `src/spot/` (the old `src/voice_control/` disappears at the end):

| Package | Owns (moved from) |
|---|---|
| `spot/app/` | `main.py` composition root, CLI args, shutdown sequence (from `client_mic.main`, `_shutdown_*`, `scripts/run_voice_control.py` internals) |
| `spot/core/` | event bus, shared state objects (6.x), `tuning`, `config`, `latency`, `startup_status`, `voice_state` |
| `spot/audio/` | input stream open/retry, audio callback, endpointer/VAD, noise floor, chime mask (`client_mic` 116–870 region), `audio_frontend`, `audio_player`, `audio_feedback` (chimes) |
| `spot/mic/` | XVF3800 control + reconnect (`mic_array`), DoA, LED ring (`ring_animator`, ring parts of `state_feedback` and `client_mic` 1120–1240), beam steering |
| `spot/wake/` | wake KWS engines (`wake/sherpa_onnx`, `wake/livekit`), phrases |
| `spot/safety/` | always-on safety KWS (`wake/safety_kws`, `client_mic.safety_kws_loop` + `_on_safety_kw` … `_stop_safety_thread`), transcript safety check, `rescue_safety`, halt/hush, safety generation counter, `src/estop.py` |
| `spot/stt/` | `SpeechToText` interface, gRPC client (`send_to_asr`), ASR server (`server.py`), backends (`asr/parakeet`, `asr/nemotron`), proto + generated stubs |
| `spot/tts/` | `TextToSpeech` interface, Kokoro, ElevenLabs, fallback logic (`spot_tts`, `tts/`), `text_segment` chunker |
| `spot/conversation/` | the conversation loop (the closures inside `client_mic.main`, 1364–2230, become a `Conversation` class), voice gate, short-utterance path, speaker lock (`speaker_lock`), voiceprint (`speaker_embed`), barge-in, `response_gate`, thinking filler, utterance log, `chatbot/conversation_log`, `chatbot/session_state` |
| `spot/brain/` | `Brain` interface, regex intent parser (`intent.py`), LLM brain (`llm_brain`), backends (`brain/llamacpp_backend`, `ollama_backend`), persona (`chatbot/persona`, `voice_discovery`), knowledge files, grammar *generation* |
| `spot/robot/` | Spot session/lease (`src/session.py`, `ensure_spot_session`, `close_spot_session`), dispatcher core (`dispatch_intent`, arm policy, motion abort, `_send_stop`, stow, restore-after-motion), robot state queries, frame capture, core actions (6.2) |
| `spot/nav/` | GraphNav plumbing shared by features: `graph_nav_utils`, `map_loader`, `location_manager`, localisation helpers from `spot_dispatch` 575–825 |
| `spot/perception/` | existing `perception/*`, `world_objects`, and a `Vision` interface with a `NullVision` implementation (no behaviour) |
| `spot/features/<name>/` | optional features (6.2) |
| `spot/operator/` | `control_server`, status snapshot, immediate commands (`_on_immediate`), web panel backend; `scripts/spotctl`, `scripts/wakespot`, `scripts/web_panel.py` stay as thin entry points with identical CLIs |

Import layout fix (structural, allowed): the bare `import asr_pb2` / `sys.path`
bootstrap in `client_mic.py` and `server.py` is replaced by normal package imports.

### 4.3 Interfaces (my call on exact signatures; builder refines)

Each is a `Protocol` in the package's `__init__` or `api.py`, with the current
implementation behind it, chosen by a factory reading config/env exactly as today
(e.g. `SPOT_ASR_BACKEND`, `SPOT_BRAIN_BACKEND`, TTS provider selection):

- `stt.SpeechToText.transcribe(pcm16: bytes, timeout) -> str`
- `tts.TextToSpeech.speak(text, voice=None, cue=True)` plus the existing streaming/chunk hooks
- `wake.WakeDetector` (current sherpa detector surface)
- `brain.Brain` (current `LLMBrain` public surface + regex fallback)
- `perception.Vision` (people/objects query surface; `NullVision` returns nothing)
- `robot.Robot` (dispatch, state, halt) — single owner of the lease

### 4.4 Event bus (my call)

- In-process, synchronous publish, handlers called in subscription order.
- Each handler wrapped in try/except: a failing or missing subscriber never breaks the
  publisher.
- **Never on the stop path.** Stop/freeze/e-stop go safety -> robot by direct call,
  exactly as today (lock-free halt first, then hush on its own thread, then the
  safety-generation bump). Events about a stop may be published *after* the halt.
- Initial events (only those needed to remove an existing cross-module reference):
  `speaking_started`, `speaking_stopped`, `voice_state_changed`, `speaker_locked`,
  `speaker_released`, `robot_stood`, `robot_sat`, `safety_halt` (post-halt notice),
  `person_seen` (reserved for vision, no publisher yet).

## 5. Shared state: owners

Today these are module globals. Each moves to exactly one owner object; other
packages get a reference passed in by `main` (no new globals, no singletons reached
by import except where a test proves identical behaviour needs it).

| State (current location) | Owner |
|---|---|
| `_safety_gen`, `_dispatch_gen`, `current_safety_gen` (dispatch) | `safety.SafetyGen` (shared by robot + features) |
| `_voice`, `_status`, `_state_leds`, `_enter_state` (client_mic) | `conversation.Conversation` + `core.voice_state` |
| `_lock` (speaker lock), voiceprint mode/threshold | `conversation` |
| `_mic`, `_mic_gated`, `_mic_reopen_at`, `_mic_dead_logged`, `_player_last_busy_at` | `audio` / `mic` |
| `_ring_*`, `_anim`, `_reject_until` | `mic.ring` |
| `_safety_q/_stop/_active/_thread/_health`, `_last_safety_at` | `safety` |
| `_operator_muted`, `_crowd_mode`, `_abort_chain` | `conversation` (set via `operator`) |
| `_audio_player` | `audio` |
| `_spot_session`, `_session_context`, `_session_lock`, `_closing` | `robot.Session` |
| `_arm_pose`, `_pose_before_walk`, `_pose_during_walk`, `_arm_lock`, `_arm_settle_until`, `_last_traj` | `robot.arm` |
| `_dispatch_lock`, `_last_brain`, `_talk_resume`, `_mouth_resume` | `robot` dispatcher |
| `_nav_thread/_stop_event/_mode/_lock`, `_home_waypoint` | `nav` |
| `_auto_arm*`, `_gaze_deg/_t`, `_gaze_q/_worker` | `features/gaze` + `features/mouth` (auto-arm = mouth feature's startup hook) |
| `_last_routine` | `features/dance` |
| `_brain_ok`, `_store_transcript`, `_speaking_hook` | `conversation` |
| `_tuning` | `core.tuning` (unchanged `get_tuning()`) |

The builder must re-check this table against the code (it was made from a grep of
module-level assignments) and add anything missed.

## 6. Features and the registry

### 6.1 Feature contract (my call)

Each `features/<name>/__init__.py` exposes one `FEATURE` object declaring:

- `actions`: action name -> `ActionSpec` (handler, GBNF params fragment, LLM prompt
  line, flags: `locomotion`, `aborting`, `unlocked`, `short_path_blocked`, `info`,
  `state_only`, `motion_free`, arm-policy entry, needs-session)
- `intent_patterns`: regex rules for the regex parser (ordered; global order is kept
  identical to today by an explicit priority value)
- `spotctl` commands / help text (if any)
- `subscriptions`: event -> handler
- `start(ctx)` / `stop(ctx)` hooks (e.g. auto-arm after stand, gaze worker)
- `requires`: other features or services it needs (load fails clearly if missing)

`spot/features/enabled.py` (or a `features:` list in config — builder's call, but
explicit names either way) lists what loads. A feature that fails to import logs a
warning and is skipped; a **core** module that fails stops startup, as today.

### 6.2 Core vs feature split of today's 35 intents

Core (in `robot` / `safety`, cannot be disabled): `stop`, `freeze`, `estop`, `stand`,
`sit`, `power_off`, `stow`, `selfright`, `status`, `battery_status`.

Features:
- `dance`: `emote` (+ clearance scan use)
- `mouth`: `enable_mouth`, `disable_mouth`, auto-arm
- `gaze`: speaker gaze (no intent; subscribes to `speaker_locked`, standing state)
- `walk`: `walk`, `turn`, `strafe`, `body_height`, `set_speed`
- `navigation`: `walk_to`, `go_to`, `tour`, `patrol`, `come_back`, `load_map`,
  `save_location`, `list_maps`, `list_locations`
- `follow`: `follow_me`
- `visual_nav`: `go_to_object`, `check_obstacles`
- `settings`: `set_persona`, `add_persona`, `set_volume`, `set_backend`

`describe` (camera description) goes in `visual_nav`. `shutdown` is core, an alias of `power_off`.
(The builder confirms the list against `_dispatch_core` and the GBNF; any intent found
there and missing here goes into the matching feature.)

### 6.3 Generated, not hand-maintained

Built from core + enabled features at startup:
- GBNF action-name rule and params rules (today `grammar/spot_action.gbnf`)
- LLM prompt action section (today in `llm_brain`, e.g. the `emote` section)
- regex intent table (today `intent.py`)
- `LOCOMOTION_INTENTS`, `SAFETY_INTENTS`, `SHUTDOWN_INTENTS`, `ABORTING_INTENTS`,
  `_UNLOCKED_INTENTS`, `INFO_INTENTS`, `SHORT_PATH_BLOCKED_INTENTS`,
  `_STATE_ONLY_INTENTS`, `_PERSONA_INTENTS`, `_MOTION_FREE_INTENTS`, `arm_policy_for`
- `spotctl` help text

**Identity requirement:** with all features enabled, every generated artifact must be
equal to today's hand-written version (grammar text compared after whitespace
normalisation; sets compared exactly; regex table compared rule-by-rule in order). A
test enforces this. Disabling a feature must remove it from all of the above.

## 7. Migration strategy

Strangler pattern, so the suite is green at every merge:

1. **Skeleton + shims.** Create `src/spot/` packages, event bus, state owners, registry
   (empty). No code moved.
2. **Move leaf modules** (no dependencies on the big files): tuning, audio_player,
   chimes, ring animator, mic_array, wake engines, TTS, ASR backends, perception.
   Old paths become one-line re-export shims (`from spot.tts.kokoro import *` style,
   plus explicit re-export of private names tests patch).
3. **Split `spot_dispatch.py`** into `robot` core + `nav` + features, building the
   registry; add the identity test. Old module becomes a shim exposing the same names.
4. **Split `client_mic.py`** into `audio`, `mic.ring`, `safety`, `conversation`,
   `operator`, `app.main`. The `main()` closures become methods of `Conversation`.
   Old module becomes a shim.
5. **Port tests** package by package to the new paths (patch targets move to the
   owner objects).
6. **Delete shims** and `src/voice_control/` once nothing imports them.

Parallelism: step 1 is one lane. Step 2 parallelises by module. Steps 3 and 4 are one
lane each (they can run in parallel with each other once step 2 is merged, since
their files are disjoint). Steps 5–6 follow. Safety review gates steps 3, 4 and the
final `main` wiring.

### 7.1 Tests during the move (my call, after builder check)

Tests set module globals through monkeypatch about 260 times (about 141 on `client_mic`, about 123 on `spot_dispatch`). Once state moves to owner objects, setting a global on a shim silently checks nothing. So:

- **Tests port in the same merge as the code they cover.**
- **A port may change only import paths and patch targets.** Assertion lines and expected values stay byte-identical. The reviewer checks the test diff for exactly that.
- **No test is deleted or skipped** during the move.
- **Where one old module maps to one new module**, the shim may alias the old name to the new module in `sys.modules`. Patching then still reaches the real code. Split modules don't get this alias.

### 7.1b Arm, mouth and feature boundary (operator decided 2026-10-29)

- **These stay in the robot core:**
  - the arm poses (stowed / talk / walk);
  - `arm_policy_for`, including the talk-is-walk-pose branch;
  - the talk/mouth resume after a safety halt;
  - close-mouth-before-tuck/stow;
  - `get_arm_pose` and the arm settle time.
- **The `mouth` feature owns** only the jaw motion (`MouthDriver`, the envelope, enable/disable intents) and auto-arm after stand.
- **With mouth disabled**, nothing ever puts the arm in the talk pose, so it stays stowed. The core paths are unchanged.
- **Same rule for `gaze` and `dance`:** core keeps the arm, stow and stop plumbing. The feature owns only its own behaviour (the gaze worker; the routines plus the clearance check call). No new hooks on the stop/stow path.

### 7.1c Scripts (operator decided 2026-10-29)

Rule: anything that runs on or against the robot or the mic is kept and ported.

- **Port:**
  - operator tools: `wakespot`, `spotctl`, `web_panel`, `estop_run`, the systemd unit;
  - setup: `setup_*`, including `prewarm_personas`, which is one-time setup that seeds `personas.yaml` from ElevenLabs;
  - recording: `record_demo_clips`, `record_corpus`, `chime_preview`;
  - mic tools: `xvf_probe`, `mic_verify`;
  - diagnostics and maps: `diag_*`, `map_waypoints`, `download_map_from_spot`, `setup_map`;
  - robot tests: `spike_gripper_rate`, `try_walk_pose`, `test_visual_nav`, `preflight_2c_capture` (useful for the vision upgrade);
  - `scripts/training`.
- **Delete** (desk-only one-offs from finished stages): `preflight_2a`, `audition_personas`, `audition_kokoro_voices`, `save_self_intro`.
- **Robot-touching scripts can't be run in CI.** Each must at least import cleanly and run `--help`, and the builder lists which ones need an on-robot check.

### 7.2 Base and concurrent changes (my call)

- **The base is the demo tag.** The restructure merges it once before step 1.
- **Freeze rule:** while steps 3–4 are in flight, the operator makes no further changes to `client_mic.py` or `spot_dispatch.py` on other branches. A fix that can't wait goes into `restructure-modular` by hand, as its own one-line commit.
- **Order:** split `spot_dispatch` and `client_mic` early. Leaf moves can run alongside.

## 8. Verification gates

1. Full test suite green at every merge into `restructure-modular` (615 at the base;
   count may only grow).
2. Replay harness (`tests/audio/replay_demo.py`, `tests/voice_control/test_replay_demo.py`)
   reproduces the numbers recorded in `docs/project/demo-tuning-results.md` (wake hits,
   gate pass/fail, voiceprint sims, safety-KWS hits) on the local recordings.
3. Registry identity test (6.3) passes, including a byte-identical generated LLM
   prompt action section.
3b. `replay_demo.py` copies the loop rather than running it, so it doesn't cover the
   conversation refactor. Add an integration check: drive `Conversation` with the
   recorded clips in dry-run (fake robot, fake TTS). Record its decisions on the base
   commit, and require the same decisions after the split.
4. Safety-reviewer sign-off on `safety`, `robot`, features with motion, and `app.main`
   wiring; specifically that stop/freeze/e-stop ordering, lock-free halt, safety-gen
   bump and hush-on-thread are unchanged.
5. Startup smoke: `wakespot --dry-run` (and the robot-free paths of `spotctl`) start
   and shut down cleanly with identical log lines for status/health.
6. On-robot checklist (the 2026-10-29 test-round-3 list: startup, stops normal/quiet/far,
   arm fold + gaze + mouth, one-word commands, noise check, dances at 1 m, voiceprint,
   mic reconnect) before anything merges toward `tour_guide_upgrade_matteo`. Merging
   there needs the operator's explicit OK.

## 9. Left to the builder

- Exact Protocol method signatures and file names inside each package.
- Whether the enabled-features list lives in Python or `config/`.
- Shim mechanics, order of leaf moves, how private names are re-exported.
- Whether `spot/mic` and `spot/audio` merge into one package if the split proves
  awkward (the rule is "owner obvious from the name", not a fixed count).
- Where docs (`pipeline-explainer.md`, `daily-operations.md`) need path updates; they
  must be updated in step 6.
