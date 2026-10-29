# Package Map

Where everything lives after the 2026-10 restructure. Code is under `src/spot/`. Each folder owns one job and exposes one small interface, so you can change or swap one part without reading the rest.

Status note: the split is done. `src/voice_control/client_mic.py` and `src/voice_control/server.py` stay as script shims (`scripts/run_voice_control.py` launches them, and its `pgrep` check matches `voice_control/server.py`); they forward to `src/spot/app/main.py` and `src/spot/stt/server.py`. Every other old path in `src/voice_control/` (including `spot_dispatch.py`) is a thin alias that re-exports the new location and is removed later.

## Packages

| Package | Owns | Interface | Swap or add |
|---|---|---|---|
| `core/` | The event bus, the feature registry, tuning values, latency, startup status, voice state | `EventBus.subscribe/publish`, `Registry`, `Tuning` | Add an event name to `EVENTS` in `core/events.py`. Tuning keys live in `config/tuning.yaml` |
| `audio/` | Playback, chimes, echo/frontend processing, barge-in, response gate, mic capture | `AudioPlayer`, `chimes`; `AudioCapture` (`capture.py`) | Replace a stage by editing that one file. New sound: add to `chimes.py` |
| `mic/` | The XVF3800 mic array: LED ring, ring animation, state feedback, Spot's LEDs | `xvf3800`, `ring_animator`, `state_feedback`, `spot_leds`; `Ring` (`ring_control.py`) | New mic hardware: write a new `xvf3800`-style module with the same calls |
| `wake/` | Wake word ("Hey Spot") | Detector with `process_frame`; `WakeDetector` protocol (`api.py`) and `make_wake()` (`factory.py`) | New engine: add a file next to `sherpa_onnx.py` and `livekit.py`, pick it by `SPOT_WAKE_BACKEND` |
| `safety/` | Stop/freeze/e-stop: safety keyword spotter, safety generation counter, e-stop, the safety listener | `SafetyGen.bump/aborted`, `kws.py`, `estop.py`; `SafetyListener` (`listener.py`) | Do not add a bus hop here. See the safety note below |
| `stt/` | Speech to text: the ASR gRPC server and its backends (Parakeet, Nemotron) | `SpeechToText.transcribe(pcm16, timeout)` and `make_stt()`; server in `stt/server.py` | New model: add `stt/backends/<name>.py`, choose with `SPOT_ASR_BACKEND` |
| `tts/` | Text to speech: Kokoro (local), ElevenLabs (cloud), the speaker, sentence chunker | `TextToSpeech.speak(text, voice, cue)` and `make_tts()` | New voice engine: add a file and register it in `tts/__init__.py`, choose with `SPOT_TTS_BACKEND` |
| `conversation/` | Turn-taking: speaker lock, voiceprint, session state, conversation log; the gate, filler and main loop (`gate.py`, `filler.py`, `loop.py`) | `Conversation.run()` and `.process_utterance()` (`loop.py`) | Change who Spot listens to in `speaker_lock.py` / `voiceprint.py` |
| `brain/` | Intent parsing (regex), the LLM brain and its backends, personas, grammar, knowledge packs | `LLMBrain`, `intent.parse_intent`; `Brain` protocol and `make_brain()` | New LLM server: add `brain/backends/<name>_backend.py`, choose with `SPOT_BRAIN_BACKEND`. Event facts go in `brain/knowledge/` |
| `robot/` | Everything that talks to Spot: connection, arm, motion, state, the dispatcher | `CONNECTION`, `ARM`, `SAFETY_GEN`, `NAV`; `dispatcher.dispatch_intent` / `dispatch_safety` | New core command: add a handler in `dispatcher.py`. New optional behaviour: make a feature instead |
| `nav/` | GraphNav: the navigation runner, map loading, saved locations, GraphNav helpers | `NavRunner.spawn/cancel`, `location_manager`, `map_loader` | New map source: change `map_loader.py` |
| `perception/` | Cameras and what Spot sees: intrinsics, world objects, obstacle queries, ray casting | Helper functions; `Vision.people()/objects()` and `NullVision` (`vision.py`); `make_vision()` in `factory.py` | New detector: implement `Vision` and return it from the factory |
| `operator/` | The operator control server (mute, status, `spotctl`) | `control_server.py`; `control.py` (the status snapshot and immediate-command handlers) | New operator command: add it in the control module |
| `features/` | Optional robot abilities, one folder each (dance, gaze, mouth, visual_nav, ...) | A `FEATURE` declaration per folder; `enabled.py` lists which are on | See the checklists below |
| `app/` | The single wiring point | `app/main.py` `main(argv)` | Rarely. Only to add a new service to the startup order |

## How the pipeline fits together

Microphone audio comes in through `audio/` and is watched by two things at once: the wake word detector (`wake/`) and the always-on safety keyword spotter (`safety/`). When Spot is addressed, the utterance goes to `stt/` for text, then to `brain/` which turns the text into one or more actions (regex rules first, LLM for the rest). Each action goes to `robot/dispatcher.py`. Core actions (stand, sit, stop, ...) are handled there; everything else is handled by a feature registered in `features/`. The reply is spoken by `tts/` through `audio/`.

`src/spot/app/main.py` is the one place that builds and connects these pieces, in a fixed 9-step order: tuning and CLI, event bus, robot owners, features, audio (capture, ring, wake, safety listener), services (stt, tts, brain, vision) with robot bring-up, conversation, control server, then `Conversation.run()`. If you want to know what runs at startup, read `main()` top to bottom. Nothing else wires services together.

Parts talk in two ways. Direct calls for commands ("do this now"). The event bus (`core/events.py`) for announcements ("this happened"), such as `speaking_started`, `robot_stood`, `speaker_locked`. Publishers do not know who listens, and a listener that crashes is logged and ignored.

## Safety note

Stop, freeze and e-stop never go through the event bus. The listener calls the halt directly, and the halt happens first. An event such as `safety_halt` is published only afterwards, for information. If a bus handler is slow or raises, the robot has already stopped. Do not move a stop path onto the bus, and do not make a halt wait for anything.

## How to add a feature

A feature is one folder under `src/spot/features/`. Example: a `wave` feature.

1. Make the folder `src/spot/features/wave/` with `__init__.py` exporting `FEATURE = Feature(name="wave", actions=(ActionSpec(...),), intent_rules=(...), ...)`, and a `handlers.py` next to it. Each `ActionSpec` names the action, its handler `(params, ctx) -> bool` (from `handlers.py`), a `prompt_line` (the line the LLM sees; leave it `None` to keep the action regex-only), grammar and prompt ranks, and flags such as `locomotion` or `info`. The `Ctx` object (`features/ctx.py`) gives the handler the connection, arm, nav, speak and the event bus. Copy `features/walk/` or `features/dance/` as a starting point.
2. Add one line, `"wave",`, to `ENABLED` in `src/spot/features/enabled.py`.
3. `Registry.add` rejects a bad declaration at load: an action named like a core intent, `aborting=True`, `unlocked` without `motion_free`, or `locomotion` combined with `unlocked`, `motion_free` or `state_only`. Core intents (stop, freeze, estop, stand, sit, ...) live only in `core/core_intents.py`.
4. Nothing else to edit. The LLM grammar, the prompt action list, the regex intent table and the intent sets (`intent_names("locomotion")` and friends) are generated from the core intents plus the enabled features (`brain/grammar/generate.py`, `core/core_intents.py`). `brain/grammar/spot_action.gbnf` is a committed snapshot of the generated grammar, not a file to edit by hand.
5. See `tests/spot/test_registry_identity.py` for how the generated output is checked against golden copies, and add a `tests/spot/features/test_<name>_declared.py` like the existing ones.

If a feature fails to import, the registry logs `[Features] <name> skipped` and the rest still starts; `main` prints the failure loudly at startup. All current features (dance, gaze, mouth, settings, walk, follow, navigation, visual_nav) are in this shape.

## How to remove a feature

1. Delete its name from `src/spot/features/enabled.py`. That alone turns it off: its actions disappear from the grammar, prompt, intent rules and spotctl help. Nothing hand-written needs removing.
2. Optionally delete the folder.
3. Check that no other feature lists it in `requires`.
4. Run the tests.

## Old to new path cheat sheet

| Old (`src/voice_control/`) | New (`src/spot/`) |
|---|---|
| `audio_player`, `audio_feedback`, `audio_frontend`, `barge_in`, `response_gate` | `audio/{player,chimes,frontend,barge_in,response_gate}` |
| `mic_array`, `ring_animator`, `state_feedback`, `spot_leds` | `mic/{xvf3800,ring_animator,state_feedback,spot_leds}` |
| `wake/*` | `wake/*`; `wake/safety_kws` is `safety/kws` |
| `asr/*`, `server.py` | `stt/backends/*`, `stt/server.py` |
| `tts/*`, `spot_tts`, `text_segment` | `tts/*`; `spot_tts` is `tts/speaker` |
| `intent`, `llm_brain`, `brain/*`, `chatbot/persona`, `grammar/`, `knowledge/` | `brain/*` |
| `chatbot/conversation_log`, `chatbot/session_state`, `speaker_lock`, `speaker_embed` | `conversation/{conversation_log,session_state,speaker_lock,voiceprint}` |
| `src/session.py`, `src/estop.py` | `robot/session`, `safety/estop` |
| `src/graph_nav_utils`, `map_loader`, `location_manager` | `nav/*` |
| `perception/*`, `world_objects` | `perception/*` |
| `visual_nav`, `emotes`, `gaze`, `animation/mouth` | `features/visual_nav/navigator`, `features/dance/emotes`, `features/gaze/math`, `features/mouth/driver` |
| `tuning`, `latency`, `startup_status`, `voice_state` | `core/*` |
| `control_server` | `operator/control_server` |
| `spot_dispatch` | split into `robot/{state,arm,motion,dispatcher,connection}`, `nav/runner`, `safety/gen` and `features/*`; the old file only re-exports |
| `client_mic` | `audio/capture`, `mic/ring_control`, `safety/listener`, `operator/control`, `conversation/*` (`gate`, `filler`, `loop`), `app/main`; `client_mic.py` stays as a script shim |
