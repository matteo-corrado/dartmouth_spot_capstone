# Index

One-page map of the repo. For details per package see `docs/architecture/package-map.md`.

## Start here

| To... | Run / read |
|---|---|
| Start Spot (E-stop + voice) | `scripts/wakespot.py` (`--dry-run` for no robot) |
| Operator commands while running | `scripts/spotctl.py` (mute, release, stop, freeze, stow, status, persona...) |
| See what runs at startup | `src/spot/app/main.py` `main()`, top to bottom |
| Add or remove a robot ability | `src/spot/features/<name>/` + one line in `src/spot/features/enabled.py` |
| Open bugs | `docs/project/bugs.md` |

## Top level

| Folder | What's in it |
|---|---|
| `src/spot/` | All runtime code (packages below) |
| `src/voice_control/` | Launch shims only (`client_mic.py`, `server.py`) + `grammar` symlink for the old systemd unit |
| `src/config.py` | Robot IP and login, read from the env file |
| `config/` | `tuning.yaml` (thresholds, speeds), `personas.yaml`, `wake_phrases.txt` |
| `configs/wake/` | Wake-word training configs |
| `models/` | Downloaded models: KWS, TTS, speaker (symlink, not in git) |
| `scripts/` | Launchers (`wakespot`, `spotctl`, `run_voice_control`), setup (`setup_*`), diagnostics, map tools, recording |
| `systemd/` | `llama-server.service` (LLM server) |
| `tests/` | `spot/` unit tests, `integration/` golden conversation replay, `audio/` lab clips + eval scripts. `robot_guard.py` blocks tests from contacting Spot |
| `docs/` | `architecture/`, `getting-started/`, `guides/`, `modules/`, `project/` (status, bugs, roadmap) |

## `src/spot/` packages

| Package | Job |
|---|---|
| `app/` | `main.py`: the one place that wires everything, fixed startup order |
| `core/` | Event bus, feature registry, core intents (stop/sit/stand...), tuning, latency, startup status |
| `audio/` | Mic capture, playback, chimes, echo frontend, barge-in, response gate |
| `mic/` | XVF3800 mic array, LED ring, Spot LEDs |
| `wake/` | "Hey Spot" wake word engines |
| `safety/` | Stop/freeze/e-stop: safety keyword spotter, listener, generation counter, e-stop |
| `stt/` | Speech to text: ASR gRPC server + backends (Parakeet, Nemotron) |
| `tts/` | Text to speech: Kokoro, ElevenLabs, speaker, sentence chunker |
| `conversation/` | Turn loop, speaker lock, voiceprint, session state, logs |
| `brain/` | Regex intents, LLM brain + backends, personas, generated grammar, knowledge packs |
| `robot/` | Everything that talks to Spot: connection, arm, motion, state, dispatcher |
| `nav/` | GraphNav runner, map loading, saved locations |
| `perception/` | Cameras, world objects, obstacle queries, vision detector |
| `operator/` | Control server behind `spotctl` |
| `features/` | Optional abilities, one folder each: `dance`, `follow`, `gaze`, `mouth`, `navigation`, `settings`, `visual_nav`, `walk`. `enabled.py` lists which are on |

## Flow

```
mic -> audio/ -> wake/ ----> stt/ -> brain/ -> robot/dispatcher -> core action
              \-> safety/ (always on, halts directly)        \-> features/<name>
reply text -> tts/ -> audio/ (speaker)
```

Stop/freeze/e-stop never go through the event bus: `safety/` halts first, then announces.
