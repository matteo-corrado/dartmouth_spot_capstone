# System Status

Snapshot of what the code actually does today, verified against the code and git on **2026-10-27**.
The forward plan lives in [roadmap.md](roadmap.md). Old per-stage plans live in `docs/superpowers/plans/`.

---

## 1. Where things stand

| | |
|---|---|
| Working branch | `tour_guide_upgrade_matteo` (never merged to `main`; local `main` is 126 ahead / 1 behind `origin/main`) |
| Last code change | 2026-06-02 (the 2026-07-23 commit was the 2C plan only) |
| Tests | `spot-env/bin/pytest tests` → 104 passed |
| Original end goal | Talking tour guide for Thayer: wake by voice, answer from knowledge packs, personas, gripper "mouth", vision, follow people |
| Next milestone | Poster/networking demo, Thursday 2026-10-29: crowd-robust conversation, clear turn signalling, operator control |

---

## 2. Pipeline, stage by stage

```
XVF3800 mic (ch0) ─► energy gate ─► Silero VAD ─► sherpa KWS "hey spot" ─► Parakeet ASR (gRPC :50055)
     ─► safety regex (stop/freeze/estop) ─► LLM brain (Gemma 4 E4B, llama.cpp :11435, GBNF JSON)
     ─► TTS (ElevenLabs Flash / Kokoro) + gripper mouth + chimes/LEDs ─► dispatcher ─► Spot SDK
```

| Stage | What runs | Key files | State |
|---|---|---|---|
| Launch | `scripts/wakespot` → `wakespot.py` (E-Stop, map) → `run_voice_control.py` → `server.py` + `client_mic.py` | `scripts/wakespot.py`, `scripts/run_voice_control.py` | Works. `--latency` and `--no-brain` are not passed through by wakespot |
| Services | `llama-server.service` (docker, port 11435) **and Ollama (11434)** | `systemd/llama-server.service` | **Hidden dependency:** `LLMBrain.is_available()` checks Ollama even on the llama.cpp backend; if Ollama is down the brain silently drops to regex-only |
| Mic | reSpeaker XVF3800, **firmware v2.1.1** (flashed 2026-10-27), 2ch 16 kHz. ch0 = processed (NS+AGC), ch1 = ASR beam (no NS/AGC). Code reads ch0 only | `client_mic.py:186` | USB control works without root (udev `99-respeaker.rules`, pyusb installed). DoA, beam steering and LED ring are available but unused |
| Noise floor | 2 s calibration, then EMA floor, gate = 2× floor | `client_mic.py:340-380`, `:1222-1229` | **Buggy:** calibration above 0.003 RMS is thrown away; floor can ratchet up to a level speech can't clear |
| VAD / endpoint | Silero (sherpa-onnx), 3-frame onset, end after 1.2 s (short) / 2.25 s silence, 20 s cap | `client_mic.py:1035-1188` | **Buggy:** VAD is only fed frames above the energy gate, so pauses cut people off and the VAD state goes stale |
| Wake word | sherpa-onnx KWS, single phrase `HEY SPOT`, gain ×8, score 1.5, threshold 0.05 | `wake/sherpa_onnx.py`, `scripts/setup_kws.py` | Loose (false fires in chatter); chime mutes the mic ~400 ms so one-breath commands lose their first word. Custom livekit models: device code done, training in progress |
| Safety words | Separate KWS for stop/freeze/estop (WAKE_WORD state only) + regex on every transcript | `wake/safety_kws.py`, `client_mic.py:543-549` | Regex fires on any sentence containing "stop" (e.g. "next stop") |
| ASR | Parakeet on GPU via gRPC, ~190 ms | `asr/parakeet.py`, `server.py` | Good; misses proper nouns ("Thayer" → "Thaya") |
| Brain | Gemma 4 E4B Q8 + mmproj via llama.cpp, GBNF `{thinking, actions, response}`, ~12K-token knowledge pack, persona prefix, history 24 | `llm_brain.py`, `brain/llamacpp_backend.py`, `grammar/spot_action.gbnf`, `knowledge/` | 2–4 s short replies, 7–11 s long; persona switch ~15 s (persona text sits before the big prompt, so the cache is lost). Streamed text decoded as Latin-1 → garbled dashes |
| TTS | ElevenLabs Flash (cloud, default) or Kokoro (local GPU), sentence-chunked | `spot_tts.py`, `tts/` | If ElevenLabs fails mid-run, that sentence is silently dropped (fallback only at startup) |
| Mouth | Gripper opens with the speech envelope at 20 fps | `animation/mouth.py`, `spot_tts.py:203-237` | Works; per-sentence normalisation, no smoothing, no sync offset, snaps shut between sentences. Arm is **not stowed** on stop/estop/shutdown |
| Turn state | WAKE_WORD → LISTENING → THINKING → RESPONDING; chimes + Spot AV LEDs; 15 s follow-up window | `voice_state.py`, `state_feedback.py`, `spot_leds.py`, `response_gate.py` | LEDs not updated on wake or on timeout; follow-up window accepts any voice (crowd loop risk) |
| Dispatcher | ~30 actions in one 835-line `if/elif` | `spot_dispatch.py:755-1589` | Info actions (battery, status, obstacles, list locations) are printed, never spoken |
| Navigation | GraphNav go-to / tour / patrol / come-back, saved locations | `graph_nav_utils.py`, `location_manager.py`, `map_loader.py` | Works; arrival/stuck/lost are not spoken |
| Vision | YOLO-World / yolov8n **on CPU**, serial camera scan, VLM describe | `visual_nav.py`, `llm_brain.py:571` | VLM gets the front image **sideways**; side/back cameras use the wrong rotation; follow mode unreliable |
| Operator control | Web panel (E-Stop + control) | `scripts/web_panel.py` | No terminal controls for mute / reset / release speaker |

---

## 3. Feature status

| Feature | Status |
|---|---|
| Voice pipeline, wake word, one-breath commands, command chaining | Done (crowd robustness poor) |
| Personas + voice switching (10 personas, Kokoro + ElevenLabs) | Done |
| Gripper mouth | Done (needs tuning + arm safety) |
| Mic gate, follow-up window, state chimes/LEDs | Done (signalling gaps above) |
| Barge-in reset, safety keywords, sentence chunker, `make eval-voice` | Done |
| Thayer knowledge pack + tour route | Done |
| GraphNav navigation, locations, maps | Done |
| Vision: describe, go-to-object, follow | Works but weak (CPU YOLO, rotation bugs) |
| Custom wake word (livekit) | Partial: training on cluster |
| Vision overhaul (GPU YOLO26/YOLOE, batched scan) — Stage 2C | Planned, not started |
| Web search, remembered facts — Stage 2D C/D | Not started |
| Speaker ID / face check / personal VAD — 2F P4/P5/P8 | Not started (WeSpeaker verified on device) |
| Scene comments — 2E.3 | Not started |
| Door opening | Abandoned (code deleted) |
| VLFM object search | Separate project at `/mnt/ssd/seg-vla-ws/seg-vla-26X` (docker); not wired to voice |

---

## 4. Known code-health issues

- **Double-loaded modules:** `client_mic.py:85-92` imports `audio_player`, `spot_tts`, `audio_feedback`, `latency` by bare name while everything else uses `src.voice_control.*`. Python loads two copies, so "set volume" never reaches the beeps and TTS latency is never recorded.
- **Giant functions:** `dispatch_intent` (835 lines), `client_mic.main` (660), `process_utterance` (372), `follow_person` (330).
- **Dead code:** `startup_status.py`, `perception/camera_intrinsics.py`, `perception/raycast_helpers.py`, ~8 unused functions in `visual_nav.py` / `llm_brain.py`.
- **Config sprawl:** ~20 env vars read ad hoc; `SPOT_DRY_RUN` read 8× and written 4× in `client_mic.py`.
- **Logs:** real conversation logs land in `src/voice_control/logs/`; tests write into the production log.
- **Stale docs:** `docs/getting-started/daily-operations.md` (still describes Ollama + two terminals).

---

## 5. Environment

| Item | Version |
|---|---|
| Jetson | AGX Orin 64GB, JetPack 6.2 / L4T R36.5, CUDA 12.6, Python 3.10 |
| bosdyn SDK | 5.2.0 (upgraded 2026-10-27; backup venv `/mnt/ssd/spot-backups/spot-env-5.1.4-backup`) |
| Spot robot OS | 5.2.1 (updated 2026-10-28), license "Explorer" (no choreography feature); 9 built-in choreography routines on the robot |
| Spot wired IP | 192.168.50.3 |
| YOLO GPU env | `/mnt/ssd/yolo-env` (needs `LD_LIBRARY_PATH` for CUDA; flask/lap not installed); weights in `/mnt/ssd/yolo-models/` |
