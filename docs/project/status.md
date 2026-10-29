# System Status

Snapshot of what the code does on the poster-demo branch (`demo-poster-2026-10-29`, base commit `cd4528d`), updated **2026-10-28**. Stage-by-stage detail with file and line references is in the [pipeline explainer](../architecture/pipeline-explainer.md); the operator procedure is in [daily-operations.md](../getting-started/daily-operations.md). The forward plan is [roadmap.md](roadmap.md); open bugs are in [bugs.md](bugs.md).

---

## 1. Where things stand

| | |
|---|---|
| Demo branch | `demo-poster-2026-10-29` (rollback tag `pre-demo-2026-10-29`); upstream working branch `tour_guide_upgrade_matteo` |
| Tests | 465 passed at `cd4528d` (per the build ledger) |
| Next milestone | Poster/networking demo, Thursday 2026-10-29. Rehearsal with babble and the final tag (`demo-ready-2026-10-29`) are still pending |
| Pending operator decisions | `WALK_POSE` (`src/session.py:186`, currently `"stow"`); fill in `knowledge/event_knowledge.md`; Wi-Fi vs Kokoro voice; map vs `--no-map` |

---

## 2. Pipeline

```
XVF3800 (ch0) -> frame VAD + noise floor + SNR gate -> wake KWS (4 phrases) -> speaker lock (DoA +/-35 deg)
   -> Parakeet ASR (gRPC :50055) -> safety rule -> LLM brain (Gemma 4 E4B, llama.cpp :11435, GBNF)
   -> TTS (ElevenLabs, Kokoro fallback) + mouth + chimes + LED ring -> dispatcher -> Spot SDK
   always-on safety KWS thread (all states)   operator: spotctl over ssh -> localhost :8770
```

| Stage | State on this branch |
|---|---|
| Launch | `scripts/wakespot` claims the E-Stop, runs `run_voice_control.py` (ASR server + `client_mic.py`). Forwards `--no-map --enable-arm --no-transcript-log --latency --mic-channel` and others. Fails closed without the safety KWS |
| Services | `llama-server.service` (port 11435). Ollama no longer required on the default backend |
| Mic | XVF3800 fw 2.1.1. Audio ch0. USB control used for DoA (20 Hz) and the LED ring. Calibrated 2026-10-28: yaw offset 179, sign -1, ring offset 6 LEDs |
| Front-end | VAD on every frame, percentile noise floor (rise-limited), endpointer (3 frames on, 1.2/1.5 s end, 20 s cap), SNR gate (>= -50 dBFS and >= 10 dB) |
| Wake | sherpa KWS: HEY SPOT, LISTEN UP SPOT, EXCUSE ME SPOT, HEY THERE SPOT; peak-safe gain; one-breath "wake + command" |
| Speaker lock | Direction lock +/-35 deg, 20 s timeout, nudges toward the talker; a new wake re-locks. Voiceprint code exists but is **not wired** |
| Safety | Always-on safety KWS (stop/freeze/estop) in every state, unaffected by mute; transcript rule (noun-use exemption); `spotctl` immediate stop; dispatcher stop-first and lock-free; arm stows on stop/sit/shutdown |
| ASR | Parakeet (GPU, about 190 ms). Misses some proper nouns |
| Brain | Gemma 4 E4B Q8 via llama.cpp, GBNF `{actions, response}`, cache-friendly prompt, event + Thayer + tour knowledge, thinking filler after 4 s. Spot never says stop/halt/freeze aloud |
| TTS | ElevenLabs Flash primary, per-sentence Kokoro fallback (4.5 s budget, 60 s sticky). Mouth smoothing keys are **not wired** |
| Signals | Ring: idle dim white, listen green sweep toward speaker, think amber breathing, speak blue, muted red; distinct chimes. Measured: no LED dead-centre front, 12 LEDs, about 1175 writes/s |
| Dispatcher | Arm tucks before walking, walk cap 0.5 m/s, arm stows on stop/sit/shutdown; emotes (3 dances + 4 gestures) refused when moving, not standing, within 1.5 m of an obstacle, or without an obstacle grid; info results spoken |
| Operator control | `scripts/spotctl` (hotkeys or one-shot over ssh): mute, unmute, listen, release, reset, hush, say, persona, stop, freeze, sit, stand, stow, status, crowd, voiceprint(flag) |
| Logs | `logs/utterances/`, `logs/conversations/`, `logs/latency-*.jsonl` (`--latency file`); `--no-transcript-log` for privacy |

---

## 3. Feature status

| Feature | Status |
|---|---|
| Voice pipeline, wake, one-breath commands, chaining, personas, knowledge packs | Done |
| Crowd robustness (frame VAD, SNR gate, direction lock, filler filter) | Done; rehearsal with babble pending |
| Turn signalling (ring + chimes) | Done, operator-approved patterns |
| Operator control (`spotctl`) | Done |
| Safety (always-on KWS, transcript rule, stop-first dispatch, arm stow, fail-closed start) | Done; safety-reviewed |
| Arm tuck + walk speed cap | Done; `WALK_POSE` choice pending |
| Emotes | Done (needs obstacle grid) |
| TTS fallback | Done |
| Arm gaze toward speaker | Code exists (`gaze_at_bearing`), **not wired** |
| Voiceprint check | Code exists (`speaker_embed.py`), **not wired** |
| Mouth smoothing | **Not wired** (`mouth_*` keys unused) |
| Vision overhaul (Stage 2C), web search, scene comments | Not started |
| Custom wake models (livekit) | Training only; sherpa is live |
| Door opening | Abandoned |

---

## 4. Known limits (see bugs.md for the full list)

- Stop during robot bring-up (first command stands the robot, about 30 s) needs the E-Stop.
- `SPOT_IP` only controls the startup probe; `BOSDYN_ROBOT_IP` is the real connection. Wrong address silently means dry-run.
- Voice "estop" is a motion stop, not the E-Stop.
- Named arm poses are not speed-limited.
- Vision (VLM image rotation, CPU YOLO) unchanged and weak.
- Code health: giant functions in `spot_dispatch.py` and `client_mic.py`, double-imported modules (E2).

---

## 5. Environment

| Item | Version |
|---|---|
| Jetson | AGX Orin 64GB, JetPack 6.2 / L4T R36.5, CUDA 12.6, Python 3.10 |
| bosdyn SDK | 5.2.0 (rollback venv `/mnt/ssd/spot-backups/spot-env-5.1.4-backup`) |
| Spot robot OS | 5.2.1; license "Explorer" (choreography routines run without a license) |
| Spot wired IP | 192.168.50.3 |
| Mic / speaker | reSpeaker XVF3800 / `UACDemoV1.0` on Spot CORE I/O |
| YOLO GPU env | `/mnt/ssd/yolo-env`; weights in `/mnt/ssd/yolo-models/` (unused by the voice path) |
