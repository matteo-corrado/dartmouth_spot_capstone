# System Status

Snapshot of what the code does on the poster-demo branch (`demo-poster-2026-10-29`, HEAD `d16cb6d`), updated **2026-11-03**. Stage-by-stage detail with file and line references is in the [pipeline explainer](../architecture/pipeline-explainer.md); the operator procedure is in [daily-operations.md](../getting-started/daily-operations.md). The forward plan is [roadmap.md](roadmap.md); open bugs are in [bugs.md](bugs.md). Build history and per-task reviews are in the ledger `.superpowers/sdd/2026-10-28-poster-demo/progress.md`.

---

## 1. Where things stand

| | |
|---|---|
| Demo branch | `demo-poster-2026-10-29` at `d16cb6d` (pushed to `origin`). Tags: `pre-demo-2026-10-29` (`ddf24c4`, before demo work) and `demo-ready-2026-10-29` (`81fedfe`, demo-day rollback point). Five commits sit on top of the tag: safety KWS stream fix and voiceprint running enrolment |
| Upstream | `tour_guide_upgrade_matteo` is still at `ddf24c4`; the demo branch has **not** been merged back |
| Tests | 641 passed, 6 skipped at `d16cb6d` (2026-11-03) |
| Milestone | Poster/networking demo held Thursday 2026-10-29. Rehearsal and demo results are not recorded in the repo |
| Open operator items | Fill in `src/voice_control/knowledge/event_knowledge.md` (still the `[FILL IN]` template); verify `gaze_sh0_sign` on the robot; decide `lock_window_deg` once voiceprint data is in; decide on merging into `tour_guide_upgrade_matteo` |
| Follow-on work | Modular restructure on branch `restructure-modular` (worktree `../dartmouth_spot_capstone-restructure`) |

---

## 2. Pipeline

```
XVF3800 (ch1, ASR beam) -> frame VAD + noise floor + loudness gate -> wake KWS (2 phrases) -> speaker lock (DoA +/-35 deg, optional voiceprint)
   -> Parakeet ASR (gRPC :50055) -> safety rule -> LLM brain (Gemma 4 E4B, llama.cpp :11435, GBNF)
   -> TTS (ElevenLabs, Kokoro fallback) + mouth + chimes + LED ring -> dispatcher -> Spot SDK
   always-on safety KWS thread (stop/freeze, all states)   operator: spotctl over ssh -> localhost :8770
```

| Stage | State on this branch |
|---|---|
| Launch | `scripts/wakespot` claims the E-Stop, runs `run_voice_control.py` (ASR server + `client_mic.py`). Forwards `--no-map --enable-arm --no-transcript-log --latency --mic-channel` and others. Fails closed without the safety KWS. Voiceprint model warms up before the mic stream opens (no startup overflows) |
| Services | `llama-server.service` (port 11435). Ollama no longer required on the default backend |
| Mic | XVF3800 fw 2.1.1, audio **ch1** (ASR beam; replay: 11/13 wakes vs 6/13 on ch0). USB control for DoA (20 Hz), LED ring, and per-beam azimuth + speech energy readback (`MicArray.read_beams`). Control link auto-reconnects with backoff; ring re-applies; `mic_ok` in status. Calibration: yaw offset 179, sign -1, ring offset 6 LEDs, not reversed. AEC is inert (TTS does not play through the XVF3800) |
| Front-end | VAD on every frame, percentile noise floor, endpointer (end silence 1.2 s / 1.0 s short, 20 s cap). Loudness gate = loudest 150 ms of speech over floor, `min_snr_db` 8. Utterances of 1.5 s or less of speech skip the gate and go to a commands-only ASR path (stop/freeze/commands, plus yes/no/okay mid-conversation); that path cannot start travel or power off |
| Wake | sherpa KWS: HEY SPOT, EXCUSE ME SPOT (LISTEN UP / HEY THERE SPOT dropped after replay 0/10 and 3/10). Peak-safe gain; one-breath "wake + command" |
| Speaker lock | Direction lock +/-35 deg, 20 s timeout; a new wake re-locks. Voiceprint (WeSpeaker, memory-only enrolment) wired, **off by default**: `spotctl voiceprint on`, threshold 0.25 (`spotctl voiceprint threshold <x>`). Running enrolment folds in accepted utterances scoring >= 0.35. Voiceprint failure falls back to direction-only |
| Safety | Always-on safety KWS, **stop + freeze only** (estop removed), every state, unaffected by mute: 3 staggered short-lived streams (a long-lived stream lost quiet stops), input gain x3, score 1.5, threshold 0.25 (clamped 0.05-0.40, logged at ready), queue-overflow drops counted (`safety_kws_dropped` in status). Transcript rule (noun-use exemption), `spotctl` immediate stop, dispatcher stop-first and lock-free, second stop/freeze has an RPC timeout, arm stows on stop/sit/shutdown, e-stop cancels pending talk-pose resume |
| ASR | Parakeet (GPU, about 190 ms). Misses some proper nouns |
| Brain | Gemma 4 E4B Q8 via llama.cpp, GBNF `{actions, response}`, cache-friendly prompt, event + Thayer + tour knowledge, thinking filler after 4 s. Spot never says stop/halt/freeze aloud |
| TTS | ElevenLabs Flash primary, per-sentence Kokoro fallback (4.5 s budget, 60 s sticky). RESPONDING (blue ring) starts just before the first reply line |
| Mouth | Gripper as mouth driven by an RMS-dB syllable envelope (closed about 50% of speech), no opening cap. Active while talking and walking in the folded pose |
| Arm | `WALK_POSE = "joints"`: folded joints pose for talking and walking; auto-fold + mouth on once standing (`auto_arm: true`). Gaze turns shoulder yaw (sh0) toward the speaker, limit `gaze_max_deg` 20 (code cap 60), held during pose moves, walk start and aborting intents. `gaze_sh0_sign` 1 **unverified** |
| Signals | Ring: idle steady dim white, listening car-indicator sweep toward speaker, thinking fast light-amber breathing, speaking blue, muted red. Operator-approved simple beep set; speaking state silent. Replay: first word kept 5/5 quick replies, 0 chime self-triggers |
| Dispatcher | Walk cap 0.5 m/s; emotes (3 dances + 4 gestures) refused when moving, not standing, within 1.0 m of an obstacle (from body centre), or without an obstacle grid; info results spoken |
| Operator control | `scripts/spotctl` (hotkeys or one-shot over ssh): mute, unmute, listen, release, reset, hush, say, persona, stop, freeze, sit, stand, stow, status, crowd, voiceprint on/off/threshold |
| Logs | `logs/utterances/` (incl. `voice_sim`, `short_path`, `rescued`), `logs/conversations/`, `logs/latency-*.jsonl` (`--latency file`); `--no-transcript-log` for privacy |
| Tooling | `tests/audio/replay_demo.py` (replay harness, KWS sweep, gain A/B, voiceprint calibration), `tests/audio/beam_ab.py` + `scripts/record_demo_clips.py` beam A/B, `scripts/chime_preview.py`, `scripts/xvf_probe.py`. Demo recordings in `tests/audio/demo/` (gitignored, this Jetson only) |

---

## 3. Feature status

| Feature | Status |
|---|---|
| Voice pipeline, wake, one-breath commands, chaining, personas, knowledge packs | Done |
| Crowd robustness (frame VAD, loudness gate, short-utterance path, direction lock, filler filter) | Done. Replay direction lock: correct 6/13 fan, 0/3 babble |
| Turn signalling (ring + chimes) | Done, operator-approved |
| Operator control (`spotctl`) | Done |
| Safety (always-on stop/freeze KWS, transcript rule, stop-first dispatch, arm stow, fail-closed start) | Done; safety-reviewed |
| Folded arm pose for talk + walk, auto fold, walk speed cap | Done |
| Arm gaze toward speaker | Done (sh0 yaw, 20 deg); sign unverified on robot |
| Mouth envelope | Done |
| Voiceprint check | Wired, off by default; running enrolment added `d16cb6d` |
| Emotes | Done (needs obstacle grid) |
| TTS fallback | Done |
| Event knowledge content | **Template only** |
| Face / body lights | Not possible: AV service is a stub on this robot, no choreography license for custom uploads |
| Vision overhaul (Stage 2C), web search, scene comments | Not on this branch (vision work on `vision/perception-v2`) |
| Custom wake models (livekit) | Training only; sherpa is live |
| Door opening | Abandoned |

---

## 4. Known limits (see bugs.md for the full list)

- Stop during robot bring-up (first command stands the robot, about 30 s) needs the E-Stop.
- No voice e-stop: the safety KWS hears stop and freeze only; use the E-Stop or `spotctl stop`.
- Quiet "freeze" from about 3 m may be missed; about 0.3 false stops per minute possible in noise.
- Babble can steal the direction lock (no AEC; per-beam speech energy readback exists but is not used by the lock yet).
- `SPOT_IP` only controls the startup probe; `BOSDYN_ROBOT_IP` is the real connection. Wrong address silently means dry-run.
- Named arm poses (stow/ready/carry) use BD default speed.
- Vision (VLM image rotation, CPU YOLO) unchanged and weak on this branch.
- Code health: giant functions in `spot_dispatch.py` and `client_mic.py`, double-imported modules (E2); being addressed on `restructure-modular`.

---

## 5. Environment

| Item | Version |
|---|---|
| Jetson | AGX Orin 64GB, JetPack 6.2 / L4T R36.5, CUDA 12.6, Python 3.10 |
| bosdyn SDK | 5.2.0 (rollback venv `/mnt/ssd/spot-backups/spot-env-5.1.4-backup`) |
| Spot robot OS | 5.2.1; license "Explorer" (built-in choreography runs without a license; custom uploads denied) |
| Spot wired IP | 192.168.50.3 |
| Mic / speaker | reSpeaker XVF3800 fw 2.1.1 / `UACDemoV1.0` on Spot CORE I/O |
| Speaker model | WeSpeaker in `models/speaker/` (gitignored) |
| YOLO GPU env | `/mnt/ssd/yolo-env`; weights in `/mnt/ssd/yolo-models/` (unused by the voice path) |
