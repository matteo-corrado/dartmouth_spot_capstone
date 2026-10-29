# Pipeline Explainer

How a spoken sentence becomes a robot action and a spoken reply, stage by stage: which module does each job, what goes in and out, what you can tune, how each stage fails, and how to debug it.

Written for someone who has not seen this code. Line references (`file:line`) are against the poster-demo branch at commit `cd4528d` ("startup status template: say safety switch instead of e-stop"). They drift as code changes; the function names are the stable handle. Paths are relative to the repo root; `src/voice_control/` is shortened to `vc/`.

Operators: the day-of runbook is [daily-operations.md](../getting-started/daily-operations.md). This page explains *why* things behave as they do.

---

## 0. One-page data flow

```
 PEOPLE ──voice──►  reSpeaker XVF3800 (USB, 16 kHz, 2 ch; ch0 = beamformed)
                     │  audio ─────────────────────────────┐    │ USB control: DoA 20 Hz, LED ring
                     ▼                                     │    ▼
        sounddevice callback  vc/client_mic.py:642         │  MicArray (vc/mic_array.py) ◄── RingAnimator (LED writes)
          1. tap ► safety queue ─► SAFETY-KWS THREAD ──┐   │        │ DoA history
             (every state; ignores mute and gates)     │   │        ▼
          2. drop if muted / Spot speaking / gated     │   │  SpeakerLock (vc/speaker_lock.py)
          3. ► audio_queue (30 ms frames)              │   │
                     ▼                                 │   │
   MAIN LOOP (client_mic.main, 30 ms frames)           │   │
     dBFS + Silero VAD every frame ► FloorTracker      │   │
     wake KWS (WAKE_WORD + LISTENING) ─ wake! ─► lock on DoA at wake time ─► LISTENING
     Endpointer ► utterance ► SNR gate ► lock check (±35°, else red flash + drop)
                     ▼                                 │
     gRPC :50055 ► server.py ► Parakeet ASR (GPU)      │
                     ▼ transcript                      │
     safety rule (intent.py) ──stop/freeze/estop──────►├──► dispatch_intent ► stop FIRST ► stow arm
     strip wake phrase ► filler filter                 │            ▲
                     ▼                                 │            │ actions[]
     LLMBrain (llm_brain.py) ► llama-server :11435 (Gemma 4 E4B, GBNF JSON {actions, response})
        │ response tokens stream ► TTSChunker ► SpotTTS ► AudioPlayer ► USB speaker (mic gated)
        │                           ElevenLabs (4.5 s budget) ─fail─► Kokoro (local)   │ envelope
        │                                                                              ▼
        └─ actions ► spot_dispatch.dispatch_intent ► Spot SDK (folded arm ► walk ≤0.5 m/s, emote, nav, ...)
                                                       mouth (gripper) follows speech

 OPERATOR ─ssh─► scripts/spotctl ─HTTP 127.0.0.1:8770─► control_server.py ─► client_mic._on_immediate / _on_queued
 LAUNCH:  scripts/wakespot ─► E-Stop claim ─► run_voice_control.py ─► server.py (ASR) + client_mic.py (everything else)
 LOGS:    logs/utterances/<date>.jsonl   logs/conversations/<date>.jsonl   logs/latency-<ts>.jsonl (with --latency file|all)
```

Turn state machine (`vc/voice_state.py`): `WAKE_WORD` (idle) → `LISTENING` (locked to a speaker, mic open) → `THINKING` (ASR + LLM, mic gated) → `RESPONDING` (TTS + actions, mic gated) → back to `LISTENING` after a 0.5 s settle (follow-up needs no wake) → `WAKE_WORD` when the 20 s conversation timeout or `spotctl release` fires.

### Signals cheat-sheet (state → ring → chime)

| State | XVF3800 ring (12 LEDs) | Chime on entry | Notes |
|---|---|---|---|
| Idle (`WAKE_WORD`) | steady dim white `0x595959` | none | Back-to-idle after a timeout/release plays the *released* cue: descending G5 to C5 |
| Listening, speaker located | green car-indicator sweep toward the speaker, 0.07 s/frame | `listen_cue`: rising E5 to A5 | Ring fills from the opposite side toward the speaker's LED, all lit, then 2 dark frames, repeat |
| Listening, no direction | solid green | `listen_cue` | No DoA readings (mic control lost, or no speech flag at wake) |
| Thinking | fast breathing amber `0xFFB000`, 0.04 s/frame | `think_cue`: A4 then D5 | Only fires once a real transcript reaches the brain; noise/empty ASR stays silent |
| Speaking (`RESPONDING`) | solid blue `0x0040FF` | `speak_cue`: one soft C6 | Mic is gated while Spot talks |
| Muted (`spotctl mute`) | solid red `0xFF0000` | none | Overrides every state. The safety keyword thread still hears "stop" |
| Off-lock speaker rejected | 200 ms red flash, then the current signal returns | none | Someone outside the ±35° window spoke |
| Pipeline ready | no change | power-on chime C4, C5, C6 | Played once, after all models loaded |
| Other cues | n/a | `command_ok` double beep (action dispatched), `error` low descending buzz, `chain_next` short tick | Spot's onboard front LEDs also pulse by state once a robot session exists (secondary) |

Ring limits measured on the device (2026-10-28): 12 LEDs 30 degrees apart; **no LED is dead-centre front** (LED0 sits at about 12:30 relative to Spot's front), so "pointing at the speaker" resolves to 30 degrees at best; ring write rate about 1175 writes/s (the animator caps itself at `ring_max_hz` = 30). Mic direction and LED index both increase clockwise seen from above. Calibration: `mic_yaw_offset_deg` 179, `mic_yaw_sign` -1, `ring_offset_leds` 6, `ring_reverse` false.

---

## 1. Launch: wakespot, run_voice_control, services

**What it does.** Starts everything in the right order and owns the shutdown choreography.

| Layer | File | Job |
|---|---|---|
| `scripts/wakespot` (shell) → `scripts/wakespot.py` | `wakespot.py:main` | Lockfile `/tmp/wakespot.pid` (refuses a second copy); resolves the map (`--no-map` skips; no argument = last-used map); **claims the software E-Stop** (`estop_session`, keep-alive heartbeat thread); spawns `run_voice_control.py` in its own process group; waits. Ctrl+C = graceful (sit, power off, release E-Stop without cutting; 45 s limit then escalate). Ctrl+C twice or Ctrl+\\ = immediate E-Stop CUT and SIGKILL |
| `scripts/run_voice_control.py` | `main`, `start_asr_server` | Loads `.env`; kills stale `voice_control/server.py`; starts the ASR gRPC bridge and waits for "listening on"; starts `client_mic.py` as a child and forwards shutdown signals. Only touches Ollama when `SPOT_BRAIN_BACKEND=ollama` |
| `vc/server.py` | `ASRServicer.StreamingRecognize` (`server.py:28`) | gRPC `:50055`; backend from `SPOT_ASR_BACKEND` (`vc/asr/__init__.py:14`, default `parakeet`; `nemotron` is the rollback). The `server.py` docstring still says nemotron is the default; the code says parakeet |
| `vc/client_mic.py` | `main` (`client_mic.py:1266`) | Everything else: devices, 2 s noise calibration, brain + warm-up, audio player, TTS, VAD, wake + safety detectors, robot probe, mic array, ring, control server, main loop |

Startup order inside `client_mic.main`: auto-detect devices by name (`XVF3800` mic, `UACDemoV1.0` speaker, `client_mic.py:112`) → calibrate noise (2 s, **stay quiet**) → connect ASR → `LLMBrain.wait_available(90 s)` then `warm_up()` → `AudioPlayer` → `SpotTTS` (warms Kokoro fallback) → Silero VAD → wake KWS → safety KWS → robot probe → open `MicArray` + `RingAnimator` → start safety thread → open input stream → `beep.ready()` (power-on chime) → `set_auto_arm` (automatic fold + mouth once standing, unless `--no-arm` / `auto_arm: false`) → control server → main loop.

Robot bring-up has two paths:

- **With a map** (`wakespot` default, or `wakespot <name>`): `ensure_spot_session(map_path)` runs *at startup* (`client_mic.py:1521`) and **powers on and stands the robot**, uploads the map, localises. Fails closed first: refuses to start without the safety KWS.
- **`--no-map`**: only a TCP probe to `SPOT_IP:443` (`_probe_host`, `client_mic.py:962`). The real session is created lazily by the **first command that needs the robot**, which stands it (about 30 s).

**Inputs/outputs.** Flags in; a running process group out. Flags: `--no-map`, `--no-arm` (`--enable-arm` is a no-op kept for old launch lines), `--no-transcript-log`, `--latency off|summary|file|all`, `--mic-channel N`, `--tts-backend kokoro|elevenlabs`, `--no-tts`, `--no-wake-word`, `--volume`, `--device/--output-device`, `--skip-services`, `--debug-audio`, `--debug-crash`. `run_voice_control.py` additionally accepts `--dry-run`, `--latency-out`, `--allow-no-safety-kws`, `--no-brain` (wakespot does not forward those four).

**Tunables.** Environment (names only): `SPOT_IP` (the TCP probe target), `BOSDYN_ROBOT_IP` (used by `src/session.py` and `src/estop.py` for the real connection and by the E-Stop claim), `SPOT_BRAIN_BACKEND` (`llamacpp` default), `SPOT_ASR_BACKEND`, `SPOT_TTS_BACKEND`, `SPOT_WAKE_BACKEND`, `SPOT_DRY_RUN`, `SPOT_NO_TRANSCRIPT_LOG`, `ELEVENLABS_API_KEY`. Services: `llama-server.service` (docker, port 11435). Ollama is **no longer required** on the default backend.

**Failure modes.**

- **Silent dry-run.** If the TCP probe to `SPOT_IP:443` fails (wrong address, robot off, not on the wired LAN) or eager bring-up throws, the pipeline *auto-enables dry-run*: it listens, thinks and talks but never touches the robot. Console says `Auto-enabling dry-run mode`. `SPOT_IP` and `BOSDYN_ROBOT_IP` must both be the robot's reachable address (wired LAN `192.168.50.3`, not the Wi-Fi AP default `192.168.80.3`).
- Safety KWS missing on the robot: process exits with "Refusing to start" (override `--allow-no-safety-kws`, operator's call).
- `Could not find mic device matching 'XVF3800'`: mic unplugged or not enumerated; exits.
- llama-server still loading (503): waits up to 90 s, then runs regex-only until it answers (re-checked each utterance).
- A stale `/tmp/wakespot.pid` from a live process blocks launch.

**Debug.** Everything prints to the wakespot terminal; look for `[Spot]`, `[Brain]`, `[SafetyKWS]`, `[WakeWord]`, `[MicArray]`, `[TTS]`. `systemctl status llama-server`, `curl -s localhost:11435/health`, `pgrep -af voice_control`.

---

## 2. Mic and XVF3800: channels, DoA, ring

**What it does.** The reSpeaker XVF3800 (firmware 2.1.1) is both the microphone (USB audio) and a controllable device (USB vendor control transfers): it reports **direction of arrival (DoA)** of speech and drives the 12-LED ring.

- Audio: stereo 16 kHz. `MIC_CHANNEL` (`Tuning.mic_channel`, default 0) picks ch0 (processed: noise suppression + AGC, beamformed) or ch1 (ASR beam, no NS/AGC). Selection happens in `audio_callback` (`client_mic.py:642`).
- Control: `MicArray` (`vc/mic_array.py:62`): `open()` (`:73`, returns `None` if absent), `read_doa()` (`:118`, degrees plus a speech flag), `start_poller(20 Hz)` (`:149`) keeping a timestamped history, `samples_between(t0, t1)` (`:171`), `set_led_effect`, `set_ring(12 colours)` (`:127`, `:133`), `set_fixed_beams` (beam steering, off by default).
- Ring: `RingAnimator` (`vc/ring_animator.py:21`) is the **single owner of all ring writes**. `show(spec)` never blocks; one daemon thread loops `spec.frames` at `frame_s`, capped to `ring_max_hz`. Specs come from `ring_for(state, lock_deg, muted, ...)` (`vc/state_feedback.py:143`).

**Hard rules in code.** The XVF3800 is only ever *written at runtime*: never `SAVE_CONFIGURATION`, never `REBOOT`, never `set_configuration`, never a kernel-driver detach (audio must keep streaming).

**Frames of reference.** DoA is in the **mic frame** (increases clockwise from above). The lock stores mic-frame degrees. Body frame (0 = ahead, +90 = left) is derived by `mic_to_body_deg` (`vc/speaker_lock.py:22`) = `sign * (mic - offset) mod 360`; measured: FRONT 179, LEFT 89, BACK 356, RIGHT 296. Body frame is used only for status, logs and gaze. The ring takes the mic frame plus `ring_offset_leds`/`ring_reverse`.

**Tunables** (`config/tuning.yaml`): `mic_channel`, `mic_yaw_offset_deg`, `mic_yaw_sign`, `ring_offset_leds`, `ring_reverse`, `ring_max_hz`, `beam_steering`.

**Failure modes.**

- USB control lost (unplugged, USB error): `MicArray.dead` goes True; console prints `[MicArray] XVF3800 control lost`; **voice continues** but with no ring and no direction lock (everything is accepted).
- USB *audio* lost mid-run: the loop just stops receiving frames (the 5 s idle line stops printing). Replug, then restart wakespot.
- A noisy room can make DoA jump; the lock uses the *median* of speech-flagged samples so single outliers do not matter.
- Beam re-convergence takes seconds after the talker moves (observed on the device), so right after a speaker change DoA can briefly point at the old one.

**Debug.** `scripts/xvf_probe.py live` (DoA 10 Hz), `clap` (re-derive yaw offset/sign), `index` (LED orientation), `show` (preview real turn signals). `spotctl status` shows `locked_deg` (body frame). `logs/utterances/*.jsonl` carries `lock_mic_deg`/`utt_mic_deg` per utterance and a 1 Hz idle record.

---

## 3. Front-end: VAD, noise floor, endpointer, SNR gate

**What it does.** Turns a continuous 30 ms frame stream into discrete utterances and drops the ones that are too quiet or too close to the room's noise.

Per frame in the main loop (`client_mic.py:1915-1990`):

1. `frame_dbfs` (`vc/audio_frontend.py:7`) = level of the frame.
2. **Silero VAD on every frame** (via sherpa-onnx; `client_mic.py:1921`). Old bug A1 fed it only loud frames; fixed.
3. `ChimeMask` (`client_mic.py:612`): while one of Spot's own *chimes* plays (and `speaker_tail_s` after), VAD speech is masked so a chime can never start an utterance. Spot's *speech* is handled by the audio callback (frames dropped).
4. `FloorTracker.update` (`audio_frontend.py:20`): the noise floor is the 10th percentile of the last 5 s of non-speech frames; falls instantly, rises at most 1 dB/s (so steady crowd noise cannot ratchet it up out of reach), and is **seeded once** by the 2 s startup calibration.
5. Frame goes into a **pre-roll** deque (25 + onset frames) on every frame in every state, so the first syllable is never clipped.
6. `Endpointer.feed` (`audio_frontend.py:48`): "start" after `onset_frames` (3) consecutive speech frames; "end" after `end_silence_s_short` (1.2 s) of silence, lengthened to `end_silence_s` (1.5 s) once the utterance passes 2 s; "cap" at `max_utterance_s` (20 s). A frame counts as silence only if the VAD says non-speech *and* its level is under floor + 6 dB.
7. At the end, `_finish_utterance` (`client_mic.py:1709`) applies the **SNR gate** `passes_gate` (`audio_frontend.py:116`): median speech level must be at least `min_speech_dbfs` (-50, plus 6 dB while the robot is moving) **and** at least `min_snr_db` (10) above the floor. Failing utterances are logged with `drop_reason: "gate"` and discarded without ASR.

Utterances shorter than 0.3 s (`MIN_SPEECH_DURATION`) are dropped before ASR.

**Mic gating.** `audio_callback` discards frames when muted, while Spot's speech plays (`AudioPlayer.is_speech_busy`), during the response gate (`_mic_gated`, set when an utterance is committed), and for `speaker_tail_s` (0.15 s) after speech. The gate lifts `reopen_delay_s` (0.5 s) after the player goes idle (`vc/response_gate.py`), at which point detectors are reset, the audio queue is drained, and the `listen_cue` fires. Chimes do **not** hold the gate.

**Inputs/outputs.** Frames in; `(utterance frames, UtteranceStats, endpoint reason)` out; metrics line to `logs/utterances/<date>.jsonl`.

**Tunables.** `min_speech_dbfs`, `min_snr_db`, `onset_frames`, `end_silence_s`, `end_silence_s_short`, `max_utterance_s`, `floor_window_s`, `floor_percentile`, `floor_rise_db_per_s`, `reopen_delay_s`, `speaker_tail_s`. Code constants: `PREROLL_FRAMES`, `TRAILING_KEEP_FRAMES`, `NAV_SPEECH_DB_BOOST`.

**Failure modes.**

- *Visitor not heard, no reaction:* gate drop (`[Gate] dropped: speech X dBFS, SNR Y dB`). Loosen `min_snr_db` / `min_speech_dbfs`, or the room is simply too loud.
- *Utterance runs to 20 s ("Max duration reached"):* loud babble keeps frames above floor + 6 dB so the silence counter never completes.
- *Cut off mid-sentence:* `end_silence_s_short` too small for the speaker's pauses.
- *Calibration happened during noise:* the floor starts high (warning printed: `High noise ... during calibration`) but falls instantly if the room quiets.
- Walking raises motor noise: the gate is 6 dB stricter while moving.

**Debug.** `--debug-audio` prints `dbfs`/`floor` every 5 s; `spotctl status` returns `floor_dbfs`; read `snr_db`, `speech_dbfs_median`, `endpoint`, `drop_reason` in the utterance log; `scripts/record_demo_clips.py` + the replay lane for offline tuning.

---

## 4. Wake word

**What it does.** A frame-level sherpa-onnx keyword spotter (zipformer 3.3M, int8, about 5 MB) listens for any of the wake phrases in `config/wake_phrases.txt`:

`HEY SPOT`, `LISTEN UP SPOT`, `EXCUSE ME SPOT`, `HEY THERE SPOT`.

- `WakeWordDetector` (`vc/wake/sherpa_onnx.py:38`): `process_frame` (`:102`) applies a **peak-safe gain** (`peak_safe_gain`, `vc/wake/phrases.py:67`: scale so the frame peaks at `kws_peak_target` 0.9, never more than x8; target <= 0 restores the legacy fixed x8) and returns True on a hit.
- It runs in `WAKE_WORD` **and** `LISTENING`, so a new speaker's wake re-locks onto them. In `WAKE_WORD` no recording or ASR happens at all (crowd chatter never reaches ASR).
- "Hey Spot, stand up" in one breath: the wake fires mid-sentence, a fresh endpointer starts, the pre-roll holds the tail, and `strip_wake_phrase` (`phrases.py:48`, fuzzy `difflib` 0.75, tolerates one leading filler such as "hey/hi/say") removes the phrase from the transcript (`process_utterance`, step 1c). A bare wake phrase with no command is ignored.
- Keyword token files are generated by `scripts/setup_kws.py` (sentencepiece fallback if sherpa's `text2token` needs `pypinyin`). Do not hand-edit `models/kws/keywords.txt`; edit `config/wake_phrases.txt` and re-run the script.

**Inputs/outputs.** PCM16 frame in; bool out → `_on_wake(t_wake)` (`client_mic.py:1669`).

**Tunables.** `kws_peak_target`. Detection constants live in `wake/sherpa_onnx.py:30-34` (`KEYWORDS_SCORE` 1.5, `KEYWORDS_THRESHOLD` 0.05, `NUM_TRAILING_BLANKS` 1). The `kws_score` and `kws_threshold` keys in `tuning.yaml` are **not read by any code yet**; editing them does nothing.

**Failure modes.** Misses at distance or in babble; false fires on chatter (a false fire just opens a LISTENING window that nobody speaks into and the lock times out after 20 s). Per-phrase score/threshold have not been tuned individually. `SPOT_WAKE_BACKEND=livekit` is an unused alternative.

**Debug.** Console `>>> Wake word detected!` then `[Lock] speaker at ...`. `spotctl listen` bypasses the wake word entirely.

---

## 5. Speaker lock (and voiceprint status)

**What it does.** Pins the conversation to one *direction* so the room's chatter is ignored.

- `SpeakerLock` (`vc/speaker_lock.py:26`): `lock(degs, now)` (`:38`) takes the circular median of speech-flagged DoA samples from the 1.5 s before the wake and stores it. `accept(degs, now)` (`:47`) allows an utterance if its median DoA is within `lock_window_deg` (35) and **nudges** the lock 30% toward it (`lock_nudge`), tracking a slowly moving talker. `expired(now)` is true 20 s (`lock_timeout_s`) after the last accepted utterance. `release()` drops it.
- Wiring: `_on_wake` locks and enters LISTENING; `_finish_utterance` calls `accept` after the SNR gate (reject → red flash + `drop_reason: "lock"`); the main loop calls `conversation_expired` and then `_release(cue=True)` (descending cue, back to idle).
- **Degrades open:** with no DoA readings (mic control lost) the lock accepts everything; a no-direction re-wake clears the old speaker's lock.
- A different person takes over only by saying a wake phrase (the lock is re-taken at that moment).
- `crowd` mode (`spotctl crowd`): after every reply drop back to `WAKE_WORD`, so each turn needs a fresh wake.
- Optional `beam_steering` points the mic's fixed beam at the locked angle (off by default; untested at the event).

**Voiceprint (not live).** `vc/speaker_embed.py` (WeSpeaker ResNet34-LM embedding, cosine similarity, `accept_combined`) exists and `scripts/setup_wespeaker.py` fetches the model, but **nothing in `client_mic.py` calls it**. `spotctl voiceprint` only toggles a status flag and the `voiceprint_enabled` / `voiceprint_threshold` tuning keys are unused. Direction is the only speaker gate at the demo.

**Gaze (live, folded arm).** On a wake lock, an accepted lock update and a release (→ centre), client_mic queues `(body_deg, safety_gen)` to one gaze worker (newest request only; the gen is taken at request time, so a stop before pickup wins). `gaze_at_bearing` (`vc/spot_dispatch.py`) turns shoulder yaw sh0 of the folded joints pose with a slow joint move (`ARM_MAX_ANGULAR_VEL`), clamped to ±`gaze_max_deg` (tuning, default 20, capped at 60 in code; `gaze_sh0_sign` flips the direction), ignoring speakers > 150° behind. It only runs in the talk pose, standing + powered, not walking/dancing/closing, at least 3 s after any pose move, when the target moved > 15° and ≥ 2 s since the last gaze. It is sent under `_arm_lock` after re-checking pose and safety gen, with a 0.5 s RPC timeout. `vc/gaze.py` (old arm_gaze_command math) is unused.

**Tunables.** `lock_window_deg`, `lock_timeout_s`, `lock_nudge`, `mic_yaw_*`, `beam_steering`.

**Failure modes.** Wrong person locked (two people at similar angles, or the beam still pointing at the previous talker at wake time); speaker walks more than 35° away between turns; two visitors side by side (indistinguishable by direction).

**Debug.** `[Lock] speaker at X deg (mic), Y deg (body)`, `[Lock] rejected utterance from X deg (locked Y deg)`, `[Conversation timeout ...]`. `spotctl status` → `locked_deg`. `spotctl release` forces a new wake.

---

## 6. ASR

**What it does.** Converts one finished utterance to text. Not streaming: the client sends the whole utterance (PCM16 @ 16 kHz) after endpointing.

- Client: `send_to_asr` (`client_mic.py:838`) opens a gRPC stream to `localhost:50055`.
- Server: `ASRServicer.StreamingRecognize` (`vc/server.py:28`) concatenates chunks and calls `backend.transcribe`. Backends: `ParakeetBackend` (`vc/asr/parakeet.py`, default, GPU, about 190 ms) and `NemotronBackend` (`asr/nemotron.py`, rollback), chosen by `SPOT_ASR_BACKEND`.
- `process_utterance` (`client_mic.py:2188`) then cleans punctuation, rejects empty/one-character transcripts, and runs the safety and wake-strip steps (below).

**Inputs/outputs.** PCM16 bytes in; transcript string out (logged as `HEARD: ...`).

**Tunables.** `SPOT_ASR_BACKEND`. Proper nouns are the weak spot ("Thayer" can come out as "Thaya"); there is no phrase-boost list yet.

**Failure modes.** ASR server dead (gRPC error; the turn is dropped, loop continues); empty transcript (`[No speech recognized]`, no THINKING signal); hallucinated text on noise.

**Debug.** `[Timing] ASR: x.xxs`, `HEARD:`/`CLEAN:` lines; `pgrep -af voice_control/server.py`; run `scripts/run_voice_control.py --server-only`.

---

## 7. Safety path

Spot should stop within a second of a spoken "stop" **in any state**. Five independent layers:

1. **Always-on safety keyword spotter** (own thread). `audio_callback` pushes every frame into `_safety_q` *before* any mute/gate check (`client_mic.py:652-655`). `safety_kws_loop` (`:443`) feeds `SafetyKeywordDetector` (`vc/wake/safety_kws.py:30`: stop / freeze / estop, easier to trigger than the wake word, fixed x8 gain); on a hit `_on_safety_kw` (`:502`) marks the safety event (aborts any action chain), starts a **hush** (cut speech) on a daemon thread, and dispatches the halt intent immediately. It runs in all states and **operator mute does not silence it**. 1.5 s refractory after a hit. Health is exposed as `safety_kws_ok` in `spotctl status`.
2. **Transcript rule** (`check_safety_command`, `vc/intent.py:129`), applied to every transcript before wake handling and again after the wake phrase is stripped, short-circuiting before the LLM. *"stop" fires unless every "stop" is a noun use* directly after `next, last, bus, pit, first, final, another` (and never after sentence punctuation); `estop`/`emergency stop` always wins; `halt` and `freeze` always fire. Errs toward stopping: "come to a stop" and "press the stop button" halt Spot; "what's the next stop" does not.
3. **`spotctl` immediate commands** (`_on_immediate`, `client_mic.py:1191`): `stop`, `freeze`, `sit`, `stand`, `stow`, `hush` run on HTTP threads without taking any lock the main loop might hold.
4. **Dispatcher** (`vc/spot_dispatch.py`): `SAFETY_INTENTS` (`:74`) bypass the dispatch lock. `_dispatch_safety` (`:1368`) sends the SDK **stop command first**, then the original branch, then stows the arm (`_stow_now`, `:194`) without waiting; `_safety_gen` aborts locomotion still being set up on another thread; with **no session** it does nothing (it never powers on a parked robot just to stop it). "Freeze" holds a stand only if already standing. Mouth is latched shut.
5. **Fail closed.** `safety_startup_ok` (`client_mic.py:552`): on a reachable robot, startup refuses unless the safety KWS loaded (`--allow-no-safety-kws` overrides; dry-run only warns).

Important distinctions:

- The voice intent `estop` sends the *same motion-stop command* as `stop`. It is **not** the E-Stop. The real motor cut is the E-Stop (the endpoint wakespot holds, the tablet, the hardware button) or Ctrl+\\ / double Ctrl+C in the wakespot terminal.
- **Spot never says stop, halt, freeze or estop aloud**: its own voice would trip layer 1. The prompt forbids it and the startup status template says "safety switch". If it happens anyway, Spot halts itself.
- Known gap: a stop issued during **session bring-up** (power-on / stand) is a no-op, because there is no session to stop. Use the E-Stop in that window. Avoid it by bringing the robot up yourself before visitors arrive (`spotctl stand`, or start with a map).

**Tunables.** None in `tuning.yaml`. Constants: `SAFETY_REFRACTORY_S`, `SAFETY_QUEUE_MAX`, `KEYWORDS_SCORE`/`THRESHOLD` in `safety_kws.py`, `_STOP_NOUN_PRECEDERS` in `intent.py`, `SAFETY_LOCK_TIMEOUT_S`/`RPC_TIMEOUT_S` in the dispatcher.

**Failure modes.** Safety KWS misses a quiet or distant "stop" (then transcript rule needs the speaker to be locked and heard); a visitor saying "stop" while a different person is locked still halts (by design); Spot's speech containing the word halts it.

**Debug.** `[SAFETY-KWS] 'stop' — executing stop immediately`, `>>> SAFETY COMMAND EXECUTED`; `[Spot] ✓ Stop command sent`; `[SafetyKWS] !!! UNHEALTHY` after 50 consecutive detector errors; `spotctl status` → `safety_kws_ok`.

---

## 8. Brain: prompt layout, knowledge, grammar

**What it does.** One LLM call per turn returns *both* what Spot does and what it says.

- `LLMBrain` (`vc/llm_brain.py:304`): `process(transcript, state)` (`:499`) builds messages with `_build_messages` (`:475`) and calls the active backend (`llamacpp` default: `vc/brain/llamacpp_backend.py`, POST to llama-server at port 11435 with the GBNF grammar; `ollama_backend.py` is the alternative via `SPOT_BRAIN_BACKEND`).
- **Prompt layout (cache-friendly, static first)**: `SYSTEM_PROMPT` (rules, action list, examples) + emote section (only if the grammar mentions `emote`) + knowledge packs → persona prefix → `KNOWN PERSONAS` list → live robot-state block (volatile keys such as uptime dropped) → history (24 messages) → user text. llama.cpp reuses the KV cache for the unchanged prefix, which is why replies take 2-4 s instead of 10+.
- **Knowledge packs** (`_load_knowledge_packs`, `llm_brain.py:264`): `vc/knowledge/event_knowledge.md` first (marked PRIMARY CONTEXT), then `thayer_knowledge.md`, `tour_route.md`. In `event_knowledge.md` every line containing `FILL IN` is dropped (**whole lines only**, so fill complete lines); if nothing real remains the pack is omitted. Fill it in before the event. The knowledge is loaded at import time: restart to pick up edits.
- **Grammar** (`vc/grammar/spot_action.gbnf`): output is forced to `{"thinking"?, "actions":[{"action":<name>,"params":{...}}], "response":"..."}`. `action-name` is an **allowlist**; adding a dispatcher action requires adding it here or the model cannot emit it. The brain renames `action` to `intent` for the dispatcher.
- Personas (`config/personas.yaml`, `vc/chatbot/persona.py`) set voice and style. `set_persona`/`add_persona` run before the reply is spoken so the announcement uses the new voice.
- Describe/vision: a `describe` action captures a camera frame and calls `brain.query_vlm` (same llama-server's mmproj); the LLM's own response text is replaced by a stock "taking a look" line.
- Conversation turns are appended to `logs/conversations/<date>.jsonl` (`vc/chatbot/conversation_log.py`).

**Inputs/outputs.** Transcript + robot state in; `{actions: [{intent, params}], response, raw_llm}` out. Response tokens stream to the TTS chunker while generating.

**Tunables.** `thinking_filler_s` (4.0; if no speech has started by then, Spot says "Hmm, let me think."). Constants: `MAX_HISTORY` 24, `REQUEST_TIMEOUT`, sampling profiles. The system prompt itself.

**Failure modes.** llama-server down or loading: `wait_available` at startup, then regex-only (`parse_intent`) and a re-check every utterance (Spot answers only simple commands); request failure: error buzz and "Sorry, I lost my train of thought"; hallucinated facts when a knowledge slot is empty (empty slots are *dropped*, so the model says it does not know); persona switch or `add_persona` is slow (up to about 30 s) and primes the cache again.

**Debug.** `[Brain-timing] backend=llamacpp profile=... total_ms=...`, `[Models] LLM only (action: ...)`, `[Brain-think]`; `curl localhost:11435/health`; `journalctl -u llama-server`.

---

## 9. Dispatcher: actions, arm policy, emotes

**What it does.** Executes each action from the brain (or from regex fallback / spotctl) on the robot, thread-safely.

- Entry: `execute_on_spot` (`client_mic.py:873`) → `dispatch_intent` (`vc/spot_dispatch.py:1312`). Honors `SPOT_DRY_RUN`. Chains run in order with waits between steps (`CHAIN_MOVEMENT_WAIT_S` 3 s, `CHAIN_POSTURE_WAIT_S` 4 s, or until navigation completes); an operator/voice stop cancels the rest.
- Routing: safety intents → `_dispatch_safety`; "unlocked" intents (persona, volume, backend, battery/status/obstacle/list info) → `_dispatch_core` directly; locomotion (`LOCOMOTION_INTENTS`, `:70`) → `_dispatch_locomotion`; everything else under `_dispatch_lock`. `ensure_spot_session` (`:460`) brings the session up lazily.
- **Arm policy** (`arm_policy_for`, `:101`), pure and unit-tested:
  - The talk pose **is** the walking pose: the folded joints pose `WALK_JOINTS` (`src/session.py`, `WALK_POSE = "joints"`, operator decision 2026-10-28). Locomotion (walk, strafe, turn, go_to, tour, patrol, come_back, go_to_object, follow_me) from the talk pose sends **no arm command** and the mouth keeps animating (a gaze-yawed shoulder is re-centred first, without waiting). With another `WALK_POSE` the old path applies: tuck first, refuse to move if the tuck fails, restore only after a natural finish.
  - stop / freeze / estop / sit / power_off / stow: arm **stows**; sit stows first.
  - emote: mouth off, arm stows; after the routine the arm folds back and the mouth comes back on.
  - Automatic fold: once the robot is standing + powered (after bring-up, a stand, or at the next turn) the arm folds and the mouth turns on (`auto_arm_if_ready`). E-stop, sit, power-off and close cancel a pending fold; "disable mouth" / `spotctl stow` turn it off for the run.
- **Walk speed cap**: `walk_speed_cap` (`src/graph_nav_utils.py:17`) = `Tuning.walk_speed_cap` (0.5 m/s) on every velocity, trajectory and GraphNav command; turning limited to 1 rad/s. Named arm poses (stow/carry) keep the SDK's default speed; the joints pose uses a 2.5 s move capped at `ARM_MAX_ANGULAR_VEL`.
- **Emotes** (`vc/emotes.py`, `_handle_emote` `spot_dispatch.py:1209`): routines `hey_join_me`, `performance_bow`, `smooth_sailing` (dances), `happy_gesture`, `tail_wag`, `upright_sit`, `wanna_play`; "dance"/"random" picks a random dance different from the last. Runs via the Choreography client (works without a license). It **refuses**: while navigating or walking ("I can't dance while I'm on the move"), unless powered and standing ("Ask me to stand up first"), if the floor-clearance scan finds an obstacle inside `dance_clearance_m` (1.5 m; "I need a little more room to dance"), or if there is **no usable obstacle grid** ("I can't check my surroundings right now", fail closed). A stop cancels the routine and stows.
- Info actions (battery, status, obstacles, locations, maps) speak their results through the `speak=` callback rather than letting the LLM guess.
- **Gripper mouth** (`vc/animation/mouth.py`): `enable_mouth` (or the automatic fold) moves the arm to the folded "talk" pose (standing + powered only) and opens/closes the gripper from the speech loudness envelope at 20 fps. The `mouth_*` keys in `tuning.yaml` (cap, attack, release, rest, lead) are **not wired yet**; the mouth uses the envelope's built-in 0.06 gate and per-sentence normalisation.

**Inputs/outputs.** `{"intent", "params"}` in; bool out (False → error buzz, chain aborts); side effects on the robot.

**Tunables.** `walk_speed_cap`, `dance_clearance_m`, `WALK_POSE` / `WALK_JOINTS` (code constants), `auto_arm` / `--no-arm`, `gaze_max_deg`, `gaze_sh0_sign`.

**Failure modes.** Robot not reachable → dry-run (see section 1); lease lost to another client (tablet) → commands fail; battery low; stop during bring-up (no-op, E-Stop needed); emote refusal lines are expected behaviour, not bugs; GraphNav go_to without localisation fails and is announced.

**Debug.** `[Spot] ...` lines; `>>> SUCCESS` / `>>> FAILED`; `[Chain] aborted by operator stop/hush`; `spotctl status`; `scripts/try_walk_pose.py` (stop voice control first: it takes the lease).

---

## 10. TTS, fallback and mouth

**What it does.** Turns the streaming reply into audio, sentence by sentence, and plays it through one serialised output stream.

- `TTSChunker` (`vc/spot_tts.py:312`) watches the LLM's JSON token stream, extracts only the `"response"` string (handles escape sequences and UTF-8), and emits each completed sentence (`vc/text_segment.py:66` `split_sentences`, pysbd-based, abbreviation-aware) to `SpotTTS.speak` (`:170`). A `describe` action suppresses the chunker.
- Rendering runs in the `AudioPlayer` worker thread: backend registry `vc/tts/__init__.py`. Primary = `SPOT_TTS_BACKEND` (default **ElevenLabs Flash**, cloud), fallback = **Kokoro** (local, GPU, voice `af_sarah`, 24 kHz), kept warm from startup.
- **Fallback rule** (`synthesize_with_fallback`, `tts/__init__.py:126`): if the primary errors, returns empty audio, or exceeds `PRIMARY_BUDGET_S` (4.5 s), that sentence is re-rendered with Kokoro and the primary is skipped for `PRIMARY_STICKY_S` (60 s). Spot never goes silent when Wi-Fi drops; the voice just changes to Kokoro.
- `AudioPlayer` (`vc/audio_player.py:117`): one persistent output stream for TTS and chimes; `is_busy` vs `is_speech_busy` (`:228`, `:241`) separate chimes from speech; `force_reset` (`:393`) is the hush; a 30 s watchdog (`:562`) un-wedges a stuck player.
- **Mouth sync**: during render, `envelope()` computes gripper-opening frames; `cb_play_start` starts `MouthDriver.play` at playback start, `cb_play_end` closes the gripper.
- While Spot speaks (and for a short tail), the mic callback drops frames; the safety tap still hears.

**Tunables.** `--volume` (0-1.5; also the `set_volume` voice action), `--tts-backend`, `thinking_filler_s`. Constants: `PRIMARY_BUDGET_S`, `PRIMARY_STICKY_S`, `DEFAULT_SPEED` 1.1.

**Failure modes.** Venue Wi-Fi flaky: first sentence is up to 4.5 s late, then Kokoro for a minute. Missing `ELEVENLABS_API_KEY`: falls back at init. Kokoro model missing: `[TTS] Not available` and silent operation. Player wedged: watchdog after 30 s, or `spotctl hush`.

**Debug.** `[DBG-render] OK backend=... used=primary|fallback voice=...`, `[TTS] ... using fallback voice af_sarah for 60s`; `--latency file` records `tts_render_*`/`tts_play_*` marks.

---

## 11. Signalling

**What it does.** Makes the turn state obvious to a crowd. The cheat-sheet at the top lists the signals; this section says where they come from.

- `_enter_state(new_state)` (`client_mic.py:1017`) is the single hook: plays the state's chime (`chime_for`, `vc/state_feedback.py:46` → `audio_feedback.beep.<cue>`), updates Spot's onboard AV LEDs (`StateLeds`, `vc/spot_leds.py`, only after a robot session exists), records the state for `spotctl status`, and calls `_update_ring`.
- `_update_ring` → `_current_ring_spec` → `ring_for(state, lock_angle, muted, ...)` → `_show_ring` → `RingAnimator.show` (the only place ring writes happen). Frames: `sweep_frames` (listening), `breathe_frames` (thinking), solid specs otherwise.
- Release/timeouts call `_release(cue=True)` which plays `released_cue`.
- `_signal_reject` flashes red for 200 ms for off-lock speakers.

**Tunables.** `ring_offset_leds`, `ring_reverse`, `ring_max_hz`, `mic_yaw_*`. Colours, frame times and chime tones are constants in `state_feedback.py` and `audio_feedback.py`.

**Failure modes.** Ring dark: mic control lost or `RingAnimator` write errors (one warning, then silent). Ring pointing at the wrong LED: recalibrate with `xvf_probe.py index`. Chimes inaudible in a loud room: raise `--volume`. Muted state shows red even while nothing else is wrong; check `spotctl status`.

**Debug.** `spotctl status` (`state`, `muted`, `locked_deg`), `[Indicator] ...` console lines, `scripts/xvf_probe.py show`.

---

## 12. spotctl: operator control

**What it does.** A laptop-side CLI for the running pipeline. The operator ssh's into the Jetson and runs `scripts/spotctl`, which POSTs to a **localhost-only** HTTP server (`127.0.0.1:8770`) started by `client_mic` (`_start_control_server`, `client_mic.py:1249`; `vc/control_server.py:117`).

- **Immediate** commands run on HTTP threads and take effect now: `hush`, `stop`, `freeze`, `sit`, `stand`, `stow`, `status`, `mute`, `unmute` (`_on_immediate`, `client_mic.py:1191`).
- **Queued** commands are handled by the main loop between frames (`_on_queued`, `:1790`): `listen`, `release`, `reset`, `say <text>`, `persona <name>`, `crowd [on|off]`, `voiceprint [on|off]`. The reply `{"queued": true}` means *accepted*, not *done*; a busy turn delays them.
- CLI: `scripts/spotctl.py` (stdlib only). One-shot `spotctl <name> [arg...]` prints JSON (exit codes: 0 ok, 1 error reply, 2 unreachable, 3 no reply in time). With no arguments it enters hotkey mode (needs `ssh -t`). Full command table is in [daily-operations.md](../getting-started/daily-operations.md).
- Motion commands (`stop/freeze/sit/stand/stow`) also set the abort flag so a chain in progress stops; `stop`/`freeze` also hush speech.

**Tunables.** Port 8770 is fixed in code; CLI `--host/--port/--timeout`.

**Failure modes.** `cannot reach control server`: pipeline not running, or port 8770 taken by an older copy (`[Control] server not started (port 8770 ...)` at startup, voice continues without spotctl). `no reply within 5s — the command may still be executing`. `voiceprint` is a flag only.

**Debug.** `spotctl status` (state, `locked_deg`, `floor_dbfs`, `last_transcript`, `last_latency_s`, `muted`, `crowd`, `safety_kws_ok`); `curl -s 127.0.0.1:8770/status` on the Jetson.

---

## 13. Logs and metrics

| File | Written by | Contents |
|---|---|---|
| `logs/utterances/<date>.jsonl` | `UtteranceLog` (`client_mic.py:262`) | One record per finished utterance (`kind: utterance`: floor, speech median/p90 dBFS, SNR, endpoint, duration, lock and utterance angles in mic and body frames, `accepted`, `drop_reason` = `gate`/`lock`, transcript, `reached_llm`, `latency_s`) plus a 1 Hz `kind: idle` record (state, floor, frame level, lock angle) |
| `logs/conversations/<date>.jsonl` | `ConversationLog` (`chatbot/conversation_log.py`) | One record per brain turn: transcript, response, actions, persona, TTS backend, voice id, LLM latency |
| `logs/latency-<ts>.jsonl` | `vc/latency.py`, with `--latency file|all` (`--latency summary` prints a percentile table at shutdown) | Per-utterance stage timestamps: VAD, ASR, LLM, dispatch, TTS render/play |

`--no-transcript-log` (sets `SPOT_NO_TRANSCRIPT_LOG=1`) writes `transcript: null` to the utterance and conversation logs and `<not logged>` in `spotctl status`; the console still shows `HEARD:`. Use it when visitors might be recorded. `logs` is a symlink to the SSD, so logs survive reboots; check free space before long sessions.

Quick looks:

```bash
tail -f logs/utterances/$(date +%F).jsonl
jq -c 'select(.kind=="utterance") | {accepted,drop_reason,snr_db,latency_s}' logs/utterances/$(date +%F).jsonl
jq -c '{transcript,response,actions}' logs/conversations/$(date +%F).jsonl
```

**Failure modes.** Disk full or I/O error disables the utterance log after one printed warning (`[Metrics] utterance log disabled`); voice continues. Tests no longer write into the production log.

---

## Tuning reference: what is live

`config/tuning.yaml` overrides `Tuning` defaults (`vc/tuning.py`); unknown keys are warned about and ignored; bad types raise at startup. Read once at import: restart to apply.

| Key | Default | Live? |
|---|---|---|
| `mic_channel`, `min_speech_dbfs`, `min_snr_db`, `onset_frames`, `end_silence_s`, `end_silence_s_short`, `max_utterance_s`, `floor_window_s`, `floor_percentile`, `floor_rise_db_per_s`, `reopen_delay_s`, `speaker_tail_s` | see file | yes (sections 2-3) |
| `kws_peak_target` | 0.9 | yes |
| `kws_score`, `kws_threshold` | 1.5 / 0.05 | **no**, constants in `wake/sherpa_onnx.py` are used |
| `lock_window_deg`, `lock_timeout_s`, `lock_nudge`, `beam_steering` | 35 / 20 / 0.3 / false | yes |
| `mic_yaw_offset_deg`, `mic_yaw_sign`, `ring_offset_leds`, `ring_reverse`, `ring_max_hz` | 179 / -1 / 6 / false / 30 | yes (measured values) |
| `thinking_filler_s`, `walk_speed_cap`, `dance_clearance_m` | 4.0 / 0.5 / 1.5 | yes |
| `voiceprint_enabled`, `voiceprint_threshold` | false / 0.5 | **no** (voiceprint not wired) |
| `mouth_lead_s`, `mouth_max_open`, `mouth_attack_s`, `mouth_release_s`, `mouth_rest_open` | see file | **no** (mouth smoothing not wired) |
