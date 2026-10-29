# Bug Register

Bugs and fragile behaviour found in the 2026-10-27 audit, updated 2026-10-28 after the poster-demo build (base commit `cd4528d`). Struck-through entries are fixed; the hash is the commit that landed the fix on the demo branch. Entries not struck are open or only partly fixed (see the note). New findings are in section F.

Severity: **S1** breaks the demo or is unsafe · **S2** visible to visitors · **S3** internal/latent.

---

## A. Audio front-end, wake word, turn-taking

| ID | Sev | Bug | Status |
|---|---|---|---|
| ~~A1~~ | S1 | ~~Silero VAD only fed frames above the energy gate~~ | Fixed `374a81c` (VAD on every frame, endpointer) |
| ~~A2~~ | S1 | ~~Startup calibration discarded above 0.003 RMS~~ | Fixed `374a81c` (measured floor seeds the tracker once; high noise only warns) |
| ~~A3~~ | S1 | ~~Floor ratchets up in steady noise~~ | Fixed `b95451c` + `374a81c` (percentile floor, rise capped at 1 dB/s) |
| ~~A4~~ | S1 | ~~Follow-up window accepts any voice; crowd loop~~ | Fixed `d2be1f4` + `374a81c` (direction lock, 20 s timeout, SNR gate) |
| ~~A5~~ | S2 | ~~Wake chime mutes the mic~~ | Fixed `f060df5` (chimes no longer hold the mic gate) |
| ~~A19~~ | S1 | ~~Mic reopen gap, no signal when it reopens~~ | Fixed `f060df5` (tuned settle, listen cue at the instant the mic opens) |
| A6 | S2 | End of speech after 1.2 s silence cuts off pauses | Partly: values now in `config/tuning.yaml` (`end_silence_s*`); semantic end-of-turn model still After |
| A7 | S2 | ch0 AGC/NS vs ratio gate | Partly: ratio gate replaced by SNR gate `374a81c`, `--mic-channel` `2231ab3`; ch0 vs ch1 A/B on babble not concluded |
| A8 | S2 | Wake KWS gain x8 hard-clipped | Wake fixed `36deb97` + `5bed13e` (peak-safe gain). Safety KWS deliberately kept at fixed x8 until re-evaluated |
| ~~A9~~ | S2 | ~~Loose wake tuning, single short phrase~~ | Fixed `a0593f8` + `1e8b224` (4 phrases, tuned flags). Per-phrase score/threshold still untuned (see F2) |
| ~~A10~~ | S2 | ~~Wake does not enter LISTENING state~~ | Fixed `374a81c` |
| ~~A11~~ | S2 | ~~Timeout drops to idle with no signal~~ | Fixed `374a81c` (`_release(cue=True)`) |
| ~~A12~~ | S2 | ~~"Your turn" chime same as power-on~~ | Fixed `5ef1463` (distinct listen/think/speak/released cues) |
| ~~A13~~ | S2 | ~~"stop" anywhere halts~~ | Fixed `74419b1` + `9c0c313` (noun-use exemption after next/last/bus/pit/first/final/another) |
| ~~A14~~ | S2 | ~~Safety KWS only in WAKE_WORD state~~ | Fixed `3fa1ec5` + `85ca770` (always-on thread, all states; spotctl `fc740e6`) |
| ~~A15~~ | S3 | ~~VAD segments never popped~~ | Fixed `374a81c` |
| ~~A16~~ | S3 | ~~Pre-roll not filled in WAKE_WORD~~ | Fixed `374a81c` (pre-roll on every frame) |
| ~~A17~~ | S3 | ~~No `--mic-channel` flag~~ | Fixed `2231ab3` |
| ~~A18~~ | S3 | ~~`setup_kws.py` overwrites hand-edited phrases~~ | Fixed `36deb97` (phrases in `config/wake_phrases.txt`) |

## B. Brain, speech output, dispatcher

| ID | Sev | Bug | Status |
|---|---|---|---|
| ~~B1~~ | S2 | ~~Latin-1 stream decoding~~ | Fixed `3ccce91` |
| ~~B2~~ | S2 | ~~TTS chunker speaks escape sequences~~ | Fixed `3ccce91` |
| ~~B3~~ | S2 | ~~Info actions never spoken~~ | Fixed `3ccce91` + `4398003` |
| ~~B4~~ | S2 | ~~Copied "battery is looking good" example~~ | Fixed `3ccce91` |
| ~~B5~~ | S2 | ~~Persona prefix before the big prompt~~ | Fixed `3ccce91` + `71385e3` (static prefix first, persona list in the dynamic tail) |
| ~~B6~~ | S2 | ~~Persona announcement in old voice~~ | Fixed `fed822b` |
| ~~B7~~ | S2 | ~~Volatile state keys in prompt~~ | Fixed `3ccce91` |
| ~~B8~~ | S2 | ~~ElevenLabs failure drops the sentence~~ | Fixed `5235b86` + `e00fe25` (per-sentence Kokoro fallback, 4.5 s budget, 60 s sticky) |
| B9 | S2 | Navigation arrival/stuck/lost not spoken | Open |
| B10 | S2 | `go_to_object` returns True when its thread starts | Open (After) |
| ~~B11~~ | S2 | ~~Brain answers filler~~ | Fixed `fed822b` + `85ca770` (zero non-filler words only; one-word commands pass) |
| ~~B12~~ | S2 | ~~No spoken ack in long waits~~ | Fixed `fed822b` ("Hmm, let me think." after `thinking_filler_s`) |
| ~~B13~~ | S3 | ~~`enable_mouth` not in prompt~~ | Fixed `3ccce91` |
| B14 | S3 | Robotic canned lines / raw keys spoken | Partly (info results and refusals are phrased); not audited fully |

## C. Robot safety, arm, mouth

| ID | Sev | Bug | Status |
|---|---|---|---|
| ~~C1~~ | S1 | ~~Arm not stowed on stop/estop/shutdown~~ | Fixed `c46b2fd` + `acd9baa` (stop first, lock-free; stow without waiting) |
| ~~C2~~ | S1 | ~~Walk with arm deployed at full speed~~ | Fixed `c46b2fd` (tuck before walking, 0.5 m/s cap) |
| C3 | S2 | Arm deploy has no velocity cap | Partly `c46b2fd`: joint/gaze moves limited; named poses (stow/ready/carry) keep SDK default speed |
| C4 | S2 | Mouth envelope jittery, uncapped, unsynced | **Open**: `mouth_*` keys exist in `tuning.yaml` but nothing reads them; mouth still uses per-sentence normalisation and a fixed gate |
| C5 | S3 | No watchdog closing the gripper on stalled playback | Open (After) |

## D. Vision

| ID | Sev | Bug | Status |
|---|---|---|---|
| D1 | S2 | VLM gets the front fisheye sideways | Open |
| D2 | S2 | Wrong per-camera rotations | Open (After, Stage 2C) |
| D3 | S2 | Front cameras crossed | Open (After) |
| D4 | S2 | YOLO on CPU, YOLO-World not preloaded | Open (After) |
| D5 | S2 | go_to_object turns fixed angles | Open (After) |
| D6 | S3 | describe has no gripper-camera option | Open |

## E. Infrastructure and code health

| ID | Sev | Bug | Status |
|---|---|---|---|
| ~~E1~~ | S1 | ~~`is_available()` checks Ollama on the llama.cpp backend~~ | Fixed `3ccce91` (active backend; `wait_available` for cold boot `71385e3`). Ollama no longer required |
| E2 | S2 | Bare-name imports in `client_mic.py` load modules twice | Open (After): voice "set volume" still may not reach chimes |
| ~~E3~~ | S3 | ~~wakespot drops `--latency`/mic-channel/no-transcript~~ | Fixed `e2a4319` + `2231ab3` (`--dry-run`, `--no-brain`, `--allow-no-safety-kws`, `--latency-out` are still not forwarded by wakespot) |
| ~~E4~~ | S3 | ~~Conversation log path relative; tests pollute logs~~ | Fixed `e2a4319` |
| ~~E5~~ | S3 | ~~`livekit-wakeword` unpinned in requirements~~ | Fixed `e2a4319` (commented out) |
| E6 | S3 | Tests import `VoiceState` from `client_mic` | Open |
| E7 | S3 | Dead code (`raycast_helpers.py`, unused functions); `startup_status` imported but never spoken at boot | Open |
| ~~E8~~ | S3 | ~~Stale `daily-operations.md`~~ | Fixed `99a3cdb` (runbook rewritten). `server.py` docstring still says nemotron is the default (code default is parakeet) |
| E9 | S3 | Branch never merged to `main` | Open (After) |

## F. New findings from the demo build (open)

| ID | Sev | Finding | Effect / workaround |
|---|---|---|---|
| F1 | S1 | Stop during robot session bring-up (power-on / stand, about 30 s) is a no-op: there is no session to stop | Use the E-Stop in that window; bring the robot up yourself before visitors (`spotctl stand`) |
| F2 | S3 | `kws_score`, `kws_threshold`, `voiceprint_*`, `mouth_*` keys in `tuning.yaml` are read by no code | Editing them does nothing; wake constants live in `wake/sherpa_onnx.py` |
| F3 | S2 | `gaze_at_bearing` and `speaker_embed` (voiceprint) exist but are not called from the voice loop | Arm does not turn toward the speaker; direction is the only speaker gate; `spotctl voiceprint` is a flag only |
| F4 | S1 | `SPOT_IP` drives only the startup TCP probe; the robot connection and wakespot's E-Stop use `BOSDYN_ROBOT_IP` | Both must be the robot's reachable address (wired 192.168.50.3). A failed probe silently enables dry-run |
| F5 | S2 | Voice "estop" sends the same motion-stop as "stop"; it is not the E-Stop | Real cut: E-Stop / Ctrl+\\ in wakespot |
| F6 | S2 | Spot's own speech containing stop/halt/freeze would trip the safety listener | Mitigated by prompt rule `6449ac4` and status template wording `cd4528d` |
| F7 | S3 | Voice stop during the main-thread arm-tuck wait is deferred up to 5 s | `spotctl stop` / E-Stop |
| F8 | S3 | Halt confirmation beep cleared by hush; `last_latency_s` ends at `process_utterance` return; TTS pool executor not shut down (exit delay up to 30 s) | Cosmetic |
| F9 | S3 | `pyusb` and `PyYAML` not listed in `requirements.txt` | Install by hand on a fresh venv |
| F10 | S3 | Event knowledge placeholders are dropped per line; a half-filled line is lost | Fill whole lines |
