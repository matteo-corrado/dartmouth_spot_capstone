# Mic and voice: validation and next steps

Context: commit `d16cb6d` on `demo-poster-2026-10-29` (ported to `restructure-modular` as `c3d4a4b`). Everything below is off by default; the live pipeline is unchanged unless an operator turns it on.

Live pipeline today: XVF3800 fw 2.1.1, audio **ch1** (ASR output of the auto-select beam, no AGC/NS), DoA from `DOA_VALUE` at 20 Hz, speaker lock +/-35 deg, voiceprint gate off (`voiceprint_threshold 0.25`), `beam_steering: false`.

---

## 1. Validate: voiceprint running enrolment

**What it does.** `VoiceprintGate.decide(..., update_threshold)` (`src/spot/conversation/voiceprint.py`): after the wake + first-utterance enrolment, every accepted utterance scoring >= max(`voiceprint_threshold`, `voiceprint_update_threshold` = 0.35) is averaged into the voiceprint. Resets on new wake / lock release. Log line shows `enrolled=yes(n=K)`.

**Offline evidence so far** (replay, ch1, all-operator recordings, only 2 other-voice segments, so treat rates as direction, not magnitude):

| | same-speaker median | other-voice median | thr 0.25 TA / FA | thr 0.30 TA / FA | thr 0.35 TA / FA |
|---|---|---|---|---|---|
| wake + first only | 0.53 | 0.16 | 96% / 17% | 93% / 7% | 90% / 4% |
| + 2 running updates | 0.65 | 0.20 | 99% / 32% | 97% / 12% | 93% / 3% |

Every score rises, so **the accept threshold has to rise with it**: 0.30 beats the old 0.25 setup on both rates.

**Steps**
1. Get real impostor data: record 3-5 different people (not the operator) doing the overlap / quick-reply clips: `scripts/record_demo_clips.py --condition people_<name> --set overlap`. The current set has one speaker, so false-accept is barely measured.
2. Re-run the calibration with running updates on: extend `voiceprint_calibration` in `tests/audio/replay_demo.py` (or reuse the simulation pattern: enrol wake + first, fold in K same-speaker clips via the real `VoiceprintGate`, score held-out clips and every other person's clips). Report TA/FA at 0.25 / 0.30 / 0.35.
3. Live check: `spotctl voiceprint on`, `spotctl voiceprint threshold 0.30`; hold a 6-8 turn conversation while a second person talks from inside the +/-35 deg lock window. Watch `[Voice] sim=` lines: own turns should climb as `n` grows, the other person's should stay below threshold.

**Pass:** at the chosen threshold, own short follow-ups accepted >= 90% and other people <= 10% after 2+ updates. Then set `voiceprint_threshold` in `config/tuning.yaml` (operator decides) and consider `voiceprint_enabled: true`.

**Fail modes to watch:** enrolment drifting toward a second person who stands in the lock window and scores just over 0.35 (raise `voiceprint_update_threshold`); a noisy room lifting everyone's scores (room/channel captured by the average). If drift shows up, the fix is a stricter update bar, not a cap or EMA.

---

## 2. Validate: beam steering (`beam_steering`)

**What exists.** `src/spot/conversation/loop.py` `on_wake` aims both XVF3800 fixed beams at the lock angle when `beam_steering: true`; `release` turns them off. Never tested. Tooling from `d16cb6d`:
- `scripts/record_demo_clips.py --beam` aims the fixed beams at each clip's speaker angle (known position, so a best case) and turns them off after. `--interferer-deg` saves where the second talker stood.
- Every clip with the control link also writes `<clip>.beams.csv`: azimuth + speech energy for beam 1, beam 2, free-running, auto-select at 10 Hz (`MicArray.read_beams`).
- `tests/audio/beam_ab.py` compares an off run against an on run.

**Not yet confirmed on the device:** that `set_fixed_beams` takes effect and reads back in the same frame as `DOA_VALUE` (the mic was unplugged when this was built). Do this first:
```
spot-env/bin/python - <<'E'
import math, time
from src.spot.mic.xvf3800 import MicArray
m = MicArray.open()
for t in (90.0, 270.0):
    m.set_fixed_beams(math.radians(t)); time.sleep(0.5)
    print(t, [round(math.degrees(a) % 360) for a in m.read_beams()[0]], m.read_doa())
m.set_fixed_beams(None); m.close()
E
```
Beams 1 and 2 should read back at the target angle. Runtime writes only; nothing is saved to the device. Make sure the voice pipeline (`wakespot`) is not running.

**Steps**
1. Same room, same session, second talker (podcast speaker) at the robot's LEFT for both runs:
   ```
   spot-env/bin/python scripts/record_demo_clips.py --condition ab_off --set overlap --interferer-deg 90
   spot-env/bin/python scripts/record_demo_clips.py --condition ab_on  --set overlap --interferer-deg 90 --beam
   ```
2. `spot-env/bin/python tests/audio/beam_ab.py --off ab_off --on ab_on`
3. Run the replay command it prints (`replay_demo.py --conditions ab_off ab_on --channel 1 --asr`, needs the ASR gRPC server) for wake / lock / ASR.
4. Optional: repeat with a babble background (`--set babble`) to see crowd suppression.

**Pass (all three):**
- Second-talker level ("interferer dBFS") drops, or visitor-over-talker ("S+I/I dB") rises, by roughly 3 dB or more with the beam on.
- Wake and ASR results on `ab_on` are no worse than `ab_off`.
- "DoA verdict" in the per-clip table says **hears interferer**, not **follows beam**. If DoA follows the fixed beam, the speaker lock would confirm itself: switch the lock to the free-running beam (section 3, item B) before enabling steering.

Then set `beam_steering: true` in `config/tuning.yaml` and do one live session.

---

## 3. Improvement directions (ranked)

### A. Give the XVF3800's echo cancellation a reference (highest impact)
- **Problem.** AEC is inert (`AEC_AECCONVERGED=0`) because TTS plays on the `UACDemoV1.0` USB speaker (hw:2,0), never through the XVF3800's own USB playback (hw:3,0, 2-out). XMOS: the far-end reference must arrive on the device's USB/I2S input. So self-speech is handled only by muting (`src/spot/audio/capture.py` drops frames during TTS + `speaker_tail_s`), there is no barge-in, and the safety KWS hears Spot's own voice.
- **Change.** Play TTS to the XVF3800 (`--output-device` = XVF3800, 16 kHz) and drive a powered speaker from its 3.5 mm line-out. Hardware question first: does the UACDemo speaker have a line-in? If not, an amplified speaker is needed.
- **Validate.** Read `AEC_AECCONVERGED` after a few sentences of TTS; record ch1 during TTS and measure residual echo. Keep the mute until the residual is low; relax it for barge-in only after on-robot data.
- **Not recommended:** mirroring TTS to both USB devices (clock drift between devices defeats AEC).

### B. Speaker lock from per-beam speech energy (low effort, crowd fix)
- **Problem.** The lock is the circular median of `DOA_VALUE` readings. Replay: 0/3 correct locks in babble (DoA pointed at the crowd), DoA sticks on one value per clip (`docs/project/demo-tuning-results.md`).
- **Change.** Poll `AEC_AZIMUTH_VALUES` + `AEC_SPENERGY_VALUES` (`MicArray.read_beams` already exists) alongside DoA; weight lock samples by speech energy; use `AUDIO_MGR_SELECTED_AZIMUTHS[0]` (NaN = no speech) instead of the speech flag. With beams steered, take "is someone else talking" from the free-running beam.
- **Validate.** Replay lock accuracy on `fan_babble` and the new A/B sets (needs beams.csv, so only clips recorded after `d16cb6d`). About 40 lines in `src/spot/conversation/speaker_lock.py` / `loop.py`.

### C. ASR output gain (cheap trial)
- `AEC_ASROUTGAIN=1.0`; ch1 runs quiet in absolute level. Try 2-4x as a runtime write (no save), then re-check `min_speech_dbfs` and floor gates on replay. Digital gain does not improve SNR; it only matters if levels sit near quantisation or gate thresholds.

### D. High-pass filter for fan and motor rumble (cheap trial)
- `AEC_HPFONOFF=2` (125 Hz). Try 3 or 4 (150 / 180 Hz), recording a short fan set for each and comparing wake / ASR on replay.

### E. Voiceprint, later
- Length-aware threshold (strict on long clips, skip the voice check under ~1 s).
- Score normalisation (AS-norm) against embeddings of venue crowd speech, so scores adapt to the room.
- Target-speaker extraction (neural filter conditioned on the voiceprint) only if same-direction talkers remain the problem after A, B and beam steering. Risk: artifacts often hurt ASR on clean audio; check for a Jetson-ready export first.
- A model swap (ReDimNet2-B3, CAM++) gains a little; the short clips are the bigger limit.

### Doc fix
- Anything still saying the live audio is ch0 is stale; ch1 is correct (ch0's AGC/NS clips the wake phrase).

---

## Device facts (read-only, 2026-10-29)
`AUDIO_MGR_OP_L=(8,0)` comms (ch0), `OP_R=(7,3)` + `AEC_ASROUTONOFF=1` ASR beam (ch1), `AEC_ASROUTGAIN=1.0`, `AEC_NUM_FARENDS=1`, `AUDIO_MGR_SYS_DELAY=12`, `PP_AGCONOFF=1` (ch0 only), `AEC_HPFONOFF=2`, fixed beams off, beam gating off. XVF3800 = hw:3,0, 2 in / 2 out, 16 kHz.

Sources: XMOS XVF3800 user guide ch. 8 (ASR) and datasheet (voice pipeline), Seeed reSpeaker XVF3800 wiki, Seeed `python_control/xvf_host.py` (parameter table).
