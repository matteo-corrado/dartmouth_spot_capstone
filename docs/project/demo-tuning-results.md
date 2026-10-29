# Demo tuning results (replay of the 2026-10-28 recordings)

Produced by `tests/audio/replay_demo.py` (Task 9, Lane D) on demo head `25b25fc`
(PREROLL_FRAMES = 25, ChimeMask with the speech-before rule). Re-run:

```
spot-env/bin/python tests/audio/replay_demo.py --clips --json main.json
spot-env/bin/python tests/audio/replay_demo.py --kws-sweep --gain-ab --endpoint-sweep --json sweep.json
```

**Read this first: the samples are small.** `fan` has 4 phrases x 4 angles, one pause clip
per phrase, 3 quick replies, 3 overlap clips and 120 s of background. `fan_babble` has
one front wake per phrase, 2 quick replies and 120 s of background. One clip moves a
rate by 20-100 points. ASR was **off** (no gRPC server on :50055), so cut-offs use the
fallback rule: the first utterance after the wake must cover >= 80 % of the Silero
speech after the wake fire, and the command must not be split into a second utterance.

## How the replay works

Each clip is fed frame by frame (30 ms) through a mirror of the `client_mic` loop:
Silero VAD (same config), `ChimeMask` (recorded chime window from `<clip>.chime.json`,
plus the `listen_cue` that `_on_wake` plays, plus `speaker_tail_s`), `FloorTracker`
seeded from 2 s of the condition's background take (same as the startup calibration),
25-frame pre-roll, the real `WakeWordDetector` (temp keyword files only), `Endpointer`,
`passes_gate`, and `SpeakerLock` on the recorded DoA. 2.5 s of the clip's own lead
noise is appended so the endpointer can close. The background take can't be used for
padding because ch0's AGC lifts it about 10 dB above clip silence.

## 1. Headline: the SNR gate drops almost every real command

At the current `min_snr_db: 10`, **every** woken utterance on ch1 (13/13 fan, 3/3
babble) and 6 of 7 on ch0 fail the gate. Live, Spot would wake and then not answer.

Cause: `passes_gate` uses the **median** dB of VAD-flagged frames. Silero flags speech
about 0.24 s late and keeps flagging it for 0.5 s (min_silence_duration) after it stops.
For a 1-2 s command, half the "speech" frames are floor-level hangover. Speech p90 sits
10-28 dB over the floor, but the median sits only 2-8 dB over it:

| set | SNR median (dB) | SNR p90 (dB) |
|---|---|---|
| fan ch0, woken commands | 3.4-13.5 (med 6.6) | 14-23 |
| fan ch1, woken commands | 1.9-5.3 (med 2.7) | 6.7-11.9 |
| babble ch1, woken commands | 3.5-4.5 | 8.9-13.2 |
| quick replies ch0 / ch1 | 2.4-4.1 / 1.4-3.0 | 25-28 / 12-15 |
| background false starts (fan, LISTENING) | ch0 2.5 / ch1 2.0, 2.1 | ch0 11.9 / ch1 4.9, 7.7 |

The median can't separate real commands from noise. Endpoint sweep (`gate drops` =
woken utterances the gate would drop; `bg passes` = noise utterances that pass while
LISTENING in 120 s):

| ch | min_snr_db 0 | 2 | 4 | 6 | 10 |
|---|---|---|---|---|---|
| fan ch1: gate drops / 13 | 0 | 0-1 | 11 | 13 | 13 |
| fan ch1: bg passes / 120 s | 2-3 | 0-2 | 0 | 0 | 0 |
| babble ch1: gate drops / 3 | 0 | 0 | 1 | 3 | 3 |
| fan ch0: gate drops / 7 | 0 | 0 | 2 | 3 | 6 |

The "bg passes" on fan ch1 fall inside 20-44 s of the background take, where Silero
also hears real talking. They'd send that audio to ASR. The cost of a false pass is
bounded: an ASR call and the empty/no-op path, and the mic reopens after 0.5 s.
The real fix is in `audio_frontend`/`client_mic` (not this lane's files): gate on p90,
or drop the hangover frames from `UtteranceStats`.

## 2. Wake phrases and channel

Wake hits split by clip type (default KWS 1.5 / 0.05, peak 0.9):

| condition / ch | phrase | wake (4 angles) | pause | overlap |
|---|---|---|---|---|
| fan ch0 | HEY SPOT | 0/4 | 1/1 | 0/3 |
| fan ch0 | EXCUSE ME SPOT | 4/4 | 1/1 | - |
| fan ch0 | LISTEN UP SPOT | 0/4 | 1/1 | - |
| fan ch0 | HEY THERE SPOT | 0/4 | 0/1 | - |
| fan ch1 | HEY SPOT | **4/4** | 1/1 | 1/3 |
| fan ch1 | EXCUSE ME SPOT | **4/4** | 1/1 | - |
| fan ch1 | LISTEN UP SPOT | 1/4 | 1/1 | - |
| fan ch1 | HEY THERE SPOT | 0/4 | 0/1 | - |
| babble ch0 | all four | 0/1 each | - | - |
| babble ch1 | HEY SPOT / EXCUSE ME / LISTEN UP | 1/1 each | - | - |
| babble ch1 | HEY THERE SPOT | 0/1 | - | - |

- **HEY THERE SPOT never fired (0/10)**, at any score or threshold. It never fired as a
  wrong keyword either (wrong-kw = 0 everywhere).
- **LISTEN UP SPOT is weak** (best 3/10, at legacy gain).
- **ch1 (ASR beam) wakes far better than ch0 (processed)**: for the two kept phrases,
  11/13 vs 6/13 fan clips (all phrases 13/23 vs 7/23), and 3/4 vs 0/4 under babble. ch0's NS/AGC output clips (peaks reach 1.0) and HEY SPOT dies on it.
- ch1's cost is lower absolute SNR (see section 1). ASR accuracy on ch1 is **untested**
  because the server was down.
- False fires: **0 in 2 x 120 s** for every phrase and every setting. The 95 % upper
  bound from 4 min of zeros is about 0.75/min, so this can't prove the brief's
  <= 0.2/min target. It only shows nothing obviously wrong.
- KWS latency: the wake fires 1.7-2.7 s into a clip, about 0.8-1.4 s after the phrase
  starts. With pre-roll 25, the command utterance starts 0.36-0.39 s *before* the fire
  (the "start lag" column is negative). Whether the first command word survives in
  one-breath "Hey Spot what is..." needs ASR to confirm.

### KWS score x threshold sweep (peak 0.9; hits are over wake + pause + overlap clips)

| score / thr | fan ch0 HEY / EXC / LIS / THERE | fan ch1 HEY / EXC / LIS / THERE | babble ch0 (4 phrases) | babble ch1 (4 phrases) |
|---|---|---|---|---|
| 1.0 / 0.05 | 1/8 5/5 0/5 0/5 | 4/8 5/5 0/5 0/5 | 0/4 | 3/4 |
| 1.0 / 0.3 | 1/8 5/5 0/5 0/5 | 3/8 5/5 0/5 0/5 | 0/4 | 2/4 |
| **1.5 / 0.05** | 1/8 5/5 1/5 0/5 | **6/8 5/5 2/5 0/5** | 0/4 | **3/4** |
| 1.5 / 0.1 | 1/8 5/5 1/5 0/5 | 6/8 5/5 2/5 0/5 | 0/4 | 3/4 |
| 1.5 / 0.2 | 1/8 5/5 1/5 0/5 | 4/8 5/5 2/5 0/5 | 0/4 | 3/4 |
| 1.5 / 0.3 | 0/8 5/5 0/5 0/5 | 0/8 5/5 1/5 0/5 | 0/4 | 2/4 |
| 2.0 / 0.05 | 1/8 5/5 1/5 0/5 | 6/8 5/5 2/5 0/5 | 1/4 | 3/4 |
| 2.0 / 0.1 | 1/8 5/5 1/5 0/5 | 5/8 5/5 2/5 0/5 | 1/4 | 3/4 |
| 2.0 / 0.2 | 1/8 5/5 1/5 0/5 | 2/8 5/5 2/5 0/5 | 1/4 | 2/4 |
| 2.0 / 0.3 | 0/8 5/5 0/5 0/5 | 0/8 5/5 0/5 0/5 | 1/4 | 1/4 |

Every row had 0 false fires and 0 wrong-keyword fires. Thresholds above 0.1 lose HEY
SPOT. Score 2.0 matches 1.5 on ch1, and with no false-fire evidence either way there's
no reason to move off 1.5 / 0.05. The 1.5 / 0.05 row reproduces the main run exactly.

### Wake gain A/B (score 1.5, threshold 0.05)

| setting | fan ch0 HEY / EXC / LIS | fan ch1 HEY / EXC / LIS | babble ch0 | babble ch1 |
|---|---|---|---|---|
| peak_target 0.9 (peak-safe) | 1/8 5/5 1/5 | **6/8** 5/5 2/5 | 0/4 | 3/4 |
| legacy x8 (<= 0) | 2/8 5/5 1/5 | 4/8 5/5 3/5 | 1/4 (EXC) | 3/4 |

On ch1, peak-safe 0.9 is better for HEY SPOT (+2) and worse for LISTEN UP (-1).
Keep 0.9. On ch0, legacy x8 is marginally better, but ch0 isn't recommended.

## 3. Endpointing (pause clips)

The pause clips have gaps of 1.13-1.2 s by Silero. **0/3 pause cut-offs and 0 wake
cut-offs at every setting tried**: end_silence_s 1.0 / 1.2 / 1.5, short 0.9 / 1.0 /
1.2 / 1.5, on both channels. The effective end-of-utterance silence is about
`end_silence + 0.5 s`, because Silero's 0.5 s hangover keeps frames flagged as speech
before the endpointer starts counting silence. That's why a 1.2 s pause survives a
1.0 s setting. The evidence is n = 3 pause clips (fan only, ch0 and ch1) with
operator-timed pauses, and no ASR, so treat the lower values cautiously.

## 4. Quick reply after the chime (first-word capture)

LISTENING from t = 0. The recorded chime is masked with sidecar timestamps plus
`speaker_tail_s`.

| condition / ch | clip | chime (s) | speech onset (s) | utterance t0 (s) | margin (s) | first word |
|---|---|---|---|---|---|---|
| fan ch0 | quick_reply_1 / 2 / 3 | 0.51-0.70 | 1.30 / 1.17 / 1.10 | 0.81 / 0.78 / 0.63 | 0.49 / 0.39 / 0.47 | yes / yes / yes |
| fan ch1 | quick_reply_1 / 2 / 3 | 0.51-0.70 | 1.30 / 1.17 / 1.10 | 0.81 / 0.66 / 0.63 | 0.49 / 0.51 / 0.47 | yes / yes / yes |
| babble ch0 | quick_reply_1 / 2 | 0.76-0.98 | 1.26 / 1.30 | 0.78 / 0.78 | 0.48 / 0.52 | yes / yes |
| babble ch1 | quick_reply_1 / 2 | 0.76-0.98 | 1.26 / 1.30 | 0.78 / 0.78 | 0.48 / 0.52 | yes / yes |

**First word captured on 10/10** with pre-roll 25 (0.39-0.52 s of margin). The same
fan ch0 clips with the old pre-roll 10 gave margin 0.04 s on clip 1 and **missed the
first word on clips 2 and 3**, so this validates the Gate 2 fix. Every one of these
utterances is still dropped by the SNR gate at 10 dB (section 1). The chime tail can
sit inside the captured audio sent to ASR (utterance t0 0.63 s is before chime end
0.70 s). Its effect on ASR is untested.

## 5. Chime self-trigger

Every `AudioFeedback` cue (11 of them) was rendered, resampled 48 kHz -> 16 kHz and
replayed on floor-level noise. Two levels were used: digital, and scaled to the
recorded cue's mic level. Counts are utterances started.

- **Mask on (real loop): 0 for all 11 cues**, every condition, both channels and both
  levels. The wake KWS fired on 0 cues.
- Mask off: `error` starts 1 utterance everywhere, and `thinking`/`ready` start 1 in
  some runs. The mask is load-bearing; keep it.
- The recorded (acoustic) chime with the mask off started 0 utterances in all 5
  quick-reply clips.

## 6. Speaker lock correctness

The lock is taken from speech DoA in the 1.5 s before the wake fire, after
`mic_to_body_deg(offset 179, sign -1)`, and counts as correct within 35 deg:

| condition | locks from the wake window | correct |
|---|---|---|
| fan ch1 (13 wakes) | front 2/2, right 2/3, back 1/2, left 0/2, pause clips 1/3, overlap 0/1 (44 deg) | **6/13** |
| fan ch0 (7 wakes) | | 4/7 |
| babble ch1 (3 front wakes) | 122, 122, 119 | **0/3** |

- In babble, the XVF3800 DoA points at the crowd (background DoA reads 60-88 deg mic,
  about 90-120 deg body), not at the visitor. **The lock will attach to the crowd.**
- In the fan set the DoA often sticks to one value for a whole clip (for example
  271 deg mic on several clips from different angles).
- Overlap: only 1 overlap clip woke (ch1 overlap_1). Its lock was wrong (44 deg body)
  but the command was accepted, since lock and command share the same wrong DoA.
- No beams-on recording exists, so there's no evidence to turn on `beam_steering`.

With a wrong lock and a 35 deg window, the visitor's follow-ups get rejected. See
the recommendation.

## RECOMMENDATION

Confidence is low to medium throughout: 1-8 clips per cell, no ASR, 4 min of
background. Values:

```yaml
# config/wake_phrases.txt
HEY SPOT
EXCUSE ME SPOT

# config/tuning.yaml
mic_channel: 1           # kept-phrase wakes 11/13 vs 6/13 (fan), 3/4 vs 0/4 (babble); ASR on ch1 untested
kws_score: 1.5           # unchanged; 2.0 ties on ch1, no false-fire evidence to justify a change
kws_threshold: 0.05      # unchanged; >= 0.2 loses HEY SPOT
kws_peak_target: 0.9     # ch1 HEY SPOT 6/8 vs 4/8 legacy x8
min_snr_db: 2.0          # STOPGAP. 10 drops 13/13 real ch1 commands; 2 drops 0-1/13, passes 0-2 noise utts/2 min
end_silence_s: 1.2       # 0 cut-offs at 1.0-1.5; 1.2 keeps margin for chained commands
end_silence_s_short: 1.0 # 0/3 pause cut-offs at 0.9-1.5 (Silero adds ~0.5 s hangover)
beam_steering: false     # no beams-on data; lock is already unreliable
# optional, outside the requested list:
lock_window_deg: 180.0   # effectively accept all directions: wake-window lock correct 6/13 (fan), 0/3 (babble)
```

- **Phrases:** drop HEY THERE SPOT (0/10, never fires) and LISTEN UP SPOT (best 3/10).
  Keep HEY SPOT and EXCUSE ME SPOT; both 5/5 on non-overlap fan ch1 clips and 1/1 in
  babble. Coach visitors to say "Hey Spot". Re-run
  `scripts/setup_kws.py --phrases-file config/wake_phrases.txt`.
- **min_snr_db** must come down or the demo answers nothing. 2.0 is chosen from ch1 rows
  only; ch0 SNRs run about 4-10 dB higher and need a different value. Follow-up code fix:
  gate on p90 or exclude Silero hangover frames.
- **Check live before the demo:** one "Hey Spot, what is this poster about" on ch1 with
  ASR running, to confirm the transcript keeps "what" and that ch1 ASR quality is fine.
  If ch1 ASR is poor, fall back to ch0 with `min_snr_db: 4` (2/7 ch0 drops) and accept
  the weaker wake.

## Appendix: per-clip notes

- `start lag` = utterance first frame minus wake fire time. It's negative because the
  25-frame pre-roll reaches back before the fire.
- Full per-clip tables: run with `--clips`. JSON from both runs: `--json`.
