# Stage 2F — Wake-Word & ASR Robustness Overhaul

**Status:** design v2 (post blind-review), pending spec review
**Branch:** `stage2f-wake-asr-overhaul` (off `main`)
**Date:** 2026-05-28
**Revision:** v2 incorporates a 4-agent blind review. Major changes: Sortformer
dropped (ONNX export broken + fails Orin RTF gate) → Personal VAD; barge-in fix
redesigned against actual code; chunker preserves the GBNF JSON interface; new
top-level Safety section; rollback-flag independence fixed; eval de-circularized;
embedding model swapped; livekit-wakeword existence promoted to a P3 gate.

## Problem

Operator-reported failures in the voice pipeline:

1. **Wake word does not reliably detect the activation phrase.** Current
   sherpa-onnx zipformer KWS (general GigaSpeech ASR repurposed as keyword
   spotter) measured 1/5 fire rate at Stage 2A T12; flagged for Stage 2B. The
   Stage 2B livekit-wakeword custom-train was scaffolded
   (`configs/wake/wakeword_hey_spot.yaml`, `scripts/training/*`,
   `src/voice_control/wake/livekit.py`) but never executed —
   `/mnt/ssd/wake-models/` does not exist.

2. **ASR chunking breaks when the operator speaks while Spot is speaking.**
   Mic-mute interlock leaks audio and corrupts detector state.

Surfaced during design (broad-scope, path B):

3. **Crowd robustness** — Spot operates in crowds; distinguish operator from
   ambient speakers.

4. **TTS chunker splits sentences oddly** — regex `[.!?]\s` splits abbreviations
   ("Dr.", "e.g.", "U.S.") and ignores clause boundaries.

## Goals

- Wake fires reliably (TPR ≥ 0.9 at FAR ≤ 1/hr on a held-out eval set).
- No audio leak / stale-detector-state around TTS playback.
- Operator-vs-crowd discrimination, tiered by difficulty, **never at the cost of
  safety** (see Safety).
- Chunker never splits mid-word/abbreviation; clause-aware low-latency streaming.
- Every change behind an env flag with a safe default that degrades to
  pass-through; reproducible eval drives decisions.

## Non-goals

- Barge-in / interrupt-Spot-mid-speech. **Dropped by design** — operator
  re-wakes after every response (Alexa-style). No AEC needed.
- Replacing Kokoro TTS, Parakeet ASR, or Gemma 4 backend.
- Multi-operator handoff, voice anti-spoof, cross-session person re-ID.
- Reverse-engineering the XVF3800 proprietary HID (DoA / LED ring not exposed).

## Safety (non-negotiable, applies to every phase)

A wake word with an over-fire bias on a 30 kg actuated robot is a hazard, not a
nuisance: a false wake opens a window where bystander speech can be transcribed
into a motion command. Rules, independent of scope:

- **E-stop / stop / freeze are NEVER gated.** The existing `safety_only` path
  (`client_mic.py:962`, processes stop/freeze/estop even in `WAKE_WORD` state
  without a wake) is preserved and extended — it must bypass wake, speaker-ID,
  and crowd gating entirely. `src/estop.py` physical e-stop is untouched.
- **Motion-causing actions gate behind speaker-ID confirm OR a confirm-chime.**
  Actions that move the robot (`move`, `walk`, `stand`, `sit`, `dock`,
  `undock`, `follow`, `go_to`) require either a confirmed speaker-ID session or,
  when speaker-ID is off/unavailable, an explicit confirm step. Non-motion
  actions (describe, set_persona, set_volume) do not.
- **Over-fire threshold only ships once the gate is live.** Until speaker-ID
  (P4) is deployed, the wake word runs at a *balanced* threshold (≈0.5), not the
  over-fire 0.42. Over-fire + gate is enabled together in P4, never wake-alone.
- **Speaker-ID must have a hard bypass.** A bad enrollment (cold, whisper,
  distance, mis-steered beamformer) must not lock the operator out. Provide
  `SPOT_SPEAKER_ID=off` (env) AND a runtime bypass phrase that is itself in the
  always-on `safety_only` set. Reject never silently drops a *safety* command.
- **Reject feedback ships WITH the drop logic, not in the indicators phase.** A
  distinct reject chime/LED fires before any utterance is dropped, so the
  operator can tell "heard-but-rejected" from "not heard". This moves out of P7
  into P4.

A dedicated safety-reviewer pass (`safety-reviewer` agent) gates the speaker-ID
and state-machine commits before merge.

## Hardware constraints (verified)

Mic = **Seeed reSpeaker XVF3800 4-Mic Array** (USB `2886:001a`, confirmed via
`lsusb` + `docs/getting-started/hardware.md`). 2 channels only (ch0 beamformed,
ch1 reference); AEC + beamforming firmware-fixed; **DoA and LED ring NOT exposed
to host** (proprietary HID, no Linux driver). Consequence: crowd robustness uses
speaker embedding + Personal VAD + visual; "listening" indicator uses Spot
onboard LEDs + web panel + chimes.

Compute: Jetson AGX Orin 64GB (Ampere ~5.32 TFLOPS FP32 / 21 TOPS INT8, CUDA
12.6, onnxruntime-gpu 1.23.0). `/mnt/ssd` (~1.8TB) for models — disk not a
constraint; VRAM ~46GB headroom. Operator has L40S/H200 cluster for training.

## Component 1 — Wake word (livekit-wakeword custom train)

**P3 GATE (do first, blocks everything else in this component):** verify
`github.com/livekit/livekit-wakeword` still exists, its license is permissive,
and it trains on the cluster + runs inference on Orin aarch64. It is NOT on PyPI
(git-clone + `--no-deps -e` install). If this fails, the whole component falls
back to "tune sherpa-onnx threshold + add keyword variants" (lower ceiling) and
the over-fire/push-to-talk safety story changes.

Three A/B candidates trained in one cluster batch (~3h):

| ID | Phrase | Hard-negative focus |
|----|--------|---------------------|
| A | "Hey Spot" | hot spot, hey scott, spotted, spotlight |
| B | "Hi Spot" | hi scott, hipster, high spot |
| C | "Big Yellow" | big fellow, big mellow, pig yellow, big yell |

Config diff from existing YAML: `model_size: small → large` (V2),
`n_samples: 25000 → 50000`, `steps: 50000 → 80000`. **Threshold staging:**
train at `target_fp_per_hour: 1.0` (high recall) but **deploy at balanced
runtime threshold ≈0.5 until P4 gate is live, then drop to 0.42** (Safety).

Training data: Piper VITS pool (904 synthetic speakers, built-in) + ~30–50 real
positives via XVF3800 (`scripts/record_wake_positives.py`) + ~50 ElevenLabs
stylings (`scripts/synth_wake_elevenlabs.py`, **verify free-tier ToS permits
training-data use** — V10). **Data-sufficiency risk:** 30–50 real positives is
low for a custom KWS; synthetic-heavy data risks train/serve mismatch. Fallback
gate: **if held-out TPR < 0.9 by the agreed date, ship sherpa-onnx + a
push-to-talk button** rather than a flaky low-threshold wake.

Deploy: eval winner → `/mnt/ssd/wake-models/active.onnx` →
`SPOT_WAKE_BACKEND=livekit`. Rollback: `SPOT_WAKE_BACKEND=sherpa_onnx`.

Open verifications: V1 (custom-positive-dir field), V2 (`model_size: large`),
V10 (ElevenLabs ToS), V11 (livekit repo existence + Orin trainability — the P3
gate above).

## Component 2 — Barge-in fix (redesigned against actual code)

Corrected diagnosis (v1 was wrong — `_drain_audio_queue()` already exists at
`client_mic.py:1043`, called at :969 and :1008, and keeps the last ~33 frames):

- **Real gap (was Leak A/D):** during *streaming* TTS the chunker issues many
  `speak()` → `enqueue_render` calls; the main loop keeps pulling the ~33
  retained frames and processing them through VAD/KWS *while* `RESPONDING`, and
  the VAD/KWS streaming state is frozen mid-decision across the mute and never
  reset.
- **Leak B does NOT exist** — `is_busy()` is backed by an inflight counter
  incremented under lock at enqueue (`audio_player.py:222-234`); no race window.
  Removed from scope.
- **Leak C (OS tail):** real but small; covered by the existing cooldown.

Fix (all single-threaded in the **main loop**, never the PortAudio callback,
never reaching across modules into the player thread):
- Detect the player busy→idle edge in the main loop. On that edge, once, reset
  `vad`, `wake_detector`, `preroll_buffer`, `pending_speech_frames`,
  `consecutive_speech`, and clear any partial `speech_buffer`, then call the
  existing `_drain_audio_queue()`.
- **Cooldown:** keep `SPEAKER_TAIL_COOLDOWN_S` modest (≤0.2, not 0.4 — a larger
  value adds dead-mic latency to every wake-fire path and double-counts with the
  state reset). Gate the cooldown on TTS tails, not short beeps, if measurement
  shows beeps over-extend the mute window.

Verify: `eval_bargein.py` asserts no ASR submission during `is_busy()`, state
reset on busy→idle edge, and a correct first post-TTS wake fire.

## Component 3 — Speaker-ID cascade (crowd robustness)

Tiered; ship cheap default, escalate only where measured data demands.

**Tier 1 (default): speaker embedding.**
- Day-one: **WeSpeaker ResNet34-LM ONNX** (0.72% EER, ~26MB, CC-BY-4.0,
  **pre-shipped ONNX** — zero export risk, drops into ort-gpu 1.23). Chosen over
  ResNet293-LM because the 293's "~15–25ms on Orin" was an unbenchmarked,
  optimistic guess for a 28-GMAC net.
- Upgrade: **ReDimNet2-B6** (0.29% EER, MIT, ~13 GMAC) — best open-weight
  embedding; **verify pretrained weights have shipped** (paper Mar 2026, weights
  often lag — V12). Self-export to ONNX.
- On wake fire: extract operator fingerprint, lock to session; gate subsequent
  utterances by cosine similarity.

**Tier 2 (`SPOT_CROWD_MODE=on`): Personal VAD (replaces Sortformer).**
- Sortformer dropped: `streaming_export()` ONNX is known-broken (NeMo
  #15077/#15536 open) and Ada→Orin extrapolation predicts RTF ~1.5–3.0, failing
  its own 0.7 gate.
- Instead train a **Personal VAD** (speaker-conditioned frame-level VAD, ~1–3M
  params, INT8, <1ms/frame) on the cluster, conditioned on the WeSpeaker
  embedding. Frame-level "is the enrolled operator speaking" → gate ASR frames.
  Handles overlap far better than clip-level embedding cosine, and is small
  enough to avoid the continuous-GPU-contention risk that killed Sortformer.
- No pretrained weights exist → ~3–5 days training (cluster). This is the main
  net-new ML effort; gated behind `SPOT_CROWD_MODE` and not on the demo path.

**Tier 3 (deferred): target-speaker extraction.**
- OpenSpeakerBeam-SS (MIT, 7.64M params, ONNX-ready) — waveform separation
  before ASR. Only if Tier-2 data shows residual WER under heavy overlap.

**Visual cascade (parallel gate on wake fire):**
- YOLOv8n person detect (existing) → MediaPipe Face Mesh → mouth-motion variance
  over a 5-frame burst. **Mouth-motion is a weak tiebreak hint only** (fires on
  chewing/smiling/back-channeling; no AV-sync) — never a sole confirmer.
- **Mic ownership during cascade:** the in-breath command path ("Hey Spot stand
  up") must NOT be delayed by visual escalation. In-breath commands use the
  embedding-only fast path (~30ms); arm-camera escalation (~700ms–1.2s) is used
  ONLY for a *separate* follow-up utterance when fisheye is ambiguous, and only
  after confirming the bounded `audio_queue` (~1s) won't overflow — otherwise
  re-prompt. Define mic ownership explicitly in the implementation.
- VLM ("who is talking") only as a manual high-stakes escalation (2–5s latency).

Thresholds (`EMBEDDING_MATCH_THRESHOLD=0.75`, `EMBEDDING_GATE_THRESHOLD=0.65`,
`MOUTH_THRESHOLD`, `ESCALATION_RATIO=2.0`) are initial guesses tuned by the crowd
eval. **Reject feedback + hard bypass per Safety ship in this component.**

Open verifications: V3 (WeSpeaker ONNX load), V4 (MediaPipe aarch64), V9
(arm-camera control + bearing latency), V12 (ReDimNet2-B6 weights).

## Component 4 — TTS chunker (preserve JSON interface)

`TTSChunker` carries load-bearing logic beyond sentence-splitting: `suppressed`
plus the GBNF-JSON-stream parsing (`_RESPONSE_KEY_RE`, `_DESCRIBE_ACTION_RE` at
`spot_tts.py:245-289`), consumed at `client_mic.py:1282,1373`. `stream2sentence`
/ BlingFire consume *plain text*, not the raw JSON token stream.

Therefore: **keep the JSON-extraction front-end and `suppressed` interface;
swap only the sentence-splitting backend** to operate on the already-extracted
`response` text:
- Driver `stream2sentence` (MIT) + detector **BlingFire** (MIT, ~1MB,
  abbreviation-safe). **Re-benchmark BlingFire's sentence API on aarch64** — the
  "0.06ms / 450×" figure is a tokenization benchmark, not segmentation (V8).
- Clause fallback: rule split on `,;:` when buffer > ~80 chars.
- SaT `sat-3l-sm` ONNX-CPU only as a last-resort for unpunctuated tails (note:
  SaT is scored by **F1**, not EER).

20-case TDD set (`tests/voice_control/tts/test_chunker.py`) gates correctness;
tests must include the `suppressed`/describe-action path, not just splitting.

## Component 5 — Eval harness (de-circularized)

Corpora under `tests/audio/wake_eval/`: positives, hard negatives (confusables +
10-min conversation + recorded TTS playback), crowd scenarios.

- **De-circularization:** training positives and eval positives must use
  **different speakers and a different room** where possible — at minimum a
  held-out speaker set. Same-operator-same-room TPR is an optimistic ceiling, not
  robustness; report it as such.
- Synthetic crowd (voices through a 2nd speaker) is a stand-in; flag measured
  crowd numbers as optimistic vs real bodies.

Runners: `eval_wake.py` (ROC sweep → TPR, FAR/hr, latency, AUC → pick
candidate+threshold at FAR ≤ 1/hr), `eval_speaker.py` (operator-accept /
crowd-reject / overlap-accept per tier → decides if Tier 2 is needed),
`test_chunker.py` (20/20), `eval_bargein.py` (leak asserts). **No Sortformer
bench** (dropped); add a Personal-VAD eval if Tier 2 is built. `make eval-voice`
runs the suite; baseline at `tests/audio/wake_eval/baseline.json`.

## Component 6 — State machine + indicators + A/B toggle

Enum migration: current `VoiceState = {WAKE_WORD, LISTENING, RECORDING}`
(`RECORDING` declared-unused). Add `SPEAKER_ID`, `THINKING`, `RESPONDING`;
remove or repurpose `RECORDING`. **Every new state has an explicit edge back to
`WAKE_WORD`** — in particular `SPEAKER_ID` reject → `WAKE_WORD` (with reject
feedback per Safety), and a `LISTENING`-timeout edge.

`SPOT_LISTEN_MODE`: `alexa` (default; post-TTS → `WAKE_WORD`, re-wake required)
vs `continuous` (legacy; post-TTS → `LISTENING` 15s).

**Rollback-flag independence (must hold):** each gate degrades to pass-through
when its upstream flag is off. `SPOT_SPEAKER_ID=off` → no session object
required; `continuous` mode must not depend on a speaker-ID session existing.
`SPOT_CROWD_MODE=off` → Personal-VAD frame gate hard-bypassed (pass all frames),
never silent-drops. The flag cross-product is a test matrix in `eval`.

Session (`embedding`, `visual_id`, timestamps, `command_count`): single-turn in
`alexa`, 15s/45s timeout in `continuous`; created on `SPEAKER_ID` confirm; absent
and not required when `SPOT_SPEAKER_ID=off`.

Indicators (4 states, graceful-degrade): web badge (`server.py`), chimes
(`audio_feedback.py` — wake / listening-close / response-done **/ reject**),
Spot onboard LEDs (`src/voice_control/spot_leds.py`, **verify BD SDK LED control
— V5**; drop channel if absent). Reject chime is NOT deferred — it ships in P4.

## Phasing

| Phase | Work | Risk | Gate |
|-------|------|------|------|
| P0 | Barge-in fix (Component 2, redesigned) | low | eval_bargein asserts |
| P1 | TTS chunker (backend swap, JSON interface kept) | low | 20/20 TDD incl. suppress path; V8 |
| P2 | Eval harness + held-out corpus | low | runners run |
| P3 | **Verify livekit (V11)** → wake train (3 candidates) | med | TPR≥0.9 @ FAR≤1 on held-out, else sherpa+PTT fallback |
| P4 | Speaker embedding Tier 1 **+ Safety (gate, bypass, reject feedback)** + over-fire enable | med | crowd-eval rates; safety-reviewer pass |
| P5 | Visual cascade (mouth-motion tiebreak only) | high | speaker-ID crowd accuracy |
| P6 | State machine + A/B + session + flag cross-product | med | both-mode smoke; rollback matrix |
| P7 | Remaining indicators (web badge, LEDs) | low | per-state visual confirm |
| P8 | Personal VAD Tier 2 (replaces Sortformer) | high | crowd-eval overlap improvement |

P0/P1/P2 ship immediately. **Demo fast path = P0 + P1 + P3** (with balanced
threshold; motion-action confirm-chime substitutes for the not-yet-live
speaker-ID gate). If P3's data-sufficiency or livekit gates fail → P0 + P1 +
sherpa + push-to-talk is the honest demo fallback.

## Rollback

`SPOT_WAKE_BACKEND=sherpa_onnx`, `SPOT_SPEAKER_ID=off`, `SPOT_CROWD_MODE=off`,
`SPOT_LISTEN_MODE=continuous`; chunker commit git-revertible. Every gate
degrades to pass-through when its flag is off (tested cross-product).

## Dependencies

| Dep | Phase | Note |
|-----|-------|------|
| `blingfire` | P1 | pin; verify aarch64 (V8) |
| `stream2sentence` | P1 | pure Python |
| `mediapipe` | P5 | pin; verify aarch64 (V4) |
| Personal VAD (custom-trained) | P8 | cluster; no pretrained weights |
| ~~`nemo_toolkit` / Sortformer~~ | — | **dropped** (ONNX broken, RTF fails gate) |

Model artifacts on `/mnt/ssd`: `wake-models/`, `spkr-embed/`, `pvad/`,
`segmenters/`, `tse/` (Tier 3, deferred).

## Open verifications (consolidated)

- V1 livekit custom-positive-dir field — P3
- V2 livekit `model_size: large` — P3
- V3 WeSpeaker ResNet34-LM ONNX on ort-gpu 1.23 — P4
- V4 MediaPipe Face Mesh aarch64 — P5
- V5 BD SDK LED control — P7
- V8 BlingFire aarch64 sentence-API re-bench — P1
- V9 arm-camera control + bearing latency — P5
- V10 ElevenLabs free-tier ToS for training data — P3
- V11 livekit-wakeword repo existence + Orin trainability (**P3 gate**) — P3
- V12 ReDimNet2-B6 pretrained weights shipped — P4 (upgrade only)

(Dropped from v1: V6/V7 Sortformer ONNX + Orin RTF — Sortformer removed.)
