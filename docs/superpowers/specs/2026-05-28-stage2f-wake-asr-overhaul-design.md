# Stage 2F — Wake-Word & ASR Robustness Overhaul

**Status:** design approved, pending spec review
**Branch:** `stage2f-wake-asr-overhaul` (off `main`)
**Date:** 2026-05-28

## Problem

Two operator-reported failures in the voice pipeline:

1. **Wake word does not reliably detect the activation phrase.** The current
   sherpa-onnx zipformer KWS (general GigaSpeech ASR repurposed as a keyword
   spotter) was measured at a 1/5 fire rate during Stage 2A T12 and flagged as
   a known gap to fix in Stage 2B. The Stage 2B custom-train (livekit-wakeword)
   was scaffolded (`configs/wake/wakeword_hey_spot.yaml`,
   `scripts/training/*`, `src/voice_control/wake/livekit.py`) but never
   executed — `/mnt/ssd/wake-models/` does not exist.

2. **ASR chunking breaks when the operator speaks while Spot is speaking.** The
   mic-mute interlock (`AudioPlayer.is_busy()` + 0.15s tail cooldown) leaks
   audio and corrupts detector state. Root causes traced below.

Two additional issues surfaced during design:

3. **Crowd robustness.** Spot operates in crowds. The system must distinguish
   the operator's voice from ambient speakers.

4. **TTS chunker splits sentences oddly,** producing phonetically broken
   playback. Root cause: regex `[.!?]\s` splits on abbreviations
   ("Dr.", "e.g.", "U.S.") and ignores clause boundaries.

## Goals

- Wake word fires reliably for the operator (TPR ≥ 0.9 at FAR ≤ 1/hour).
- No audio leak or stale-detector-state bugs around TTS playback.
- Operator-vs-crowd voice discrimination, tiered by environment difficulty.
- TTS chunker never splits mid-word or mid-abbreviation; emits clause-aware
  chunks for low-latency streaming.
- Every change behind an env flag with a safe default and a rollback path.
- Decisions driven by a reproducible eval harness, not vibes.

## Non-goals

- Barge-in / interrupt-Spot-mid-speech. **Dropped by design** — the operator
  re-wakes after every response (Alexa-style). Robustness beats interrupt;
  acoustic echo cancellation is therefore unnecessary.
- Replacing Kokoro TTS or the Parakeet/Gemma backends.
- Multi-operator handoff, voice anti-spoof, person re-ID across sessions.
- Reverse-engineering the XVF3800 proprietary HID (DoA / LED ring not exposed
  — see Hardware Constraints).

## Hardware constraints (verified)

Mic is a **Seeed reSpeaker XVF3800 4-Mic Array** (USB `2886:001a`, confirmed via
`lsusb` and `docs/getting-started/hardware.md`).

- Exposes **2 channels only**: ch0 beamformed, ch1 reference. AEC + beamforming
  are firmware-fixed; no host control of AGC/NS.
- **Direction-of-arrival is NOT exposed** to the host. A proprietary HID
  interface (190-byte report) exists but has no Linux driver; decoding it is
  days of reverse-engineering with no guarantee.
- **LED ring is NOT host-controllable** via documented USB.

Consequence: the original "DoA + mic LED indicator" idea is infeasible on this
hardware. Crowd robustness uses speaker embedding + visual cascade instead; the
"listening" indicator uses Spot's onboard LEDs + web panel + chimes.

Compute target: Jetson AGX Orin 64GB (Ampere ~5.32 TFLOPS FP32 / 21 TOPS INT8,
CUDA 12.6, onnxruntime-gpu 1.23.0). `/mnt/ssd` (~1.8TB) available for models —
disk size is not a constraint; VRAM is generous (~46GB headroom); the only real
compute risk is continuous Sortformer load (Tier 2, see below).

## Architecture overview

```
XVF3800 (ch0 beamformed) → 16kHz PCM16
  → Wake word (livekit-wakeword custom .onnx, A/B-selected phrase)
  → Speaker-ID cascade (embedding + visual, on wake fire)
  → ASR (Parakeet TDT v3 CUDA, existing) — embedding-gated frames
  → LLM Brain (Gemma 4 E4B, existing)
  → TTS chunker (stream2sentence + BlingFire) — NEW
  → Kokoro TTS (existing) — mic muted during playback
  → state → WAKE_WORD (Alexa-style re-wake)
Indicators: web panel + chimes + Spot onboard LEDs
```

## Component 1 — Wake word (livekit-wakeword custom train)

Execute the dormant Stage 2B plan, with three A/B candidate phrases trained in
one cluster batch (operator has L40S / H200 access — ~3h wall for all three).

Candidates:

| ID | Phrase | Rationale | Hard-negative focus |
|----|--------|-----------|---------------------|
| A | "Hey Spot" | control, branding | hot spot, hey scott, spotted, spotlight |
| B | "Hi Spot" | same brand, less collision | hi scott, hipster, high spot |
| C | "Big Yellow" | distinctive, Dartmouth "Big Green" pun | big fellow, big mellow, pig yellow, big yell |

Config (`configs/wake/wakeword_<id>.yaml`), diff from existing:
`model_size: small → large`, `n_samples: 25000 → 50000`,
`n_samples_val: 5000 → 10000`, `steps: 50000 → 80000`,
`target_fp_per_hour: 0.1 → 1.0` (over-fire bias per operator request —
"rather wake too much than not enough"). Runtime
`LIVEKIT_WAKE_THRESHOLD: 0.55 → 0.42`, tunable post-deploy with no retrain.

Training data (no human-recording-marathon required):
- Piper VITS SLERP pool (904 synthetic speakers) — livekit-wakeword built-in.
- ~30–50 real positives per candidate via XVF3800 (`scripts/record_wake_positives.py`),
  a 5-distance × 3-angle × 3-style grid (~10 min recording).
- ~50 ElevenLabs expressive stylings per candidate (whisper/shout/fast/slow),
  ~1.2k chars total — free tier (`scripts/synth_wake_elevenlabs.py`).
- ACAV100M negatives + per-candidate hard negatives — built-in.

Deploy: eval harness picks winner → `scp` to `/mnt/ssd/wake-models/active.onnx`
→ `SPOT_WAKE_BACKEND=livekit`. Rollback: `SPOT_WAKE_BACKEND=sherpa_onnx`.

Open verifications: livekit YAML custom-positive-dir field name (V1);
`model_size: large` exists (V2).

## Component 2 — Barge-in leak fix

The mic-mute interlock leaks audio and corrupts detector state. Verified leaks:

- **Leak A (dominant):** `audio_queue` (33 frames ≈ 1s) is not drained when TTS
  starts. Pre-TTS frames are processed by the main loop *during* TTS playback,
  triggering spurious ASR submissions / wake fires.
- **Leak B (minor):** ~30ms race window between callback and `is_busy()` flip.
- **Leak C (tight margin):** OS playback tail 30–100ms on Jetson USB audio;
  0.15s cooldown barely covers it.
- **Leak D (structural):** VAD + KWS streaming state freezes mid-decision during
  mute and is never reset; first 0.5–2s after TTS ends are noisy.

Fix (≈30 lines, two files, no AEC, no new threads):
- Drain `audio_queue` when `_inflight` rises 0→1 in
  `AudioPlayer.enqueue_render`/`enqueue_raw` (under lock).
- Reset `vad`, `wake_detector`, `preroll_buffer`, `pending_speech_frames`,
  `consecutive_speech`, `speech_buffer` once when the callback transitions from
  cooldown to forwarding.
- Bump `SPEAKER_TAIL_COOLDOWN_S` 0.15 → 0.4.

The threading architecture itself (non-blocking callback → bounded queue → main
loop, separate TTS worker, watchdog) is correct and unchanged.

## Component 3 — Speaker-ID cascade (crowd robustness)

Tiered: ship cheap default, escalate only where measured data demands it.

**Tier 1 (default): speaker embedding.**
- Model: **WeSpeaker ResNet293-LM ONNX** (0.45% EER VoxCeleb1-O, ~110MB,
  CC-BY-4.0, pre-shipped ONNX, ~15–25ms/clip on Orin CUDA, ~150MB VRAM).
  Day-one pick. ResNet34-LM (0.72% EER, lighter) is the fallback if V3 fails.
- On wake fire: extract operator fingerprint, lock to session. Gate each
  subsequent utterance by cosine similarity.
- Upgrade path (v2): ReDimNet-B6-LM (0.37% EER, MIT, self-export).

**Tier 2 (`SPOT_CROWD_MODE=on`): target-speaker diarization.**
- Embedding alone fails under simultaneous speech (overlap contaminates the
  embedding). Tier 2 adds **NVIDIA Streaming Sortformer 4spk-v2** (CC-BY-4.0,
  online overlap-aware diarization, ONNX export via NeMo `streaming_export()`).
  Enroll operator as speaker-0; gate ASR frames by the speaker-0 mask.
- **Gated on Orin RTF bench (V7).** If RTF > 0.7, fall back to INT8/TRT, or
  train a Personal VAD (~3–5 days), or stay Tier-1-only. Sortformer is the only
  continuous GPU workload — it is the sole real compute risk.

**Tier 3 (deferred): target-speaker extraction.**
- OpenSpeakerBeam-SS (MIT, 7.64M params, ONNX-ready) — actual waveform
  separation before ASR. Only if Tier 2 data shows residual WER under heavy
  overlap.

**Visual cascade (parallel gate, runs on wake fire):**
- YOLOv8n person detect (existing) → MediaPipe Face Mesh → mouth-motion
  variance over a 5-frame burst spanning the wake window.
- Decision: clear single mouth-mover → confirm (~250ms total); ambiguous
  multi-person → escalate by rotating the arm camera toward the strongest
  candidate and re-meshing on the high-res hand camera (~700ms–1.2s).
- VLM ("who is talking") only as a manual high-stakes escalation — 2–5s latency
  is too slow for per-utterance gating.

Thresholds (`EMBEDDING_MATCH_THRESHOLD=0.75` initial,
`EMBEDDING_GATE_THRESHOLD=0.65` subsequent, `MOUTH_THRESHOLD`,
`ESCALATION_RATIO=2.0`) are initial guesses tuned by the crowd eval (Section 5).

Dropped utterances are silently discarded — the operator sees no response and
naturally re-wakes, avoiding a "robot refused my command" failure mode.

Open verifications: WeSpeaker ONNX load (V3); MediaPipe aarch64 (V4); arm-camera
control + bearing latency (V9).

## Component 4 — TTS chunker

Replace the regex with the production-proven streaming pattern (every
commercial streaming-TTS vendor — Cartesia, ElevenLabs, AssemblyAI, Pipecat,
vLLM-Omni — uses regex + buffer-delay, not neural segmenters; ML segmenters are
too slow for streaming and reserved for offline batch).

- **Driver:** `stream2sentence` (MIT) — purpose-built LLM-stream → sentence
  yields, with first-fragment-ASAP and deadline knobs.
- **Detector:** **BlingFire** (Microsoft, MIT, ~1MB) — 0.06ms/text, abbreviation-
  safe ("Dr.", "U.S.", "p. 55", "3 p.m."), 450× faster than SaT-CPU.
- **Clause fallback:** rule split on `,;:` when the buffer exceeds ~80 chars.
- **Last-resort:** SaT `sat-3l-sm` ONNX-CPU only for ambiguous/unpunctuated
  tails (rare). Total footprint <60MB, <2ms typical, zero CUDA contention.

New `src/voice_control/tts/chunker.py` (`StreamingClauseChunker`) replaces
`TTSChunker` internals in `spot_tts.py`. 20-case TDD set
(`tests/voice_control/tts/test_chunker.py`) gates correctness before integration.

Open verification: BlingFire aarch64 wheel (V8) — else pysbd pure-Python.

## Component 5 — Eval harness

Recorded-once corpora under `tests/audio/wake_eval/`: positives (45/candidate
grid), hard negatives (confusables + 10min conversation + recorded TTS
playback), crowd scenarios (operator solo / +1 / +crowd / overlap). Crowd
without many people: play synthetic voices through a second speaker while the
operator speaks live.

Runners:
- `scripts/eval_wake.py` — ROC sweep per candidate × threshold → TPR, FAR/hr,
  latency, AUC → pick (candidate, threshold) at FAR ≤ 1/hr.
- `scripts/eval_speaker.py` — operator-accept / crowd-reject / overlap-accept
  rates per tier → decides whether Tier 2 is needed for this environment.
- `tests/voice_control/tts/test_chunker.py` — 20/20 gate.
- `scripts/eval_bargein.py` — asserts no ASR during `is_busy()`, queue drained
  at TTS start, state reset post-cooldown, post-TTS wake fires.
- `scripts/bench_sortformer_orin.py` — exports Sortformer ONNX, measures Orin
  RTF at 0.32s and 1.04s buffers (tegrastats GR3D_FREQ), gates Tier 2.

`make eval-voice` runs the suite; baseline stored at
`tests/audio/wake_eval/baseline.json`; changes must not regress.

## Component 6 — State machine + indicators + A/B toggle

Alexa-style: `WAKE_WORD → SPEAKER_ID → LISTENING → THINKING → RESPONDING →
(branch)`. Branch on `SPOT_LISTEN_MODE`:

| Mode | Post-TTS | Use case |
|------|----------|----------|
| `alexa` (default) | → WAKE_WORD, re-wake required | robust, intentional, crowd-safe |
| `continuous` (legacy) | → LISTENING 15s | fast multi-turn, A/B comparison |

Session (`embedding`, `visual_id`, timestamps, `command_count`) created on
SPEAKER_ID confirm; single-turn in alexa mode, 15s/45s timeout in continuous.

Indicators (all reflect the 4 states, graceful-degrade if a channel is
unavailable):
- Web panel badge (extend `server.py`).
- Chimes (extend `audio_feedback.py`): add listening-close + response-done.
- Spot onboard LEDs (new `src/voice_control/spot_leds.py`): off / green /
  pulsing amber / blue. **Verify BD SDK LED control exists (V5)** — else drop
  LED channel.

## Phasing

| Phase | Work | Risk | Gate |
|-------|------|------|------|
| P0 | Barge-in leak fix | low | eval_bargein asserts |
| P1 | TTS chunker | low | 20/20 TDD |
| P2 | Eval harness + corpus | low | runner runs |
| P3 | Wake-word train (3 candidates) | med | TPR≥0.9 @ FAR≤1 |
| P4 | Speaker embedding Tier 1 | med | crowd-eval rates |
| P5 | Visual cascade | high | speaker-ID crowd accuracy |
| P6 | State machine + A/B + session | med | both-mode smoke |
| P7 | Indicators | low | per-state visual confirm |
| P8 | Sortformer Tier 2 | high | Orin RTF ≤ 0.7 |

P0/P1/P2 ship immediately (no deps). P3 trains while P4/P5 develop. P8 gated on
bench. **Demo fast path: P0 + P1 + P3** covers the bulk of the felt pain.

## Rollback

Every subsystem behind an env flag with a safe default:
`SPOT_WAKE_BACKEND=sherpa_onnx`, `SPOT_SPEAKER_ID=off`, `SPOT_CROWD_MODE=off`,
`SPOT_LISTEN_MODE=continuous`; chunker commit is git-revertible in isolation.

## Dependencies

| Dep | Phase | Note |
|-----|-------|------|
| `blingfire` | P1 | pin; verify aarch64 (V8) |
| `stream2sentence` | P1 | pure Python |
| `mediapipe` | P5 | pin; verify aarch64 (V4) |
| `nemo_toolkit` | P8 | only if Sortformer ONNX export fails (V6); SSD pip cache |

Model artifacts on `/mnt/ssd`: `wake-models/`, `spkr-embed/`, `diar/`,
`segmenters/`, `tse/`.

## Open verifications (consolidated)

- V1 livekit custom-positive-dir field — P3
- V2 livekit `model_size: large` — P3
- V3 WeSpeaker ResNet293-LM ONNX on ort-gpu 1.23 — P4
- V4 MediaPipe Face Mesh aarch64 — P5
- V5 BD SDK LED control — P7
- V6 Sortformer NeMo `streaming_export()` ONNX — P8
- V7 Sortformer Orin RTF ≤ 0.7 — P8
- V8 BlingFire aarch64 wheel — P1
- V9 arm-camera control + bearing latency — P5
