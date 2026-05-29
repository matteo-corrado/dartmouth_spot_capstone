# Stage 2F — Wake-Word & ASR Robustness Overhaul

**Status:** design v3 (post second blind-review), pending spec review
**Branch:** `stage2f-wake-asr-overhaul` (off `main`)
**Date:** 2026-05-29 (v3); 2026-05-28 (v2); 2026-05-27 (v1)
**Revision:** v3 incorporates a second 4-agent blind review (safety / ML-feasibility
/ architecture-code-claims / eval-completeness), two findings re-verified against
code by the author. Major changes vs v2:
- **C1 (verified):** cold-stop safety is dead code in the shipped detector config
  → new always-on frame-level safety KWS, moved into P0.
- **C2 (verified):** motion-action gate listed phantom verbs and missed 11 real
  motion verbs → gate inverted to a programmatically-derived non-motion allowlist.
- **C3:** FAR ≤ 1/hr is unmeasurable from a 10-min corpus → corpus sized for the
  rate; FAR reported with a confidence interval.
- **C4:** de-circularization collapsed to circular with one operator → mandatory
  speaker-disjoint synthetic eval split.
- **C5:** no enrollment UX → dedicated enrollment step defined.
- **C6:** no end-to-end latency budget → fast-path p95 budget, allocated + asserted.
- HIGHs: Alexa-rewake × chaining × per-sentence-TTS-reset; livekit 86.1% recall
  reality; non-audio hard bypass; confirm-chime is not a gate; threshold bound to
  gate-liveness in code; runtime-model-load failure (fail-closed-to-safe);
  eval decision criteria; voice-drift adaptation; field observability.
- Citation fixes: NeMo Sortformer issues are CLOSED (workaround exists) — drop
  rationale moved to RTF/contention (labeled estimate); BlingFire figure is a
  segmentation benchmark and BlingFire is NOT abbreviation-safe; ReDimNet2-B6
  weights exist (pin a commit); OpenSpeakerBeam license TBD; WeSpeaker pulled from
  the CC-BY-4.0 source.

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
   Detector streaming state is frozen mid-decision across the TTS mute and never
   reset; pre-TTS frames are re-evaluated against stale state on resume.

Surfaced during design (broad-scope, path B):

3. **Crowd robustness** — Spot operates in crowds; distinguish operator from
   ambient speakers.

4. **TTS chunker splits sentences oddly** — regex `[.!?]\s` splits abbreviations
   ("Dr.", "e.g.", "U.S.") and ignores clause boundaries.

Surfaced by the v2 blind review (pre-existing, not introduced by 2F, but worsened
by the over-fire plan):

5. **Cold safety commands do not work in WAKE_WORD state when a dedicated wake
   detector is active** (verified — see Safety C1). This already affects the
   shipped sherpa detector; the over-fire wake plan makes it more dangerous.

## Goals

- Wake fires reliably (**target** TPR ≥ 0.9 at FAR ≤ 1/hr on a held-out eval
  set; see C3/C4 — this is aspirational, livekit's own best is 86.1%, so the
  sherpa+push-to-talk fallback is an accepted outcome, not a failure).
- Cold safety commands (stop/freeze/estop) halt the robot from WAKE_WORD state
  with no wake required, with any wake backend (C1).
- No audio leak / stale-detector-state around TTS playback; chaining survives.
- Operator-vs-crowd discrimination, tiered by difficulty, **never at the cost of
  safety** (see Safety).
- Chunker never splits mid-word/abbreviation; clause-aware low-latency streaming.
- End-to-end fast-path latency within budget (C6).
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

### C1 — Cold safety commands must work in WAKE_WORD state (VERIFIED BUG)

The v2 claim that the `safety_only` path (`client_mic.py:962/1001`) processes
stop/freeze/estop in `WAKE_WORD` state without a wake **is false whenever a
dedicated detector is active**. Verified flow:

- `client_mic.py:914-917`: in `WAKE_WORD` state with `wake_detector` set, the
  speech-onset branch does `consecutive_speech = 0; pending_speech_frames.clear();
  continue` **before** `is_speaking` is set. Recording never starts.
- Therefore `process_utterance()` is never called, so `check_safety_command()`
  (`:1140`, itself correctly placed before the `safety_only` discard) never sees
  the audio. The comment at `:862-863` is defeated by the code 50 lines below.
- Both Stage 2F backends are dedicated detectors (`livekit`, and the
  `sherpa_onnx` rollback). So **today** the operator cannot say "stop" cold; they
  must say the wake word first. This is a latent safety regression already live
  with the sherpa detector.

**Fix (ships in P0):** an **always-on frame-level safety KWS** running in
`WAKE_WORD` state regardless of `SPOT_WAKE_BACKEND`:
- Reuse the sherpa-onnx `KeywordSpotter` (already a dependency, ~5MB, CPU, ~0ms)
  loaded with keywords `stop`, `freeze`, `estop` (+ the runtime bypass phrase).
- In `WAKE_WORD` state, feed each frame to the safety KWS **before** the
  `:914-917` early-continue. On a safety hit, execute the safety command directly
  (`execute_on_spot` of the mapped intent) without recording or ASR.
- The wake detector and safety KWS run in parallel on the same frames; neither
  gates the other. The early-continue for the *wake* path is preserved (keeps the
  crowd-noise-latency optimization), but it no longer swallows safety.
- **Test gate:** `eval_bargein.py` / a safety test asserts "stop" halts the robot
  from a cold `WAKE_WORD` state with a dedicated detector live and no prior wake.
  The P3 over-fire enable is blocked on this test passing.

### Gating rules

- **E-stop / stop / freeze are NEVER gated.** Detected by the always-on safety KWS
  above (C1). **Note (verified architecture):** the *voice* `estop` action
  (`spot_dispatch.py`) is a soft `RobotCommandBuilder.stop_command()` requiring a
  live session — it is NOT the hardware cut. `client_mic.py` runs as a
  `subprocess.Popen(..., start_new_session=True)` **child** of `wakespot.py`, in a
  separate process group; the `EstopKeepAlive` hardware-cut handle
  (`wakespot.py:375-391`) lives **only in the parent** and the child has no
  reference to it. The existing cross-process channel is parent→child signals
  only. So "fall through to `keepalive.stop()`" is NOT a local call the child can
  make. **Fix (wireable design):** the child raises the hard cut via a one-way
  child→parent control channel — `wakespot.py` installs a handler (a dedicated
  `SIGUSR1`, or a small control FIFO/socket the parent reads) that calls the
  parent's `EstopKeepAlive.stop()`. The voice `stop`/`freeze`/`estop` path: (1)
  attempts the soft `stop_command()` if the command client is live, AND (2)
  unconditionally signals the parent to trigger the hardware cut. Do NOT create a
  second `EstopEndpoint` in the child (two endpoints competing on one robot is its
  own hazard). The **tablet/hardware e-stop remains the independent primary** and
  is untouched; this fix only makes the *voice* "estop" reach the parent's cut
  instead of silently no-op'ing when the session is down.

- **Motion-causing actions gate behind speaker-ID confirm OR a confirm step.**
  The v2 gate list was wrong (verified against
  `src/voice_control/grammar/spot_action.gbnf:14`): it
  named 4 phantom actions (`move`, `dock`, `undock`, `follow` — do not exist) and
  omitted 11 real motion verbs. **Fix: invert the gate.** Define a
  programmatically-derived classification over the *actual* grammar action set:
  - `SAFETY_ACTIONS = {stop, freeze, estop}` — always-on, never gated (C1).
  - `NON_MOTION_ALLOWLIST = {describe, status, battery_status, check_obstacles,
    list_locations, list_maps, load_map, save_location, set_volume, set_speed,
    set_persona, add_persona, set_backend}` — no gate.
  - `DESTRUCTIVE = {power_off}` — always requires confirm, even with a confirmed
    speaker-ID session.
  - **Gated motion = every grammar action not in the three sets above**
    (currently `stand, sit, selfright, walk, strafe, turn, body_height, go_to,
    tour, patrol, come_back, go_to_object, follow_me`). Derived at import from the
    grammar so it cannot drift.
  - **Test:** every action in `spot_action.gbnf` is classified into exactly one
    bucket; a new grammar action with no classification fails the test (ties into
    the GBNF-allowlist memory).
  - **Bucketing caveats (verified):** `set_speed` is in the non-motion allowlist
    because its handler is a **no-op today** (prints only; persists no mobility
    cap) — if it is ever wired to persist a speed cap that affects future motion,
    re-classify it as gated. `load_map` is non-motion because its handler is a
    localization-pose RPC (`graph_nav_utils.set_localization`) with no actuation.
    Both are safe now; the classification test must be re-evaluated if either
    handler changes.

- **Confirm step is identity-bearing, not just a delay.** A confirm-chime
  followed by an unauthenticated spoken "yes" can be satisfied by the same
  bystander/crowd that issued the false command — it is **not** a substitute for
  speaker-ID. When speaker-ID is off/unavailable (incl. the demo path), a gated
  motion action requires a **physical/web-panel confirm** (button click on
  `server.py`), not a spoken confirm. The demo fast path either uses the physical
  confirm OR is explicitly documented as having no crowd-safety guarantee (run
  only in a controlled space with a human on the hardware e-stop).

- **Over-fire threshold is bound to gate-liveness in code, not by convention.**
  v2 rule "over-fire only ships once the gate is live" was contradicted by the
  demo path (P3 wake without P4 gate). **Fix:** the wake loader refuses to load
  the over-fire threshold (`0.42`) unless the motion-gate reports itself live
  (single source of truth, asserted at startup, logged). "Wake-alone at over-fire"
  is made unrepresentable. Until then the wake runs at a balanced threshold
  (≈0.5, re-derived from the DET curve — see Component 1).

- **`continuous` mode requires the motion gate regardless of phase.** Continuous
  mode re-opens a 15 s ungated ASR window after every response and is the rollback
  default — its wake protection is weakest. Gated motion actions in `continuous`
  mode always require the speaker-ID-or-physical-confirm gate; `continuous` +
  over-fire is prohibited by the same loader check above.

- **Hard bypass is a NON-AUDIO channel.** A spoken bypass phrase is routed through
  the very speaker-ID/crowd path it is meant to escape and fails in the loud-crowd
  / bad-enrollment case it exists for. **Fix:** the guaranteed escape hatch is
  `SPOT_SPEAKER_ID=off` (env, out-of-band) AND a web-panel toggle. A spoken bypass
  phrase MAY exist as a best-effort convenience (recognized by the always-on
  safety KWS) but is documented as best-effort, not the guarantee.

- **Runtime model-load failure fails closed to a safe path.** Deliberate env-flag
  rollback (below) is distinct from a model failing to load at runtime. The
  cold-stop safety guarantee is satisfied by the always-on safety KWS **OR** by a
  non-voice stop that does not depend on any ONNX/sherpa model (the parent's
  hardware-cut channel from the voice-estop fix above, plus a web-panel STOP button
  on `server.py`). Per-model, at a startup health check:
  - Wake ONNX load failure → auto-fall-back to `sherpa_onnx`, log loudly; never to
    an ungated path. If sherpa-onnx itself cannot load → push-to-talk only (no
    over-fire window).
  - Speaker-ID ONNX load failure → behave as `SPOT_SPEAKER_ID=off` (motion still
    gated behind physical confirm), log loudly.
  - Safety KWS load failure → the **voice** cold-stop channel is down. Because the
    safety KWS and the `sherpa_onnx` wake fallback share the same sherpa-onnx
    library, a sherpa-onnx import failure takes out **both** at once — so this case
    is handled jointly with the wake fallback: drop to push-to-talk AND **refuse
    all motion/destructive actions** (non-motion still allowed) until a working
    safety channel exists. The non-voice stop (parent hardware cut + web-panel STOP
    button) must be verified reachable at startup; only if neither sherpa-onnx nor
    the web STOP button can initialize is startup **fatal** (refuse the run loop).
    This reconciles the wake-path "degrade to PTT" with the safety requirement:
    the system may run without voice cold-stop only while motion is blocked and a
    non-voice stop is confirmed live.

- **Reject feedback ships WITH the drop logic (P4), not the indicators phase.** A
  distinct reject chime fires before any utterance is dropped, so the operator can
  tell "heard-but-rejected" from "not heard".

A dedicated `safety-reviewer` agent pass gates the C1 fix, the motion-gate
classification, the speaker-ID gate, and the state-machine commits before merge.

## Hardware constraints (verified)

Mic = **Seeed reSpeaker XVF3800 4-Mic Array** (USB `2886:001a`, confirmed via
`lsusb` + `docs/getting-started/hardware.md`). 2 channels only (ch0 beamformed,
ch1 reference); AEC + beamforming firmware-fixed; **DoA and LED ring NOT exposed
to host** (proprietary HID, no Linux driver). Consequence: crowd robustness uses
speaker embedding + Personal VAD + visual; "listening" indicator uses Spot
onboard LEDs + web panel + chimes.

Compute: Jetson AGX Orin 64GB (Ampere ~5.32 TFLOPS FP32 / 21 TOPS INT8, CUDA
12.6, onnxruntime-gpu 1.23.0). `/mnt/ssd` (~1.8TB) for models — disk not a
constraint; VRAM ~46GB headroom. **VRAM is not the binding constraint — power /
thermal / GPU-contention is** (see Component 7). Operator has L40S/H200 cluster
for training.

## Component 1 — Wake word (livekit-wakeword custom train)

**P3 GATE (do first, blocks everything else in this component):** verify
`github.com/livekit/livekit-wakeword` still exists, its license is permissive
(verified Apache-2.0 in the v2 review), and it trains on the cluster + runs
inference on Orin aarch64. It is NOT on PyPI (git-clone + `--no-deps -e`
install). If this fails, the whole component falls back to "tune sherpa-onnx
threshold + add keyword variants + push-to-talk" (lower ceiling).

**Recall reality (verified):** livekit-wakeword's own headline result is **86.1%
recall at 0.08 FPPH**, trained on **15,000 positive clips** at optimal threshold
**0.68**. So the 0.9 TPR target is aspirational and the 0.42/0.5 deploy thresholds
are well below livekit's own optimum. **Do not fix thresholds a priori — derive
the deploy operating point from the trained candidate's DET curve** at the
measured FAR target. The sherpa+push-to-talk fallback is the *probable* demo
outcome, not the exception.

Three A/B candidates trained in one cluster batch (~3h):

| ID | Phrase | Hard-negative focus |
|----|--------|---------------------|
| A | "Hey Spot" | hot spot, hey scott, spotted, spotlight |
| B | "Hi Spot" | hi scott, hipster, high spot |
| C | "Big Yellow" | big fellow, big mellow, pig yellow, big yell |

Config diff from existing YAML: `model_size: small → large` (V2),
`n_samples: 25000 → 50000`, `steps: 50000 → 80000`. **Threshold staging:** train
at `target_fp_per_hour: 1.0` (high recall) but **deploy at a DET-curve-derived
balanced threshold (≈0.5–0.68) until P4 gate is live, then lower toward over-fire
ONLY via the gate-liveness-bound loader** (Safety).

Training data: Piper VITS pool (904 synthetic speakers, built-in) + ~30–50 real
positives via XVF3800 (`scripts/record_wake_positives.py`) + ~50 ElevenLabs
stylings (`scripts/synth_wake_elevenlabs.py`, **verify free-tier ToS permits
training-data use** — V10). **Data-sufficiency risk:** 30–50 real positives is
low for a custom KWS; synthetic-heavy data risks train/serve mismatch. Fallback
gate: **if held-out TPR < 0.9 (or whatever the DET curve yields at FAR ≤ 1/hr) by
the agreed date, ship sherpa-onnx + a push-to-talk button** rather than a flaky
low-threshold wake.

Deploy: eval winner → `/mnt/ssd/wake-models/active.onnx` →
`SPOT_WAKE_BACKEND=livekit`. Rollback: `SPOT_WAKE_BACKEND=sherpa_onnx`.

Open verifications: V1 (custom-positive-dir field), V2 (`model_size: large`),
V10 (ElevenLabs ToS), V11 (livekit repo existence + Orin trainability — verified
exists/Apache-2.0/`conv_attention`/`model_size: large` valid; **Orin
trainability + inference still to confirm on-device**).

## Component 2 — Barge-in fix (redesigned against verified code)

Corrected diagnosis (v1 wrong; v2 mechanism refined by the review):

- **`_drain_audio_queue()` already exists** at `client_mic.py:1043`, called at
  `:969` and `:1008`, keeps the last ~33 frames (`:1058`). Verified.
- **`is_busy()` has no race** — inflight counter incremented under lock at enqueue
  (`audio_player.py:222-234`, `:253-256`, `:286-289`), decremented in the worker
  (`:519`). "Leak B does not exist." Verified. Removed from scope.
- **Real gap:** during streaming TTS the chunker issues many non-blocking
  `speak()` → `enqueue_render` calls and `process_utterance` returns; the player
  thread plays while the main loop continues. The PortAudio callback discards
  frames while `is_busy()` (`:218-220`), so the queue empties and the main loop
  spins on `audio_queue.get(timeout=1.0)` → `except queue.Empty: continue`
  (`:840-842`). The VAD/KWS streaming state is frozen mid-decision across the mute
  and never reset; the ~33 retained frames are **pre-TTS** audio (captured before
  playback, not speaker bleed) re-evaluated against stale state on resume.
- **Leak C (OS tail):** real but small; covered by the existing cooldown
  (`SPEAKER_TAIL_COOLDOWN_S = 0.15` today — already within budget).

**Fix (all single-threaded in the main loop, never the PortAudio callback, never
reaching across modules into the player thread):**
- **Detect the player busy→idle edge at the TOP of the `while True:` loop, before
  the blocking `audio_queue.get()`** (NOT next to the existing `:969/:1008`
  drains — those live inside the inner frame loop, which does not run while the
  queue is empty during TTS). On the edge, **once**: reset `vad`,
  `wake_detector`, `preroll_buffer`, `pending_speech_frames`, `consecutive_speech`,
  clear `speech_buffer`/`speech_float_buffer`, then call `_drain_audio_queue()`.
  (All these names exist in `main()` scope — verified.)
- **Edge fires on the FINAL TTS tail only, not every inter-sentence gap.**
  Streaming TTS issues many `speak()`/idle cycles per response; a reset on every
  cycle would bounce the state machine and corrupt multi-sentence playback and
  command chaining. Track a "response in progress" flag set when the chunker opens
  and cleared when the chunker closes + the player reports idle; reset only on the
  close→idle edge.
- **Cooldown:** keep `SPEAKER_TAIL_COOLDOWN_S` ≤ 0.2 (currently 0.15). Gate the
  cooldown on TTS tails, not short beeps, if measurement shows beeps over-extend
  the mute window.

Verify: `eval_bargein.py` asserts (a) no ASR submission during `is_busy()`,
(b) state reset on the response-close→idle edge, (c) a correct first post-TTS wake
fire, (d) **a multi-sentence streaming-TTS response does not reset mid-response or
kick the operator back to WAKE_WORD**, and (e) the C1 cold-stop test above.

## Component 3 — Speaker-ID cascade (crowd robustness)

Tiered; ship cheap default, escalate only where measured data demands.

### Enrollment (C5 — was missing)

The cascade conditions on a reference operator embedding that v2 never defined.
- **Dedicated enrollment step** `scripts/enroll_operator.py`: capture N≥5 clean
  utterances via the XVF3800, compute per-utterance WeSpeaker embeddings, store the
  **L2-normalized mean** to `/mnt/ssd/spkr-embed/operator.npy` (+ per-sample
  embeddings for re-scoring). Runs at first boot / on demand; re-runnable.
- **Match vs self-enroll rule:** if an enrolled reference exists, the wake-fire
  utterance is **matched** against it (`EMBEDDING_MATCH_THRESHOLD`), and the
  session embedding is initialized to the reference — NOT to whatever fired the
  wake (which on an over-fire could be a bystander). If no reference exists,
  speaker-ID is treated as `off` (motion gated behind physical confirm); the
  system never silently self-enrolls a stranger.
- **Single-flight session lock:** with the over-fire threshold in a crowd, two
  near-simultaneous wake fires are plausible. The first fire claims the session;
  any further fires within the embedding-gate window are **matched against the
  claimed session**, never allowed to open a second session or re-enroll. The lock
  releases on session end (Alexa single-turn, or `continuous` timeout).
- **Drift adaptation:** on a command confirmed by the confirm gate, EMA-update the
  session embedding toward the accepted utterance (slow `α`). A sustained
  reject-streak (cold/shouting/distance) auto-degrades the session to
  confirm-gate mode rather than locking the operator out, and surfaces a
  "re-enroll?" prompt on the web panel.

**Tier 1 (default): speaker embedding.**
- Day-one: **WeSpeaker ResNet34-LM ONNX** (0.72% EER, ~26MB, CC-BY-4.0,
  **pre-shipped ONNX** — zero export risk, drops into ort-gpu 1.23). **Pull from
  the CC-BY-4.0 source** (`Wespeaker/...`, `pyannote/...`) — NOT the CC-BY-NC
  mirrors (`Revai/...`) or differently-licensed Apache mirrors.
- Upgrade: **ReDimNet2-B6** (0.29% EER, MIT, ~13 GMAC, 12.3M params) — best
  open-weight embedding. **Weights exist** (`PalabraAI/redimnet2`, B0–B6 `.pt`,
  v1.0.0); repo is under active development → **pin a commit**; self-export to
  ONNX (V12 downgraded from "may lag" to "exists, pin + export").
- On wake fire: match operator fingerprint (see Enrollment), lock to session;
  gate subsequent utterances by cosine similarity.

**Tier 2 (`SPOT_CROWD_MODE=on`): Personal VAD (replaces Sortformer).**
- **Sortformer dropped — corrected rationale.** v2 cited NeMo issues
  #15077/#15536 as "open / ONNX export impossible." Both are **CLOSED**, and a
  community `torch.onnx.export` workaround (bypassing the broken built-in
  `streaming_export()` helper) is reported working. The drop still stands, but on
  **GPU-contention / RTF** grounds, not ONNX impossibility: a continuous diarizer
  competing with Parakeet + Gemma for the GPU is the risk. The "Ada→Orin RTF
  1.5–3.0" figure is an **author estimate, not a measurement** — labeled as such.
- Instead train a **Personal VAD** (speaker-conditioned frame-level VAD,
  conditioned on the WeSpeaker embedding) on the cluster. The canonical Personal
  VAD (Ding et al., arXiv 1908.04284) is ~130K params; this design budgets larger
  (~1–3M) for crowd robustness — a deliberate capacity choice, not the paper's
  size. INT8, target <1ms/frame on Orin (**unbenchmarked target, gated**).
  Frame-level "is the enrolled operator speaking" → gate ASR frames. Handles
  overlap better than clip-level cosine; small enough to avoid the continuous-GPU
  contention that killed Sortformer.
- No pretrained weights exist (correct) → ~3–5 days training incl. overlap-sim
  dataset construction (the longer pole). Gated behind `SPOT_CROWD_MODE`, not on
  the demo path.

**Tier 3 (deferred): target-speaker extraction.**
- OpenSpeakerBeam-SS (~7.64M params, ONNX-ready; **license is "TBD (likely
  MIT/Apache)" per its README — do not assert MIT**) — waveform separation before
  ASR. Only if Tier-2 data shows residual WER under heavy overlap.

**Visual cascade (parallel gate on wake fire):**
- YOLOv8n person detect (existing) → MediaPipe Face Mesh → mouth-motion variance
  over a 5-frame burst. **Mouth-motion is a weak tiebreak hint only** (fires on
  chewing/smiling/back-channeling; no AV-sync) — never a sole confirmer.
- **Mic ownership during cascade:** the in-breath command path ("Hey Spot stand
  up") uses the embedding-only fast path (~30ms); arm-camera escalation
  (~700ms–1.2s) is used ONLY for a *separate* follow-up utterance when fisheye is
  ambiguous, and only after confirming the bounded `audio_queue` (~1s) won't
  overflow — otherwise re-prompt. Mic ownership defined explicitly in impl.
- VLM ("who is talking") only as a manual high-stakes escalation (2–5s latency).

Thresholds (`EMBEDDING_MATCH_THRESHOLD=0.75`, `EMBEDDING_GATE_THRESHOLD=0.65`,
`MOUTH_THRESHOLD`, `ESCALATION_RATIO=2.0`) are initial guesses tuned on the
**threshold-tuning split**, distinct from the **decision split** that decides Tier
2 (see Component 5). **Reject feedback + non-audio hard bypass per Safety ship in
this component.**

Open verifications: V3 (WeSpeaker ONNX load), V4 (MediaPipe aarch64), V9
(arm-camera control + bearing latency), V12 (ReDimNet2-B6 ONNX self-export).

## Component 4 — TTS chunker (preserve JSON interface)

`TTSChunker` carries load-bearing logic beyond sentence-splitting: `suppressed`
(`spot_tts.py:266`) plus the GBNF-JSON-stream parsing (`_RESPONSE_KEY_RE`,
`_DESCRIBE_ACTION_RE` at `spot_tts.py:245-246`, suppression through `:289`),
consumed at `client_mic.py:1282,1373` (verified). `stream2sentence` / BlingFire
consume *plain text*, not the raw JSON token stream.

Therefore: **keep the JSON-extraction front-end and `suppressed` interface; swap
only the `_SENTENCE_END_RE` split (`spot_tts.py:306-312`) backend** to operate on
the already-extracted `response` text:
- Driver `stream2sentence` (MIT) + detector **BlingFire** (MIT, ~1MB).
  **BlingFire is NOT abbreviation-safe** (corrected: its default `sbd.bin`
  mis-splits `Mr./Ms.` and decimals — exactly the `Dr./e.g./U.S.` class in the
  Problem statement). It is better than `[.!?]\s` but not a guarantee — rely on
  the clause-fallback + the TDD set, not on BlingFire being abbreviation-proof.
- **Re-benchmark BlingFire's sentence API on aarch64** (V8). Correction: the
  "0.06ms / 450×" figure IS a sentence-segmentation benchmark (not tokenization,
  as v2 stated) — the legitimate need is the aarch64 re-bench, since the published
  figure is x86.
- Clause fallback: rule split on `,;:` when buffer > ~80 chars.
- SaT `sat-3l-sm` ONNX-CPU only as a last-resort for unpunctuated tails (scored by
  **F1**, not EER).

20-case TDD set (`tests/voice_control/tts/test_chunker.py`) gates correctness;
tests MUST include the `suppressed`/describe-action path and abbreviation cases,
not just splitting.

## Component 5 — Eval harness (de-circularized + measurable)

Corpora under `tests/audio/wake_eval/`: positives, hard negatives, crowd scenarios.

### C3 — FAR must be measurable
FAR ≤ 1/hr cannot be estimated from a 10-min negative sample (rule-of-three: 0
fires in 10 min → 95% upper bound ≈18/hr). **Concrete corpus sizing (rule-of-three,
95% CI):** the CI upper bound is ≈ `3 / T_hours` when zero fires are observed, so a
≤ 1/hr gate needs **≥ 3 h of negatives with zero false fires** (5 h gives margin);
each observed fire requires proportionally more (e.g. 1 fire → need ≈ 4.7 h for the
same bound). Sourced from the Stage 2B lab corpus + recorded long-form ambient/
conversation, **including Spot's own Kokoro `af_sarah` output as an explicit
negative class** (the highest-prior false-fire source). Report FAR as
fires-per-corpus-hour **with the 95% CI**, not a point estimate; the gate is the CI
upper bound ≤ 1/hr. P2 records the actual corpus duration in `baseline.json`.

### C4 — De-circularization (no loophole)
The v2 "where possible" clause collapses to circular with one operator (30–50
positives, one speaker — a held-out *human* speaker set is arithmetically
impossible). Two distinct biases must be controlled separately:
- **Speaker overfit** → **mandatory speaker-disjoint synthetic split**: Piper/
  ElevenLabs synthetic speaker IDs held out of training, used as the
  speaker-generalization eval. This controls speaker overfit but is **domain-shared**
  (synthetic TTS, not real-mic-through-XVF3800) — it does NOT control mic-domain
  shift, and the spec must not present a synthetic TPR as real-world robustness.
- **Mic-domain shift** → a **real-human held-out set (N≥3 non-operator speakers
  recorded through the XVF3800) is REQUIRED for the ship/robustness claim** — not
  "if feasible." If real held-out humans cannot be recruited, the wake ships on the
  honest fallback (sherpa + push-to-talk), because a synthetic-only TPR cannot
  justify the over-fire/gate deployment.
- The **same-operator-same-room TPR is reported separately and labeled "optimistic
  ceiling"** — never the headline number. Headline = real held-out; synthetic split
  = speaker-generalization ceiling.

### Synthetic crowd caveat
Synthetic crowd (voices through a 2nd speaker) is a single point source with one
room transfer function and no independent beamformer steering — it does NOT
reproduce the spatial/embedding path the cascade depends on. It is an **upper
bound on the easy case only.** Any Tier-2 ship decision is gated on ≥ 2–3 real
multi-body recording sessions, not the synthetic rig alone.

### Decision criteria (were undefined)
- `eval_wake.py`: ROC/DET sweep → TPR, FAR/hr (+CI), latency (p50/p95), AUC. Pick
  candidate+threshold at the **single operating point** TPR @ FAR ≤ 1/hr on the
  **common held-out negative set** (AUC is not comparable across the three
  phrases' differing negative distributions). Tie-break: latency, then
  false-fire-transcript severity.
- `eval_speaker.py`: operator-accept / crowd-reject / overlap-accept per tier on
  the **decision split** (disjoint from the threshold-tuning split). **Tier 2 is
  built iff** Tier-1 crowd-reject rate < X OR overlap-accept rate < Y at the
  operating threshold. **Provisional X = 95% crowd-reject, Y = 90% overlap-accept**
  (i.e. build Tier 2 if Tier-1 lets through > 5% of crowd utterances or misses
  > 10% of overlapped operator speech). These are set from the P4 Tier-1 baseline
  measurement and **recorded in `baseline.json` before the decision run** so the
  decision is not made on the same data that tuned the thresholds.
- A dedicated runner asserts **safety-phrase recall** (stop/freeze/estop via the
  always-on KWS) under the crowd corpus — safety recall matters most where it is
  hardest.
- `test_chunker.py` (20/20 incl. suppress + abbreviation), `eval_bargein.py` (leak
  + C1 cold-stop + multi-sentence-no-reset asserts).

### Regression gate (was just a file)
`baseline.json` defines compared metrics (TPR, FAR/hr, p50/p95 latency, chunker
20/20, bargein asserts) **with tolerances** (e.g. TPR may not drop > 2 pts, FAR CI
upper bound may not exceed 1/hr, latency p95 may not regress > agreed ms).
`make eval-voice` **exits non-zero on violation**; baseline updates only on an
approved intentional change (never silently ratcheted to match a regression).

## Component 6 — State machine + indicators + A/B toggle

Enum migration: current `VoiceState = {WAKE_WORD, LISTENING, RECORDING}`
(`RECORDING` declared-unused — zero `.RECORDING` references repo-wide, verified).
Add `SPEAKER_ID`, `THINKING`, `RESPONDING`; remove `RECORDING`. **Every new state
has an explicit edge back to `WAKE_WORD`** — `SPEAKER_ID` reject → `WAKE_WORD`
(with reject feedback), and a `LISTENING`-timeout edge.

`SPOT_LISTEN_MODE`: `alexa` (default; post-TTS → `WAKE_WORD`, re-wake required) vs
`continuous` (legacy; post-TTS → `LISTENING` 15s, **motion gate always required**
per Safety).

**Command chaining × re-wake (C-HIGH):** chained multi-step commands ("Hey Spot,
stand then walk forward then turn left") are parsed in **one utterance pre-dispatch**
— re-wake is irrelevant *within* a chain. Re-wake applies only *between* operator
turns. The Component-2 busy→idle reset fires only on the response-close→idle edge,
so per-sentence streaming TTS within one response does NOT bounce the state machine
mid-chain.

**Confirm step lives in the state machine.** A gated motion action enters a
`CONFIRM` sub-state: physical/web confirm (or speaker-ID-confirmed session) within
a timeout → execute; else → `WAKE_WORD`. The confirm does NOT require a fresh wake
(no extra wake-cycle latency per move on the demo path).

**Rollback-flag independence (must hold; forward-looking — none of these flags
exist in code today, verified):** each gate degrades to pass-through when its
upstream flag is off. `SPOT_SPEAKER_ID=off` → no session object required;
`continuous` mode must not depend on a speaker-ID session existing.
`SPOT_CROWD_MODE=off` → Personal-VAD frame gate hard-bypassed (pass all frames),
never silent-drops. The flag cross-product is a test matrix in `eval`.

Session (`embedding`, `visual_id`, timestamps, `command_count`): single-turn in
`alexa`, 15s/45s timeout in `continuous`; created on `SPEAKER_ID` confirm; absent
and not required when `SPOT_SPEAKER_ID=off`.

Indicators (4 states + reject, graceful-degrade): web badge (`server.py`), chimes
(`audio_feedback.py` — wake / listening-close / response-done **/ reject**; the
reject chime is **net-new**, no `reject` method exists today — ships in P4, not
P7), Spot onboard LEDs (**`src/voice_control/spot_leds.py` is net-new, not an
existing module**; **verify BD SDK LED control — V5**; drop channel if absent).

## Component 7 — Power / thermal / GPU contention (was missing)

The full stack (wake KWS + safety KWS + WeSpeaker + Personal VAD + YOLOv8n +
MediaPipe + VLM + Gemma 4 + Kokoro + Parakeet) runs on one Orin. VRAM is not the
constraint; sustained power/thermal is, and thermal throttling silently inflates
every latency number measured in isolation.
- State the Orin power mode the targets assume (MAXN vs capped).
- Add a **concurrent-load `tegrastats` measurement** in a worst-case scenario
  (continuous wake + safety KWS + Personal-VAD frame gate + a VLM escalation
  firing simultaneously); re-measure the C6 latency budget **under that sustained
  load**, not per-component.
- **GPU ownership:** define whether the continuous Personal-VAD frame gate and the
  burst consumers (VLM 2–5s, YOLO+MediaPipe 0.7–1.2s) share a CUDA
  stream/context, the priority, and the continuous gate's behavior while a burst
  model holds the GPU (a 2–5s VLM call must not starve the operator's frame gate).

## C6 — End-to-end latency budget (was missing)

Two distinct p95 budgets for the path the operator feels, separated because they
mix a fast perception path with a slow LLM path:

- **Perception ack** (wake fire → embedding gate → state transition → "listening"
  chime): **provisional gate p95 ≤ 150 ms.** Allocation: wake event ~50 ms,
  embedding extract+match ~50 ms, gate+state transition ~20 ms, chime ~30 ms.
- **Action dispatch** (wake fire → GBNF parse → dispatch decision → action ack):
  LLM-bound, measured separately; **provisional gate p95 ≤ 1.5 s** for the
  in-breath fast path.

The provisional numbers above are **set in P2 from first measurement and recorded
in `baseline.json`** (same mechanism as the FAR/TPR baseline); the eval then
asserts against the recorded values, not against a hard-coded guess. Both budgets
are measured **under Component-7 sustained concurrent load**, not per-component in
isolation. Assertions live in `eval_wake.py` (perception) / `eval_speaker.py`
(dispatch).

## Observability — field false-reject decision log (was missing)

When the operator says "it didn't hear me" in the field there must be a recorded
signal to tell a wake miss from an embedding reject from a VAD gate from a
beamformer miss — the offline eval cannot diagnose a live failure. Required:

- **Structured per-utterance decision log** written to the logs SSD symlink, one
  record per wake/utterance: timestamp, wake backend + score, safety-KWS hit,
  embedding cosine vs threshold, which tier/gate made the accept/reject decision,
  Personal-VAD frame %, final state transition + action (or reject reason), and the
  C6 latency breakdown. This is what the reject chime (Safety) corresponds to on
  the developer side.
- **Optional rolling audio ring-buffer dump on reject** (last ~N seconds) so a
  field false-reject can be replayed offline against `eval_speaker.py` /
  `eval_wake.py`. Gated by a flag (privacy + disk); off by default.
- Ships incrementally: the decision log lands in P4 (with the gate it describes);
  the ring-buffer dump is optional and may defer to P5.

## Phasing

| Phase | Work | Risk | Gate |
|-------|------|------|------|
| P0 | Barge-in fix (Comp 2) **+ C1 always-on safety KWS** | low | eval_bargein: leak + C1 cold-stop + multi-sentence-no-reset |
| P1 | TTS chunker (backend swap, JSON interface kept) | low | 20/20 TDD incl. suppress + abbrev; V8 aarch64 re-bench |
| P2 | Eval harness + held-out corpus (C3/C4 sized + disjoint) | low | runners run; FAR-CI measurable; safety-recall runner |
| P3 | **Verify livekit (V11 on-device)** → wake train (3 cand) | med | DET-curve op-point @ FAR≤1/hr on held-out, else sherpa+PTT |
| P4 | Speaker embedding Tier 1 + Enrollment + **Safety (inverted motion gate, non-audio bypass, reject feedback, gate-liveness-bound over-fire, runtime-fail-closed)** | med | crowd-eval rates; safety-reviewer pass |
| P5 | Visual cascade (mouth-motion tiebreak only) | high | speaker-ID crowd accuracy |
| P6 | State machine + CONFIRM sub-state + A/B + session + flag cross-product | med | both-mode smoke; rollback matrix; chaining-survives test |
| P7 | Remaining indicators (web badge, LEDs) | low | per-state visual confirm |
| P8 | Personal VAD Tier 2 (replaces Sortformer) | high | crowd-eval overlap improvement; Comp-7 contention OK |

P0/P1/P2 ship immediately. **C1 (cold-stop) is in P0 — it is a live safety bug,
not a new feature.** **Demo fast path = P0 + P1 + P3** with the DET-derived
balanced threshold and a **physical/web confirm** for motion actions (the
confirm-chime alone is not a crowd-safety guarantee). If P3's data-sufficiency or
livekit gates fail → P0 + P1 + sherpa + push-to-talk is the honest demo fallback.

## Rollback

`SPOT_WAKE_BACKEND=sherpa_onnx`, `SPOT_SPEAKER_ID=off`, `SPOT_CROWD_MODE=off`,
`SPOT_LISTEN_MODE=continuous`; chunker commit git-revertible. Every gate degrades
to pass-through when its flag is off (tested cross-product). **Deliberate rollback
(env flags) is distinct from runtime model-load failure (fail-closed-to-safe per
Safety).** The always-on safety KWS (C1) is NOT behind a rollback flag — it is
always on; its load failure is handled per the Safety runtime-fail-closed rule
(motion refused until a non-voice stop channel is verified reachable; fatal only
if no safety channel can initialize), never by a flag.

## Dependencies

| Dep | Phase | Note |
|-----|-------|------|
| `blingfire` | P1 | pin; verify aarch64 (V8); NOT abbreviation-safe |
| `stream2sentence` | P1 | pure Python |
| `mediapipe` | P5 | pin; verify aarch64 (V4) |
| Personal VAD (custom-trained) | P8 | cluster; no pretrained weights |
| `redimnet2` weights | P4 (upgrade) | exists, MIT, pin a commit, self-export ONNX |
| ~~`nemo_toolkit` / Sortformer~~ | — | **dropped** (RTF/GPU-contention; ONNX export works via workaround) |

Model artifacts on `/mnt/ssd`: `wake-models/`, `spkr-embed/` (incl.
`operator.npy`), `pvad/`, `segmenters/`, `tse/` (Tier 3, deferred).

## Open verifications (consolidated)

- V1 livekit custom-positive-dir field — P3
- V2 livekit `model_size: large` — P3 (valid per review; confirm on-device)
- V3 WeSpeaker ResNet34-LM ONNX on ort-gpu 1.23 — P4
- V4 MediaPipe Face Mesh aarch64 — P5
- V5 BD SDK LED control — P7
- V8 BlingFire aarch64 sentence-API re-bench — P1
- V9 arm-camera control + bearing latency — P5
- V10 ElevenLabs free-tier ToS for training data — P3
- V11 livekit-wakeword repo (verified exists/Apache-2.0/conv_attention) —
  **on-device Orin train + inference still to confirm** — P3 gate
- V12 ReDimNet2-B6 ONNX self-export (weights confirmed to exist) — P4 (upgrade)

(Dropped from v1: V6/V7 Sortformer ONNX + Orin RTF — Sortformer removed.)
