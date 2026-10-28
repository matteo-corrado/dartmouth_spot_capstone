# Stage 2E.2 Design — Gripper Puppet Mouth

> **Status:** Design approved 2026-06-02. Implements slice 2E.2 from the Stage 2E personality spec (`2026-05-21-stage2e-personality-design.md` §6). 2E.1 (TTS backends + persona routing) has shipped: `spot_tts.speak()` routes through `TTSBackend` and the `AudioPlayer` worker exposes `on_play_start` / `on_play_end` callbacks. Branch `tour_guide_upgrade_matteo`. Exit tag `stage2e2-complete`.

## Goal

Spot's gripper opens and closes in sync with TTS audio — the Boston Dynamics "talking robot" puppet trick — so an utterance looks like the robot is mouthing the words. Mouth-only for this slice.

## Scope

**In scope:** gripper amplitude animation driven by TTS PCM, runtime enable/disable, per-persona intensity, safe arm deploy/stow, feasibility spike, review gates.

**Out of scope (deferred):** arm gaze toward nearest person, `WorldObjectClient` person queries, `ArmCartesianCommand` orientation, `gaze_target` / `person_present` / `person_count` state. These remain in §6 of the parent spec for a later slice.

## Key architectural decision

§6.2 of the parent spec proposed a *parallel streaming PCM tap* feeding an envelope extractor while audio plays. That is unnecessary here. The existing `AudioPlayer` worker **fully materializes a TTS chunk's PCM before playback begins** (`_materialize` → `(samples, rate)`), and `on_play_start` fires immediately before the first audio sample is written to the device stream.

Therefore the envelope is **precomputed per chunk** at render time, and the gripper is driven in lockstep with playback by pacing against the `on_play_start` monotonic timestamp. This removes the parallel-tap machinery, the cross-thread PCM queue, and the resync-per-chunk-boundary logic from the original sketch.

## Architecture

```
spot_tts._render(chunk)            # already returns full (samples float32, 24000)
   └─> envelope(samples, rate, fps, gate, intensity)   # pure fn → np.ndarray of open_fractions
         └─> frames = [(t_offset_sec, open_fraction), ...]   at MOUTH_FPS

AudioPlayer worker plays the chunk:
   on_play_start()  ──> MouthDriver.play(frames, t0=monotonic())   # hand off + start pacing
   ...audio plays...
   on_play_end()    ──> MouthDriver.close()                        # gripper to 0, idle
```

`MouthDriver` runs one paced thread: pop frame, sleep until `t0 + t_offset`, send `gripper_open_fraction`. Single audio worker plays chunks sequentially and in order, so timing is deterministic per chunk.

## Build order — spike-first (HARD GATE)

Audio-rate gripper streaming is unverified on our Spot. No animation code is written until the spike proves the rate.

**Step 0 — `scripts/spike_gripper_rate.py`.** Power Spot, deploy arm to safe pose, modulate `RobotCommandBuilder.claw_gripper_open_fraction_command(frac)` by a sine wave for 30 s at each of 5 / 10 / 20 / 30 Hz. Measure: achieved command rate (does the SDK rate-limit, and at what cap?), jitter, visible smoothness. Confirm the command mechanism (individual `robot_command` calls vs a single streaming `ArmCommand`).

**Output:** `MOUTH_FPS` = the proven practical ceiling, used everywhere downstream.

**Branch:** if the cap is below ~10 Hz, the envelope-streaming approach is abandoned in favor of a **syllable-trigger variant**: open/close on word/phone timestamps from sherpa-onnx Kokoro or ElevenLabs alignment metadata. Same `MouthDriver`, different frame source. This fork is decided by the spike result, not now.

## Components

| Action | Path | Purpose |
|---|---|---|
| Create | `scripts/spike_gripper_rate.py` | Feasibility spike + rate measurement (Step 0 gate) |
| Create | `src/voice_control/animation/__init__.py` | Module marker |
| Create | `src/voice_control/animation/mouth.py` | `envelope()` pure fn + `MouthDriver` class |
| Modify | `src/session.py` | `deploy_arm_safe()` / `stow_arm()` helpers (velocity-capped soft pose) |
| Modify | `src/voice_control/spot_tts.py` | Wrap `_render`: compute envelope, pass frames to `MouthDriver` via `on_play_start` / `on_play_end` |
| Modify | `src/voice_control/spot_dispatch.py` | `enable_mouth` / `disable_mouth` actions |
| Modify | `grammar/spot_action.gbnf` | Add the two new actions to the allowlist (REQUIRED — omitting it makes the LLM silently misfire; known gap) |
| Modify | `src/voice_control/chatbot/session_state.py` | Add `arm_deployed: bool`, `mouth_enabled: bool` |
| Modify | `config/personas.yaml` | Add `mouth_intensity` per persona (butler 0.7, pirate 1.3, default 1.0) |

### `envelope(samples, rate, fps, gate, intensity) -> np.ndarray`

Pure function, no robot dependency.

1. Rectify `|samples|`.
2. IIR lowpass (`scipy.signal.lfilter`, cutoff ~50 Hz) → smooth amplitude.
3. Downsample to `fps` (= `MOUTH_FPS`).
4. Noise gate: values below `gate` → 0.0 (silence keeps gripper closed).
5. Normalize to [0, 1], multiply by `intensity`, clamp to [0, 1].

Returns one `open_fraction` per frame. Deterministic, unit-testable.

### `MouthDriver`

- Owns the arm `RobotCommandClient` (or `dry_run` sink).
- `.play(frames, t0)` — schedule frames against monotonic clock `t0`.
- One paced worker thread; `.close()` forces gripper to 0.0.
- `.enabled` flag — `disable_mouth` flips it off; driver becomes a no-op while playback continues normally.
- `dry_run=True` — logs `open_fraction` stream to file/stdout instead of commanding the robot. Used by unit tests and the spike harness; makes the whole module runnable without Spot.

## Safety (arm deployed near humans — non-negotiable)

- Gripper-only animation by default; the arm body holds a fixed soft pose, no lateral motion during speech.
- Arm deploy gated behind motor power-on **and** `session_state.arm_deployed`. Never command the gripper if the arm is stowed.
- `MouthDriver` subscribes to the existing e-stop / freeze and barge-in signals → immediate stop + gripper close.
- Velocity caps on the deploy/stow pose transitions; soft 45°-forward chest-height pose chosen away from the collision-likely volume.
- `on_play_end` and every exception path force the gripper closed — no stuck-open failure mode.
- Battery: arm deployed throughout a demo drains faster; budget ~30 min before swap. Operator keeps e-stop in hand.

## Review gates (built into the plan)

1. **Spike review** — after Step 0, review measured rate/jitter against the ≥10 Hz threshold before committing to envelope-streaming vs syllable-trigger.
2. **safety-reviewer agent** — mandatory pass on `session.py` arm helpers, `MouthDriver` stop/close paths, and `spot_dispatch.py` action handling, *before* any live full-pipeline test. Audits e-stop/freeze coverage, shutdown sequencing, stuck-open and unsafe-motion paths.
3. **code-reviewer agent** — review the `mouth.py` + `spot_tts.py` integration diff for correctness before merge.
4. **Live validation** — full TTS→mouth run on Spot only after gates 1–3 pass.

## Testing

- **`envelope()` unit test** — sine input → expected fraction curve; pure silence → all-zero; gate threshold respected; intensity scales linearly and clamps at 1.0.
- **`MouthDriver` pacing test** — `dry_run=True`, feed a known frame list, assert each `open_fraction` is emitted within tolerance of its scheduled wall-clock offset. No robot.
- **Spike script** — live hardware rate/jitter measurement; serves as the empirical gate.
- **Persona intensity** — `mouth_intensity` from `personas.yaml` multiplies the envelope; verify butler vs pirate produce different swing amplitudes.

## Rollback

Single env/flag kill switch: `mouth_enabled` defaults can be forced off, and the `spot_tts` wrapper is a no-op when no `MouthDriver` is attached. Append a 2E.2 rollback layer to `docs/project/stage2-rollback.md`. Removing the slice = drop the `on_play_start`/`on_play_end` mouth hooks; TTS and audio playback are unchanged.
