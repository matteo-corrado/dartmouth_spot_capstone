# Stage 2A Scaffolding Cleanup Plan

**Status:** pending — runs AFTER `stage2a-voice-complete` tag (d9b927b) verified in real-robot use.
**Don't run before:** Stage 2B kicks off — keeping rollback paths intact protects the 2026-05-26 demo deadline.

## Goal

Remove the dead scaffolding that 2A leaves behind: stale docs/comments referencing
qwen2.5/webrtcvad/old wake-detector paths, redundant smoke logs, comparison
artifacts that only had meaning during T0–T12, and a small set of unused source
files. Net result: a clean repo at `stage2a-voice-complete` whose every file
traces to runtime, tests, or rollback.

## What to keep (do not delete)

- `src/voice_control/asr/nemotron.py` — Layer B rollback in stage2-rollback.md.
- `src/voice_control/wake/livekit.py` + `configs/wake/wakeword_hey_spot.yaml` +
  `scripts/training/*` — pre-scaffolded for the Stage 2B custom wake-word train.
- `scripts/setup_nemotron.sh` — needed by Layer B.
- `models/kws/` artifacts — current default wake backend.
- `docs/project/stage2-rollback.md` — runbook is canonical.
- `tests/audio/*_lab.wav` — corpus consumed by mic-verify.

## Tasks (14)

### Cleanup

- [ ] **C1 — Drop qwen2.5 mentions from runtime comments.**
  `src/voice_control/llm_brain.py:161` references "qwen2.5:7b's 32K context"
  but the brain is now Gemma 4 E4B Q8_0 with 32 k context. Update the comment
  to cite Gemma 4 (or remove — context is documented elsewhere).
- [ ] **C2 — Drop webrtcvad mention from runtime comments.**
  `src/voice_control/client_mic.py:660` says "Silero via sherpa-onnx — replaces
  webrtcvad 2012-era model". Trim the parenthetical; the swap is canonical now.
- [ ] **C3 — Sweep docs for qwen2.5/webrtcvad/openwakeword references.**
  Files: `docs/index.md`, `docs/POST_FLASH_MIGRATION.md`, `docs/project/status.md`,
  `docs/project/archive/upgrade-plan.md`, `docs/project/archive/knowledge_pack_upgrade_handoff.md`,
  `docs/architecture/services.md`, `docs/architecture/overview.md`,
  `docs/architecture/voice-pipeline.md`, `docs/architecture/stage2-architecture.md`,
  `docs/modules/llm-brain.md`, `docs/modules/wake-word.md`,
  `docs/guides/troubleshooting.md`, `docs/commands/reference.md`.
  For each: update to reflect Gemma 4 / Silero / sherpa-onnx KWS or delete the
  paragraph if it is purely historical. Do NOT touch stage1-rollback.md or
  stage2-rollback.md — those are runbooks, the references are intentional.
  Update README.md last (after architecture docs settle).
- [ ] **C4 — Remove `/tmp/asr_server_*.log` references and the old `_smoke`
  variants if any survive in scripts.** Check for hardcoded `/tmp` log paths
  outside the README troubleshooting section.
- [ ] **C5 — Decide on `scripts/wakespot.py`.** This was an early "Hey Spot"
  one-shot test from before the wake subpackage existed. If it still imports
  the old wake_word path, fix or delete. Run `grep -n wake_word scripts/wakespot.py`
  and act accordingly.
- [ ] **C6 — Verify `scripts/preflight_2a.py` is still useful or delete it.**
  T0 baseline lives at `/tmp/stage2a_preflight_*.json`. Either re-point it at
  the new pipeline or move it under `scripts/archive/` for historical reference.

### Documentation

- [ ] **D1 — Write `docs/architecture/voice-pipeline.md` rewrite reflecting 2A
  actual state.** Sections: Mic capture (XVF3800 ch0) → Silero VAD → sherpa-onnx
  KWS wake → Parakeet ASR (gRPC :50055) → LLMBrain → unified GBNF → llama.cpp
  (Gemma 4 E4B Q8_0 :11435) → Kokoro streaming TTS. Diagram each hop.
- [ ] **D2 — Append a "Stage 2A measured baselines" section to
  `docs/architecture/voice-pipeline.md`** citing T12 numbers: ASR median 195 ms /
  p95 337 ms, brain TTFT p50 (re-measure if not captured in T0), wake fire 1/5
  (deferred to 2B). Link the report at `/tmp/stage2a_mic_verify_*.json`.
- [ ] **D3 — Update `docs/modules/wake-word.md` to describe the wake subpackage**
  (`src/voice_control/wake/{sherpa_onnx,livekit}.py`) and the
  `SPOT_WAKE_BACKEND` env switch. Reference the LiveKit yaml at
  `configs/wake/wakeword_hey_spot.yaml`.
- [ ] **D4 — Update `docs/modules/llm-brain.md` to describe LLMBrain (renamed
  from SpotBrain), the unified GBNF, the SSE streaming + TTSChunker hand-off,
  and the `SPOT_BRAIN_BACKEND` env switch.**
- [ ] **D5 — Mark all completed 2A tasks in
  `docs/superpowers/plans/archive/2026-05-15-stage2a-voice-pipeline.md`** with the
  commit SHA they landed in. Keep deviations (Q8_0 vs Q6_K, Parakeet default,
  --dry-run vs full rewrite) called out at the top so future readers see what
  shipped vs what the plan called for.

### Sanity gates (verify before merging cleanup)

- [ ] **V1 — Re-run mic-verify.** ASR pass must stay at 17/17, latency within
  ±30 ms of T12. Wake numbers expected to stay 1/5 until LiveKit train lands.
- [ ] **V2 — Run pin-guardian.** Confirm numpy 1.26.4, onnxruntime-gpu 1.23.0,
  kokoro-onnx 0.4.9, opencv-python 4.11.0.86 still pinned.
- [ ] **V3 — Confirm rollback drills.** From `docs/project/stage2-rollback.md`
  Layer B, run `SPOT_ASR_BACKEND=nemotron` boot the server, verify it listens
  on :50055. From Layer A, verify `SPOT_BRAIN_BACKEND=ollama` mode still
  resolves Gemma 4 (the qwen pair was removed in T14, so this only works if
  `gemma4:e4b` was kept).

## Execution order

1. C1, C2 (mechanical comment edits — safe).
2. V1 (mic-verify still 17/17).
3. C3 doc sweep (architecture first, then modules, then README).
4. D1–D5 (after docs are factually correct).
5. C5, C6 (script decisions — keep or archive).
6. V2 + V3 (full validation).
7. Final commit + push, no tag (this is cleanup, not a milestone).

## Out of scope

- Re-architecting wake word — that's 2B work.
- Phonetic-rewrites map growth — only add entries as real leaks appear.
- Removing `gemma4:e4b` Ollama model — that's the Layer A rollback target.
- Touching `models/kws/` — current default wake backend, do not move.

## How to know cleanup is done

- `grep -rn "qwen2\.5\|webrtcvad\|openwakeword" docs/ src/ scripts/` returns
  zero non-runbook hits.
- `scripts/mic_verify.py --room lab` returns 17/17 ASR pass and exit 0.
- All four pin-guardian targets verified.
- Both rollback layers (B and A) successfully boot in a smoke run.
