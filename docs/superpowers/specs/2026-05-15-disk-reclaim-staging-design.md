# Disk-Reclaim Staging Design — Stage 1 (peripheral-free) and Stage 2 (peripheral-dependent)

**Date:** 2026-05-15
**Branch:** `tour_guide_upgrade_matteo`
**Pre-Stage-1 commit:** `625184b` (`upgrade-plan.md: consolidate Phase 0 + Phase 9 + elegant-moseying-robin`)
**Author context:** Matteo Corrado (working solo on the Jetson without mic/speaker/Spot connected). Disk at 53 GB used / 787 MB free / 99% on `/dev/mmcblk0p1` per `project_jetson_disk_state.md` (refreshed 2026-05-15 by parallel session).

---

## Context

The master upgrade plan (`docs/project/upgrade-plan.md`) sequences nine phases of work to swap out the voice pipeline (Riva → Parakeet, qwen2.5/qwen2.5vl → gemma4:e4b, Kokoro v0.19 CPU → v1.0 GPU [done], etc.). The plan was written assuming peripherals were available.

The user is currently working without mic, speaker, or robot. Stage 1 of this design captures everything the user CAN do right now to (a) reclaim disk and (b) make productive code progress without peripherals. Stage 2 captures everything that has to wait for peripherals to return.

The constraint that drives the staging is the disk: only 787 MB free, and the master plan's biggest enabler (Phase 4 Riva removal, ~26 GB reclaim) doesn't actually need peripherals at all. Doing it now unblocks the rest of the plan.

A separate SSD is mid-install for an unrelated project. When it lands, a brief Stage 1.5 relocates Ollama models off the eMMC. Stage 1 does not depend on the SSD.

## Goals

1. Reclaim ~27 GB on the eMMC without using the mic/speaker/Spot.
2. Land all peripheral-free code work from the master plan that gates measurement credibility for later phases.
3. Swap the LLM/VLM brain to gemma4:e4b with text-only verification — keeping the qwen pair on disk as the one-line rollback until Stage 2 mic-verifies the swap in production.
4. Document a runbook to roll back to pre-Stage-1 state, with GitHub holding the canonical "this worked" reference via tags.
5. Define Stage 2 with two open evaluation gates (LiveKit Wakeword vs openWakeWord; YOLO26/YOLOE26 vs YOLO11n/YOLOE-11s) where the master plan's picks are now demonstrably stale.

## Non-goals

- Keeping voice control functional during/between Stage 1 and Stage 2. (User accepted indefinite downtime — peripherals are away anyway.)
- Re-litigating the Phase 2 model pick (`gemma4:e4b` is confirmed best fit per separate research summary; see Appendix A).
- Migrating to fine-tuned community gemma4 weights (none align with the use case; the two prominent variants are abliterated/uncensored, actively wrong for a campus tour-guide robot).
- Pre-deploying LiveKit Wakeword or YOLO26/YOLOE26 in Stage 1 — both are Stage 2 evaluation gates.
- Moving anything to SSD during Stage 1 — Stage 1.5 placeholder, fires when SSD is physically live.
- Pre-building Phase 7 unified audio integration — defers; noted as a potential earlier unlock now that gemma4 E4B has audio modality.

---

## Stage 1 — Peripheral-free disk reclaim + code work + brain swap

### Stage 1 Section A — Disk reclaim sequence

Five reclaim operations, ordered by safety and reversibility:

| # | Operation | Reclaim | Risk | Rollback |
|---|---|---|---|---|
| A.1 | `rm -rf ~/riva_quickstart_arm64_v2.17.0/` | -2.1 GB | Low — tarball already drained | Re-download tarball from NVIDIA NGC |
| A.2 | `docker stop riva-speech || true` then `docker rm riva-speech` | -380 KB | Trivial | `docker run` recreates |
| A.3 | `docker rmi nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64` | **-24.04 GB** | Low — voice broken anyway | Re-pull from `nvcr.io` (24 GB download) |
| A.4 | VS Code CLI prune: keep `Stable-0958016b2af9f09bb4257e0df4a95e2f90590f9f` (most recent per `lru.json`); `rm -rf` the other three (`Stable-8b640...`, `Stable-41dd7...`, `Stable-e7fb5...`) | ~-975 MB | Low — VS Code re-downloads if needed | VS Code reinstalls cli on next remote connect |
| A.5 | `rm ~/riva_quickstart_arm64_v2.17.0.tar.gz` (already 0 bytes) | 0 | None | n/a |

**Subtotal: ~27.1 GB freed.** Pre-state 787 MB free → post ~27.9 GB free.

**Verification gates after Section A:**
- `docker system df` shows Images=0, no `riva-speech` container.
- `df -h /` shows ≥ 26 GB recovered vs pre-state.
- `ls ~/.vscode-server/cli/servers/` shows exactly one `Stable-*` directory.
- `code-server` (or VS Code remote) still launches; `~/.vscode-server/cli/servers/lru.json` still references the kept version.

### Stage 1 Section B — Code work (peripheral-free)

Three blocks of pure code edits, sequenced so every commit leaves the tree compilable.

**B.1 — Phase 0.2 sub-fixes** (3 surgical edits)
- `scripts/run_voice_control.py:48,74,86`: replace bare `except Exception: return False` in `_port_open` / `_docker_container_running` / `_docker_container_exists` with `except Exception as e: print(f"[startup] {fn.__name__} probe failed: {type(e).__name__}: {e}", file=sys.stderr); return False`. Diagnostic on Docker socket permission errors.
- `src/voice_control/spot_dispatch.py:434-435`: `_navigate_waypoint_sequence`: replace silent `except Exception: pass` on `navigation_feedback` with log + `break` so caller can re-issue.
- New helper `_normalize_location_name(name)` in `src/voice_control/spot_dispatch.py`. Call from `save_location` (`:1257`), `tour` (`:1372`), `patrol` (`:1428`), and `src/location_manager.py:144,169,185`. Audit existing `locations.json` for pre-existing space-named entries needing migration.

Verification: `grep -nE 'except Exception:\s*\b(return False|pass)\b' scripts/run_voice_control.py src/voice_control/spot_dispatch.py` returns no matches in patched lines.

Commit message: `phase 0.2: log on probe failure, break on nav feedback failure, normalize location names`

**B.2 — Phase 0.3 Wave 2 latency wiring** (5 callsite edits)
Producers (`src/voice_control/latency.py`, `src/voice_control/startup_status.py`) already shipped. Wave 2 wires them in:
- `src/voice_control/client_mic.py` — argparse `--latency` / `--latency-out` flags; `init_recorder()` in the `--latency` branch; greet block calls `tts.speak_sync(startup_status.build_template_status(...))`; `with time_phase("brain_warmup"):` around `client_mic.py:545-553`; `with trace_span("vlm"):` around `brain.query_vlm(...)` in `process_utterance`; `with trace_span("dispatch"):` around the action loop; `try / finally` with `recorder.begin_utterance() / complete()` brackets per turn.
- `src/voice_control/spot_tts.py` `speak()` (~L161): fetch `get_recorder().current()`, build closures for `on_render_start/end/play_start/play_end`, pass to `enqueue_render`. The `_Task` dataclass already accepts these.
- `src/voice_control/spot_dispatch.py` `get_robot_state_dict()`: add `_PROCESS_START_TS` constant + `uptime_s` / `estop_holder` / `num_saved_locations` / `process_pid` fields.
- `README.md` — append `## Latency Metrics` section.

Verification (no peripherals): `python -c "import src.voice_control.client_mic; import src.voice_control.spot_tts; import src.voice_control.spot_dispatch"` imports clean. `grep -n 'init_recorder\|time_phase\|trace_span' src/voice_control/client_mic.py` shows wires. Real JSONL trace validation deferred to Stage 2 (needs mic interaction).

Commit message: `phase 0.3: wire latency producers into client_mic, spot_tts, spot_dispatch`

**B.3 — Phase 4 cosmetic code cleanup (excluding `server.py`)**
- `scripts/run_voice_control.py:31-32, 103, 294`: delete `RIVA_CONTAINER`, `RIVA_PORT`, `ensure_riva()`, the Riva startup branch in `main()`. Keep `start_asr_server()` because `server.py` still runs as a subprocess.
- `requirements.txt:15`: drop `nvidia-riva-client>=2.17.0`.
- Delete `scripts/setup_riva.sh`.
- Update `README.md` and `docs/architecture/overview.md`: new pipeline diagram, no Riva quickstart step.
- Update `docs/project/status.md`: drop Riva from feature/model lineup.

Deferred to Stage 2: `server.py` `RivaBackend` deletion (Phase 4 step 1) — coupled to ParakeetBackend existing.

Verification: `grep -rn -i 'riva\|nvidia-riva' --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' .` returns only intentional references (e.g., `server.py` Backend wrapper note, this design doc, `docs/project/stage1-rollback.md`). `python -c "import scripts.run_voice_control"` imports clean.

Commit message: `phase 4 (partial): remove Riva orchestration from non-server code`

**B.4 — Phase 8.x cleanup (deferred to Stage 2 or later)**
The plan's god-function refactor and `tour`/`patrol` collapse have high blast radius and would conflict with Phase 6 Stage A's `SYSTEM_PROMPT` rewrite. Listed for completeness; do not start in Stage 1 unless explicitly requested.

### Stage 1 Section C — LLM/VLM brain swap (gemma4:e4b)

**C.1 Ollama in-place upgrade 0.16.1 → ≥0.20.0**

```
sudo systemctl stop ollama
sudo cp /usr/local/bin/ollama ~/ollama-0.16.1.backup
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
ollama --version    # confirm >= 0.20.0
ollama list         # confirm qwen pair still resident
```

The systemd override at `/etc/systemd/system/ollama.service.d/override.conf` (`OLLAMA_MAX_LOADED_MODELS=2`, `KEEP_ALIVE=-1`, `NUM_PARALLEL=1`) is user-owned and survives the upgrade.

Verification: `ollama run qwen2.5:7b "say hi"` returns a token stream (proves the upgraded daemon talks to existing models).

Rollback: documented in `stage1-rollback.md` Section 5.2.

**C.2 Disk pre-flight + gemma4 pull**
Hard gate: `df -h /` must show ≥ 12 GB free (9.6 GB model + Ollama scratch). After Section A reclaim we'll have ~28 GB free.

```
ollama pull gemma4:e4b
ollama list   # three models now resident
```

Verification: `ls -lh /usr/share/ollama/.ollama/models/blobs/` shows the new gemma4 manifest; `df -h /` confirms ~9.6 GB consumed.

**C.3 Phase 2 brain edits** in `src/voice_control/llm_brain.py`:
- `:28-29` — `DEFAULT_MODEL = "gemma4:e4b"` and `VLM_MODEL = "gemma4:e4b"`.
- `:249-304` — `warm_up_vlm()` becomes a no-op (single model is already warm). Delete the 1×1 JPEG bytes and `_vlm_warmed` state machine.
- Drop the eager warm thread at `src/voice_control/client_mic.py:548`.

Defer the optional `query_vlm` consolidation (line 511 reusing the `process()` path) — adds diff surface without Stage 1 verification value.

Verification: `python -c "from src.voice_control.llm_brain import SpotBrain; b = SpotBrain(); print(b.model)"` prints `gemma4:e4b`. `grep -n '_vlm_warmed\|warm_up_vlm' src/voice_control/llm_brain.py` returns only the no-op stub.

**C.4 Text-only verification** (no peripherals)
Run `python src/voice_control/llm_brain.py gemma4:e4b` against representative typed transcripts via the existing CLI test at `:608-657`. Confirm per utterance:
- JSON parses cleanly.
- Action selection matches one of the 28-action catalog names.
- Params match the action's expected schema.
- `response` field is non-empty and stylistically plausible.

VLM path: pass any sample JPEG (e.g., from `models/maps/` or `git log --all -- '*.jpg'`) to `query_vlm("describe this image")` and confirm coherence.

Resource check: `tegrastats --interval 1000 --logfile /tmp/tegra.gemma4.log` for 30 s during inference. Confirm GPU memory growth within the 6-8 GB e4b forecast.

**Pass criteria:** all 4 above on the typed corpus + VLM smoke. **Fail = roll back per Section 5 BEFORE committing the brain edits.** Text-verify is suggestive, not production-grade — Stage 2 mic verification is still required before reclaiming the qwen pair.

**Notes captured during the gemma4 lookup (Appendix A):**
- gemma4 E4B has audio modality natively. Phase 7 unified audio (originally deferred awaiting `ollama#15333`) may now be reachable through the standard chat API. Out of scope for Stage 1; flagged for future planning.
- gemma4 E4B supports 128K context (vs the plan's conservative 8K forecast). Default Ollama `num_ctx` is 2048 — explicit `num_ctx` bump in `llm_brain.py` would unlock Phase 6 MAX_HISTORY headroom. Out of scope for Stage 1; noted for Phase 6.

Commit message: `phase 2: collapse LLM + VLM to gemma4:e4b (text-verified, mic-verify pending)`

**C.5 Qwen pair stays put**
Explicitly do NOT run `ollama rm qwen2.5:7b qwen2.5vl:7b` in Stage 1. Reclaim of those 10.7 GB is Stage 2 work, gated on real-mic verification of gemma4 in production.

---

## Stage 1.5 — SSD migration (when SSD is physically mounted)

Fires when the user's separately-installed SSD is mounted and writable. Decoupled from Stage 1's critical path because the SSD is mid-install.

**1.5.1 Ollama model store relocation**
- `sudo systemctl stop ollama`
- `sudo mv /usr/share/ollama/.ollama/models /mnt/ssd/ollama-models` (path placeholder — adjust to actual SSD mount).
- Edit `/etc/systemd/system/ollama.service.d/override.conf` to add `Environment="OLLAMA_MODELS=/mnt/ssd/ollama-models"`.
- `sudo systemctl daemon-reload && sudo systemctl start ollama`
- Verify: `ollama list` shows all models still present; pulling a new model lands on SSD (`ls /mnt/ssd/ollama-models/blobs/`).
- Reclaims ~10-20 GB on eMMC depending on which models are resident.

**1.5.2 CLIP cache symlink**
- `mv ~/.cache/clip /mnt/ssd/clip-cache && ln -s /mnt/ssd/clip-cache ~/.cache/clip`
- Reclaims 343 MB on eMMC. Load-bearing constraint dissolves.

**1.5.3 Logs directory symlink**
- `mkdir -p /mnt/ssd/spot-logs && ln -s /mnt/ssd/spot-logs ~/spot/dartmouth_spot_capstone/logs` (or move existing `logs/` contents in first if present).
- Sets up Phase 0.3 latency JSONL, Phase 6 Stage A conversation log, Phase 6 Stage C web_search audit log to write to SSD endurance.

**1.5.4 Rollback note**
Once Ollama models live on SSD, the gemma4-fail rollback to qwen is instant (1-line edit in `llm_brain.py`). The Riva re-acquisition path can target SSD (`docker pull` writes to `/var/lib/docker` on eMMC by default — relocating Docker storage is a bigger ask, deliberately not in scope here).

What stays on eMMC: OS, the repo + venv, Docker container runtime.

---

## Stage 2 — Peripheral-dependent work

Fires when peripherals + Spot are back. Two flagged decision points where the master plan's picks are stale (LiveKit Wakeword vs openWakeWord; YOLO26/YOLOE26 vs YOLO11n/YOLOE-11s) need re-measurement before commit.

### 2.1 Mic-verify gemma4 (gates qwen reclaim)

Run `scripts/run_voice_control.py` with mic + speaker (no robot needed). Walk through the Phase 1 regression utterance set (stand/sit/walk-1m/go-to-X/follow/stop/battery/describe-surroundings/etc.). Confirm action selection ≥ baseline, JSON parses, latency feels reasonable, VLM returns coherent fisheye descriptions.

- **Pass:** `ollama rm qwen2.5:7b qwen2.5vl:7b` reclaims **-10.7 GB**.
- **Fail:** flip `DEFAULT_MODEL = "qwen2.5:7b"` and `VLM_MODEL = "qwen2.5vl:7b"` in `llm_brain.py:28-29`, restart. Diagnose gemma4 in isolation. Stay on qwen until fixed.

### 2.2 Phase 1 audio corpus capture

10-15 representative clean utterances → `tests/audio/*.wav`. 5 motor-idle clips for hallucination check. `tegrastats --interval 1000 --logfile /tmp/tegra.before.log` baseline during a normal voice interaction.

### 2.3 Phase 3 Parakeet STT swap

Per `~/.claude/plans/elegant-moseying-robin.md`:
- Pre-flight checks PF-1..PF-6.
- Phase A: `pip install --no-deps onnx-asr==0.11.0 huggingface-hub>=0.30.2`, `scripts/setup_parakeet.py`, refactor `server.py` into `Backend` interface, `ParakeetBackend` (~80 LOC), `RivaBackend` extracted (kept as code rollback even though Riva itself is gone — until Phase C green).
- Phase B: Silero VAD swap.
- Phase C: flip `SPOT_ASR_BACKEND=parakeet`, run `scripts/test_stt.py` against the audio corpus.

Assumption check (validated 2026-05-15): Parakeet TDT 0.6b v3 confirmed running on Jetson AGX Orin (HF discussion #21), 10× faster than Whisper Large V3 Turbo, zero hallucinations (trained on 36k hours of pure non-speech with empty-string targets). Sticking with the master plan's pick.

### 2.4 Phase 4 final code cleanup + Phase E wake-word swap

- Step 1 (deferred from Stage 1): delete `RivaBackend` from `server.py`. Safe now — Parakeet is the live backend.
- **Phase E wake-word swap (open evaluation gate):** the master plan locks in `openWakeWord` as the replacement for sherpa-onnx KWS. Before committing, also benchmark **`livekit/livekit-wakeword`** on the same "Hey Spot" custom-trained model. LiveKit claims:
  - 100× fewer false positives per hour
  - 17× more wake-word detections (60× lower AUT)
  - Standard ONNX export, drop-in compatible with openWakeWord loaders
  - Same training pipeline (Piper synthetic positives), same model-size class (~5 MB)
  - Apache 2.0
- Verify on a quiet-room baseline + a noisy demo-floor recording. Bias tiebreaker toward LiveKit if metrics validate (claimed gains are structural — conv-attention head — not just numerical). openWakeWord is the rollback if LiveKit underdelivers.

### 2.5 Phase 5 Spot end-to-end

Robot powered, E-Stop in another terminal:
- stand → walk forward 1m → sit
- "what do you see?" (VLM via gemma4)
- "go to <known location>" (GraphNav)
- "follow me" then "stop"
- Smoke door + follow last (riskier paths per `docs/project/status.md`)
- If all pass: merge `swap-models` to `main`.

### 2.6 Phase 9.B vision overhaul (Stages 0-3 unchanged; Stages 4-5 expanded as benchmark matrix)

Stages 0-3 from the master plan unchanged: per-camera rotation table + `Frame` namedtuple, gripper plumbing, batched 5-camera scan, `look_around` action.

**Stages 4-5 (revised as a per-role-sized benchmark matrix):**

The vision stack has two distinct roles with different latency budgets:

| Role | Frequency | Latency budget | Accuracy bias |
|---|---|---|---|
| Person follow loop | 10 Hz target | ~50 ms / frame | Speed > accuracy (single-class person) |
| Open-vocab object scan | Once per command, ~1-2 s | ~500 ms / 5-camera batch | Accuracy > speed |

The right answer is likely two different models, not one unifier. The CPU benchmark already showed YOLOE is not a unifier (4× slower than YOLO11n on CPU); GPU may not flip cleanly enough to justify the integration cost.

**Closed-vocab (person follow) candidates:**

| Candidate | Approx size | Notes |
|---|---|---|
| YOLOv8n (current) | 7 MB | CPU baseline 3.04 FPS / 329 ms |
| YOLO11n | 7 MB | CPU 3.21 FPS — only 5% faster on CPU; GPU may differ |
| YOLO11s | 22 MB | If 11n is plenty fast on GPU, step up for accuracy headroom |
| **YOLO26-N** | ~7 MB | NMS-free, DFL removed, +43% CPU vs YOLO11-N — likely winner |
| YOLO26-S | ~22 MB | Larger sibling for accuracy headroom on GPU |

**Open-vocab (object scan) candidates:**

| Candidate | Approx size | Notes |
|---|---|---|
| YOLO-Worldv2-s (current) | ~28 MB + 343 MB CLIP cache | CPU 0.97 FPS / 1034 ms |
| YOLOE-11s | ~25 MB + 572 MB MobileCLIP cache | CPU 0.82 FPS — slower on CPU but seg-capable |
| YOLOE-11m | ~50 MB | Higher accuracy; GPU may make latency acceptable |
| **YOLOE26-S** | similar to YOLOE-11s | NMS-free, +1.6% LVIS mAP over YOLOE-L |
| YOLOE26-M | similar to YOLOE-11m | Larger, accuracy ceiling on GPU |

**Why this expansion exists:** the plan's existing YOLO11n / YOLOE-11s pick was made on a CPU benchmark (`project_yolo_cpu_benchmarks.md`, 2026-04-08) where no model swap helped (5% gain). YOLO26 was on the radar but never benchmarked because the bar to switch on CPU was deemed unmeetable. Stage 4's GPU port is exactly the moment that deferred decision needs to be made — and YOLO26's claimed wins (NMS-free, structural simplification, 43% CPU speed-up) are the kind that translate to GPU. Disk cost is not the gating constraint post-Stage-1 reclaim.

**Benchmark protocol:**
1. ONNX-export each candidate via `model.export(format="onnx", dynamic=True, batch=5)`.
2. Load via `onnxruntime.InferenceSession(..., providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`.
3. Measure with `tegrastats --interval 1000` running — GPU utilization MUST spike. If it stays at 0%, ONNX silently fell back to CPU; debug before measuring.
4. Per role, capture: ms/frame, FPS, mAP on a held-out set of 20-30 captured Spot frames.
5. Pick winner per role. Bias tiebreakers toward NMS-free YOLO26 family for structural simplicity. Document why if not.
6. Update this design doc with measured numbers; write a benchmark-result memory analogous to `project_yolo_cpu_benchmarks.md`.

### 2.7 Phase 6 Stage A/C/D smart-chatbot

Stage B (knowledge packs) already shipped. Stages A (persona + MAX_HISTORY 12→24 + conversation log), C (web_search with Tavily + 5 safety mitigations), D (FactsStore SQLite). These benefit from real mic interaction for the persona/free-form tests.

### 2.8 Stage 2 ordering

- 2.1 must run before 2.4 step 1 (server.py RivaBackend deletion).
- 2.4 wake-word eval can run in parallel with 2.5/2.6 — different files.
- 2.6 benchmark matrix and 2.7 are independent — different files.
- 2.5 must complete before merging `swap-models` to `main`.

---

## Stable-state preservation + rollback

### Code rollback (git tags + GitHub mirror)

Three tags total on `tour_guide_upgrade_matteo`, all pushed to origin:
- `pre-stage1-stable` — created BEFORE any Stage 1 edit, at commit `625184b`. Absolute restore point.
- `stage1-complete` — after the last Stage 1 commit lands and text-verify passes.
- `stage2-complete` — after Phase 5 mic+robot e2e passes, before merging to `main`.

```
git tag -a pre-stage1-stable -m "Pre-Stage 1: Riva live, Ollama 0.16.1 + qwen pair, no gemma4, no Stage 1 code edits"
git push origin pre-stage1-stable
git push origin tour_guide_upgrade_matteo
```

After every Stage 1 commit, also `git push origin tour_guide_upgrade_matteo` so each commit is mirrored to GitHub immediately. No silent local-only drift. Per project CLAUDE.md, the user manually pushes (no credential helper on Jetson).

Rollback to `pre-stage1-stable`: `git checkout pre-stage1-stable` (detached HEAD inspection) or `git reset --hard pre-stage1-stable` on the branch (destructive — only with explicit user approval at the time).

### Model state rollback

- **Ollama binary**: backup created in Section C.1 at `~/ollama-0.16.1.backup`. Revert: `sudo systemctl stop ollama && sudo cp ~/ollama-0.16.1.backup /usr/local/bin/ollama && sudo systemctl start ollama && ollama --version` returns `0.16.1`. Models on disk survive (under `/usr/share/ollama/.ollama/models/`, untouched by binary swap).
- **Brain model default**: 2-line edit at `llm_brain.py:28-29`:
  ```python
  DEFAULT_MODEL = "qwen2.5:7b"
  VLM_MODEL = "qwen2.5vl:7b"
  ```
  Restart `run_voice_control.py`, qwen pair takes over. gemma4:e4b stays in Ollama as fallback's-fallback unless `ollama rm gemma4:e4b` to free 9.6 GB.

### Off-disk artifact re-acquisition

| Artifact | Re-acquire | Cost |
|---|---|---|
| Riva Docker image | `docker pull nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64` (requires NGC API key — `ngc config set` first if not already) | ~24 GB download |
| Riva quickstart dir | `cd ~ && wget <NGC_URL_CAPTURED_AT_RUNBOOK_TIME> && tar -xzf riva_quickstart_arm64_v2.17.0.tar.gz` | ~2.1 GB download |
| VS Code old CLIs | Re-downloaded automatically on next remote-VSCode connect to `~/.vscode-server/cli/servers/Stable-<sha>/` | ~325 MB per version |

The Riva quickstart URL must be captured from the live NGC catalog at the time the runbook is written, because NGC URLs occasionally rotate.

### The runbook itself: `docs/project/stage1-rollback.md`

Single source of truth. Sections:
1. **What "stable" means** — pre-Stage-1 inventory (branch HEAD + tag, Riva location, Ollama version, qwen models, VS Code CLI versions).
2. **Code rollback** — tag commands, restore-from-tag, verification.
3. **Ollama binary rollback** — cp + systemctl sequence, verification.
4. **Brain rollback** — 2-line edit, verification (`grep -n DEFAULT_MODEL src/voice_control/llm_brain.py`).
5. **Riva re-acquisition** — captured URLs + commands, verification (`docker images | grep riva-speech` shows 24 GB).
6. **VS Code restoration** — "do nothing, next connect re-downloads."
7. **Disk-pressure guard** — before any rollback that re-acquires artifacts, run `df -h /` first; if free space < 30 GB, abort and `ollama rm gemma4:e4b` first.
8. **Decision tree** — gemma4 acts up → 5.2 brain revert (cheap, 1 minute). Stage 1 code edits broke something → 5.1 code revert (cheap, 1 minute). Need full Riva → 5.3 re-acquisition (slow, 30-60 min, NGC auth).

The runbook is committed and pushed to GitHub as part of Stage 1's first commit so it's findable from origin if the local checkout dies.

A placeholder section for `stage2-rollback.md` (Parakeet model files, openWakeWord/LiveKit Wakeword model, ONNX YOLO exports, possibly Tavily API key) lives at the bottom of `stage1-rollback.md`. Real `stage2-rollback.md` is written as part of Stage 2 work.

---

## Ordering, risks, parallelization

### Hard ordering constraints

1. Audit → reclaim (read-only audit before any `rm`/`docker rmi`).
2. Riva quickstart deletion → Riva Docker rmi (logged in this order for runbook fidelity).
3. Ollama 0.16.1 backup → in-place upgrade → version check.
4. Ollama upgrade → gemma4 pull (gemma4 needs ≥ 0.20.0).
5. gemma4 pull → brain code edits.
6. Phase 0.2 → 0.3 wiring → Phase 4 cosmetic edits (small, reviewable diffs).
7. Stage 1 commits → push to origin (per commit; no silent local-only drift).
8. `pre-stage1-stable` tag → first Stage 1 commit (tag the rollback point BEFORE editing anything).
9. Stage 2 mic-verify of gemma4 → `ollama rm qwen2.5*` (qwen stays until peripherals confirm).
10. Phase 3 ParakeetBackend exists → Phase 4 step 1 (`server.py` needs SOME backend at all times).

### Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ollama in-place upgrade breaks daemon | Low | High | Binary backup + verified rollback procedure |
| gemma4 text-verify passes, mic-verify regresses | Medium | Medium | qwen pair on disk; 1-line revert |
| Phase 0.3 latency wiring breaks `client_mic.py` import | Low | High | `--latency` off-by-default; producers no-op without recorder; `python -c "import"` test |
| Phase 4 disk reclaim done; later edit introduces stale Riva ref | Medium | Low | Phase 4 cosmetic edits land in same Stage 1; `grep -rn riva` verification |
| LiveKit Wakeword claims don't hold up on Spot's mic | Medium | Low | Stage 2 eval gate; openWakeWord is rollback |
| YOLO26/YOLOE26 ONNX export issues on Jetson | Medium | Medium | Eval matrix includes YOLO11n/YOLOE-11s known-working candidates; tiebreaker to YOLO26 only if export + GPU spike validate |
| User pushes the wrong way and overwrites origin | Low | High | Runbook spells exact `git push origin tag-name`; no force-push |

### Parallelization

Stage 1: nothing parallelizes meaningfully — sequential.

Stage 2:
- 2.1 → 2.2 → 2.3 sequential.
- 2.4 wake-word eval independent of 2.6 / 2.7 — can interleave.
- 2.6 benchmarks and 2.7 are independent — can interleave.
- 2.5 must be the last gate before merging to main.

---

## Appendix A — gemma4 lookup summary (2026-05-15)

**Variants on Ollama:**
| Tag | Size | Active params | Modalities |
|---|---|---|---|
| `gemma4:e2b` | 7.2 GB | 2.3B effective | Text + Image + Audio |
| **`gemma4:e4b`** | **9.6 GB** | **4.5B effective** (8B w/ embeddings) | Text + Image + Audio |
| `gemma4:26b` | 18 GB | ~3.8B active per token (MoE, 25.2B total VRAM) | Text + Image |
| `gemma4:31b` | 20 GB | 30.7B dense | Text + Image |
| `gemma4:31b-cloud` | cloud | 30.7B | Text + Image |

**New info vs the master plan:**
- E4B has audio modality (potential earlier Phase 7 unlock).
- 128K context window (master plan said 8K).
- Quantization on Ollama is Q4_K_M (default).
- Ollama tag last updated ~1 week ago — actively maintained.
- No newer Gemma variant exists as of 2026-05-15.

**Community new-weights variants (not pure quants):** the two with traction (`HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`, `OBLITERATUS/gemma-4-E4B-it-OBLITERATED`) are abliterated/uncensored — actively wrong for a campus tour-guide robot. Not used. If domain regression appears in Stage 2 mic-verify, options are: (a) try a vetted community quant for slightly better quality at same size, (b) QLoRA fine-tune on collected (transcript, action) pairs from `logs/conversations.jsonl` (10 GB VRAM, single-day job), (c) fall back to qwen pair (still on disk).

Sources:
- [google/gemma-4-E4B-it on Hugging Face](https://hf.co/google/gemma-4-E4B-it)
- [Gemma 4 release blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Ollama gemma4 model page](https://ollama.com/library/gemma4)
- [Ollama gemma4:e4b tag](https://ollama.com/library/gemma4:e4b)
- [Unsloth Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train)

## Appendix B — Disk audit results (2026-05-15)

Pre-Stage-1 state (refreshed 2026-05-15 by parallel session per `project_jetson_disk_state.md`):
- `/dev/mmcblk0p1` 57 GB total / 53 GB used / **787 MB free / 99%**.
- Riva Docker image 24.04 GB at `/var/lib/docker` (single image, 100% reclaimable).
- Riva quickstart dir 2.1 GB at `~/riva_quickstart_arm64_v2.17.0/`.
- Riva tarball 0 bytes (already drained).
- Ollama 14 GB total = 10 GB models in `/usr/share/ollama/.ollama/models` (qwen2.5vl:7b 6.0 + qwen2.5:7b 4.7) + 4 GB binary in `/usr/local/lib/ollama`.
- VS Code 1.3 GB in `~/.vscode-server/cli/servers/` across 4 versions; most-recent per `lru.json` is `Stable-0958016b2af9f09bb4257e0df4a95e2f90590f9f`.
- `~/.cache/clip` 343 MB (load-bearing — YOLO-World needs it; do NOT delete in Stage 1).
- `~/.vscode-server/{data,extensions}` 139 + 240 MB (in-use; do NOT touch).
- pip cache 0 (already empty per parallel-session note).
- HuggingFace cache absent.
- Old `models/tts/kokoro-en-v0_19/` already removed (Phase 2.5 step 13).

## Appendix C — Stage 2 assumption-check results (2026-05-15)

**Phase 3 (Parakeet TDT v3 ASR) — STILL THE RIGHT PICK ✅** Validated stronger: confirmed running on Jetson AGX Orin (HF discussion #21), 10× faster than Whisper Large V3 Turbo, zero hallucinations (trained on 36k h pure non-speech / empty-string targets). Possible future-look: `Nemotron-Speech-Streaming-en-0.6b` for true streaming; defer.

**Phase 4 (Riva removal) — VALIDATED MUCH STRONGER ✅** NVIDIA has deprecated Jetson Orin in favor of Jetson Thor; Riva TTS NIM docs migrated March 2026 to Speech NIM (x86-only). Riva is on a hard sunset path.

**Phase 4 step 8 (openWakeWord) — POSSIBLY DETHRONED ⚠️** LiveKit Wakeword (`livekit/livekit-wakeword`) claims 100× fewer false positives, 17× more detections, drop-in ONNX compatibility with openWakeWord. Stage 2 eval gate.

**Phase 9.B Stage 4-5 (YOLO11n + YOLOE-11s) — POSSIBLY DETHRONED ⚠️** YOLO26 / YOLOE26 released 2026-01-14: YOLO26-N 43% faster CPU than YOLO11-N, NMS-free end-to-end, DFL removed. YOLOE26 +1.6% LVIS mAP over YOLOE-L. Stage 2 eval matrix.

**Phase 2 (LLM/VLM collapse to gemma4) — STILL RIGHT ✅** gemma4 31B scores 86.4% on τ2-bench (agentic tool use) vs Gemma 3 27B's 6.6%. No newer model on the horizon.

Sources:
- [Riva NIM TTS support matrix (x86-only)](https://docs.nvidia.com/nim/riva/tts/latest/support-matrix.html)
- [Riva release notes — TTS NIM doc migration](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/release-notes.html)
- [LiveKit Wakeword on GitHub](https://github.com/livekit/livekit-wakeword)
- [LiveKit blog — open-source wake word training](https://livekit.com/blog/livekit-wakeword)
- [openWakeWord repo](https://github.com/dscripka/openWakeWord)
- [YOLO26 — Ultralytics docs](https://docs.ultralytics.com/models/yolo26)
- [YOLOE — Ultralytics docs](https://docs.ultralytics.com/models/yoloe)
- [Parakeet TDT v3 — Jetson AGX Orin discussion](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/21)
- [project_yolo_cpu_benchmarks.md (memory, 2026-04-08)](file:///home/spotdog/.claude/projects/-home-spotdog/memory/project_yolo_cpu_benchmarks.md) — original CPU benchmark
- [project_jetson_disk_state.md (memory, 2026-05-15)](file:///home/spotdog/.claude/projects/-home-spotdog/memory/project_jetson_disk_state.md) — refreshed disk inventory
