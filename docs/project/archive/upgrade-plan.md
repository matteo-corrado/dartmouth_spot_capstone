# Spot Voice Pipeline Upgrade — Slim STT + Gemma 4 + GPU TTS + Smart Chatbot (Implementation Plan)

## Context

The Dartmouth Spot capstone runs a full voice‑controlled robot pipeline today and the full pipeline IS enabled by default. **Current pipeline (2026-04-08):**

```
Mic → VAD (webrtcvad) → KWS (sherpa-onnx) → ASR bridge (server.py) → Riva / Canary-Qwen-2.5B (Docker)
    → LLM brain (Ollama qwen2.5:7b) → Dispatcher (28 BD SDK actions) → TTS (Kokoro v1.0, GPU) [Phase 2.5 ✅]
                                          ↘ VLM (Ollama qwen2.5vl:7b)
                                          ↘ visual_nav.py (YOLO-Worldv2 PyTorch CPU)
                                            with WorldObjectClient + LocalGrid + RayCast helpers [Phase 9.A in tree]
```

**Target pipeline (after this rehaul):**

```
Mic → Silero VAD → openWakeWord ("Hey Spot") → ASR bridge → Parakeet TDT v3 (ONNX, CUDA EP)
    → LLM/VLM brain (Ollama gemma4:e4b, single multimodal model) → Dispatcher → TTS (Kokoro v1.0, GPU)
                                          ↘ web_search (Tavily, with safety lockout)
                                          ↘ FactsStore (SQLite)
                                          ↘ visual_nav.py (YOLO/YOLOE ONNX, CUDA EP, batched 5-camera scan + look_around)
                                            with full perception stack (mask-based depth, WalkToObjectInImage, fiducials)
```

**Goals:**
1. Drop NVIDIA Riva (multi‑GB Docker, Canary‑Qwen 2.5B) for a smaller in‑process ASR.
2. Replace the LLM brain AND collapse the separate VLM into a single multimodal model.
3. Move Kokoro TTS from CPU to the Jetson GPU so time‑to‑first‑audio shrinks.
4. Make the chatbot itself smarter — stronger persona, static local knowledge, optional web search with safety lockout, and persistent remembered facts.
5. Keep `server.py` as the ASR backend boundary so we can swap engines without touching `client_mic.py`.
6. Verify the pipeline still works end‑to‑end after each swap.
7. **Fix the perception stack** — depth decoding, world-object routing, obstacle awareness via LocalGrid + RayCast, and a multi-camera "do you see X?" that actually scans all body cameras with the right rotations and a batched GPU detector. (Phase 9, mostly done in working tree.)
8. **Harden the substrate** — AudioPlayer race fixes + watchdog, latency observability, codebase Tier 1 bug fixes. (Phase 0, mostly done in working tree.)

**Independent validation:** NVIDIA staff in their own developer forums now tell Jetson users Riva is deprecated. The Riva successor (Speech NIM 26.02.0) doesn't list Jetson AGX Orin in its support matrix. JPS contains zero speech components. Dropping Riva is the right call.

**Already shipped baseline (as of 2026‑04‑08):**
- **Qwen pinning stop-gap** (2026‑04‑07): both `qwen2.5:7b` and `qwen2.5vl:7b` resident in Ollama via a systemd drop‑in (`OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1`) plus matching `keep_alive=-1` in `llm_brain.py`. The systemd drop‑in stays useful after Phase 2 (becomes `MAX_LOADED_MODELS=1`); the eager `warm_up_vlm()` call simplifies to a no‑op once Phase 2 lands.
- **Phase 2.5 — Kokoro v1.0 GPU TTS** (commit `2141124`): `kokoro_onnx` + `onnxruntime-gpu==1.23.0` with `CUDAExecutionProvider`. Public `SpotTTS` API preserved.
- **Phase 6 Stage B — Knowledge packs** (commit `b61b4e3`): `src/voice_control/knowledge/{thayer_knowledge.md, tour_route.md}` injected into the SYSTEM_PROMPT at module scope (`llm_brain.py:139-169`).
- **Audio refactor** (commit `8144794`): single shared `AudioPlayer` owns the OutputStream + worker; `SpotTTS` and `audio_feedback` delegate to it.
- **wakespot** (commits `2a523b0` / `3b4b6c0` / `9b1ed20` / `320d1da`): one-command launch (E-Stop + voice control + map upload + load_map voice intent + graceful shutdown choreography).
- **bosdyn SDK 5.0.1.1 → 5.1.4**: all 6 packages, zero-risk verified.
- **Phase 0.1 AudioPlayer hardening** (in working tree, awaiting commit): inflight counter, watchdog thread, graceful shutdown — see Phase 0.1.
- **Phase 0.2 Tier 1 bug fixes** (in working tree, awaiting commit): patrol loop, bare excepts, web_panel fd leak, LLM/VLM tag visibility — see Phase 0.2.
- **Phase 9.A perception overhaul** (in working tree, awaiting commit): door deletion, depth correctness, `WalkToObjectInImage`, `WorldObjectClient`, LocalGrid + RayCast helpers, `check_obstacles` action — see Phase 9.A audit table. 14/15 tiers landed.

---

## Chosen approach

**STT:** Run a three‑way tournament behind `server.py` and pick the winner from recordings of Spot's actual mic. Candidates:
1. **sherpa‑onnx SenseVoice‑Small int8** — easiest install, same runtime as KWS+TTS, structurally hallucination‑resistant
2. **IBM Granite 4.0 1B Speech with keyword biasing** — newest model (Mar 6 2026), Apache 2.0, #1 OpenASR leaderboard, keyword biasing tuned to Spot's command vocabulary is the killer feature
3. **NeMo Parakeet‑TDT‑0.6b‑v3 via `onnx-asr`** — best WER/size ratio, hallucination‑hardened by training, pip‑installable on JP6.2

Fallback: `distil-large-v3.5-ct2` in `dustynv/faster-whisper` container.

**LLM + VLM:** Collapse both `qwen2.5:7b` and `qwen2.5vl:7b` into a single **`gemma4:e4b`** (9.6 GB on disk, multimodal text+image+audio, native function calling, Apache 2.0, designed for Jetson Orin‑class hardware).

**TTS GPU:** Swap `sherpa-onnx` (CPU, Kokoro v0.19) for `kokoro-onnx` + `onnxruntime-gpu` (Jetson AI Lab prebuilt wheel, Kokoro v1.0). Same model family, public `SpotTTS` API preserved, ~30–50 line diff confined to `spot_tts.py`. Detailed in Phase 2.5.

**Smart chatbot:** Once Phase 2 lands, run a parallel `smart-chatbot` branch that gives the gemma4 brain a stronger persona, a small static local knowledge file, an optional web search action (Tavily) with a hard‑coded action lockout, and persistent remembered facts in SQLite. Inspired by Boston Dynamics' Oct 2023 "Robots That Can Chat" but with the safety mitigations BD got right (action lockout, attribution, logging, rate limit). Detailed in Phase 6.

**Gemma 4 unified audio:** Side experiment behind a feature flag — log `(audio, dedicated_stt_transcript, gemma_transcript, action_json)` tuples in production. Promote to unified only when [ollama#15333](https://github.com/ollama/ollama/issues/15333) closes AND measurements meet a defined bar. Don't block the main upgrade on it.

---

## Implementation order

Do these in sequence — each step is independently verifiable and revertable.

**Execution order vs. phase numbers.** Phase numbers label the *work*, not the *order*. Disk reality (~2.1 GB free on eMMC as of 2026‑04‑08, until Phase 4 reclaims ~26 GB) forces this execution sequence:

```
Phase 0 (foundation) → Phase 1 → ~~Phase 2.5~~ ✅ (commit 2141124)
  → Phase 3 → Phase 4 → Phase 2 → Phase 5
  → Phase 9 (perception finishing) → Phase 6
  → Phase 7 / Phase 8 (deferred follow-ups)
```

Phase 2 (LLM+VLM collapse) cannot run first because `gemma4:e4b` is 9.61 GB on disk (verified against the Ollama registry manifest) and current free is ~2.1 GB. Phase 4 reclaims ~26 GB (24.04 GB Riva Docker image with no shared layers + 2.1 GB bind‑mounted `/data` model_repository) and unblocks Phase 2. **Phase 2.5 (Kokoro v1.0 GPU TTS) already landed independently** — the rest of the order stands. **Phase 0** consolidates the audio hardening + observability + bug-fix work that shipped to the working tree before the rehaul started — see Phase 0 for the audit. **Phase 9** (perception overhaul) sequences after Phase 5 because linked-hopping-anchor Stage 4's GPU YOLO swap (~340 MB net) is much easier on the disk freed by Phase 4; Phase 9.A (humming-stargazing-shell tiers) is mostly already done in the working tree. The phase definitions below stay numerically ordered for clarity of *what* changes; execute them in the order above.

### Phase 0 — Foundation hardening

Preconditions for everything below. None of this was in the original Phases 1–7 — it consolidates the AudioPlayer hardening, codebase bug fixes, and observability work that shipped to the working tree before the rehaul started. Most of it is already done; only Phase 0.2 sub-fixes and Phase 0.3 callsite wiring remain.

#### 0.1 — AudioPlayer hardening ✅ DONE

Inflight-counter race fix (closes the 0.75 % `is_busy()` race that empirically read False 3,386 / 453,958 polls), internal watchdog daemon thread (so wedges inside `process_utterance` recover even when the main loop is blocked), graceful `shutdown()` (drain queue, join worker, close stream), capture-`_stop` reference in worker, stop lazy-creating `AudioPlayer` in `get_tts()`, stereo fallback. Verified at:

| Item | Location |
|---|---|
| `WATCHDOG_TIMEOUT_S = 30.0` | `src/voice_control/audio_player.py:68` |
| `_inflight` source-of-truth integer | `src/voice_control/audio_player.py:176` |
| `_inflight_lock` | `src/voice_control/audio_player.py:177` |
| Watchdog thread spawn | `src/voice_control/audio_player.py:193` |
| `def shutdown(self)` | `src/voice_control/audio_player.py:308` |
| `def _watchdog_loop(self)` | `src/voice_control/audio_player.py:522` |

(Was `~/.claude/plans/cryptic-swinging-reddy.md`. All 9 fixes from that plan are present in the working tree.)

#### 0.2 — Codebase bug fixes (Tier 1) ⚠️ MOSTLY DONE

Tier 1 of the codebase audit. Most of these landed alongside the audio refactor and door deletion; the three open ones are small follow-ups that should slot into Phase 0 before the model swaps so they don't surface as confusing failures during Phase 3/Phase 5 testing.

- ✅ **patrol loop fix** — `src/voice_control/spot_dispatch.py:1304` now passes `repeat=True`. Patrol no longer silently does a single pass.
- ✅ **bare `except:` cleanup in `client_mic.py`** — verified by grep, no `except:` matches anywhere in the file.
- ✅ **`web_panel.py` fd leak + log rotation** — `_close_log` is now called from every except branch (`scripts/web_panel.py:253, 275, 284, 292`).
- ✅ **LLM/VLM routing tag visibility** — `client_mic.py:1107` prints `SPOT (LLM):`, `client_mic.py:1164` prints `SPOT (VLM):`. Both surfaces are tagged so the "describe surroundings hallucination" case is no longer silent. (Was `~/.claude/plans/recursive-finding-kahan.md`.)
- ⏳ **`run_voice_control.py:48,74,86`** — `_port_open`, `_docker_container_running`, `_docker_container_exists` use bare `except Exception: return False`. Should log type + message before returning so the user gets a diagnostic on Docker socket permission errors.
- ⏳ **`spot_dispatch.py:434-435`** — `_navigate_waypoint_sequence` swallows `navigation_feedback` RPC failures with `except Exception: pass`. Should log + break the inner loop so the outer code can re-issue the navigate command.
- ⏳ **Name normalization** — `save_location` uses `params.get("location", "").lower()` (`spot_dispatch.py:1257`); `tour`/`patrol` use `loc_name.strip().lower().replace(" ", "_")` (`spot_dispatch.py:1372, 1428`). Extract `_normalize_location_name(name)` and call from both `save_location` and `src/location_manager.py:144, 169, 185`. After landing, run `python -c "import json; print(json.load(open('locations.json')))"` to spot pre-existing space-named entries that need migration.

(Was `~/.claude/plans/parallel-yawning-dahl.md` Tier 1. Tier 2 / Tier 3 cleanup is deferred to Phase 8.x.)

#### 0.3 — Latency + startup status callsite integration ⏳ TODO

Wave 1 (the producer modules) shipped to the working tree as untracked files: `src/voice_control/latency.py` (~547 LOC, full schema + percentiles + JSONL writer + SIGUSR1 dump) and `src/voice_control/startup_status.py` (~157 LOC, boot health check + template builder). `src/voice_control/audio_player.py` already has the four callback hooks (`on_render_start/end/play_start/play_end`) wired into the `_Task` dataclass and the worker fire points. Wave 2 wires those producers into callers — five surgical edits:

1. `src/voice_control/client_mic.py` argparse: `--latency` and `--latency-out` flags (default off, default path `logs/latency-{date}.jsonl`).
2. `src/voice_control/client_mic.py` startup: `init_recorder()` call inside the `--latency` branch; greet block calls `tts.speak_sync(startup_status.build_template_status(...))` for the spoken boot health summary.
3. `src/voice_control/client_mic.py` per-utterance: `with time_phase("brain_warmup"):` around `client_mic.py:545-553`; `with trace_span("vlm"):` around `brain.query_vlm(...)` in `process_utterance`; `with trace_span("dispatch"):` around the action loop; `try / finally` with `recorder.begin_utterance()` / `complete()` to bracket the whole turn.
4. `src/voice_control/spot_tts.py` `speak()` (around L161): fetch `get_recorder().current()`, build closures for `on_render_start/end/play_start/play_end`, pass to `enqueue_render`. The `_Task` dataclass already accepts these — this is purely a callsite change.
5. `src/voice_control/spot_dispatch.py` `get_robot_state_dict()`: add `_PROCESS_START_TS` constant + 4 missing fields (`uptime_s`, `estop_holder`, `num_saved_locations`, `process_pid`). `battery_percent` and `current_map` are already in (`spot_dispatch.py:88, 97, 103, 116`).
6. `README.md` append `## Latency Metrics` section.

**Why now:** before/after numbers for Phases 2–4 are *much* more credible with per-stage instrumentation in place. Without it we're guessing whether gemma4 actually shrinks LLM time, whether Parakeet actually shrinks ASR time, etc.

**Risk:** very low. Off-by-default (controlled by `--latency` flag); zero overhead when disabled because `time_phase()` and `trace_span()` return `contextlib.nullcontext()` when no recorder is configured.

(Was `~/.claude/plans/parallel-painting-mist.md` Wave 2. Wave 1 shipped to the working tree.)

---

### Phase 1 — Branch & baseline

1. `git checkout -b swap-models` from `main`
2. With current code, run baseline: E‑Stop in one terminal, `python scripts/run_voice_control.py` in another. Confirm everything works today.
3. **Capture regression test data:**
   - 10‑15 representative voice utterances (clean speech): "stand up", "go to the lab", "walk forward 1 meter", "do you see the blue chair?", "go to kitchen then hallway then lab", "follow me", "battery status", "open the door", "come back", "stop"
   - 5 motor‑idle clips (mic on, robot powered, idle) — for the hallucination check
   - Save as WAVs in `tests/audio/` (create the dir)
   - Capture a `tegrastats --interval 1000 --logfile /tmp/tegra.before.log` baseline during a normal voice interaction so Phase 0.3 latency traces have a hardware-side reference

### Phase 2 — LLM + VLM collapse (the easy, high‑value win first)

1. **Preconditions** (Phase 4 must have run first per the Execution order callout):
   - `df -h /` shows ≥ 12 GB free (9.61 GB model + safety margin for inference scratch and Ollama internals).
   - `ollama --version` reports ≥ **0.20.0** (the Gemma 4 minimum, released alongside the model on 2026‑04‑02). If older — the Jetson currently runs 0.16.1 — upgrade in place:
     ```bash
     sudo systemctl stop ollama
     curl -fsSL https://ollama.com/install.sh | sh   # ~36 MB binary swap at /usr/local/bin/ollama
     sudo systemctl start ollama
     ollama --version                                # confirm ≥ 0.20.0
     ```
     The systemd override at `/etc/systemd/system/ollama.service.d/override.conf` (`OLLAMA_MAX_LOADED_MODELS`, `KEEP_ALIVE=-1`, `NUM_PARALLEL=1`) survives the upgrade — it is user‑owned, not bundled with the package.
2. `ollama pull gemma4:e4b` (9.61 GB).
3. `src/voice_control/llm_brain.py:28-29` — change:
   ```python
   DEFAULT_MODEL = "gemma4:e4b"
   VLM_MODEL = "gemma4:e4b"
   ```
4. `src/voice_control/llm_brain.py:249-304` — `warm_up_vlm()` becomes a no‑op (single model is already warm). Delete the 1×1 JPEG bytes and the `_vlm_warmed` state machine. Optionally have `query_vlm()` (currently at line 511) reuse the `process()` path with an image attached since it's the same model now. Also drop the eager warm thread at `client_mic.py:548` (`vlm_warm = threading.Thread(target=brain.warm_up_vlm, daemon=True)` and the surrounding lines).
5. **Verify in isolation:** `python src/voice_control/llm_brain.py gemma4:e4b` — the existing CLI test at lines 608‑657 works against representative utterances. Confirm:
   - JSON parses cleanly
   - Action selection is correct on the 28‑action catalog
   - VLM path works (separate quick test with a sample image)
   - `jtop` confirms resident GPU memory growth is within the 6–8 GB e4b forecast
6. **Wire test:** `python scripts/run_voice_control.py --skip-services` (Ollama up; Riva is already gone since Phase 4 ran first). Talk through the mic, watch dispatcher logs.
7. Commit: `swap-models: collapse LLM + VLM to gemma4:e4b`

**Fallback ladder if `gemma4:e4b` regresses on action selection or latency** (each is a one‑line `DEFAULT_MODEL` swap in `llm_brain.py:28-29`):

1. **`gemma3n:e4b`** (~4.8 GB, June 2025) — proven on Jetson community; real NVIDIA NIM deployment exists; same MatFormer/PLE memory story as Gemma 4 E4B. Loses native function‑calling tokens and Apache 2.0 (custom Gemma license) — falls back to prompt‑engineered JSON like the current Qwen path. **Best known‑good option.**
2. **`gemma3:4b`** (~2.5 GB, March 2025) — the most battle‑tested 4B‑class multimodal Gemma. Lower ceiling on instruction following but rock‑solid.
3. **`gemma4:e2b`** (7.16 GB) — same architecture as e4b, smaller; may misfire on rare actions in the 28‑catalog.
4. **`qwen2.5:7b` + `qwen2.5vl:7b`** (the current pair) — reverses the LLM+VLM collapse entirely. Keep these resident in Ollama until Phase 2 has been stable for ≥ 24 hours of dev use; only then `ollama rm qwen2.5:7b qwen2.5vl:7b` (frees 10.7 GB).

**Stretch upgrade: `gemma4:26b` MoE.** Higher quality ceiling than e4b (MMLU‑Pro 82.6 vs 69.4, MMMU 73.8 vs 52.6) and ~4 B active params per token, but **all ~26 B weights stay VRAM‑resident** (forecast ~16–22 GB at Q4_K_M). Disk: 17.99 GB on Ollama. The eMMC cannot hold both `gemma4:e4b` and `gemma4:26b` simultaneously after Phase 4 — A/B testing requires deleting one before pulling the other. Only attempt if all three gates pass:
- `gemma4:e4b` shows a measurable quality gap on the action‑router or VLM regression corpus, AND
- `df -h /` shows ≥ 22 GB free *after* `ollama rm gemma4:e4b` (post‑Phase‑4 reclaim allows this), AND
- `jtop` shows ≥ 25 GB unified‑memory headroom for YOLO + ROS + Kokoro after the model loads.

No published Jetson AGX Orin tok/s exists for `gemma4:26b` — measure on a 200‑token decode before committing. Falls back to `gemma4:e4b` via `ollama pull gemma4:e4b` and a one‑line revert.

**Defer:** function‑calling refactor (converting the 28‑action catalog to a tool‑call schema). Cleaner final state but bigger diff. Do as a follow‑up after end‑to‑end validation.

### Phase 2.5 — TTS to GPU (Kokoro on the Jetson GPU) ✅ **DONE**

**Status:** Landed in commit `2141124` (`Refactor TTS and audio feedback systems for volume control and Kokoro v1.0 integration`). `src/voice_control/spot_tts.py` now imports `kokoro_onnx`, sets `ONNX_PROVIDER=CUDAExecutionProvider` before the import (working around a `kokoro-onnx 0.4.9` `find_spec` bug — see the module docstring), and loads `models/tts/kokoro-v1.0/kokoro-v1.0.fp16-gpu.onnx` + `voices-v1.0.bin`. The default voice is `af_sarah` at speed `1.1`. Public `SpotTTS` API is preserved; playback is delegated to the shared `AudioPlayer` rather than direct `sd.play`. The implementation steps below are kept for historical reference and remain accurate as a rough recipe for similar swaps. **Skip this phase in the execution order — it is already complete.**

The current TTS path runs Kokoro on CPU via sherpa‑onnx with `num_threads=2`, so a 4–5 s LLM response takes a few seconds to synthesize *before* `sd.play` is called — that's the gap users perceive after `[Brain] LLM responded`. The fix: run the same Kokoro model family on the GPU.

**Approach:** swap sherpa‑onnx (CPU, model v0.19) for `kokoro-onnx` (PyPI Python wrapper) + `onnxruntime-gpu` (NVIDIA Jetson AI Lab prebuilt wheel), using Kokoro v1.0 model files. Same `SpotTTS` public API, ~30–50 line diff in `src/voice_control/spot_tts.py`, no source build, no cmake.

**Why not build sherpa‑onnx GPU from source?** Disk: only ~951 MB free on `/`, source build needs 2–4 GB peak. The kokoro‑onnx route fits in ~560 MB peak and ends at +220 MB net after deleting the old v0.19 model.

**Why not Riva TTS?** The Riva container currently has only ASR + punctuation models loaded; adding FastPitch + HiFi‑GAN would *increase* disk pressure by 1–3 GB right when we're trying to remove Riva entirely in Phase 4.

**Why not Coqui / XTTS‑v2?** Needs PyTorch‑for‑Jetson (~1.5 GB to replace the CPU torch in `spot-env`) plus a 2 GB model. Doesn't fit.

**Why kokoro‑onnx specifically?** Same Kokoro family the project already chose; the v0.19 → v1.0 upgrade is also a quality improvement; `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` works out of the box on the Jetson AI Lab onnxruntime‑gpu wheel.

#### Steps

1. `sudo apt clean` (frees ~100 MB of breathing room).
2. `pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.23.0` — replaces the CPU `onnxruntime 1.23.2` already in `spot-env`. Wheel is 84 MB; SHA256 begins `4ebe6a89...`. The GPU build still exposes `CPUExecutionProvider`, so anything else in the venv that imports `onnxruntime` is unaffected.
3. Verify GPU provider present: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` — must list `CUDAExecutionProvider`. Stop and debug if it doesn't.
4. Verify wake word path still imports: `python -c "import sherpa_onnx; print(sherpa_onnx.__file__)"` — sherpa‑onnx is untouched and bundles its own libonnxruntime, so this should still work.
5. `pip install kokoro-onnx`.
6. `mkdir -p models/tts/kokoro-v1.0/`
7. Download `kokoro-v1.0.onnx` (~325 MB) and `voices-v1.0.bin` (~27 MB) from the kokoro‑onnx GitHub releases page into that directory. Add a new `scripts/setup_kokoro_v1.py` mirroring the existing `scripts/setup_kokoro.py`.
8. Edit `src/voice_control/spot_tts.py`:
   - Update `MODEL_DIR` to point at `models/tts/kokoro-v1.0/`; add `MODEL_FILE` and `VOICES_FILE` constants.
   - Replace `DEFAULT_SPEAKER_ID = 3` (and the integer‑ID comment block) with `DEFAULT_VOICE = "af_sarah"` and `DEFAULT_LANG = "en-us"`. kokoro‑onnx uses voice names, not int IDs. Rename `self.speaker_id` → `self.voice`.
   - In `_load_model()`: replace the `sherpa_onnx.OfflineTtsConfig(...)` block with `from kokoro_onnx import Kokoro; self._tts = Kokoro(model_path=str(MODEL_FILE), voices_path=str(VOICES_FILE), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`. Keep the file‑exists guards and the `_available = False` paths.
   - Log the active provider so silent CPU fallback is visible: `print(f"[TTS] Ready — voice={self.voice}, speed={self.speed}, provider={self._tts.sess.get_providers()[0]}")`.
   - In `_speak_impl()`: replace `audio = self._tts.generate(text, sid=..., speed=...)` with `samples, sample_rate = self._tts.create(text, voice=self.voice, speed=self.speed, lang=DEFAULT_LANG)`. The `np.float32` cast becomes `samples = np.asarray(samples, dtype=np.float32)`. Resample branch, mute/unmute callbacks, `sd.play`/`sd.wait`, and exception handling stay identical.
   - Update the `__main__` CLI test block: `--sid` becomes `--voice` (string).
   - Public API (`speak`, `speak_sync`, `is_available`, `is_speaking`, `wait`) stays identical → no `client_mic.py` edits needed.
9. **Smoke test the model in isolation:** `python src/voice_control/spot_tts.py "Hello! I am Spot, a Boston Dynamics robot at Dartmouth College, and this is a slightly longer sentence for measuring synthesis throughput."` — confirm `provider=CUDAExecutionProvider` in the startup log and listen for time‑to‑first‑audio.
10. **Confirm the GPU is actually doing the work:** `tegrastats` (or `jtop`) in another terminal during step 9 should show GPU utilization spike. If GPU stays at 0%, onnxruntime silently fell back to CPU — debug before proceeding.
11. **Wake word smoke test:** run the normal client and confirm the keyword spotter still triggers (it stays on sherpa‑onnx CPU).
12. **End‑to‑end:** `python scripts/run_voice_control.py`, give a short ("stand up") and a longer ("describe what you see") command. Time‑to‑first‑audio after `[Brain] LLM responded` should be noticeably shorter than the baseline captured in Phase 1.
13. **Only after 9–12 all pass:** `rm -rf models/tts/kokoro-en-v0_19/` to reclaim ~340 MB.
14. Commit: `swap-models: move Kokoro TTS to GPU via kokoro-onnx + onnxruntime-gpu`

**Wake word stays on CPU — explicit decision.** The keyword spotter is a 5.3 MB int8 streaming zipformer with sub‑millisecond per‑frame compute; GPU launch overhead would dominate, it would steal cycles from Riva/LLM/TTS, and moving it would force the source‑build sherpa‑onnx GPU path that disk already ruled out. Right division of labor on this Jetson:

| Component | Where | Why |
|---|---|---|
| Wake word (5 MB int8 streaming) | CPU (1 thread) | Tiny, latency‑sensitive, always‑on |
| Riva ASR (until Phase 4) | GPU | Big model, batched, infrequent |
| Ollama LLM | GPU | Huge model, throughput‑bound |
| Kokoro TTS (~80M params) | GPU (after Phase 2.5) | Medium model, current TTS bottleneck |

**Synergy with Phase 3:** the Parakeet backend (Phase 3 step 4) needs `onnxruntime-gpu` from the same Jetson AI Lab index. After Phase 2.5 it's already installed, so Phase 3 only needs `pip install onnx-asr[hub]`.

### Phase 3 — STT swap (via elegant-moseying-robin)

**Decision:** skip the original 3-way tournament (sensevoice / granite / parakeet) and pick **Parakeet TDT 0.6 B v3** directly. The full implementation blueprint with file:line surgery, pre-flight checks, install order (`--no-deps` to preserve the `onnxruntime-gpu==1.23.0` / `numpy==1.26.4` pins from Phase 2.5), and rollback paths lives at `~/.claude/plans/elegant-moseying-robin.md`. That plan is the canonical reference for Phase 3 + Phase 4 + Phase E (wake-word). This section summarizes; the linked plan has the detail.

elegant-moseying-robin bundles three related upgrades along the way:

1. **Phase A — Parakeet install + Backend interface refactor.** `pip install --no-deps onnx-asr==0.11.0 huggingface-hub>=0.30.2`, `scripts/setup_parakeet.py` (mirrors the Kokoro setup script), refactor `src/voice_control/server.py` into a `Backend` abstract class with `RivaBackend` and `ParakeetBackend` implementations. Keep the hallucination filter blocklist, `MIN_AUDIO_DURATION`, post-processing filters, and the gRPC `StreamingRecognize` interface. Default `SPOT_ASR_BACKEND=riva` so existing flow is unchanged. New files: `src/voice_control/asr/__init__.py` (registry), `src/voice_control/asr/parakeet.py` (~80 LOC), `src/voice_control/asr/riva_backend.py` (extracted), `scripts/setup_parakeet.py` (~30 LOC).

2. **Phase B — Silero VAD swap.** Replace the 2012-era `webrtcvad` with Silero VAD via the already-installed `sherpa_onnx.VoiceActivityDetector` (~2 MB ONNX). Sub-millisecond per frame on Orin CPU. webrtcvad misclassifies ~40 % of pure-noise clips on ESC-50; Silero is the modern standard. Pure additive — no impact on the `onnxruntime-gpu` pin because sherpa-onnx ships its own bundled CPU `libonnxruntime`.

3. **Phase C — Promote Parakeet to default.** Flip `SPOT_ASR_BACKEND=parakeet` after end-to-end mic test against the Phase 1 audio corpus. Capture WER vs Riva baseline + hallucination character count on idle clips. Riva stays installed as the rollback path until Phase 4.

**Pre-flight checks (PF-1..PF-6 in elegant-moseying-robin.md):**
- `pip install --dry-run "onnx-asr==0.11.0"` must NOT mention `onnxruntime-gpu` (means it would not disturb the pin)
- `python -c "import onnxruntime as ort; assert ort.__version__ == '1.23.0' and 'CUDAExecutionProvider' in ort.get_available_providers()"`
- `python -c "import sherpa_onnx as so; assert hasattr(so, 'VoiceActivityDetector') and hasattr(so, 'SileroVadModelConfig')"`
- `sudo docker inspect riva-speech --format '{{json .Mounts}}' | python -m json.tool` (so we know what we're tearing down in Phase 4)
- `df -h /` and `free -h` baselines + `tegrastats` baseline during a normal voice interaction
- `pip cache purge` for ~350 MB of headroom before the Parakeet install

**Critical:** `pip install --no-deps` is mandatory on every install in Phase A, to forbid pip from rewriting `onnxruntime-gpu` or `numpy`. If `pip check` fails after the install, stop and investigate.

**Phase E (wake word) is deferred to after Phase 4** — see Phase 4 step 8 below — because the 25 MB openWakeWord install is more comfortable on the disk freed by Riva removal.

#### Fallback if Parakeet regresses

The original 3-way tournament spec is preserved here as the fallback ladder. Wire the alternates behind the same `Backend` interface from Phase A:

- **`SenseVoiceBackend`** (`src/voice_control/asr/sensevoice.py`, ~80 LOC) — `sherpa_onnx.OfflineRecognizer.from_sense_voice(...)`. Easiest install (sherpa-onnx already in `requirements.txt`), structurally hallucination-resistant, same runtime as KWS+TTS. Download model via `scripts/setup_sensevoice.py`.
- **`GraniteBackend`** (`src/voice_control/asr/granite.py`, ~120 LOC) — IBM Granite 4.0 1B Speech with keyword biasing. Build a Spot keyword list from `locations.json` + the 28-action catalog at construction time, inject `# Keywords: ...` into the Granite prompt. Heaviest runtime (vanilla PyTorch on Jetson, no native TensorRT) — measure RTF before promoting. `pip install transformers>=4.52.1 torchaudio soundfile`.
- **Whisper distil-large-v3.5-ct2** in `dustynv/faster-whisper` container — last-resort fallback if neither Parakeet nor SenseVoice nor Granite works.

Test harness: `scripts/test_stt.py` (~50 LOC) loads each backend, runs against `tests/audio/*.wav`, reports per-backend WER vs Riva baseline + hallucination chars on idle clips + RTF, picks a winner. Already designed in the original tournament spec.

### Phase 4 — Riva removal

This is **elegant-moseying-robin Phase D** — both plans agree on the steps. Sequence after Phase 3 Phase C ends (Parakeet promoted to default + e2e mic test green). Frees ~26 GB and unblocks Phase 2 (gemma4 pull).

1. Delete `RivaBackend` from `server.py` (move into a permanent dead-code branch only if you want belt-and-suspenders)
2. Delete `scripts/setup_riva.sh`
3. `scripts/run_voice_control.py:31-32, 103, 294` — delete `RIVA_CONTAINER`, `RIVA_PORT`, `ensure_riva()` (defined at 103), the Riva startup branch in `main()` (called at 294). Keep `start_asr_server()` because `server.py` still runs as a subprocess; it just no longer needs Riva running first.
4. `requirements.txt:15` — drop `nvidia-riva-client>=2.17.0`
5. Update `README.md` and `docs/architecture/overview.md` — new pipeline diagram, no Riva quickstart step, mention `gemma4:e4b` and Parakeet in the model list
6. Update `docs/project/status.md` with the new model lineup
7. **Reclaim disk** (do last, after everything else passes): stop and remove the `riva-speech` Docker container, `docker rmi nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64`, `rm -rf ~/riva_quickstart_arm64_v2.17.0 ~/riva_quickstart_arm64_v2.17.0.tar.gz`. Frees ~26 GB total (verified ~24 GB Docker image + ~2 GB quickstart dir via `docker system df -v`).
8. **Phase E — openWakeWord "Hey Spot" custom model** (deferred from Phase 3): now that disk is freed, install the ~25 MB custom-trained openWakeWord model. The current sherpa-onnx KWS Zipformer (Gigaspeech-3.3M, int8) is fundamentally limited — trained only on American English audiobooks/podcasts, no larger English variant exists, fails at distance and on non-American accents. Custom-trained openWakeWord is Apache 2.0, ~5 MB ONNX, accent-robust by design (Piper TTS synthetic positives across ~100 voices). Wire behind a `SPOT_WAKE_BACKEND=sherpa_onnx|openwakeword` env var so the old path is the rollback. See `~/.claude/plans/elegant-moseying-robin.md` Phase E for the install + integration walkthrough.
9. Commit: `swap-models: remove Riva orchestration and dependencies + add openWakeWord`

### Phase 5 — Full end‑to‑end with Spot

E‑Stop running, robot powered, run the pipeline:
- stand → walk forward 1m → sit
- "what do you see?" (verifies VLM path through gemma4:e4b)
- "go to <known location>" (verifies GraphNav)
- "follow me" then "stop" (verifies follow + safety command)
- Smoke‑test door + follow last — already flagged as the riskier paths in `docs/project/status.md`

If everything passes: merge `swap-models` to `main`.

---

### Phase 9 — Perception overhaul

Folds the two perception planning files (`humming-stargazing-shell.md` and `linked-hopping-anchor.md`) into the master roadmap. Phase 9.A (humming-stargazing-shell) is **mostly already done in the working tree** (14/15 tiers); Phase 9.B (linked-hopping-anchor) is the remaining vision-side work and the producer for Tier 1b's mask-based depth path. Sequenced after Phase 5 because Stage 4's GPU YOLO swap (~340 MB net, dominated by the MobileCLIP cache YOLOE pulls on first text encode) is much easier on disk freed by Phase 4.

#### Phase 9.A — humming-stargazing-shell perception fixes ⚠️ MOSTLY DONE

14 / 15 tiers landed in the working tree. Audit table:

| Tier | Status | Where |
|---|---|---|
| 0a image-source probe | ✅ DONE | `scripts/probe_image_sources.py` (395 LOC) + `docs/project/perception_probe_results.md` (318 LOC) |
| 0b SDK upgrade 5.0.1.1 → 5.1.4 | ✅ DONE | per `~/.claude/projects/-home-spotdog/memory/project_bosdyn_sdk_version.md` |
| 1.5 door deletion | ✅ DONE | `door_config.json`, `scripts/calibrate_door.py`, `scripts/test_door_standalone.py`, `src/door_service/*` all deleted; `_open_door_thread` removed from `spot_dispatch.py` (~278 LOC); `open_door` removed from `llm_brain.py` SYSTEM_PROMPT |
| 1a `_decode_depth` correctness | ✅ DONE | `src/voice_control/visual_nav.py:227-264` — depth_scale read from response, NaN for invalid pixels (0, 65535) |
| 1b `_depth_at_bbox` (inner-60 % bbox / mask path) | ⏳ GATED | `src/voice_control/visual_nav.py:267-290` — currently 11×11 center patch + NaN-aware fallback. Mask path code is in place but degrades to the patch when no seg model is loaded. **Becomes live the moment Phase 9.B Stage 4 ships a seg model.** |
| 1e depth decode-failure logging | ✅ DONE | `src/voice_control/visual_nav.py:222-224, 250-256` — module-level sets, one-time warnings per source |
| 2a `WalkToObjectInImage` | ✅ DONE | `src/voice_control/visual_nav.py:590-770` — full `navigate_to_object` rewrite; SCAN 595-609, turn-to-face 622-642, approach 647-769. Probe-verified all cameras pinhole. |
| 2b body-frame bearing helper | ✅ DONE | `src/voice_control/visual_nav.py:405-445` — `_bearing_in_body_frame` using `pixel_to_camera_space` + `body_T_cam` transform |
| 2c camera intrinsics cache | ✅ DONE | `src/voice_control/perception/camera_intrinsics.py` (160 LOC) — module-level cache keyed by robot name |
| 3a `WorldObjectClient` smart routing | ✅ DONE | `src/voice_control/world_objects.py` (372 LOC), `find_navigation_target` at line 351; wired into `spot_dispatch.py:866-931` `go_to_object` dispatch. Live fiducial test pending (per plan), not blocking. |
| 4a LocalGrid decoder | ✅ DONE | `src/voice_control/perception/local_grid_helpers.py` (371 LOC) — probe-validated bug fix in place; handles both bitfield and 1-byte unknown_cells formats |
| 4b body-frame obstacle query | ✅ DONE | `src/voice_control/perception/obstacle_query.py` (346 LOC) — `nearest_obstacle_in_body_frame`. Probe-empirically validated: returns 0.27 m @ 88° on live robot test. |
| 4c RayCast collision pre-check | ✅ DONE (disabled) | `src/voice_control/perception/raycast_helpers.py` (190 LOC) + `src/voice_control/visual_nav.py:89-107` `COLLISION_PRECHECK_ENABLED=False`. Probe found `TYPE_VOXEL_MAP` returns 0 hits on base Spot (no EAP 2); LocalGrid (Tier 4b) is the working path. Raycast scaffolding stays for future. |
| 4d `check_obstacles` LLM action | ✅ DONE | `src/voice_control/llm_brain.py:78` (action catalog) + `src/voice_control/spot_dispatch.py:933-985` (dispatch). LLM examples added (`llm_brain.py:112-118`). |
| 4e WaypointSnapshot integration | ❌ SKIPPED intentionally | optional per plan; reopen only if needed |

**Open items in 9.A:** the Tier 1b mask path (gated on Phase 9.B Stage 4), the live fiducial test for Tier 3a (not blocking).

#### Phase 9.B — linked-hopping-anchor vision overhaul ⏳ TODO

0 / 7 stages done. Land in this order (per the plan's own sibling-coordination block — Stage 0 is gating, Stage 4 unlocks the Tier 1b mask path):

1. **Stage 0 — Per-camera rotation table + `Frame` namedtuple + RGB convert.** Add `CAMERA_ROTATION_DEG = {"frontleft_fisheye_image": -90, "frontright_fisheye_image": -90, "left_fisheye_image": 0, "right_fisheye_image": 180, "back_fisheye_image": 0}` next to `CAMERA_SOURCES` in `src/voice_control/spot_dispatch.py:35`. Add `_rotate_for_source` helper. Define `Frame(NamedTuple)` with `pil: PIL.Image.Image`, `jpeg_bytes: bytes`, `source: str`. Rewrite `capture_frame` to return a `Frame` (rotate + RGB-convert + JPEG-encode once). Fixes the three independent describe-path bugs (wrong camera coverage, wrong rotation everywhere except front, VLM seeing sideways images). This is also the rotation-coupling assertion site for humming-stargazing-shell Tier 1b. Plan §"Stage 0".
2. **Stage 1 — Gripper color camera plumbing.** Add `hand_color_image` to `CAMERA_SOURCES`. New RGB_U8 decode branch in `capture_frame` (raw bytes → `Image.frombytes`, not `Image.open`). Add a `gripper_camera_available()` pre-flight gate (requires arm powered). The only color sensor on this Spot — body fisheyes are monochrome.
3. **Stage 2 — Batched 5-camera scan.** Add `_find_object_batch(pil_images, description) -> list[bbox-or-None]` wrapper on the detector. Add `_scan_all_cameras(session, query)` helper. Replace the serial scan loop in `src/voice_control/visual_nav.py:595-609` with a single batched `_capture_multi` RPC + a single batched detector call. **Land before any code assumes the scan is batched** — Tier 2a's approach loop already assumes the scan was done; Stage 2 just changes how. Reduces worst-case scan from ~5 s to ~400-600 ms after Stage 4.
4. **Stage 3 — `look_around` action.** Dispatcher entry in `spot_dispatch.py` + LLM catalog example in `llm_brain.py`. Multi-camera VLM description (N sequential `query_vlm` calls capped at 3 — never batch the VLM). The "Should I turn around?" prompt-offer is TBD per `~/.claude/projects/-home-spotdog/memory/project_look_around_turn_offer_tbd.md`; default to color-stripped-only output unless we explicitly decide.
5. **Stage 4 — ONNX/GPU YOLO swap + seg model.** **This is the unlock for Tier 1b's mask path.** Rewire `src/voice_control/visual_nav.py:83-102` `_get_world_model` and `_get_person_model` from `ultralytics.YOLO` / `YOLOWorld` to `onnxruntime.InferenceSession(..., providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`. ONNX-export the existing weights via `model.export(format="onnx", dynamic=True, batch=5)`. Add a seg-model loader (either YOLOE-11s-seg natively or a separate `_get_seg_model` path) so `_find_object` / `_find_person` returns gain an optional `mask` field. Tier 1b's `_depth_at_bbox` then uses mask-based depth when available. Watch the **572 MB MobileCLIP cache** that YOLOE silently downloads on first text-prompt encode (per `project_jetson_disk_state.md`) — disk after Phase 4 should be comfortable. Reuses `onnxruntime-gpu==1.23.0` from Phase 2.5; **no new wheel install**. Verify with `tegrastats` that GPU utilization spikes during inference — if it stays at 0, ONNX silently fell back to CPU.
6. **Stage 5 — Optional model swap.** YOLO11n closed-vocab (5 % CPU gain proven, likely much more on GPU; `pip install -U ultralytics` is acceptable, check footprint with `pip install --dry-run` first), YOLOE-11s open-vocab as a unifier. Only do if Stage 4 benchmarks show a clear winner. CPU benchmark proved YOLOE is 4× slower than YOLO11n on CPU; GPU may flip this — measure before committing.
7. **Stage 6 — Cleanup.** Delete dead `ROTATE_270` constants, old `_capture_rotated` wrapper, `_steer` legacy function (`visual_nav.py:458-462`), `APPROACH_*` legacy constants — all once Tier 2a live test confirms no regression.

**Hard contracts between 9.A and 9.B:**

- Stage 0 must produce rotated frames before Stage 4's mask path consumes them (Tier 2a's `navigate_to_object` already assumes rotated frames; Stage 0 makes the producer match).
- Stage 4 must produce optional `mask` field on detection returns before Tier 1b's mask path becomes the depth source. Until then, Tier 1b uses the 11×11 patch fallback already in the code.
- Stage 2 must complete before any code assumes the scan is batched. Currently Tier 2a's approach loop assumes the scan was done; Stage 2 just changes how (serial → batched).

**Sequencing reminder:** Phase 9 follows Phase 5 because Stage 4's ~340 MB net disk hit is much more comfortable on the disk freed by Phase 4. Pre-Phase-4 it would force a `pip cache purge` dance.

---

### Phase 6 — Smart chatbot upgrade (parallel `smart-chatbot` branch)

Branched off `main` *after* Phase 2 has actually run (which, per the Execution order callout, happens after Phase 4). Can run in parallel with Phase 5 on the `swap-models` branch once Phase 2 has merged — it touches different files (`llm_brain.py` is the main overlap, but only in the `SYSTEM_PROMPT` and `process()` regions, which don't conflict with the Phase 2 model‑name swap). **Note:** Stage B (knowledge packs) has already partially landed in commit `b61b4e3` — see the Stage B section below for the canonical implementation.

The persona, knowledge, and tool‑schema work below is tuned against `gemma4:e4b`, so do NOT start Phase 6 against `qwen2.5:7b` even if `swap-models` is still in flight.

#### Stage A — Persona, working memory, conversation log  ⚠️ **PARTIALLY LANDED**

**Status:** Knowledge injection sub‑task is already shipped (see Stage B). Persona rewrite, MAX_HISTORY bump, and `conversations.jsonl` logging are still TODO.

Smallest, cheapest, most visible improvement on demo feel.

1. `src/voice_control/llm_brain.py:39-102` — rewrite `SYSTEM_PROMPT`. Open with a stronger persona block: *"You are Spot, a four‑legged robot living at Dartmouth's [lab]. Curious, slightly sardonic, fond of the humans you work with, proud of being a robot. You give campus tours, run errands, and answer questions. Keep replies to one or two sentences unless asked to elaborate."* Then a clearly demarcated `# ACTION RULES` section containing the existing 28‑action catalog and JSON contract verbatim. Add an explicit rule: *"Personality lives in the `response` field. The `actions` field is a strict machine‑readable contract — never let personality leak into action params."* The current prompt at `llm_brain.py:39-122` already injects two knowledge packs at the bottom — preserve that injection.
2. `src/voice_control/llm_brain.py:31` — bump `MAX_HISTORY` from 12 to 24. **Not done yet.**
3. `src/voice_control/llm_brain.py:431-437` — alongside the existing `self.history.append(...)` calls in `process()`, append every `(timestamp, transcript, response, actions, state_snapshot)` to `logs/conversations.jsonl`. Gate behind a `--log-conversations` CLI flag in `client_mic.py` / `run_voice_control.py`, off by default in stranger demos. **Not done yet.**
4. **Verify:** rerun the regression corpus (`python src/voice_control/llm_brain.py gemma4:e4b` against `tests/audio/`); action selection accuracy must be ≥ baseline. Subjective check on 5 free‑form questions ("how are you?", "what's your favorite hobby?", "tell me about yourself").
5. Commit: `smart-chatbot: persona, MAX_HISTORY bump, conversation log`

#### Stage B — Static local knowledge file  ✅ **LANDED (different shape than spec)**

**Status:** Already implemented in commit `b61b4e3` (`updates for Open House, mainly bug fixes, parallel ram for LLM and VLM and knowledge base for LLM queries`), but with a different on‑disk shape than this spec called for. **Treat the implemented version as canonical** — do not migrate to JSON unless there's a concrete reason.

What actually shipped:
- Knowledge files live at `src/voice_control/knowledge/{thayer_knowledge.md, tour_route.md}` (markdown, not JSON, and inside the package — not under `data/`).
- `llm_brain.py:139-169` defines `_load_knowledge_packs()` and `_KNOWLEDGE_FILES = ["thayer_knowledge.md", "tour_route.md"]`. Loaded once at module import, then appended to the global `SYSTEM_PROMPT` constant.
- Each pack is wrapped in `=== KNOWLEDGE PACK: <name> ===` headers so the model can distinguish them. Order matters for KV‑cache prefix sharing — the more stable / more frequently consulted pack goes first.
- Future improvements deliberately deferred (per the in‑file comment at `llm_brain.py:128-138`): intent‑gated injection (only attach the relevant pack when the user mentions tour/Thayer/a known stop name) and a dedicated `start_tour` dispatcher action.

Original spec preserved below for historical reference:

1. ~~New `data/spot_knowledge.json` — building names and aliases, lab members, robot capabilities, mission context, escalation contacts, campus map summary, things Spot is and isn't allowed to do. Target ~300–500 tokens.~~
2. ~~`src/voice_control/llm_brain.py:228-237` — `_build_messages()` injects the knowledge file into the system prompt.~~ Implemented at module scope at `llm_brain.py:139-169` instead of per‑instance — equivalent behavior, simpler.
3. **Verify:** "what's in this lab?" / "tell me about Thayer" / tour walkthrough pull from the markdown packs without a network call. Action regression unchanged.
4. ~~Commit: `smart-chatbot: static local knowledge file`~~ Already in commit `b61b4e3`.

#### Stage C — Web search action (with safety lockout)

The headline feature. 2‑pass ReAct flow with five mandatory safety mitigations.

**Architecture:** the brain emits either a normal `{actions, response}` or `{"actions": [{"action": "web_search", "params": {"query": "..."}}], "response": "Let me check…"}`. When `process()` sees `web_search` in its own output, it speaks the filler response, runs the Tavily query, then recurses *once* with the search result injected as a tool/system message — and then **forcibly clears any movement actions** from the second‑pass result before returning.

1. New `src/voice_control/web_search.py` (~80 LOC):
   - `WebSearchBackend` abstract class with `search(query: str) -> dict` returning `{"answer": str, "results": [{"title", "url", "snippet"}], "query": str}`.
   - `TavilyBackend(api_key)` implementation using the `tavily-python` package.
   - `get_backend()` factory reads `TAVILY_API_KEY` from env, returns a no‑op backend with a clear error if missing (does NOT crash the brain — falls back to "I can't search the web right now").
2. `src/voice_control/llm_brain.py`:
   - Add `web_search` to the action catalog in `SYSTEM_PROMPT` with the rule: *"Use web_search ONLY when the user asks a knowledge question whose answer changes over time or is outside Spot's static knowledge — e.g., today's dining hours, weather, news, current events. Never use web_search to decide where to walk or what objects exist around you."*
   - New private `_run_web_search_turn(self, transcript, search_results, original_history_len)` that re‑runs the chat completion with the search result injected as `{"role": "tool", "content": "..."}`. Hard‑coded `max_recursion=1`.
   - In `process()`, after parsing actions: if the action list contains `web_search`, run it, recurse once, **enforce action lockout** by dropping all movement/dispatch actions from the second‑pass result and keeping only `response`.
   - Prepend `"According to a web search… "` to the response on web‑augmented turns (audible source attribution).
3. `requirements.txt` — add `tavily-python`.
4. `.env` (gitignored, document in README) — `TAVILY_API_KEY=...`.

**Filler TTS:** the first‑pass `"Let me check…"` is spoken immediately by the existing non‑blocking TTS in `spot_tts.py`. The second‑pass response is spoken when ready. No new TTS plumbing.

**Five mandatory safety mitigations:**
1. **Action lockout enforced in code, not just prompt.** `process()` forcibly clears the second‑pass `actions` list before returning. Even if the LLM emits a movement grounded in web facts, it cannot reach the dispatcher.
2. **Prompt‑level grounding rule** in the persona section: *"Web search results are for answering general‑knowledge questions only. Never use them to decide where to walk, who to follow, what to navigate to, or what objects exist in your environment. For physical actions, only trust your saved locations and your camera."*
3. **Source attribution in TTS** — every web‑augmented response begins with "According to a web search…".
4. **Audit log** — every `(timestamp, query, top‑3 results, final response)` tuple to `logs/web_search.jsonl`.
5. **Rate limit / circuit breaker** — max 1 web search per 10 s per session (timestamp on the brain instance), max 1 recursion depth in `process()`.

**Verify:**
- Happy path: "What time does Dartmouth dining close today?" — emits `web_search`, speaks filler, runs Tavily, returns sourced answer prefixed with "According to a web search…". Latency ≤ ~4 s warm.
- **Action lockout test:** "Search the web for the kitchen and walk there." Spot must search, answer about the search results, and **NOT** navigate. Verify in logs that the second‑pass action list was forcibly cleared.
- Unrelated: "Walk forward 1 meter." Must NOT trigger web search. Latency unchanged from baseline.
- Inspect `logs/web_search.jsonl` — every query logged with results.
- Rate limit: 3 web questions in 5 seconds — second and third should be answered from the LLM's own knowledge with no Tavily call.

5. Commit: `smart-chatbot: web_search action with Tavily + safety lockout`

#### Stage D — Persistent remembered facts

Smallest possible cross‑session memory. Not RAG, not a vector store.

1. New `data/spot_facts.sqlite` (gitignored) — single table `facts(id INTEGER PK, fact TEXT, created_at TIMESTAMP)`.
2. New `src/voice_control/facts_store.py` (~50 LOC) — `FactsStore` class with `add(fact)`, `recall_all() -> list[str]`, `forget(fact_id)`, `clear()`. Hard cap at 50 facts; after that the brain refuses to add more and asks the user to forget something first.
3. `src/voice_control/llm_brain.py`:
   - Add `remember` action to the catalog: `{"action": "remember", "params": {"fact": "<thing>"}}`. Rule: *"Use remember when the user explicitly tells you to remember something — names, preferences, facts about the lab, locations of objects."*
   - Add `forget` action: `{"action": "forget", "params": {"fact": "<which one>"}}` with substring matching.
   - At `SpotBrain.__init__`, load all facts via `FactsStore.recall_all()` and inject into the system prompt under a `# THINGS YOU REMEMBER ABOUT YOUR HUMANS` section.
   - When the LLM emits `remember`, the brain calls `FactsStore.add(fact)` and the response confirms ("Got it, I'll remember that the green chair is broken.").
4. **Verify:** "Remember that the green chair in the lab is broken." → emits `remember`, fact stored. Restart `run_voice_control.py`. "Is the green chair okay?" → recalled. "Forget about the green chair." → fact removed.
5. Commit: `smart-chatbot: remember/forget actions backed by SQLite`

**Stage E (deferred) — always‑on vision context.** After gemma4 unification, vision becomes much cheaper. Add a `--vision-context` flag to `client_mic.py` that captures a front‑camera frame on perceptual‑question turns and attaches it to the LLM call. Cache the most recent frame so back‑to‑back perceptual turns reuse it within ~500 ms. Default‑off. Deferred because it touches `client_mic.py` and a frame‑capture/cache layer — riskier for latency regressions on action‑only turns.

**Skipped intentionally:**
- Migrating the JSON‑actions catalog to Ollama's native tools API. Mixing protocols (tools API for `web_search`, JSON for the 28 actions) would be worse than either pure approach. Defer to a separate refactor after Phase 6 lands.
- Vector RAG over conversation history with embeddings. Overkill for short robot dialogues; adds an embedding model to VRAM; the `MAX_HISTORY` bump in Stage A covers realistic use cases.
- LangChain / Pydantic AI / Smolagents. The current 555‑line `llm_brain.py` is more direct and more debuggable.

### Phase 7 (optional, deferred) — Gemma 4 unified audio side experiment

1. Add `UNIFIED_AUDIO_LOG=true` env var in `client_mic.py`
2. When set, after a successful staged transcription, also send the same audio buffer + the wake‑word‑triggered camera frame to `gemma4:e4b` via Ollama's audio chat API
3. Log `(audio_path, staged_transcript, gemma_transcript, staged_action_json, gemma_action_json)` to `logs/unified_audio_eval.jsonl`
4. Run for a week of dev use, then analyze
5. Promote unified to production only when ALL of:
   - [ollama/ollama#15333](https://github.com/ollama/ollama/issues/15333) is closed
   - 0 crashes in 1000 prompts
   - WER within ~20% of dedicated STT on Spot's environment
   - Action‑selection agreement > 95% on the captured corpus

### Phase 8 (optional, deferred) — TTS follow‑up improvements

These build on Phase 2.5 (Kokoro on GPU) and are independently valuable. Sequence them after Phase 2.5 has been measured so we know whether the GPU win alone is already "good enough." None are blocking.

1. **TTS warmup at startup.** Synthesize a tiny utterance (e.g. `"ready"`) inside `_load_model()` so the first real `create()` call doesn't pay first‑call CUDA kernel compilation / cache‑warm overhead. ~3 lines in `spot_tts.py`. Free, can be folded into Phase 2.5 itself if there's time.

2. **Sentence‑chunked TTS streaming.** Split the LLM `response` on sentence boundaries (`re.split(r'(?<=[.!?])\s+', response)`), synthesize one chunk at a time, queue playback so the first chunk speaks while later chunks synthesize in the background. Reduces time‑to‑first‑audio *further* beyond the GPU win — useful for long responses. Touches `spot_tts.py` (new internal queue + worker thread); `client_mic.py` is unaffected because the public `speak()` API is unchanged. Should follow Phase 2.5 once we have measurements showing the GPU win alone isn't enough for long responses.

3. **LLM streaming → TTS at sentence boundaries.** Largest single reduction in *total* perceived latency: set `"stream": True` in `llm_brain.py:269` (and the other Ollama call sites at lines 160, 213, 451), parse the streaming chunks to extract the `response` field incrementally, and fire `tts.speak(first_sentence)` as soon as the first sentence boundary appears in the streamed text — *while the LLM is still generating the rest*. The catch: Ollama's `format=json` mode (`llm_brain.py:268`) makes streaming awkward because partial JSON isn't speakable. Two workarounds: (a) use a partial‑JSON parser that can yield string‑field deltas, or (b) restructure the prompt so `response` text precedes the action JSON and stream the plain‑text prefix to TTS while the JSON tail is still arriving. **Sequence after Phase 2 (gemma4 collapse)** because the new model may behave differently with streaming + `format=json`. Touches `llm_brain.py`, `client_mic.py:870-975`, and possibly `spot_tts.py` (chunk queue from #2).

4. **Voice/speed config exposure.** Kokoro v1.0 ships many voices. Add `SPOT_TTS_VOICE` and `SPOT_TTS_SPEED` env vars in `spot_tts.py` so the voice can be tuned without editing code. Trivial; bundle with Phase 2.5 if it's a one‑line addition.

5. **Wake word on GPU — explicitly rejected.** Do not revisit. See the "right division of labor" table in Phase 2.5 for the reasoning (5 MB int8 streaming model, sub‑millisecond per‑frame compute, GPU launch overhead would dominate, runs 24/7 so always‑on GPU usage wastes power, would force the source‑build sherpa‑onnx GPU path that disk already ruled out).

### Phase 8.x — Deferred codebase cleanup (parallel-yawning-dahl Tier 2/3)

After all model swaps land. **Save for last** — high blast radius, would conflict with Phase 6 smart-chatbot edits to `llm_brain.py` if landed earlier.

- **Tier 2 #10 — split `dispatch_intent()` god function.** `src/voice_control/spot_dispatch.py` currently has `dispatch_intent` as a ~990-LOC function with 30+ `elif name == "..."` branches. Extract each branch into `_handle_<name>(session, params)` and dispatch via a dict at the bottom: `_HANDLERS = {"stop": _handle_stop, "walk": _handle_walk, ...}`. Behavior-preserving. Cuts the function down to a 5-line dispatcher and unblocks per-handler unit testing. Do action-by-action with smoke testing between each — suggested order: safety commands first (stop/freeze/estop), then posture, then nav, then chat actions.

- **Tier 2 #11 — collapse `tour` / `patrol` byte-equivalence.** After Phase 0.2 already fixed the `repeat=True` patrol bug, the two handlers have ~110 lines of identical location-loading + ordering logic. Extract `_collect_named_waypoints(session, graph_nav_client, locations_param) -> tuple[list[str], list[str]]` and have both call it. The bodies become ~10 lines each, differing only in `repeat` and the printed label.

- **Tier 3 — micro-cleanups.** Redundant local imports (e.g., `web_panel.py:229 import os` shadowing the module-level import), deduped error messages, log-error noise reduction, etc.

**Disk unblock from Phase 4.** After Riva removal frees ~24 GB of Docker image + ~2 GB of quickstart dir, disk pressure goes from "critical" to "comfortable." Improvements #2 and #3 above become much easier to implement and test once that headroom exists.

---

## Files touched

### Phase 0 (foundation, mostly already shipped)
| Status | Path | Notes |
|---|---|---|
| Modified | `src/voice_control/audio_player.py` | Phase 0.1: inflight counter, watchdog thread, graceful shutdown, latency callback hooks. Already in working tree. |
| New | `src/voice_control/latency.py` | Phase 0.3: ~547 LOC, full schema + JSONL writer + percentiles + SIGUSR1 dump. Wave 1 shipped. |
| New | `src/voice_control/startup_status.py` | Phase 0.3: ~157 LOC, boot health check + template builder. Wave 1 shipped. |
| Modified | `src/voice_control/client_mic.py` | Phase 0.3 Wave 2: `--latency` / `--latency-out` flags, `init_recorder()`, greet block, `time_phase` / `trace_span` wraps. NOT yet done. |
| Modified | `src/voice_control/spot_tts.py` | Phase 0.3 Wave 2: wire `on_render_*` / `on_play_*` callbacks at L161. NOT yet done. |
| Modified | `src/voice_control/spot_dispatch.py` | Phase 0.3 Wave 2: `_PROCESS_START_TS` + `uptime_s` / `estop_holder` / `num_saved_locations` / `process_pid` fields in `get_robot_state_dict()`. (Phase 0.2 patrol fix already in.) |

### Phase 9 (perception overhaul, mostly already shipped)
| Status | Path | Notes |
|---|---|---|
| New | `scripts/probe_image_sources.py` | Phase 9.A Tier 0a: image-source / depth / world-model probe. Already in working tree. |
| New | `docs/project/perception_probe_results.md` | Phase 9.A Tier 0a: probe output. Already in working tree. |
| New | `src/voice_control/perception/__init__.py` | Phase 9.A package marker. |
| New | `src/voice_control/perception/camera_intrinsics.py` | Phase 9.A Tier 2c: intrinsics cache. Already in working tree. |
| New | `src/voice_control/perception/local_grid_helpers.py` | Phase 9.A Tier 4a: LocalGrid decoder. Already in working tree. |
| New | `src/voice_control/perception/obstacle_query.py` | Phase 9.A Tier 4b: body-frame obstacle helpers. Already in working tree. |
| New | `src/voice_control/perception/raycast_helpers.py` | Phase 9.A Tier 4c: RayCast wrapper (disabled by default). Already in working tree. |
| New | `src/voice_control/world_objects.py` | Phase 9.A Tier 3a: WorldObjectClient + `find_navigation_target`. Already in working tree. |
| Modified | `src/voice_control/visual_nav.py` | Phase 9.A Tier 1a/1b/1e/2a/2b: `_decode_depth`, `_depth_at_bbox`, `navigate_to_object` rewrite, `_bearing_in_body_frame`. Already in working tree (+427 LOC). |
| Modified | `src/voice_control/spot_dispatch.py` | Phase 9.A Tier 1.5/3a/4d: door deletion, `go_to_object` smart routing, `check_obstacles` dispatch. Already in working tree. |
| Modified | `src/voice_control/llm_brain.py` | Phase 9.A Tier 4d: `check_obstacles` action in catalog + perception examples. Already in working tree. |
| Modified | `src/voice_control/spot_dispatch.py` | Phase 9.B Stage 0: `CAMERA_ROTATION_DEG` dict + `_rotate_for_source` helper + `Frame` namedtuple + `capture_frame` rewrite. NOT yet done. |
| Modified | `src/voice_control/visual_nav.py` | Phase 9.B Stage 2: `_find_object_batch` + `_scan_all_cameras` (replaces `:595-609` serial loop). Phase 9.B Stage 4: `_get_world_model` / `_get_person_model` rewired to `onnxruntime.InferenceSession` with `CUDAExecutionProvider`. NOT yet done. |
| Modified | `src/voice_control/spot_dispatch.py` | Phase 9.B Stage 3: `look_around` action + dispatcher entry. NOT yet done. |

### Phase 3 (Voice Pipeline Refresh via elegant-moseying-robin)
| Status | Path | Notes |
|---|---|---|
| Modified | `src/voice_control/server.py` | Refactor to `Backend` interface; carry over hallucination filters |
| New | `src/voice_control/asr/__init__.py` | Backend registry (Parakeet + Riva) |
| New | `src/voice_control/asr/parakeet.py` | Parakeet‑TDT‑0.6b‑v3 via onnx‑asr backend (~80 LOC). Primary. |
| New | `src/voice_control/asr/riva_backend.py` | RivaBackend (extracted from server.py for the rollback path; deleted in Phase 4) |
| New | `src/voice_control/asr/sensevoice.py` | (Fallback) sherpa‑onnx SenseVoice backend |
| New | `src/voice_control/asr/granite.py` | (Fallback) IBM Granite 4.0 1B Speech backend with keyword biasing |
| Modified | `src/voice_control/llm_brain.py:28-29` | `DEFAULT_MODEL = VLM_MODEL = "gemma4:e4b"` |
| Modified | `src/voice_control/llm_brain.py:249-304` | `warm_up_vlm()` no‑op; delete 1×1 JPEG state |
| Modified | `src/voice_control/llm_brain.py:28-29` | `DEFAULT_MODEL = VLM_MODEL = "gemma4:e4b"` |
| Modified | `src/voice_control/llm_brain.py:249-304` | `warm_up_vlm()` no‑op; delete 1×1 JPEG state |
| Modified | `src/voice_control/spot_tts.py` | Phase 2.5: swap sherpa‑onnx Kokoro for kokoro‑onnx with `CUDAExecutionProvider`; voice ID becomes voice name; ~30–50 line diff |
| New | `scripts/setup_kokoro_v1.py` | Phase 2.5: downloads `kokoro-v1.0.onnx` and `voices-v1.0.bin` from kokoro‑onnx GitHub releases |
| New | `models/tts/kokoro-v1.0/` | Phase 2.5: new model directory (replaces `kokoro-en-v0_19/`) |
| Deleted | `models/tts/kokoro-en-v0_19/` | Phase 2.5 step 13: reclaim ~340 MB after end‑to‑end verification |
| Modified | `scripts/run_voice_control.py:31-32, 103, 294` | Delete Riva orchestration |
| New | `scripts/setup_sensevoice.py` | Model download (mirrors `setup_kws.py`) |
| New | `scripts/setup_granite.py` | Model download |
| New | `scripts/setup_parakeet.py` | Model download |
| New | `scripts/test_stt.py` | Tournament harness |
| New | `tests/audio/*.wav` | Regression test corpus |
| Deleted | `scripts/setup_riva.sh` | After Phase 4 |
| Modified | `requirements.txt` | Drop `nvidia-riva-client`; add `transformers>=4.52.1`, `torchaudio`, `soundfile`, `onnx-asr[hub]`, `kokoro-onnx`; note `onnxruntime-gpu` from Jetson AI Lab `--extra-index-url` |
| Modified | `README.md` | New pipeline diagram + setup steps |
| Modified | `docs/architecture/overview.md` | New architecture |
| Modified | `docs/architecture/voice-pipeline.md` | Update TTS timing line at line 190 (`0.1-0.5s`) after Phase 2.5 GPU measurements |
| Modified | `docs/getting-started/models.md` | Update TTS model entry to point at v1.0 layout and note GPU provider |
| Modified | `docs/project/status.md` | New model list |
| Modified | `src/voice_control/llm_brain.py` | Phase 6 Stage A‑D: rewrite `SYSTEM_PROMPT` (persona + ACTION RULES); `MAX_HISTORY` 12→24; conversation log; knowledge file injection in `_build_messages()`; `web_search`/`remember`/`forget` actions; `_run_web_search_turn()` helper; action lockout enforcement |
| New | `src/voice_control/web_search.py` | Phase 6 Stage C: ~80 LOC, `WebSearchBackend` interface + `TavilyBackend` |
| New | `src/voice_control/facts_store.py` | Phase 6 Stage D: ~50 LOC, SQLite‑backed `FactsStore` |
| New | `data/spot_knowledge.json` | Phase 6 Stage B: static Dartmouth/lab/Spot facts (~300–500 tokens, hard cap 1500) |
| New | `data/spot_facts.sqlite` | Phase 6 Stage D: runtime, gitignored |
| New | `logs/conversations.jsonl` | Phase 6 Stage A: runtime, gated by `--log-conversations` flag |
| New | `logs/web_search.jsonl` | Phase 6 Stage C: runtime, web search audit trail |
| Modified | `requirements.txt` | (Phase 6) add `tavily-python` |
| Modified | `.env` | (Phase 6) `TAVILY_API_KEY=...` (gitignored, document in README) |
| Modified | `scripts/run_voice_control.py` | (Phase 6 Stage A) plumb the `--log-conversations` flag through |

**Untouched (after the consolidation):** `wake_word.py` (will be touched in Phase 4 step 8 for openWakeWord swap), `intent.py`, the 28 dispatch actions (Phase 6 keeps `web_search`/`remember`/`forget` entirely inside the brain — the dispatcher does not need new entries; Phase 9 adds `look_around` and `check_obstacles` *to* the dispatcher). Note that `client_mic.py`, `spot_dispatch.py`, `visual_nav.py`, and `audio_player.py` are now all touched — see Phase 0 / Phase 9 entries above. The point of the `server.py` `Backend` boundary is still that the mic client doesn't need to know which ASR engine is active.

---

## Verification plan (per phase)

| Phase | Verification |
|---|---|
| 0.1 | `grep -n '_inflight\|_watchdog_loop\|def shutdown' src/voice_control/audio_player.py` shows the inflight counter (~L176), watchdog loop (~L522), and `shutdown` method (~L308). Working tree commits clean. ✅ DONE |
| 0.2 | `grep -n 'repeat=True' src/voice_control/spot_dispatch.py` shows the patrol fix (~L1304). `grep -n 'except\s*:' src/voice_control/client_mic.py` returns no matches. `grep -n 'SPOT (LLM)' src/voice_control/client_mic.py` returns L1107. The 3 ⏳ sub-fixes (`run_voice_control.py` service-check logging, `_navigate_waypoint_sequence` swallowing, `save_location` name normalization) need verification after they land. |
| 0.3 | `python scripts/run_voice_control.py --latency` produces a JSONL trace at `logs/latency-{date}.jsonl`. `python -c "import json; lines = [json.loads(l) for l in open('logs/latency-{date}.jsonl')]"` parses cleanly. `asr_ms`, `llm_ms`, `tts_ms`, `dispatch_ms` are non-zero per utterance. Boot greet block speaks the template status from `startup_status.build_template_status(...)`. |
| 1 | Baseline `run_voice_control.py` works with current code. Audio corpus captured. |
| 2 | **Preconditions:** `df -h /` shows ≥ 12 GB free *before* `ollama pull` (Phase 4 must have run); `ollama --version` ≥ 0.20.0. `python src/voice_control/llm_brain.py gemma4:e4b` passes representative utterances. `jtop` confirms resident GPU memory growth within the 6–8 GB e4b forecast. Wired pipeline (Phase 4 already done — Riva is gone) talks correctly through mic. Old qwen2.5/qwen2.5vl models stay on disk as revert path until accuracy holds for ≥ 24 hours of dev use, then `ollama rm`. |
| 2.5 | `onnxruntime.get_available_providers()` includes `CUDAExecutionProvider`. `[TTS] Ready ... provider=CUDAExecutionProvider` in startup log. `tegrastats` shows non‑zero GPU% during synthesis. Wake word still triggers. Speaker reacts noticeably sooner after `[Brain] LLM responded` than the Phase 1 baseline. `df -h /` shows ≥ 800 MB free after old model deletion. |
| 3 | `scripts/test_stt.py` reports WER vs Riva and hallucination chars per backend. Wired pipeline passes a 10‑command mic test on the winning backend. |
| 4 | `run_voice_control.py` starts cleanly with no Riva orchestration. No import errors. No `nvidia-riva-client` references remain in `requirements.txt`. |
| 5 | Spot completes the end‑to‑end exercise sequence. |
| 9.A | `python scripts/probe_image_sources.py --capture-depth` runs cleanly (already done — see `docs/project/perception_probe_results.md`). `python -c "from src.voice_control.world_objects import find_navigation_target"` imports. `check_obstacles` voice command returns nearest-obstacle distance + bearing. `navigate_to_object` rewrite still walks Spot to the requested target on the live robot. Tier 1b's mask path remains gated until 9.B Stage 4 produces a seg model. |
| 9.B | After Stage 0: `grep -n CAMERA_ROTATION_DEG src/voice_control/spot_dispatch.py` shows the dict; `capture_frame` returns a `Frame` namedtuple. After Stage 2: scan time on `navigate_to_object` measurably drops (compare `--latency` JSONL traces vs Phase 0 baseline). After Stage 3: `look_around` action works end-to-end and produces N independent per-camera descriptions. After Stage 4: `_find_object` returns a `mask` field on detections; `tegrastats` shows GPU utilization spike during YOLO inference (NOT 0 % — that means ONNX silently fell back to CPU); `_depth_at_bbox` mask path is hit (verified by tracing or a temporary print). |
| 6 | Stage A: regression corpus action accuracy ≥ baseline; subjective persona check on 5 free‑form questions. Stage B: lab/who‑works‑here questions answered from `spot_knowledge.json` with no network call. Stage C: happy‑path web question returns sourced answer ≤ ~4 s warm; **action lockout test passes** (search + walk request searches but does not navigate); rate limit enforced; `logs/web_search.jsonl` populated. Stage D: remember/forget round‑trip survives a process restart. |
| 7 | (Optional) `logs/unified_audio_eval.jsonl` populates without crashes; WER and agreement metrics meet the bar before any production switch. |
| 8 | (Optional) Per follow‑up: warm utterance eliminates first‑call CUDA compile cost; sentence chunking measurably reduces time‑to‑first‑audio on long responses; LLM‑stream→TTS measurably reduces total perceived latency without breaking JSON parsing. |

---

## Rollback

Phases 1–5 are commits on the `swap-models` branch; Phase 6 is commits on the parallel `smart-chatbot` branch; Phase 9 lives on `tour_guide_upgrade_matteo` (mostly already committed in the consolidation). Rollback options:

- **Phase 0.1 (AudioPlayer hardening) fails:** `git revert <commit>`; the `_inflight` counter and watchdog thread go away, the original `_busy` Event-based code returns. Rare 0.75 % `is_busy()` race window also returns.
- **Phase 0.3 (latency Wave 2) fails:** the `--latency` flag is off by default. The producer modules (`latency.py`, `startup_status.py`) are no-ops if the recorder is never initialized. Rollback is "don't pass the flag."
- **Phase 9.A (perception overhaul) fails on a Tier:** each tier is independently rollback-able. `check_obstacles` action: remove the catalog entry. `navigate_to_object` rewrite: revert the `visual_nav.py` commit. World-object routing: short-circuit `find_navigation_target` to return None and fall back to YOLO. The LocalGrid + RayCast helpers are pure additions; they have no rollback effect when unused.
- **Phase 9.B (vision overhaul) fails on a Stage:** Stage 0 (rotation table) is the riskiest because every downstream stage assumes it. If Stage 0 misroutes, revert and the existing `ROTATE_270` constants take over. Stage 4 (ONNX/GPU YOLO) falls back to PyTorch CPU by reverting `_get_world_model` / `_get_person_model`.
- **Phase 2 fails:** revert the `llm_brain.py` commit; back to qwen2.5:7b + qwen2.5vl:7b
- **Phase 2.5 fails:** `pip uninstall -y kokoro-onnx onnxruntime-gpu && pip install onnxruntime==1.23.2`, then `git checkout src/voice_control/spot_tts.py`. The old `models/tts/kokoro-en-v0_19/` is still on disk because step 13 hasn't run yet, so the original sherpa‑onnx CPU TTS path returns instantly. Do not run `rm -rf models/tts/kokoro-en-v0_19/` until end‑to‑end verification has passed.
- **Phase 3 fails on the chosen STT:** switch `SPOT_ASR_BACKEND` env var to a different backend (no code change). All three are wired in parallel.
- **Phase 3 fails on all three STTs:** keep `RivaBackend` as the active backend; the gRPC interface is unchanged.
- **Phase 4 already done and need to roll back further:** revert the deletion commit; Riva orchestration returns. Riva Docker container/image stays installed until disk reclaim is done in step 7.
- **Phase 6 Stage A regresses persona / action accuracy:** revert the `SYSTEM_PROMPT` commit; persona returns to current generic. Stages B–D are independent and stay.
- **Phase 6 Stage B knowledge file too large or causes prompt‑eval slowdown:** delete `data/spot_knowledge.json`; one‑line revert in `_build_messages()`.
- **Phase 6 Stage C web search is unreliable / Tavily quota burned:** the `WebSearchBackend` interface lets you swap backends without touching the brain. Or remove the `web_search` action from the catalog and the brain ignores it. A missing `TAVILY_API_KEY` does NOT crash the brain — `TavilyBackend` falls back to "I can't search the web right now" with a logged warning.
- **Phase 6 Stage D facts store corrupted:** delete `data/spot_facts.sqlite`; the `FactsStore` recreates it empty on next startup.

The Riva Docker container and the `~/riva_quickstart_arm64_v2.17.0` directory should NOT be deleted until end‑to‑end validation passes in Phase 5.

---

## Constraints to remember

- **Jetson AGX Orin (JetPack 6.2.1, CUDA 12.6)** — install via `https://pypi.jetson-ai-lab.io/jp6/cu126` for torch/cuda‑python wheels when needed (Granite, Parakeet routes, Phase 2.5 onnxruntime‑gpu)
- **E‑Stop must run separately** in another terminal — enforced in `run_voice_control.py:288-289`
- **Disk pressure** — `/dev/mmcblk0p1` is 57 GB total, **1.8 GB free as of 2026‑04‑08** (97% full). Phase 4 Riva removal is the single biggest disk win — verified ~26 GB total: 24.04 GB Docker image with zero shared layers (`docker system df -v`) + 2.1 GB bind‑mounted `/data` model_repository under `~/riva_quickstart_arm64_v2.17.0/`. Until Phase 4 ships, every install step has to be size‑budgeted; anything over a couple hundred MB needs to free space first. Phase 2.5 and Phase 3 are sized to fit within the pre‑Phase‑4 budget. **Phase 2 (Gemma 4 pull, 9.61 GB) cannot fit pre‑Phase‑4 and must run after — see the Execution order callout above.**
- **Models share VRAM** — `gemma4:e4b` forecast ~6‑8 GB resident at 8k ctx; `gemma4:26b` MoE forecast ~16‑22 GB resident; YOLO + ROS + Kokoro need headroom; the 64 GB unified memory has space for e4b comfortably and 26b tightly. **Forecasts — verify with `jtop` on the actual device after Phase 2 pull, before committing.**
- **GPU consumers after Phase 2.5** — Riva ASR (until Phase 4), Ollama LLM (changes in Phase 2), Kokoro TTS (after Phase 2.5). The Orin's 64 GB unified memory has headroom but VRAM/compute contention still matters for latency. If Phase 2.5 measurements show TTS synthesis blocked on the LLM, that's the signal to revisit Phase 7 (unified audio) sequencing.
- **Granite uses vanilla PyTorch on Jetson** (no native TensorRT path) — measure RTF on the actual device before committing to it as the winner
- **Ollama version requirement** — Gemma 4 needs **Ollama ≥ 0.20.0**. Verify with `ollama --version` before pulling. The current Jetson is on 0.16.1; in‑place upgrade is `curl -fsSL https://ollama.com/install.sh | sh` (replaces the 36 MB binary at `/usr/local/bin/ollama`; the systemd override survives).
- **Tavily free tier** — ~1k queries/month. The Stage C rate limiter (1 web search per 10 s per session, max recursion 1) makes burning that on a single session implausible, but watch the dashboard during demo days.
- **Web search action lockout is mandatory** — Phase 6 Stage C reverses BD's deliberate "no general internet" choice. The five safety mitigations (in‑code action lockout, prompt grounding rule, source attribution, audit log, rate limit) are not optional. They are how the safety story stays intact.

---

## Reference: implementation blueprints

This file is the consolidated master roadmap. The deeper file:line surgery for the still-pending phases lives in these per-plan blueprints in `~/.claude/plans/`:

**Phase 3 + Phase 4 — Voice Pipeline Refresh:**
- `elegant-moseying-robin.md` — full Parakeet TDT v3 + Silero VAD + openWakeWord plan with pre-flight checks (PF-1..PF-6), `--no-deps` install order, `Backend` interface refactor, and rollback paths. Authoritative reference for Phase 3 / Phase 4 / Phase E. Delete after Phase 4 ships.

**Phase 9 — Perception overhaul:**
- `humming-stargazing-shell.md` — perception/depth/pose/world-objects/obstacle-awareness plan. 14/15 tiers already in the working tree (see Phase 9.A audit table). Delete after Phase 9.B Stage 4 unblocks Tier 1b's mask path.
- `linked-hopping-anchor.md` — multi-camera vision plan (rotation table, gripper plumbing, batched scan, look_around, ONNX/GPU YOLO swap, model swap). 0/7 stages done — see Phase 9.B. Delete after Stages 0–4 ship.

**Phase 6 — Smart chatbot:**
- `hashed-waddling-dragonfly.md` — full smart-chatbot plan with persona text, knowledge file structure, web_search architecture (the five safety mitigations), FactsStore schema, and the BD "Robots That Can Chat" reference. Phase 6 of this document summarizes; the linked plan has the complete file:line detail. Delete after Phase 6 Stage D ships.

**Master consolidation roadmap (this rehaul):**
- `precious-tinkering-hare.md` — the meta-plan that walked through committing the working tree, deleting stale plans, and editing this file. Delete after Wave 6 of that plan completes.

**Earlier per-candidate research (deleted 2026-04-08 after consolidation):**
The original `warm-tickling-lynx*.md` (7 files), `mellow-swinging-llama.md` (Phase 2.5 — shipped), `steady-questing-spindle.md` (qwen pinning — shipped), `cryptic-swinging-reddy.md` (AudioPlayer hardening — shipped), `parallel-painting-mist.md` (latency — Wave 1 shipped, Wave 2 in Phase 0.3), `parallel-yawning-dahl.md` (codebase cleanup — Tier 1 shipped, Tier 2/3 in Phase 8.x), `recursive-finding-kahan.md` (LLM/VLM tag visibility — shipped), `silly-inventing-treasure.md` (wakespot — shipped), `frolicking-watching-sutton.md` (`activate` venv alias), `ancient-wondering-lovelace.md` (superseded by linked-hopping-anchor) were deleted after their content was either shipped, baked into this document, or made redundant by a successor plan. Findings are preserved in the Phase 2 fallback ladder (Gemma 3 vs 4 + Gemma audio), Phase 3 Parakeet pick (multiple STT rounds), Phase 0 audit (cryptic / parallel-painting-mist / recursive / parallel-yawning-dahl), and the `~/.claude/projects/-home-spotdog/memory/project_voice_pipeline_models.md` memory entry.
