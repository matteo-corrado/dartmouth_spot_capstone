# Stage 2 Design — Voice Loop + Spot E2E + Perception + Smart Chatbot

> **Stage 1 + 1.5 status:** Complete. Branch `tour_guide_upgrade_matteo` at tag `stage1.5-complete`. Disk freed (eMMC 16 GB → 35 GB), Ollama models live on SSD, gemma4:e4b runs as primary brain, qwen pair kept as Stage 1 rollback. `[Brain-timing]` instrumentation in place. Voice loop is **currently broken** because `src/voice_control/server.py` still imports `riva.client` (Riva removed in Stage 1) — Stage 2 unblocks it.

## Goal

Stage 2 brings the rest of the voice + perception + chatbot pipeline up to a coherent baseline and ships an integrated tour-guide demo. End state = `tour_guide_upgrade_matteo` merged to `main` after Spot e2e regression passes.

## Scope decision

User confirmed during brainstorming: **single comprehensive design spec, multiple implementation plans** — one per subsystem. Each sub-plan ships independently testable software.

Sub-plan map (and exit tags):

| Plan | Sub-system | Sections covered | Exit tag |
|---|---|---|---|
| 2A | Voice pipeline | 2A.0 + 2A.1 + 2A.2 + 2A.3 + 2A.4 | `stage2a-voice-complete` |
| 2B | Spot end-to-end gate | 2B.1 | `stage2b-e2e-complete` (also: merge to `main`) |
| 2C | Vision overhaul | 2C.0 + 2C.0.5 + 2C.1 + 2C.2 + 2C.3 + 2C.4 | `stage2c-vision-complete` |
| 2D | Smart chatbot | 2D.A + 2D.C + 2D.D | `stage2d-chatbot-complete` |
| Final | All green | n/a | `stage2-complete` |

Execution order: **2A → 2B sequential, then 2C ∥ 2D in parallel**. 2B is the merge moment because Spot e2e demonstrates the voice loop works on the real robot. Vision and chatbot are independent files (`visual_nav.py` vs `chatbot/`) so they parallelize post-merge.

## Cross-cutting decisions (research-backed, 2026-05-15)

Five research agents validated original-spec picks against the current ecosystem. Findings:

### Brain runtime — swap Ollama → llama.cpp + HTTP (NEW Stage 2A.0)

- **Original spec:** kept Ollama 0.24.0, accepted 5 s warm latency on gemma4:e4b
- **Problem:** Ollama bug cluster #15260 / #15416 / #15502 / #15539 / #15595 / #15288 — `think:false` + `format:json` silently drops JSON constraint. Forces `think:on` for JSON path → ~5 s median warm. Same bug class in vLLM (#39130) — runtime swap to vLLM does NOT fix it.
- **Decision:** Build llama.cpp from source with CUDA backend. Use `llama-server` for OpenAI-compatible HTTP. GBNF grammar (`--grammar-file`) enforces JSON at the sampler level — bypasses the bug entirely (`--json-schema` is broken on gemma4 per llama.cpp #22396, but hand-written GBNF works fine).
- **Why not vLLM / TRT-LLM / MLC:** vLLM has same bug class. TRT-LLM Jetson v0.12.0 doesn't confirm gemma4 support (#2974). MLC-LLM gemma4 not first-class. llama.cpp is the only path that bypasses the bug, has zero ABI risk, and ships gemma4 day-one (Apr 2 2026) including vision via `--mmproj`.
- **Expected speedup:** community reports ~35-60 tok/s decode for gemma4:e4b Q4_K_M on AGX Orin 64 GB (vs Ollama ~15-25 tok/s estimated). With GBNF + no thinking tax + E2B as speculative drafter: **<2 s warm latency target**.
- **Critical gotcha:** `llama-cpp-python` Python bindings re-introduce numpy dep → would break ABI pins. Use HTTP server only.
- **Rollback:** Ollama 0.24.0 stays installed. `llm_brain.py` reads `SPOT_BRAIN_BACKEND=llamacpp|ollama` env var. One-env-var flip reverts. gemma4 GGUF on disk = same model in both runtimes.

### ASR — Parakeet TDT 0.6 B v3 (unchanged)

- Voice research confirmed Parakeet-TDT-0.6b-v3 is still optimal — no v4. Canary-Qwen-2.5B more accurate but heavier; Parakeet wins on speed/accuracy/VRAM trade for tour-guide.
- Anti-hallucination claim documented (trained on 36 k hours of non-speech with empty-string targets).
- Future note (out of scope): `nvidia/parakeet_realtime_eou_120m-v1` emits `<EOU>` tokens — could replace VAD+ASR with a single model for an "interruptible mode" future plan.

### VAD — Silero VAD via sherpa-onnx (unchanged)

- TEN-VAD is faster but license is "Apache 2.0 + additional clause forbidding competing with Agora" — not OSI-clean Apache 2.0. Skip per project license constraint.
- Silero is MIT, ~2 MB ONNX, integrates trivially via `sherpa_onnx.VoiceActivityDetector`.

### Wake word — openWakeWord → LiveKit Wakeword (SWAP)

- **Original spec:** openWakeWord ~25 MB custom-trained "Hey Spot" model
- **Decision:** Swap to LiveKit Wakeword (Apache 2.0, conv-attention head, drop-in ONNX-compatible with openWakeWord loaders).
- **Why:** LiveKit's headline numbers (60× lower AUT, 100× fewer false positives, +17 % detections) are vendor-reported on their own validation set; haircut to 2-3× and the architectural wins (focal loss, embedding mixup, 3-phase training, conv-attention) are still real. Tour-guide demo with 20 students chatting nearby = false-wake-sensitive environment.
- **Risk:** vendor numbers are not independently reproduced. Mitigation: record a small Dartmouth-specific eval set before full commit.

### Vision — YOLO26 family + YOLOE26-S + TensorRT engine export

- **Original spec follow loop:** YOLOv8n current; YOLO11n / YOLO26-N as candidates
- **Decision:** YOLO26-N for follow loop. YOLO26 released 2026-01-14 (Ultralytics). NMS-free, DFL-removed, MuSGD optimizer, ProgLoss, STAL. Ultralytics' published AGX Orin 64 GB benchmark: TRT FP16 = 2.62 ms (≈ 382 FPS) — 10 Hz follow budget met 20× over.
- **Original spec object scan:** YOLO-Worldv2-s; YOLOE-11s / YOLOE26-S as candidates
- **Decision:** **Swap to YOLOE26-S** (paper Feb 2026). +11.4 LVIS AP over YOLO-World-S. Prompt-free mode with built-in 4585-class RAM++ vocab — no manual class-list maintenance needed.
- **Runtime decision:** Export both YOLO26-N and YOLOE26-S as TensorRT `.engine` (FP16). ORT 1.23.0 already exposes `TensorrtExecutionProvider` (verified on this Jetson), so no new runtime install. Ultralytics ships first-class `model.export(format='engine', half=True, device=0)`. Engine cache to `/mnt/ssd/cfm_mppi_caches/yolo_engines/`. Engine load via the same `YOLO()` API (auto-detects `.engine`).
- **Why TRT, not pure ONNX:** Ultralytics' own benchmark shows TRT FP16 = 2.62 ms vs ONNX = 9.87 ms for YOLO26n on AGX Orin 64 GB — 3.8× speedup. Engine cache invalidates on JetPack/TRT bump (acceptable for single-Jetson research robot).
- **License:** YOLO26 / YOLOE26 are AGPL-3.0 — same as current YOLOv8 baseline, no regression. Capstone/research use is unambiguously permitted; document explicitly in the spec for any future publication.

### Fisheye undistortion — added (NEW Stage 2C.0.5)

- BD's `ImageSource` proto carries three distortion models (`pinhole`, `pinhole_brown_conrady`, `kannala_brandt`). Existing `src/voice_control/perception/camera_intrinsics.py` (160 LOC) already caches these per-robot.
- BD's `bosdyn.client.image` ships zero undistortion helpers — must wire cv2 ourselves.
- `opencv-python==4.11.0.86` is pinned and present; `cv2.fisheye.undistortImage` + `cv2.undistort` both available (verified 2026-05-15).
- New helper `_undistort_for_source` alongside the planned `_rotate_for_source`. Precompute `cv2.initUndistortRectifyMap` per camera at startup → per-frame `cv2.remap` (~3-5 ms CPU). Order in `capture_frame`: capture → rotate → undistort → output `Frame`.
- **Implication:** YOLO26-N on undistorted frames becomes the primary follow path (no fisheye-trained checkpoint needed). AprilTag follow stays as exploratory backup, not primary.

### Pin status (pin-guardian agent enforces)

| Pin | Status | Why |
|---|---|---|
| numpy==1.26.4 | HOLD | ABI-load-bearing; segfault risk on upgrade |
| onnxruntime-gpu==1.23.0 | HOLD | Jetson AI Lab publishes ONLY 1.23.0-cp310 for `jp6/cu126` (verified). Already targets cuDNN 9 (matches system 9.3.0.75). Already exposes TRT EP. No upgrade path available. |
| kokoro-onnx==0.4.9 | HOLD | Later versions drag librosa/numba → numpy 2.x |
| opencv-python==4.11.0.86 | HOLD | 4.12+ requires numpy 2.x |
| bosdyn-*==5.0.1.1 | HOLD | Robot firmware match |
| sherpa-onnx | bump within `>=1.12.0` to latest with Silero VAD class support | needed for VAD swap |
| ultralytics | bump for YOLO26 + YOLOE26 + engine export | needed for vision swap |
| onnx-asr | NEW `==0.11.0` | Parakeet (uses ORT-CUDA EP) |
| huggingface-hub | NEW `>=1.0,<2.0` | tightened upper bound; needed for onnx-asr HF download |
| livekit-wakeword loader | NEW (Apache 2.0, ONNX runtime) | wake word swap |

`pip install --no-deps` mandatory on every new install to prevent rewriting `onnxruntime-gpu` / `numpy` pins. If `pip check` fails after install, stop and investigate.

---

## Stage 2A — Voice pipeline

Goal: unblock the voice loop end-to-end, ship Parakeet ASR, mic-verify gemma4 via the new runtime, swap wake word. Exit tag: `stage2a-voice-complete`.

### 2A.0 Brain runtime swap (Ollama → llama.cpp + HTTP)

#### Build + install

- Build llama.cpp from source with CUDA backend (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=ON -DGGML_NATIVE=ON`). Output `build/bin/llama-server` and `build/bin/llama-cli`. If from-source build fails on JP6.2.1 + CUDA 12.6 + cuDNN 9.3, fall back to a `dustynv/jetson-containers` prebuilt.
- Add `scripts/setup_llamacpp_models.py` (~30 LOC, mirrors `scripts/setup_parakeet.py`/`scripts/setup_kws.py` patterns) — downloads + version-pins GGUFs from HuggingFace (unsloth or ggml-org repos) to `/mnt/ssd/llamacpp-models/`. Models pinned in the script for reproducibility:
  - `gemma-4-e4b-Q4_K_M.gguf` (~5 GB, primary)
  - `mmproj-BF16.gguf` (~946 MB, vision projector for gemma4)
  - `gemma-4-e2b-Q4_0.gguf` (~1.7 GB, speculative drafter)
- llama.cpp has no `ollama list`-style model registry; GGUFs are just files on disk. `setup_llamacpp_models.py` is the canonical source of truth for what's installed.

#### Server daemon

- Run llama-server as `systemd --user` unit (or alongside Ollama's existing system unit) — MUST be a long-lived daemon, NOT spawned per request. Cold load = 10-30s (GGUF mmap + KV cache alloc + draft model load), then sub-second per warm request.
- Server command: `llama-server -m gemma-4-e4b-Q4_K_M.gguf --mmproj mmproj-BF16.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 11435 --model-draft gemma-4-e2b-Q4_0.gguf --draft-max 16 --log-format json --metrics`. `-ngl 99` fully offloads to GPU. Speculative decoding with E2B drafter for ~1.5-2× decode speedup. `--log-format json` and `--metrics` enable structured logs + `/v1/metrics` (Prometheus shape) for the `[Brain-timing]` instrumentation to consume.
- NO server-side `--grammar-file` flag — grammar is per-call (see "Adaptive thinking" below).
- VRAM budget: gemma4-e4b Q4_K_M (~7 GB) + mmproj (~1 GB) + E2B drafter (~2 GB) + KV cache @ 8192 ctx (~2 GB) ≈ 12 GB. Coexists with Parakeet (~1-2 GB), Kokoro TTS (~330 MB), YOLO TRT engines (~200 MB), system overhead — well under AGX Orin 64 GB unified memory.
- Single model per server process: switching to a different brain model (e.g., Stage 3 two-brain split, or qwen2.5:3b sidecar) requires either restarting llama-server with a different `-m` flag, or running a second server on a different port. Out of scope for Stage 2A.

#### Adaptive thinking (per-call grammar selection)

The user does NOT want thinking disabled globally. Action dispatch needs to be fast (no thinking → < 2 s); VLM captions, knowledge queries, and free-form chat benefit from thinking (higher quality, latency acceptable). Strategy: route per-call based on intent type, selecting grammar accordingly.

- Two grammar files, both in `src/voice_control/`:
  - `action.gbnf` — strict JSON schema for action dispatch. Top-level production: `root ::= "{" ws "\"action\":" ws action_name ( "," ws optional_field )* ws "}"`. The `action_name` production enumerates valid actions (`"stand" | "sit" | "walk" | "go_to" | "follow" | "stop" | "describe" | "look_around" | ...`) so the model cannot emit `action: "quack"`. No `<think>...</think>` prefix permitted because the grammar starts with `{`. Forces direct JSON emission.
  - `freeform.gbnf` — minimal grammar that allows optional `<think>...</think>` prefix followed by free-form text. Used for paths where thinking is desirable: `root ::= ( "<think>" thought "</think>" ws )? response`. (Or skip grammar entirely on free-form paths — same effect, simpler.)
- Router in `llm_brain.py.process()`: classify intent FIRST (cheap keyword regex on the transcript or a small pre-prompt routing call). Examples:
  - "go to the kitchen", "stand up", "stop", "battery" → action dispatch → send `action.gbnf` → fast (no thinking)
  - "what do you see", "describe the surroundings", "look around" → VLM caption → no grammar (thinking allowed)
  - "what's the library called", "tell me about Dartmouth Hall" → knowledge query → no grammar (thinking allowed)
  - Smart-chatbot persona free-chat (Stage 2D.A) → no grammar (thinking allowed)
- Each HTTP request includes the per-call `grammar` field in the request body (verbose JSON, but flexible). NOT using `--grammar-file` server-side so grammars can evolve without server restart.
- Latency expectation: action path < 2 s warm (no thinking), VLM/chat path 3-6 s warm (thinking on, acceptable for the use case).

#### Dispatcher integration

- Add `SPOT_BRAIN_BACKEND=llamacpp|ollama` env var in `llm_brain.py`. Default `llamacpp`.
- Implement two code paths in `llm_brain.py`:
  - **llama.cpp path** — talks to `http://localhost:11435/v1/chat/completions`. Request includes `grammar` field (loaded from `action.gbnf` or omitted) plus standard OpenAI fields. Response is OpenAI-shape but llama.cpp extensions (`grammar`, `cache_prompt`, `n_predict`, `samplers`) are llama.cpp-specific. **OpenAI API compat is ~90 %, not 100 % — branch on backend, not just URL.**
  - **Ollama path** — existing `requests.post('/api/chat')` flow. Keeps `format: json` + thinking-workaround comments for now (Ollama-specific quirks documented in current llm_brain.py).
- Drop the `format: json` + thinking-workaround code on the llama.cpp path — GBNF makes them obsolete.

#### Speculative drafter caveats

- E2B drafter (`--model-draft gemma-4-e2b-Q4_0.gguf --draft-max 16`) runs IN THE SAME llama-server process. Single VRAM allocation, no separate daemon.
- If output quality regresses vs target-only generation, disable via `--draft-max 0` server flag flip. Trade decode latency for quality.
- Drafter and target must share tokenizer + vocab (E2B and E4B do; verified per llama.cpp #22735).

#### Smoke test + pass criteria

- Smoke test: run `llama-cli` with `action.gbnf` against 5 representative action utterances; confirm valid JSON output + warm-call latency.
- Repeat smoke test WITHOUT grammar against 3 free-form utterances ("what do you see in the room", "describe Dartmouth Hall to me"); confirm thinking happens + response is coherent.
- **Pass criteria:**
  - Action path: valid JSON for 5/5 smoke utterances; warm latency < 2 s for typical 60-token response
  - Free-form path: thinking observed; response coherent; warm latency < 6 s
  - Mic-verify in 2A.3 is the production validation gate.

#### Rollback

- Flip `SPOT_BRAIN_BACKEND=ollama`, restart `run_voice_control.py`. Ollama is unchanged on disk. gemma4:e4b is the same GGUF (Ollama wraps llama.cpp internally), so model behavior should be equivalent — only the bug-cluster latency tax returns.
- llama-server systemd unit can be `systemctl stop` and disabled; doesn't need to be uninstalled.

### 2A.1 Phase 1 audio corpus capture

- 10-15 representative clean utterances → `tests/audio/*.wav` (16 kHz mono, PCM16). Examples: `stand`, `sit`, `walk forward one meter`, `go to the kitchen`, `what do you see`, `follow me`, `stop`, `battery status`, `describe surroundings`, `come here`, `hey spot, look around`.
- 5 motor-idle clips for hallucination check (Spot powered + breathing, no speech).
- `tegrastats --interval 1000 --logfile /tmp/tegra.before.log` baseline during a normal voice interaction.
- Commit `tests/audio/` (small WAVs OK in git; if > 5 MB total, push to SSD + gitignore).

### 2A.2 Parakeet STT swap (Phase A → B → C)

- **Phase A — Backend interface refactor + ParakeetBackend.** New files:
  - `src/voice_control/asr/__init__.py` — backend registry
  - `src/voice_control/asr/riva_backend.py` — extracted from current `server.py` (kept as code rollback only — Riva server itself is gone)
  - `src/voice_control/asr/parakeet.py` (~80 LOC) — `Backend` subclass wrapping `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")`. Reuses the existing hallucination filter blocklist, `MIN_AUDIO_DURATION`, and `_is_repetitive` post-processing from `server.py`. Keeps the gRPC `StreamingRecognize` interface unchanged.
  - `scripts/setup_parakeet.py` (~30 LOC) — mirrors Kokoro setup script; downloads Parakeet model to `/mnt/ssd/parakeet-models/`.
  - Install `onnx-asr==0.11.0` + `huggingface-hub>=1.0,<2.0` with `--no-deps`. Verify `pip check` clean.
  - Default `SPOT_ASR_BACKEND=riva` so existing flow is unchanged for now.
- **Phase B — Silero VAD swap.** Replace `webrtcvad` with `sherpa_onnx.VoiceActivityDetector` (~2 MB Silero ONNX). Sub-millisecond per frame on Orin CPU. Pure additive — no impact on `onnxruntime-gpu` pin (sherpa-onnx ships its own bundled CPU `libonnxruntime`).
- **Phase C — Promote Parakeet.** Flip default `SPOT_ASR_BACKEND=parakeet`. End-to-end mic test against the Phase 1 audio corpus (2A.1). Capture WER vs Riva baseline notes + hallucination character count on idle clips. Riva backend stays as the `riva_backend.py` code path for rollback until 2A.4 deletes it.

### 2A.3 Mic-verify gemma4 (via llama.cpp + Parakeet)

- Define `tests/utterances/action_corpus.yaml` — 15-20 representative tour-guide utterances with expected `(action, location?, object?, args?)` tuples. Categories: navigation (`go to X`, `come here`, `follow me`, `stop`), perception (`what do you see`, `describe surroundings`, `find a chair`), low-level (`stand`, `sit`, `walk forward 1m`, `battery status`).
- Build `scripts/mic_verify.py` — replays audio corpus through Parakeet (or live mic) → llama.cpp brain → records actual `(action, location?, object?, args?)`. Diff against expected.
- **Pass criteria (split by per-call grammar route):**
  - Action-dispatch path (utterances routed to `action.gbnf`):
    - 90 %+ exact action match across the action utterances
    - 90 %+ correct location/object extraction where applicable
    - JSON parses on 100 % of calls (GBNF guarantees this)
    - Warm-call latency p50 < 2 s, p95 < 3.5 s (measured via existing `[Brain-timing]` instrumentation)
  - Free-form path (perception/knowledge/chat utterances, no grammar, thinking allowed):
    - 100 % coherent output (manual eval on 3 "what do you see" + 3 knowledge utterances)
    - Warm-call latency p95 < 6 s (thinking on; acceptable for these intent types)
  - Router accuracy: 95 %+ correct intent classification (action vs free-form) — verified by tracing per-utterance `grammar=action|none` decisions in `[Brain-timing]` logs and diffing against expected route from `action_corpus.yaml`.
- **Fail recovery (cascade):**
  1. If GBNF grammar is too strict and rejects valid JSON variants, loosen `action.gbnf`. Re-run.
  2. If router misclassifies (e.g., sends "describe surroundings" to action path), tighten the intent regex/classifier. Re-run.
  3. If action selection regresses (model picks wrong action despite GBNF enum constraint), try gemma4:e4b at higher quant (Q6_K or Q8_0). Re-run.
  4. If still failing, fall back to qwen2.5:7b in Ollama (Stage 1 rollback path — flip `DEFAULT_MODEL` + `SPOT_BRAIN_BACKEND=ollama`). Document why. Mic-verify on qwen2.5:7b. If passes, accept slower latency; the qwen pair stays on disk.
  5. If qwen2.5 also fails (unlikely — it was working pre-Stage-1), escalate. The original gemma4 swap may need to be reverted entirely.
- **Pass unlocks:** `ollama rm qwen2.5:7b qwen2.5vl:7b` — reclaims 10.7 GB. Update `stage2-rollback.md` to remove the brain-rollback layer that references the qwen pair.

### 2A.4 Phase 4 final cleanup + LiveKit Wakeword swap (Phase E)

- Delete `src/voice_control/asr/riva_backend.py` — Parakeet is the live backend, no need to keep the dead code path now that mic-verify passed. Update `run_voice_control.py` to remove any Riva startup branches that remained.
- Drop `nvidia-riva-client` from `requirements.txt` if still there (Stage 1 dropped it; verify).
- Wake-word swap:
  - Install LiveKit Wakeword loader (verify install path on PyPI; if not on PyPI, build from source `livekit/livekit-wakeword` repo per their README). Apache 2.0.
  - Train (or download from LiveKit hub if available) a "Hey Spot" model via their Piper-synthetic-positives pipeline. Output ONNX (~5 MB).
  - Add `SPOT_WAKE_BACKEND=sherpa_onnx|livekit` env var in `src/voice_control/wake_word.py`. Default `livekit`. Sherpa-onnx path stays as rollback.
- Evaluation: record ~30 minutes of Baker-Berry-style background noise + 50 "Hey Spot" positives. Compare false-positive rate + true-positive rate between `sherpa_onnx` and `livekit` paths. Pick the winner. Expect 2-3× improvement, not 60×.

---

## Stage 2B — Spot end-to-end gate (merge moment)

Goal: validate the voice loop end-to-end on the real robot before merging `tour_guide_upgrade_matteo` → `main`. Exit tag: `stage2b-e2e-complete`. Requires Spot powered, E-stop in another terminal, and a clear lab space.

### 2B.1 Phase 5 regression matrix

- stand → walk forward 1 m → sit
- "what do you see?" — VLM via llama.cpp + gemma4 + mmproj
- "go to <known location>" — GraphNav with a known waypoint
- "follow me" then "stop" — uses current (broken) YOLO follow path; verify the safety stop works even if perception flakes. Vision swap (2C) follows separately.
- Smoke door + smoke follow last (riskier paths per `docs/project/status.md`); these are exploratory and may not pass.
- Safety review: dispatch the `safety-reviewer` agent on any new code touching `spot_dispatch.py`, `intent.py`, `llm_brain.py` action handling, or `estop_run.py`. E-stop coverage must remain comprehensive.
- **Pass criteria:** all five non-exploratory items succeed without manual intervention. Door + follow can fail without blocking the merge (will be addressed in future plans).
- **On pass:** merge `tour_guide_upgrade_matteo` → `main` via PR. Tag `stage2b-e2e-complete` on the merge commit on `main`.

---

## Stage 2C — Vision overhaul

Goal: bring perception up to the Phase 9.B revised baseline — per-camera rotation, fisheye undistortion, gripper plumbing, batched 5-camera scan, `look_around` action, TensorRT YOLO26-N + YOLOE26-S integration. Exit tag: `stage2c-vision-complete`. Parallel-safe with 2D (different files).

### 2C.0 Per-camera rotation table + Frame namedtuple

- Add `CAMERA_ROTATION_DEG = {"frontleft_fisheye_image": -90, "frontright_fisheye_image": -90, "left_fisheye_image": 0, "right_fisheye_image": 180, "back_fisheye_image": 0}` next to `CAMERA_SOURCES` in `src/voice_control/spot_dispatch.py:35`.
- Add `_rotate_for_source(pil_image, source_name)` helper.
- Define `Frame(NamedTuple)` with `pil: PIL.Image.Image`, `jpeg_bytes: bytes`, `source: str`.
- Rewrite `capture_frame` to return a `Frame` (rotate + RGB-convert + JPEG-encode once). Fixes the three independent describe-path bugs documented in `upgrade-plan.md` (wrong camera coverage, wrong rotation everywhere except front, VLM seeing sideways images).

### 2C.0.5 Fisheye undistortion (NEW)

- Add `_undistort_for_source(pil_image, source_name)` helper alongside `_rotate_for_source`.
- At startup (e.g., in `session.py` or `spot_dispatch.py` init), iterate `CAMERA_SOURCES` and precompute per-camera `cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (cols, rows), cv2.CV_16SC2)` using the intrinsics + distortion params already cached by `src/voice_control/perception/camera_intrinsics.py`. Branch on `ImageSource.HasField("pinhole_brown_conrady")` vs `kannala_brandt` vs `pinhole`.
- Per-frame `cv2.remap(numpy_array, map1, map2, cv2.INTER_LINEAR)` — ~3-5 ms CPU on Orin. (No `cv2.cuda` dependency; runtime agent confirmed cv2 build is the standard pinned `opencv-python==4.11.0.86`.)
- Update `capture_frame` order: capture → rotate → undistort → output `Frame`.
- For `pinhole` cameras (no distortion params), skip the undistort step.
- Implication: YOLO and VLM downstream both consume undistorted, rotated frames. No code in `visual_nav.py` needs to change for this — the upstream `Frame` change carries the fix.

### 2C.1 Gripper color camera plumbing

- Add `hand_color_image` to `CAMERA_SOURCES`.
- New RGB_U8 decode branch in `capture_frame` (raw bytes → `Image.frombytes`, NOT `Image.open`).
- Add `gripper_camera_available()` pre-flight gate (requires arm powered). The only color sensor on this Spot — body fisheyes are monochrome.

### 2C.2 Batched 5-camera scan

- Add `_find_object_batch(pil_images, description) -> list[bbox-or-None]` wrapper on the detector.
- Add `_scan_all_cameras(session, query)` helper.
- Replace the serial scan loop in `src/voice_control/visual_nav.py:595-609` with a single batched `_capture_multi` RPC + a single batched detector call. **Must land before any code assumes the scan is batched** — Tier 2a's approach loop already assumes the scan was done; Stage 2C just changes how. Reduces worst-case scan from ~5 s to ~400-600 ms after 2C.4.

### 2C.3 `look_around` action

- Add `look_around` to the action enum + dispatcher in `spot_dispatch.py`.
- Implementation: capture frames from all 5 body cameras + gripper if available → batched VLM call → return N independent per-camera descriptions.
- Wire to llm_brain action-selection prompt + GBNF grammar (2A.0) so it is dispatchable from a "look around" utterance.

### 2C.4 TensorRT YOLO26 + YOLOE26 swap

- `pip install ultralytics --upgrade` (verify version supports YOLO26 + YOLOE26 export; pin once verified). Use `--no-deps` if it tries to rewrite numpy.
- Download checkpoints: `yolo26n.pt` (~7 MB, follow loop) + `yoloe26s-seg.pt` (~25 MB, object scan).
- Export both to TensorRT FP16:
  ```python
  from ultralytics import YOLO
  YOLO('yolo26n.pt').export(format='engine', half=True, device=0, dynamic=True)
  YOLO('yoloe26s-seg.pt').export(format='engine', half=True, device=0, dynamic=True)
  ```
- Move resulting `.engine` files to `/mnt/ssd/cfm_mppi_caches/yolo_engines/`.
- In `visual_nav.py`, replace `_get_world_model()` / `_get_person_model()` to load via `YOLO('/mnt/ssd/cfm_mppi_caches/yolo_engines/yoloe26s-seg.engine')` / `YOLO('.../yolo26n.engine')`. Ultralytics auto-detects engine load.
- **Verify TRT FP16 numerics:** ms/frame and mAP on a held-out set of 20-30 captured Spot frames. Expect < 1 % mAP delta vs the `.pt` baseline. `tegrastats --interval 1000` MUST show GPU utilization spike during inference (NOT 0 % — 0 % means engine silently failed and Python fell back to CPU; debug before merging).
- **Engine cache invalidation:** any JetPack / TRT version bump invalidates `.engine` files. Document in `stage2-rollback.md` that re-export is required on JetPack changes.
- Person-detection bias: YOLO26-N at 382 FPS leaves enormous headroom. If accuracy on Spot fisheye+undistort is insufficient, step up to YOLO26-S (~22 MB, still well under budget).
- AprilTag follow: stays as exploratory backup, NOT primary. The vision agent's "no native fisheye YOLO checkpoint" concern is addressed by 2C.0.5 undistortion. If YOLO26-N follow proves unreliable on the actual robot, AprilTag becomes a Stage 3 spike.

---

## Stage 2D — Smart chatbot

Goal: Phase 6 Stages A, C, D (Stage B knowledge packs already shipped). Persona, web search, FactsStore. Parallel-safe with 2C (different files). Exit tag: `stage2d-chatbot-complete`.

### 2D.A Persona + MAX_HISTORY + conversation log

- Persona prompt prefix that frames gemma4 as a Dartmouth tour guide for Spot. First-person, friendly, knowledgeable about campus.
- Raise `MAX_HISTORY` from 12 to 24 in `llm_brain.py` so multi-turn dialogue persists context.
- Append every (user_transcript, brain_response, action) to `logs/conversations.jsonl` (now symlinked to `/mnt/ssd/spot-logs/` via Stage 1.5).
- Wire `[Brain-timing]` lines to also land in conversations.jsonl so latency telemetry is queryable.

### 2D.C Web search with Tavily + safety mitigations

- Add `tavily-python` dep. Tavily API key in `.env` (gitignored).
- Implement `web_search(query)` tool callable by the brain.
- Five safety mitigations (per upgrade-plan.md original spec — defer to that doc for the exact list; expand here when 2D plan is written):
  1. Allowlist domain filtering for Dartmouth-relevant sources
  2. Rate limiting per session
  3. Result truncation (top 3 hits, 500 chars each)
  4. Prompt-injection scrub on retrieved snippets
  5. Audit log of every search query + result to `logs/web_search.jsonl`

### 2D.D FactsStore SQLite

- New `src/voice_control/facts_store.py` — SQLite at `/mnt/ssd/spot-logs/facts.db`. Schema: `(key TEXT PRIMARY KEY, value TEXT, source TEXT, captured_at TIMESTAMP)`.
- Round-trip API: `facts.remember(key, value, source)`, `facts.recall(key)`, `facts.search(query) -> list[(key, value)]`.
- Hook the brain so a user can say "remember that Dartmouth's library is called Baker-Berry" and the next "where's the library?" query uses the fact.

---

## Open eval gates (decision data captured during execution)

| Gate | Decision data | Owner sub-plan |
|---|---|---|
| llama.cpp vs Ollama as primary brain | Mic-verify pass/fail + latency p50/p95 (action + free-form paths) | 2A.3 |
| Adaptive-thinking router accuracy | % of utterances classified to correct grammar route (95 % gate) | 2A.3 |
| LiveKit Wakeword vs openWakeWord vs sherpa-onnx (rollback) | FP rate + TP rate on Baker-Berry recording | 2A.4 |
| YOLO26-N follow on undistorted fisheye vs AprilTag spike | Track quality + lab walk-around test | 2B.1 (initial), Stage 3 spike if 2B.1 fails follow |
| YOLOE26-S prompt-free mode vs explicit class list | Detection accuracy on 30 tour-guide objects | 2C.4 |
| Tavily web_search safety mitigation effectiveness | Manual red-team queries | 2D.C |

---

## Rollback architecture

- **Code:** tags `stage1.5-complete` (current) → `stage2a-voice-complete` → `stage2b-e2e-complete` (also merge-to-main moment) → `stage2c-vision-complete` ∥ `stage2d-chatbot-complete` → `stage2-complete`. Each tag is a destructive-restore point on `tour_guide_upgrade_matteo` (or on `main` after merge).
- **Brain runtime:** `SPOT_BRAIN_BACKEND=llamacpp|ollama` env var. llama.cpp is default; Ollama is hot fallback.
- **Brain model:** `DEFAULT_MODEL` + `VLM_MODEL` in `llm_brain.py` (2-line edit) flips between `gemma4:e4b` (primary) → `qwen2.5:7b` + `qwen2.5vl:7b` (Stage 1 rollback, on disk until 2A.3 passes).
- **ASR backend:** `SPOT_ASR_BACKEND=parakeet|riva` env var. Riva backend code stays in `asr/riva_backend.py` until 2A.4 deletes it (Riva server itself is gone; backend code is a deletion deferral).
- **Wake word:** `SPOT_WAKE_BACKEND=livekit|sherpa_onnx` env var. Sherpa-onnx stays as rollback.
- **Vision models:** `.pt` checkpoints stay on disk alongside `.engine` files. Loading branches on file extension via ultralytics.
- **Runbook:** `docs/project/stage2-rollback.md` written incrementally as each sub-plan ships. Layers: code, brain runtime, brain model, ASR backend, wake-word backend, vision engines, SSD failure (covered in stage1-rollback.md Stage 1.5 update).

---

## Out of scope (deferred to future plans)

- **AprilTag follow mode** — exploratory backup only. If 2B.1 follow regression fails on YOLO26-N+undistort, a Stage 3 spike validates AprilTag tag36h11 perception via Spot's built-in `WorldObjectClient`. Out of Stage 2 because the unblock path (YOLO26-N on undistorted frames) is direct.
- **`parakeet_realtime_eou_120m-v1`** — emits `<EOU>` tokens for interruptible-mode voice agent. Future plan once basic voice loop works.
- **vLLM / TRT-LLM brain runtime** — vLLM has the same JSON bug class as Ollama; TRT-LLM Jetson v0.12.0 has no confirmed gemma4 support. Revisit when either ships a fix or gemma4 support, respectively.
- **Two-brain split** (fast text LLM for action dispatch + heavy VLM only when "describe" intents fire) — research agent flagged as Stage 3 optimization once latency telemetry confirms whether single-brain llama.cpp falls short.
- **qwen3-vl:8b-instruct as primary brain** — research agent recommended swap (no thinking-mode bug, unified VLM, BFCL top-tier). User opted to keep gemma4 as primary and note qwen3-vl as Stage 3 alternative.
- **MiniCPM-V 4.6 / InternVL3.5 as VLM sidecar** — Stage 3 if single-brain VLM proves insufficient.
- **Door opening** — on ice per `MEMORY.md`; needs Spot testing with `--vlm --depth 0.5`. Separate plan.
- **Stair-aware follow mode** — Spot ascends forward, descends backwards. Future plan after basic follow ships.
- **Fisheye YOLO fine-tune** — option only if AprilTag spike also fails. Fine-tune YOLO26-N on FishEye8K (~10 k labeled images, free).
- **Ollama 0.25+ / PR #15678 landing for bug #15260 fix** — irrelevant once llama.cpp is primary.
- **`onnxruntime-gpu` upgrade past 1.23.0** — no upgrade path available (no newer wheel on Jetson AI Lab for `jp6/cu126/cp310`). Cascade-breaks ABI pins if attempted.

---

## Hard ordering constraints

1. 2A.0 (brain runtime swap) MUST come before 2A.3 (mic-verify) — mic-verify validates the new runtime.
2. 2A.2 (Parakeet swap) MUST come before 2A.3 (mic-verify) — mic-verify uses Parakeet as the front of the pipeline.
3. 2A.3 (mic-verify pass) MUST come before 2A.4 step 1 (`riva_backend.py` deletion) — Riva backend stays as rollback until Parakeet is proven.
4. 2A.3 (mic-verify pass) MUST come before any `ollama rm qwen2.5*` — qwen pair is the brain rollback.
5. 2A complete MUST come before 2B — Spot e2e needs working voice.
6. 2B complete MUST come before merge to `main` — merge gate.
7. 2C and 2D are independent of each other; both depend on 2B complete (so they run on `main`).

---

## Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| llama.cpp build fails on JP6.2.1 + CUDA 12.6 + cuDNN 9.3 | Low | Medium (delays 2A) | dustynv/jetson-containers has prebuilt llama.cpp images; fall back to a prebuilt if from-source fails |
| llama-server crashes / OOMs mid-conversation | Low | Medium | systemd `Restart=on-failure` + 5 s backoff. Dispatcher catches HTTP 5xx + connection-refused and surfaces to user as "brain restarting, retry". Ollama fallback via env var if llama.cpp is unstable for the session. |
| OpenAI API compat is ~90 % not 100 % (llama-server extensions vs Ollama quirks) | High | Low | `llm_brain.py` branches on `SPOT_BRAIN_BACKEND` for request shape, not just URL. Document llama.cpp-specific fields (`grammar`, `cache_prompt`, `n_predict`, `samplers`) in code comments. |
| Adaptive-thinking router misclassifies intent (sends "describe" to action path or vice versa) | Medium | Medium | Router is a small regex/keyword classifier — easy to iterate. Mic-verify 2A.3 measures router accuracy directly (95 % gate). On miss, tighten classifier patterns. |
| `action.gbnf` enum-of-actions drifts out of sync with `spot_dispatch.py` action list | Medium | Medium | Generate `action.gbnf` from the same source of truth as `spot_dispatch.py` action enum (script in `scripts/generate_action_gbnf.py`). Or assert at startup that both match. |
| GBNF grammar rejects valid JSON the brain produces | Medium | Medium | Iterate GBNF on smoke utterances before mic-verify; loosen rules where the grammar over-constrains |
| Per-call grammar bandwidth (verbose JSON in request) impacts latency | Low | Low | Localhost loopback handles MB-scale JSON in sub-ms. Grammar text is ~2-5 KB — negligible. |
| Speculative decoding (E2B drafter) regresses output quality | Low | Medium | Disable speculative (`--draft-max 0`) — fall back to non-speculative latency, still bypasses Ollama bug |
| Cold-start spike (10-30 s GGUF mmap + KV alloc) blocks first request after restart | Medium | Low | llama-server runs as long-lived daemon (systemd unit). `run_voice_control.py` startup waits for `/health` endpoint OK before accepting voice input. |
| gemma4:e4b mic-verify still fails (action quality, not latency) | Medium | High (forces rollback to qwen pair) | Cascade in 2A.3 fail-recovery: tighten router → loosen GBNF → try Q6_K → Q8_0 → rollback to qwen2.5 in Ollama |
| Parakeet WER worse than Riva baseline in noisy environment | Low | Medium | Tune VAD threshold; fall back to Riva via `SPOT_ASR_BACKEND` env (Riva backend code present until 2A.4) |
| LiveKit Wakeword Hey-Spot model fails on student-crowd recording | Medium | Low | Sherpa-onnx KWS stays as rollback via env var |
| YOLO26 / YOLOE26 not yet pip-installable via ultralytics | Low | Medium | Confirm pip support during 2C.4; if not, install ultralytics from git main; fall back to YOLO11 + YOLO-Worldv2 if necessary |
| TRT FP16 numerics regress YOLO mAP > 1 % | Low | Low | Re-export FP32 engines; pay the latency cost |
| Engine cache invalidates on JetPack bump | Medium | Low | Document in stage2-rollback.md; engine re-export is a 2-minute step |
| cv2 fisheye undistortion crops too much of corners | Medium | Low | Use `cv2.getOptimalNewCameraMatrix(K, D, ...)` with `alpha=1` to preserve full FOV (black borders OK for YOLO) |
| Tavily API key compromise / unexpected billing | Medium | Low | Rate limit per session; alert on > 50 queries/day; key in .env not committed |
| Robot firmware update during Stage 2 breaks bosdyn 5.0.1.1 pins | Low | High | Coordinate firmware updates outside Stage 2 window; pin-guardian enforces |
| Vision swap regresses follow mode on Spot demo | Medium | High | 2B.1 catches before merge; rollback via env-var-swappable model paths |
| Disk pressure resumes (SSD models bloat) | Low | Medium | Stage 1.5 reclaim leaves 851 GB SSD headroom; monitor with `df` in conversations.jsonl |

---

## Self-review

**Spec coverage:** Every section of the original spec's "Stage 2 — Peripheral-dependent work" (2.1-2.7) is mapped: 2.1 mic-verify → 2A.3, 2.2 audio corpus → 2A.1, 2.3 Parakeet swap → 2A.2, 2.4 cleanup + wake-word → 2A.4, 2.5 Phase 5 → 2B.1, 2.6 vision overhaul → 2C.0+0.5+1+2+3+4, 2.7 chatbot → 2D.A+C+D. New additions: 2A.0 (brain runtime swap from research), 2C.0.5 (fisheye undistortion). No section dropped.

**Placeholder scan:** No TBD / TODO / "implement later" anywhere in tasks. The "five safety mitigations" in 2D.C are listed enumerated; full detail deferred to the 2D implementation plan, which is acceptable per the design-spec / implementation-plan split.

**Internal consistency:** llama.cpp + per-call GBNF in 2A.0 is consistent with the JSON-parses-on-100 %-of-calls pass criterion in 2A.3 (GBNF guarantees it on the action path). Adaptive-thinking routing in 2A.0 is consistent with the per-route latency split in 2A.3 (action path < 2 s no thinking, free-form path < 6 s with thinking). HTTP server architecture is consistent with the `SPOT_BRAIN_BACKEND` env-var rollback in the rollback architecture section. Parakeet backend interface in 2A.2 Phase A is the same `Backend` shape referenced by the env-var rollback in 2A.4. The `Frame` namedtuple introduced in 2C.0 carries the undistortion fix from 2C.0.5 to all downstream visual_nav consumers transparently. Pin status table is consistent with `requirements.txt` + pin-guardian agent. AGPL-3.0 YOLO posture is consistent with current YOLOv8 baseline.

**Scope check:** 4 sub-plans, each independently shippable. 2A is the broadest (5 sub-tasks); could split to 2A-brain (2A.0 + 2A.3) and 2A-asr (2A.1 + 2A.2 + 2A.4) if the writing-plans skill prefers smaller chunks. Defer that split to the writing-plans phase.

**Ambiguity check:** "valid JSON for 5/5 smoke utterances" in 2A.0 = no parse errors on the brain's output. "90 % exact action match" in 2A.3 = string-equal on the action name field. "GPU utilization spike" in 2C.4 = `tegrastats --interval 1000` shows > 0 % GR3D_FREQ during inference. All numeric thresholds are explicit.
