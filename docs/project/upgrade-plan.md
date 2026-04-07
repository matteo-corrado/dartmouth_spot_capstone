# Spot Voice Pipeline Upgrade — Slim STT + Gemma 4 (Implementation Plan)

## Context

The Dartmouth Spot capstone runs a full voice‑controlled robot pipeline today and the full pipeline IS enabled by default:

```
Mic → VAD → KWS (sherpa-onnx) → ASR bridge (server.py) → Riva / Canary-Qwen-2.5B (Docker)
    → LLM brain (Ollama qwen2.5:7b) → Dispatcher (28 BD SDK actions) → TTS (Kokoro)
                                          ↘ VLM (Ollama qwen2.5vl:7b)
```

**Goals:**
1. Drop NVIDIA Riva (multi‑GB Docker, Canary‑Qwen 2.5B) for a smaller in‑process ASR.
2. Replace the LLM brain AND collapse the separate VLM into a single multimodal model.
3. Keep `server.py` as the ASR backend boundary so we can swap engines without touching `client_mic.py`.
4. Verify the pipeline still works end‑to‑end after each swap.

**Independent validation:** NVIDIA staff in their own developer forums now tell Jetson users Riva is deprecated. The Riva successor (Speech NIM 26.02.0) doesn't list Jetson AGX Orin in its support matrix. JPS contains zero speech components. Dropping Riva is the right call.

---

## Chosen approach

**STT:** Run a three‑way tournament behind `server.py` and pick the winner from recordings of Spot's actual mic. Candidates:
1. **sherpa‑onnx SenseVoice‑Small int8** — easiest install, same runtime as KWS+TTS, structurally hallucination‑resistant
2. **IBM Granite 4.0 1B Speech with keyword biasing** — newest model (Mar 6 2026), Apache 2.0, #1 OpenASR leaderboard, keyword biasing tuned to Spot's command vocabulary is the killer feature
3. **NeMo Parakeet‑TDT‑0.6b‑v3 via `onnx-asr`** — best WER/size ratio, hallucination‑hardened by training, pip‑installable on JP6.2

Fallback: `distil-large-v3.5-ct2` in `dustynv/faster-whisper` container.

**LLM + VLM:** Collapse both `qwen2.5:7b` and `qwen2.5vl:7b` into a single **`gemma4:e4b`** (9.6 GB on disk, multimodal text+image+audio, native function calling, Apache 2.0, designed for Jetson Orin‑class hardware).

**Gemma 4 unified audio:** Side experiment behind a feature flag — log `(audio, dedicated_stt_transcript, gemma_transcript, action_json)` tuples in production. Promote to unified only when [ollama#15333](https://github.com/ollama/ollama/issues/15333) closes AND measurements meet a defined bar. Don't block the main upgrade on it.

---

## Implementation order

Do these in sequence — each step is independently verifiable and revertable.

### Phase 1 — Branch & baseline

1. `git checkout -b swap-models` from `main`
2. With current code, run baseline: E‑Stop in one terminal, `python scripts/run_voice_control.py` in another. Confirm everything works today.
3. **Capture regression test data:**
   - 10‑15 representative voice utterances (clean speech): "stand up", "go to the lab", "walk forward 1 meter", "do you see the blue chair?", "go to kitchen then hallway then lab", "follow me", "battery status", "open the door", "come back", "stop"
   - 5 motor‑idle clips (mic on, robot powered, idle) — for the hallucination check
   - Save as WAVs in `tests/audio/` (create the dir)

### Phase 2 — LLM + VLM collapse (the easy, high‑value win first)

1. `ollama pull gemma4:e4b` — verify Ollama version supports it (`ollama --version` — needs recent build with Gemma 4 support).
2. `src/voice_control/llm_brain.py:26-27` — change:
   ```python
   DEFAULT_MODEL = "gemma4:e4b"
   VLM_MODEL = "gemma4:e4b"
   ```
3. `src/voice_control/llm_brain.py:175-226` — `warm_up_vlm()` becomes a no‑op (single model is already warm). Delete the 1×1 JPEG bytes and the `_vlm_warmed` state machine. Optionally have `query_vlm()` reuse the `process()` path with an image attached since it's the same model now.
4. **Verify in isolation:** `python src/voice_control/llm_brain.py gemma4:e4b` — the existing CLI test at lines 506‑553 works against representative utterances. Confirm:
   - JSON parses cleanly
   - Action selection is correct on the 28‑action catalog
   - VLM path works (separate quick test with a sample image)
5. **Wire test:** `python scripts/run_voice_control.py --skip-services` (Ollama up, Riva still up). Talk through the mic, watch dispatcher logs.
6. Commit: `swap-models: collapse LLM + VLM to gemma4:e4b`

**If gemma4:e4b regresses on action selection,** fall back to `gemma4:e2b` (7.2 GB, smaller but may misfire on rare actions) or `qwen2.5:7b` (the current). Same one‑line revert.

**Defer:** function‑calling refactor (converting the 28‑action catalog to a tool‑call schema). Cleaner final state but bigger diff. Do as a follow‑up after end‑to‑end validation.

### Phase 3 — STT tournament

Wire all three candidates behind `server.py` as selectable backends via env var (`SPOT_ASR_BACKEND=sensevoice|granite|parakeet`).

1. **Refactor `server.py` to a backend interface.** Extract a `Backend` abstract class with `transcribe(pcm_bytes: bytes) -> str`. Move the existing Riva logic into a `RivaBackend` (kept temporarily as the rollback path). Keep:
   - The hallucination filter blocklist at lines 35‑55 (`HALLUCINATION_PHRASES`, `_is_repetitive`)
   - `MIN_AUDIO_DURATION` at line 42
   - The post‑processing filters at lines 196‑202
   - The gRPC service interface (`StreamingRecognize`)

2. **`SenseVoiceBackend`** (`src/voice_control/asr/sensevoice.py`, ~80 LOC):
   - `pip install sherpa-onnx` is already in `requirements.txt`
   - Download model via new `scripts/setup_sensevoice.py` (mirrors `setup_kws.py` / `setup_kokoro.py`)
   - Wraps `sherpa_onnx.OfflineRecognizer.from_sense_voice(...)`

3. **`GraniteBackend`** (`src/voice_control/asr/granite.py`, ~120 LOC):
   - `pip install transformers torchaudio soundfile` (Granite requires Transformers ≥ 4.52.1)
   - Download via `scripts/setup_granite.py`
   - Build a Spot keyword list from `locations.json` + the 28‑action catalog at construction time
   - Inject `# Keywords: stand, sit, walk, ..., podium, couch, lab, ...` into the Granite prompt
   - Heaviest of the three runtimes (vanilla PyTorch, no native TensorRT) — measure RTF on Jetson before promoting

4. **`ParakeetBackend`** (`src/voice_control/asr/parakeet.py`, ~80 LOC):
   - `pip install onnx-asr[hub]` + `onnxruntime-gpu` from `https://pypi.jetson-ai-lab.io/jp6/cu126`
   - Loads `parakeet-tdt-0.6b-v3` ONNX export via `onnx_asr.load_model(...)`
   - Should work without NeMo or PyTorch

5. **Test harness** — `scripts/test_stt.py` (~50 LOC):
   - Loads each backend, runs against `tests/audio/*.wav`
   - Reports per‑backend: WER vs Riva baseline transcript, hallucination chars on idle clips, RTF
   - Picks a winner

6. **Run the tournament** on Jetson with the captured audio. Pick the winner.

7. **Set the default** — `SPOT_ASR_BACKEND=<winner>` becomes the default in `server.py`. Other backends stay as switchable options.

8. **Verify wired pipeline:** `python scripts/run_voice_control.py --skip-services`. Talk through the mic.

9. Commit: `swap-models: replace Riva with <winner> behind server.py backend interface`

### Phase 4 — Riva removal

Only after end‑to‑end Spot test passes (Phase 5).

1. Delete `RivaBackend` from `server.py`
2. Delete `scripts/setup_riva.sh`
3. `scripts/run_voice_control.py:31-32, 90-116, 269-273` — delete `RIVA_CONTAINER`, `RIVA_PORT`, `ensure_riva()`, the Riva startup branch in `main()`. Keep `start_asr_server()` because `server.py` still runs as a subprocess; it just no longer needs Riva running first.
4. `requirements.txt:13` — drop `nvidia-riva-client>=2.17.0`
5. Update `README.md` and `docs/architecture/overview.md` — new pipeline diagram, no Riva quickstart step, mention `gemma4:e4b` and the chosen STT in the model list
6. Update `docs/project/status.md` with the new model lineup
7. **Reclaim disk** (do last, after everything else passes): stop and remove the `riva-speech` Docker container, remove the Riva image (`docker rmi`), delete `~/riva_quickstart_arm64_v2.17.0` and `~/riva_quickstart_arm64_v2.17.0.tar.gz`. Frees several GB.
8. Commit: `swap-models: remove Riva orchestration and dependencies`

### Phase 5 — Full end‑to‑end with Spot

E‑Stop running, robot powered, run the pipeline:
- stand → walk forward 1m → sit
- "what do you see?" (verifies VLM path through gemma4:e4b)
- "go to <known location>" (verifies GraphNav)
- "follow me" then "stop" (verifies follow + safety command)
- Smoke‑test door + follow last — already flagged as the riskier paths in `docs/project/status.md`

If everything passes: merge `swap-models` to `main`.

### Phase 6 (optional, deferred) — Gemma 4 unified audio side experiment

1. Add `UNIFIED_AUDIO_LOG=true` env var in `client_mic.py`
2. When set, after a successful staged transcription, also send the same audio buffer + the wake‑word‑triggered camera frame to `gemma4:e4b` via Ollama's audio chat API
3. Log `(audio_path, staged_transcript, gemma_transcript, staged_action_json, gemma_action_json)` to `logs/unified_audio_eval.jsonl`
4. Run for a week of dev use, then analyze
5. Promote unified to production only when ALL of:
   - [ollama/ollama#15333](https://github.com/ollama/ollama/issues/15333) is closed
   - 0 crashes in 1000 prompts
   - WER within ~20% of dedicated STT on Spot's environment
   - Action‑selection agreement > 95% on the captured corpus

---

## Files touched

| Status | Path | Notes |
|---|---|---|
| Modified | `src/voice_control/server.py` | Refactor to backend interface; carry over hallucination filters |
| New | `src/voice_control/asr/__init__.py` | Backend registry |
| New | `src/voice_control/asr/sensevoice.py` | sherpa‑onnx SenseVoice backend |
| New | `src/voice_control/asr/granite.py` | IBM Granite 4.0 1B Speech backend with keyword biasing |
| New | `src/voice_control/asr/parakeet.py` | Parakeet‑TDT‑0.6b‑v3 via onnx‑asr backend |
| Modified | `src/voice_control/llm_brain.py:26-27` | `DEFAULT_MODEL = VLM_MODEL = "gemma4:e4b"` |
| Modified | `src/voice_control/llm_brain.py:175-226` | `warm_up_vlm()` no‑op; delete 1×1 JPEG state |
| Modified | `scripts/run_voice_control.py:31-32, 90-116, 269-273` | Delete Riva orchestration |
| New | `scripts/setup_sensevoice.py` | Model download (mirrors `setup_kws.py`) |
| New | `scripts/setup_granite.py` | Model download |
| New | `scripts/setup_parakeet.py` | Model download |
| New | `scripts/test_stt.py` | Tournament harness |
| New | `tests/audio/*.wav` | Regression test corpus |
| Deleted | `scripts/setup_riva.sh` | After Phase 4 |
| Modified | `requirements.txt` | Drop `nvidia-riva-client`; add `transformers>=4.52.1`, `torchaudio`, `soundfile`, `onnx-asr[hub]` |
| Modified | `README.md` | New pipeline diagram + setup steps |
| Modified | `docs/architecture/overview.md` | New architecture |
| Modified | `docs/project/status.md` | New model list |

**Untouched:** `client_mic.py`, `spot_dispatch.py`, `wake_word.py`, `spot_tts.py`, `intent.py`, `visual_nav.py`, `web_panel.py`, `door_mission_service.py`, the 28 dispatch actions. The point of the `server.py` boundary is precisely that the mic client doesn't need to know we swapped backends.

---

## Verification plan (per phase)

| Phase | Verification |
|---|---|
| 1 | Baseline `run_voice_control.py` works with current code. Audio corpus captured. |
| 2 | `python src/voice_control/llm_brain.py gemma4:e4b` passes representative utterances. Wired pipeline (Riva still up) talks correctly through mic. |
| 3 | `scripts/test_stt.py` reports WER vs Riva and hallucination chars per backend. Wired pipeline passes a 10‑command mic test on the winning backend. |
| 4 | `run_voice_control.py` starts cleanly with no Riva orchestration. No import errors. No `nvidia-riva-client` references remain in `requirements.txt`. |
| 5 | Spot completes the end‑to‑end exercise sequence. |
| 6 | (Optional) `logs/unified_audio_eval.jsonl` populates without crashes; WER and agreement metrics meet the bar before any production switch. |

---

## Rollback

Every phase is a single commit on the `swap-models` branch. Rollback options:

- **Phase 2 fails:** revert the `llm_brain.py` commit; back to qwen2.5:7b + qwen2.5vl:7b
- **Phase 3 fails on the chosen STT:** switch `SPOT_ASR_BACKEND` env var to a different backend (no code change). All three are wired in parallel.
- **Phase 3 fails on all three STTs:** keep `RivaBackend` as the active backend; the gRPC interface is unchanged.
- **Phase 4 already done and need to roll back further:** revert the deletion commit; Riva orchestration returns. Riva Docker container/image stays installed until disk reclaim is done in step 7.

The Riva Docker container and the `~/riva_quickstart_arm64_v2.17.0` directory should NOT be deleted until end‑to‑end validation passes in Phase 5.

---

## Constraints to remember

- **Jetson AGX Orin (JetPack 6.2.1, CUDA 12.6)** — install via `https://pypi.jetson-ai-lab.io/jp6/cu126` for torch/cuda‑python wheels when needed (Granite, Parakeet routes)
- **E‑Stop must run separately** in another terminal — enforced in `run_voice_control.py:264`
- **Models share VRAM** — Gemma 4 e4b uses ~6‑8 GB at Q4_K_M; YOLO + ROS need headroom; the 64 GB unified memory has space
- **Granite uses vanilla PyTorch on Jetson** (no native TensorRT path) — measure RTF on the actual device before committing to it as the winner
- **Ollama version requirement** — Gemma 4 needs a recent Ollama build; verify before pulling

---

## Reference: research files

Full per‑candidate writeups with sources, sizes, licenses, install recipes, and benchmark numbers live in `~/.claude/plans/`:

- `warm-tickling-lynx-agent-a4f9746143ff4295e.md` — STT round 1 (initial shortlist)
- `warm-tickling-lynx-agent-af13db9dca1f5afe4.md` — STT round 2 (broader Jetson survey including Whisper variants)
- `warm-tickling-lynx-agent-ae3ddd41278a6e1e1.md` — STT round 3 (cutting‑edge 2025/2026 releases — Granite 4.0, Kyutai STT, Moonshine v2)
- `warm-tickling-lynx-agent-a3da18f8ff79e733b.md` — JPS investigation
- `warm-tickling-lynx-agent-a5a0df3a83d24ff06.md` — Gemma 3 vs 4 comparison and license analysis
- `warm-tickling-lynx-agent-aa29a47c34e178710.md` — Multimodal-audio LLM survey (Gemma 4 audio, Phi‑4‑MM, Voxtral, Qwen Audio, etc.)
