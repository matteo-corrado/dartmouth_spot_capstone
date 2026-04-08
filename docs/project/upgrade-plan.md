# Spot Voice Pipeline Upgrade — Slim STT + Gemma 4 + GPU TTS + Smart Chatbot (Implementation Plan)

## Context

The Dartmouth Spot capstone runs a full voice‑controlled robot pipeline today and the full pipeline IS enabled by default:

```
Mic → VAD → KWS (sherpa-onnx) → ASR bridge (server.py) → Riva / Canary-Qwen-2.5B (Docker)
    → LLM brain (Ollama qwen2.5:7b) → Dispatcher (28 BD SDK actions) → TTS (Kokoro, CPU)
                                          ↘ VLM (Ollama qwen2.5vl:7b)
```

**Goals:**
1. Drop NVIDIA Riva (multi‑GB Docker, Canary‑Qwen 2.5B) for a smaller in‑process ASR.
2. Replace the LLM brain AND collapse the separate VLM into a single multimodal model.
3. Move Kokoro TTS from CPU to the Jetson GPU so time‑to‑first‑audio shrinks.
4. Make the chatbot itself smarter — stronger persona, static local knowledge, optional web search with safety lockout, and persistent remembered facts.
5. Keep `server.py` as the ASR backend boundary so we can swap engines without touching `client_mic.py`.
6. Verify the pipeline still works end‑to‑end after each swap.

**Independent validation:** NVIDIA staff in their own developer forums now tell Jetson users Riva is deprecated. The Riva successor (Speech NIM 26.02.0) doesn't list Jetson AGX Orin in its support matrix. JPS contains zero speech components. Dropping Riva is the right call.

**Already shipped baseline (2026‑04‑07):** A stop‑gap landed today that pins both `qwen2.5:7b` and `qwen2.5vl:7b` resident in Ollama via a systemd drop‑in (`OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1`) plus matching `keep_alive=-1` in `llm_brain.py` and an eager `brain.warm_up_vlm()` call in `client_mic.py:399-400`. This eliminates first‑describe VLM warm‑up latency on the *current* model pair while Phase 2 below collapses both models into one. The systemd drop‑in stays useful after Phase 2 (becomes `MAX_LOADED_MODELS=1`); the eager `warm_up_vlm()` call simplifies to a no‑op once Phase 2 lands.

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

### Phase 2.5 — TTS to GPU (Kokoro on the Jetson GPU)

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
   - `pip install onnx-asr[hub]` — `onnxruntime-gpu` is already installed by Phase 2.5; do not reinstall.
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

### Phase 6 — Smart chatbot upgrade (parallel `smart-chatbot` branch)

Branched off `main` *after* Phase 2 (gemma4:e4b collapse) is merged. Can run in parallel with Phases 2.5–5 on the `swap-models` branch — it touches different files (`llm_brain.py` is the main overlap, but only in the `SYSTEM_PROMPT` and `process()` regions, which don't conflict with the Phase 2 model‑name swap).

The persona, knowledge, and tool‑schema work below is tuned against `gemma4:e4b`, so do NOT start Phase 6 against `qwen2.5:7b` even if `swap-models` is still in flight.

#### Stage A — Persona, working memory, conversation log

Smallest, cheapest, most visible improvement on demo feel.

1. `src/voice_control/llm_brain.py:37-102` — rewrite `SYSTEM_PROMPT`. Open with a stronger persona block: *"You are Spot, a four‑legged robot living at Dartmouth's [lab]. Curious, slightly sardonic, fond of the humans you work with, proud of being a robot. You give campus tours, run errands, and answer questions. Keep replies to one or two sentences unless asked to elaborate."* Then a clearly demarcated `# ACTION RULES` section containing the existing 28‑action catalog and JSON contract verbatim. Add an explicit rule: *"Personality lives in the `response` field. The `actions` field is a strict machine‑readable contract — never let personality leak into action params."*
2. `src/voice_control/llm_brain.py:29` — bump `MAX_HISTORY` from 12 to 24.
3. `src/voice_control/llm_brain.py:334-338` — append every `(timestamp, transcript, response, actions, state_snapshot)` to `logs/conversations.jsonl`. Gate behind a `--log-conversations` CLI flag in `client_mic.py` / `run_voice_control.py`, off by default in stranger demos.
4. **Verify:** rerun the regression corpus (`python src/voice_control/llm_brain.py gemma4:e4b` against `tests/audio/`); action selection accuracy must be ≥ baseline. Subjective check on 5 free‑form questions ("how are you?", "what's your favorite hobby?", "tell me about yourself").
5. Commit: `smart-chatbot: persona, MAX_HISTORY bump, conversation log`

#### Stage B — Static local knowledge file

Web search is non‑deterministic and adds latency. Things that don't change daily about Spot's own world should not require a web round trip.

1. New `data/spot_knowledge.json` — building names and aliases, lab members, robot capabilities, mission context, escalation contacts, campus map summary, things Spot is and isn't allowed to do. Target ~300–500 tokens.
2. `src/voice_control/llm_brain.py:228-237` — `_build_messages()` injects the knowledge file into the system prompt. Loaded once at `SpotBrain.__init__` and cached on the instance. Refuses to load if file is over a hard 1500‑token cap (forces it to stay tight).
3. **Verify:** "what's in this lab?" and "who works here?" pull from `spot_knowledge.json` without a network call. Action regression unchanged.
4. Commit: `smart-chatbot: static local knowledge file`

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

**Disk unblock from Phase 4.** After Riva removal frees ~24 GB of Docker image + ~2 GB of quickstart dir, disk pressure goes from "critical" to "comfortable." Improvements #2 and #3 above become much easier to implement and test once that headroom exists.

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
| Modified | `src/voice_control/spot_tts.py` | Phase 2.5: swap sherpa‑onnx Kokoro for kokoro‑onnx with `CUDAExecutionProvider`; voice ID becomes voice name; ~30–50 line diff |
| New | `scripts/setup_kokoro_v1.py` | Phase 2.5: downloads `kokoro-v1.0.onnx` and `voices-v1.0.bin` from kokoro‑onnx GitHub releases |
| New | `models/tts/kokoro-v1.0/` | Phase 2.5: new model directory (replaces `kokoro-en-v0_19/`) |
| Deleted | `models/tts/kokoro-en-v0_19/` | Phase 2.5 step 13: reclaim ~340 MB after end‑to‑end verification |
| Modified | `scripts/run_voice_control.py:31-32, 90-116, 269-273` | Delete Riva orchestration |
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

**Untouched:** `client_mic.py`, `spot_dispatch.py`, `wake_word.py`, `intent.py`, `visual_nav.py`, `web_panel.py`, `door_mission_service.py`, the 28 dispatch actions. The point of the `server.py` boundary is precisely that the mic client doesn't need to know we swapped backends. (Phase 2.5 touches `spot_tts.py` but the public `SpotTTS` API is preserved, so `client_mic.py` still doesn't need edits. Phase 6 keeps `web_search`/`remember`/`forget` action handling entirely inside the brain — the dispatcher does not need new entries.)

---

## Verification plan (per phase)

| Phase | Verification |
|---|---|
| 1 | Baseline `run_voice_control.py` works with current code. Audio corpus captured. |
| 2 | `python src/voice_control/llm_brain.py gemma4:e4b` passes representative utterances. Wired pipeline (Riva still up) talks correctly through mic. |
| 2.5 | `onnxruntime.get_available_providers()` includes `CUDAExecutionProvider`. `[TTS] Ready ... provider=CUDAExecutionProvider` in startup log. `tegrastats` shows non‑zero GPU% during synthesis. Wake word still triggers. Speaker reacts noticeably sooner after `[Brain] LLM responded` than the Phase 1 baseline. `df -h /` shows ≥ 800 MB free after old model deletion. |
| 3 | `scripts/test_stt.py` reports WER vs Riva and hallucination chars per backend. Wired pipeline passes a 10‑command mic test on the winning backend. |
| 4 | `run_voice_control.py` starts cleanly with no Riva orchestration. No import errors. No `nvidia-riva-client` references remain in `requirements.txt`. |
| 5 | Spot completes the end‑to‑end exercise sequence. |
| 6 | Stage A: regression corpus action accuracy ≥ baseline; subjective persona check on 5 free‑form questions. Stage B: lab/who‑works‑here questions answered from `spot_knowledge.json` with no network call. Stage C: happy‑path web question returns sourced answer ≤ ~4 s warm; **action lockout test passes** (search + walk request searches but does not navigate); rate limit enforced; `logs/web_search.jsonl` populated. Stage D: remember/forget round‑trip survives a process restart. |
| 7 | (Optional) `logs/unified_audio_eval.jsonl` populates without crashes; WER and agreement metrics meet the bar before any production switch. |
| 8 | (Optional) Per follow‑up: warm utterance eliminates first‑call CUDA compile cost; sentence chunking measurably reduces time‑to‑first‑audio on long responses; LLM‑stream→TTS measurably reduces total perceived latency without breaking JSON parsing. |

---

## Rollback

Phases 1–5 are commits on the `swap-models` branch; Phase 6 is commits on the parallel `smart-chatbot` branch. Rollback options:

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
- **E‑Stop must run separately** in another terminal — enforced in `run_voice_control.py:264`
- **Disk pressure** — `/dev/mmcblk0p1` is 57 GB total, ~951 MB free (99% full) at the time of writing. Phase 4 Riva removal is the single biggest disk win. Until Phase 4 ships, every install step has to be size‑budgeted; anything over a couple hundred MB needs to free space first. Phase 2.5 is sized to fit within this constraint.
- **Models share VRAM** — Gemma 4 e4b uses ~6‑8 GB at Q4_K_M; YOLO + ROS need headroom; the 64 GB unified memory has space
- **GPU consumers after Phase 2.5** — Riva ASR (until Phase 4), Ollama LLM (changes in Phase 2), Kokoro TTS (after Phase 2.5). The Orin's 64 GB unified memory has headroom but VRAM/compute contention still matters for latency. If Phase 2.5 measurements show TTS synthesis blocked on the LLM, that's the signal to revisit Phase 7 (unified audio) sequencing.
- **Granite uses vanilla PyTorch on Jetson** (no native TensorRT path) — measure RTF on the actual device before committing to it as the winner
- **Ollama version requirement** — Gemma 4 needs a recent Ollama build; verify before pulling
- **Tavily free tier** — ~1k queries/month. The Stage C rate limiter (1 web search per 10 s per session, max recursion 1) makes burning that on a single session implausible, but watch the dashboard during demo days.
- **Web search action lockout is mandatory** — Phase 6 Stage C reverses BD's deliberate "no general internet" choice. The five safety mitigations (in‑code action lockout, prompt grounding rule, source attribution, audit log, rate limit) are not optional. They are how the safety story stays intact.

---

## Reference: research files

Full per‑phase / per‑candidate writeups with sources, sizes, licenses, install recipes, and benchmark numbers live in `~/.claude/plans/`:

**STT + LLM/VLM (Phases 2, 3, 7):**
- `warm-tickling-lynx.md` — original parent plan for the swap‑models branch (Phases 1–5, Phase 7 unified audio)
- `warm-tickling-lynx-agent-a4f9746143ff4295e.md` — STT round 1 (initial shortlist)
- `warm-tickling-lynx-agent-af13db9dca1f5afe4.md` — STT round 2 (broader Jetson survey including Whisper variants)
- `warm-tickling-lynx-agent-ae3ddd41278a6e1e1.md` — STT round 3 (cutting‑edge 2025/2026 releases — Granite 4.0, Kyutai STT, Moonshine v2)
- `warm-tickling-lynx-agent-a3da18f8ff79e733b.md` — JPS investigation
- `warm-tickling-lynx-agent-a5a0df3a83d24ff06.md` — Gemma 3 vs 4 comparison and license analysis
- `warm-tickling-lynx-agent-aa29a47c34e178710.md` — Multimodal‑audio LLM survey (Gemma 4 audio, Phi‑4‑MM, Voxtral, Qwen Audio, etc.)

**TTS to GPU (Phase 2.5 + Phase 8 follow‑ups):**
- `mellow-swinging-llama.md` — full kokoro‑onnx + onnxruntime‑gpu plan with disk budget, code‑change walkthrough, and the deferred TTS improvements (sentence chunking, LLM‑stream→TTS, voice/speed env vars, wake‑word‑on‑GPU rejection)

**Smart chatbot (Phase 6):**
- `hashed-waddling-dragonfly.md` — full smart‑chatbot plan with persona text, knowledge file structure, web_search architecture, the five safety mitigations, FactsStore design, and the BD "Robots That Can Chat" reference

**Already‑shipped baseline (referenced in Context):**
- `steady-questing-spindle.md` — qwen pinning stop‑gap (`OLLAMA_MAX_LOADED_MODELS=2` + `keep_alive=-1` + eager VLM warm). Status: implemented 2026‑04‑07.
