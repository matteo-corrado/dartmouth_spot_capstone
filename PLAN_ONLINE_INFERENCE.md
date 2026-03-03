# Plan: Migrate Spot Voice Pipeline to Dartmouth Cloud APIs

## Summary

Replace all local GPU inference (Riva ASR + Ollama LLM/VLM) with Dartmouth API
calls, freeing ~30-40 GB disk and ~12 GB VRAM on the Jetson AGX Orin 64GB. STT
moves to the Dartmouth speech-recognition REST endpoint. LLM command parsing
moves to `ChatDartmouth` with on-prem `meta.llama-3-1-8b-instruct` (free,
unlimited). VLM scene description moves to a cloud vision model (e.g.
`openai.gpt-4.1-mini-2025-04-14`) via the same `ChatDartmouth` interface. TTS
stays local (Kokoro on CPU, zero VRAM). Add `--output-device` CLI flag for
Bluetooth speaker routing. Remove Riva and Ollama entirely.

**Key decisions** (from planning session):

- STT: Dartmouth API only. No Riva, no offline fallback. Always on campus WiFi.
- LLM: On-prem `meta.llama-3-1-8b-instruct` via `ChatDartmouth` (free, unlimited).
- VLM: Cloud `openai.gpt-4.1-mini-2025-04-14` via `ChatDartmouth` (best vision quality).
- TTS: Keep Kokoro local on CPU. No Dartmouth TTS endpoint exists.
- Remove Riva + Ollama completely. No fallback. Maximum storage recovery.
- Use `langchain_dartmouth` library for LLM/VLM (handles auth automatically).
- Use raw REST for STT (no langchain wrapper exists for speech-recognition).

---

## Dartmouth API Inventory

Reviewed at https://ai-tools.dartmouth.edu/openapi/index.html,
https://dartmouth.github.io/langchain-dartmouth/,
https://dartmouth.github.io/langchain-dartmouth-cookbook

### Endpoints

| Capability | Endpoint / Access | Auth | Notes |
|---|---|---|---|
| **STT** | `POST /api/ai/speech-recognition` | JWT via `DARTMOUTH_API_KEY` | Raw REST, multipart file upload |
| **LLM (on-prem, free)** | `ChatDartmouth(model_name="meta.llama-3-1-8b-instruct")` | `DARTMOUTH_CHAT_API_KEY` | OpenAI-compatible, unlimited |
| **VLM (cloud)** | `ChatDartmouth(model_name="openai.gpt-4.1-mini-2025-04-14")` | `DARTMOUTH_CHAT_API_KEY` | Multimodal, daily token budget |
| **VLM (on-prem, free)** | `ChatDartmouth(model_name="meta.llama-3-2-11b-vision-instruct")` | `DARTMOUTH_CHAT_API_KEY` | Free fallback if budget is tight |
| **TTS** | None | N/A | **Not available** -- keep Kokoro local |
| **Object detection** | `POST /api/ai/object-detection` | JWT | Available (bonus, not used yet) |
| **Embeddings** | `DartmouthEmbeddings(model_name="baai.bge-large-en-v1-5")` | `DARTMOUTH_CHAT_API_KEY` | Available (bonus, not used yet) |

### Authentication (Two Separate Key Systems)

**1. Developer API key** (for raw REST endpoints: STT, object-detection, etc.):
- Obtain from https://developer.dartmouth.edu/keys
- Exchange for JWT: `POST https://api.dartmouth.edu/api/jwt` with key in `Authorization` header
- Use JWT as `Authorization: Bearer <JWT>` on subsequent calls
- JWT expires ~1 hour; code auto-refreshes 5 min early

**2. Chat API key** (for `ChatDartmouth` LLM/VLM, `DartmouthEmbeddings`):
- Obtain from https://chat.dartmouth.edu (see https://rcweb.dartmouth.edu/~d20964h/2024-12-11-dartmouth-chat-api/api_key/)
- Set as env var `DARTMOUTH_CHAT_API_KEY`
- `langchain_dartmouth` handles auth internally
- Cloud models have daily per-user token budget; on-prem models are free

### Available Cloud + On-Prem Models (via `ChatDartmouth.list()`)

| Model ID | Local? | Cost | Capabilities |
|---|---|---|---|
| `meta.llama-3-1-8b-instruct` | Yes | free | text |
| `meta.llama-3-2-11b-vision-instruct` | Yes | free | text, vision |
| `openai.gpt-oss-120b` | No | $ | text (default) |
| `openai.gpt-4.1-mini-2025-04-14` | No | $ | text, vision |
| `anthropic.claude-4-5-sonnet-20250929` | No | $$$ | text, vision |
| `google_genai.gemini-2.5-flash` | No | $ | text |

---

## Architecture After Changes

```
Mic (XVF3800 stereo) --> sounddevice callback --> audio_queue
  --> Wake Word (sherpa-onnx KWS, CPU) -- UNCHANGED
  --> WebRTC VAD + energy gating -- UNCHANGED
  --> PCM buffer accumulation -- UNCHANGED
  --> process_utterance():
      1. Safety regex fast-path (<5ms) -- UNCHANGED
      2. STT: PCM16 --> WAV in memory --> POST to Dartmouth /api/ai/speech-recognition
             (REPLACES: gRPC to server.py --> Riva Docker on port 50051)
      3. LLM: transcript + state --> ChatDartmouth("meta.llama-3-1-8b-instruct")
             (REPLACES: POST to Ollama localhost:11434/api/chat with qwen2.5:7b)
      4. TTS: response text --> Kokoro on CPU -- UNCHANGED (zero VRAM)
      5. Action dispatch --> spot_dispatch.py -- UNCHANGED
      6. VLM (on describe): JPEG + prompt --> ChatDartmouth("openai.gpt-4.1-mini-2025-04-14")
             (REPLACES: POST to Ollama qwen2.5vl:7b)
```

**What stays local (CPU only, ~380 MB total):**
- Wake word: sherpa-onnx zipformer KWS (~5 MB)
- VAD: WebRTC VAD (negligible)
- TTS: Kokoro kokoro-en-v0_19 (~340 MB)
- YOLO: yolov8s-worldv2 + yolov8n (~35 MB) for visual nav
- Audio feedback beeps (numpy-generated)

**What moves to cloud (0 VRAM, 0 disk for models):**
- STT: Dartmouth speech-recognition API
- LLM: Dartmouth on-prem llama-3-1-8b-instruct
- VLM: Dartmouth cloud openai.gpt-4.1-mini-2025-04-14

---

## Implementation Steps

### Prerequisites (Manual, Before Code)

**P1. Obtain DARTMOUTH_API_KEY**
- Go to https://developer.dartmouth.edu/keys
- Create an API key
- This is used for the STT speech-recognition endpoint (raw REST + JWT)

**P2. Obtain DARTMOUTH_CHAT_API_KEY**
- Go to https://chat.dartmouth.edu
- Follow instructions at https://rcweb.dartmouth.edu/~d20964h/2024-12-11-dartmouth-chat-api/api_key/
- This is used for ChatDartmouth LLM/VLM calls

**P3. Test API connectivity from Jetson**
- Verify the Jetson can reach `api.dartmouth.edu` and `chat.dartmouth.edu`
- Run: `curl -X POST https://api.dartmouth.edu/api/jwt -H "Authorization: YOUR_KEY"`
- Run: quick `ChatDartmouth.list()` to confirm model access

**P4. Test STT endpoint format**
- Record a short WAV file
- POST it to `/api/ai/speech-recognition` with JWT bearer token
- Confirm response format (expect `{"text": "..."}` or similar)

---

### Step 1: Extend `src/config.py` (~25 lines added)

Add all new environment variables with sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `DARTMOUTH_API_KEY` | `""` | Developer API key for JWT exchange (STT) |
| `DARTMOUTH_CHAT_API_KEY` | `""` | Chat API key for ChatDartmouth (LLM/VLM) |
| `LLM_MODEL` | `"meta.llama-3-1-8b-instruct"` | On-prem LLM model name |
| `VLM_MODEL` | `"openai.gpt-4.1-mini-2025-04-14"` | Cloud vision model name |
| `DARTMOUTH_STT_URL` | `"https://api.dartmouth.edu/api/ai/speech-recognition"` | STT endpoint |
| `DARTMOUTH_JWT_URL` | `"https://api.dartmouth.edu/api/jwt"` | JWT exchange endpoint |
| `LLM_TEMPERATURE` | `0.3` | Temperature for command parsing |
| `LLM_MAX_TOKENS` | `150` | Max tokens for LLM response |
| `VLM_MAX_TOKENS` | `200` | Max tokens for VLM description |

---

### Step 2: Create `src/voice_control/dartmouth_auth.py` (new, ~60 lines)

JWT authentication helper for the STT endpoint. The `langchain_dartmouth`
library handles auth for LLM/VLM automatically, but STT is raw REST.

- `DartmouthAuth` class:
  - `__init__(api_key, jwt_url)` -- stores key and cached JWT
  - `get_jwt() -> str` -- POST to JWT URL, cache result, auto-refresh 5 min before expiry
  - `auth_header() -> dict` -- returns `{"Authorization": f"Bearer {jwt}"}`
- Module-level `get_auth()` singleton
- Retry logic: if 401 comes back, force-refresh JWT and retry once

Dependencies: `requests` (already in requirements.txt). No new packages.

---

### Step 3: Create `src/voice_control/dartmouth_stt.py` (new, ~90 lines)

Replace the Riva gRPC ASR path with HTTP POST to Dartmouth speech-recognition.

- `DartmouthSTT` class or module-level `transcribe()` function:
  - Accept `pcm_bytes: bytes` and `sample_rate: int = 16000`
  - Wrap raw PCM16 in a WAV container using stdlib `wave` + `io.BytesIO` (in-memory)
  - POST as `multipart/form-data` with `files={"file": ("audio.wav", wav_buf, "audio/wav")}`
  - Auth: bearer JWT via `dartmouth_auth.get_auth().auth_header()`
  - Parse response JSON for transcript text (handle both `{"text": "..."}` and `{"transcript": "..."}`)
  - Apply existing hallucination filtering (reuse filter list from `server.py`)
  - Apply repetitive text detection
  - Return cleaned transcript string, or `""` on failure
- Timeout: 10 seconds (configurable)
- Error handling: log and return `""` on failure (same contract as current `send_to_asr`)

**Important**: The exact multipart format needs verification against the live API.
If the endpoint expects JSON with base64 audio instead of multipart, adjust accordingly.

Dependencies: None new -- uses `requests` + stdlib `io`, `wave`.

---

### Step 4: Refactor `src/voice_control/llm_brain.py` (~200 lines changed)

Replace Ollama HTTP calls with `ChatDartmouth` from `langchain_dartmouth`.

**4a. LLM (text command parsing)**
- Import `ChatDartmouth` from `langchain_dartmouth.llms`
- Import `SystemMessage`, `HumanMessage`, `AIMessage` from `langchain_core.messages`
- Replace `requests.post(OLLAMA_URL + "/api/chat", ...)` with:
  ```python
  from langchain_dartmouth.llms import ChatDartmouth
  llm = ChatDartmouth(model_name=LLM_MODEL, temperature=0.3, max_tokens=150)
  response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=transcript)])
  raw_json = response.content
  ```
- Keep existing `SYSTEM_PROMPT` with full action catalog + JSON schema unchanged
- For structured JSON: parse `response.content` as JSON (same as current flow).
  If `meta.llama-3-1-8b-instruct` doesn't reliably produce JSON, use
  `with_structured_output()` with a Pydantic model or `create_agent` with
  `response_format` (as shown in langchain-dartmouth cookbook recipe 15).
- Convert existing `_build_messages()` list format to LangChain message objects
- Keep same 12-message history window

**4b. VLM (scene description)**
- Replace Ollama VLM call with cloud vision model:
  ```python
  vlm = ChatDartmouth(model_name=VLM_MODEL, max_tokens=200)
  msg = HumanMessage(content=[
      {"type": "text", "text": prompt},
      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
  ])
  response = vlm.invoke([SystemMessage(content=vlm_system_prompt), msg])
  ```
- Keep existing YOLO hint injection and first-person perspective prompt
- VLM warm-up becomes a no-op (no local model to preload)

**4c. Remove Ollama-specific code**
- Remove `OLLAMA_URL`, `warm_up()`, `warm_up_vlm()`, Ollama health check in `is_available()`
- Remove `keep_alive`, `num_gpu` parameters (Ollama-specific)
- Replace `is_available()` with a lightweight API check (e.g., `ChatDartmouth.list()` or try/except invoke)
- Initialize `ChatDartmouth` instances once in `__init__`, reuse across calls

---

### Step 5: Update `src/voice_control/client_mic.py` (~80 lines changed)

**5a. Replace ASR path**
- Remove `send_to_asr(stub, pcm_bytes)` function
- Remove gRPC imports: `grpc`, `ASRStub`, `StreamingRequest`, `StreamingConfig`, `AudioChunk`
- Remove gRPC channel creation to `localhost:50055`
- In `process_utterance()`, replace:
  ```python
  # OLD:
  transcript = send_to_asr(stub, bytes(speech_buffer))
  # NEW:
  from voice_control.dartmouth_stt import transcribe
  transcript = transcribe(bytes(speech_buffer))
  ```
- Remove `stub` parameter from `process_utterance()` and all call sites

**5b. Add `--output-device` CLI argument**
- Add to argparse:
  ```python
  parser.add_argument("--output-device", type=str, default=None,
      help="Audio output device name/index for TTS and beeps")
  parser.add_argument("--list-output-devices", action="store_true",
      help="List output-capable audio devices and exit")
  ```
- Add device resolver helper:
  ```python
  def _parse_device(dev_arg):
      if dev_arg is None: return None
      try: return int(dev_arg)
      except ValueError: return dev_arg  # string name
  ```
- Pass `output_device` to `SpotTTS(output_device=_parse_device(args.output_device), ...)`
- Reinitialize `beep` singleton: `init_audio_feedback(_parse_device(args.output_device))`
- Add `--list-output-devices` handler that prints output-capable devices and exits

**5c. Remove gRPC server dependency**
- The ASR bridge server (`server.py`) and gRPC channel are no longer needed
- Remove all gRPC channel and stub setup code

---

### Step 6: Update `src/voice_control/audio_feedback.py` (~10 lines changed)

Make the module-level `beep` singleton re-assignable after arg parsing:

```python
# Change from:
beep = AudioFeedback()

# To:
beep = AudioFeedback()  # default (system output)

def init_audio_feedback(output_device=None):
    """Reinitialize the beep singleton with a specific output device."""
    global beep
    beep = AudioFeedback(output_device=output_device)
```

The `AudioFeedback` class already accepts `output_device` and passes it to
`sd.play()`. No changes needed to the class itself.

---

### Step 7: Wire `--output-device` for TTS (actual output code)

`SpotTTS` in `spot_tts.py` already accepts `output_device` and passes it to
`sd.play(device=self.output_device)`. The only change is in the **wiring**.

In `client_mic.py`, TTS initialization (around line 365) changes from:
```python
tts = SpotTTS(on_mute=_mute_mic, on_unmute=_unmute_mic)
```
to:
```python
tts = SpotTTS(
    output_device=_parse_device(args.output_device),
    on_mute=_mute_mic,
    on_unmute=_unmute_mic
)
```

Users invoke as:
```bash
python run_voice_control.py --output-device "bluez_sink.AA_BB_CC_DD_EE_FF"
# or by device index:
python run_voice_control.py --output-device 7
```

To discover devices: `python run_voice_control.py --list-output-devices`

---

### Step 8: Update `scripts/run_voice_control.py` (~100 lines removed/changed)

**Remove entirely:**
- `ensure_riva()` function (Docker container management)
- `ensure_ollama()` function (systemctl start ollama)
- `start_asr_server()` function (subprocess for server.py on port 50055)
- Port-check utilities for 50051, 50055, 11434

**Add:**
- `check_dartmouth_api()` -- quick connectivity test (JWT exchange or HTTP GET)
- `--output-device` argument, forwarded to `client_mic.py`
- Print active configuration at startup (which models are selected)

**Keep:**
- `--skip-services`, `--no-brain`, `--no-tts`, `--no-wake-word`, `--debug-audio`, `--device` flags
- Subprocess launch of `client_mic.py` with flag forwarding

---

### Step 9: Update `requirements.txt`

**Remove:**
- `nvidia-riva-client>=2.17.0` (Riva client no longer needed)
- `grpcio>=1.60.0` (no more gRPC ASR bridge)
- `grpcio-tools>=1.60.0` (no more proto compilation)
- `bosdyn-choreography-client==5.0.1.1` (unused)
- `bosdyn-orbit==5.0.1.1` (unused)

**Add:**
- `langchain-dartmouth>=0.3.0` (ChatDartmouth, model listing, auth)
- `langchain-core>=0.3.0` (message types: SystemMessage, HumanMessage, AIMessage)

**Keep:**
- `bosdyn-client==5.0.1.1`, `bosdyn-mission==5.0.1.1` (Spot SDK)
- `python-dotenv==1.0.1`
- `numpy>=1.19.0,<2.0.0`
- `requests==2.26.0` (STT REST calls, dartmouth_auth)
- `sounddevice>=0.4.6`
- `webrtcvad>=2.0.10`
- `sherpa-onnx>=1.12.0` (Kokoro TTS + wake word)
- `ultralytics>=8.1.0` (YOLO on CPU)

---

### Step 10: Create `.env.template`

```env
# ------ Robot ------
BOSDYN_ROBOT_IP=192.168.80.3
BOSDYN_CLIENT_USERNAME=
BOSDYN_CLIENT_PASSWORD=

# ------ Dartmouth API (developer.dartmouth.edu/keys) ------
DARTMOUTH_API_KEY=your-developer-api-key-here

# ------ Dartmouth Chat API (chat.dartmouth.edu) ------
DARTMOUTH_CHAT_API_KEY=your-chat-api-key-here

# ------ Model selection ------
LLM_MODEL=meta.llama-3-1-8b-instruct
VLM_MODEL=openai.gpt-4.1-mini-2025-04-14

# ------ Audio output (optional) ------
# AUDIO_OUTPUT_DEVICE=bluez_sink.AA_BB_CC_DD_EE_FF
```

---

### Step 11: Create `scripts/cleanup_local_inference.sh`

Cleanup script to remove Riva and Ollama from Jetson:

```bash
#!/bin/bash
set -e
echo "=== Cleaning up local inference stack ==="

# 1. Stop and remove Riva containers + images
echo "Removing Riva..."
docker stop riva-speech 2>/dev/null || true
docker rm riva-speech 2>/dev/null || true
docker rmi nvcr.io/nvidia/riva/riva-speech:2.18.0-l4t-aarch64 2>/dev/null || true
docker rmi nvcr.io/nvidia/riva/riva-speech:2.18.0-servicemaker-l4t-aarch64 2>/dev/null || true
docker volume prune -f
rm -rf ~/riva_quickstart

# 2. Remove Ollama and all its models
echo "Removing Ollama..."
ollama rm qwen2.5:7b 2>/dev/null || true
ollama rm qwen2.5vl:7b 2>/dev/null || true
sudo systemctl stop ollama 2>/dev/null || true
sudo systemctl disable ollama 2>/dev/null || true
sudo apt remove -y ollama 2>/dev/null || sudo rm -f /usr/local/bin/ollama
rm -rf ~/.ollama

# 3. Docker layer cleanup
echo "Pruning Docker..."
docker system prune -af

echo ""
echo "=== Done ==="
echo "Estimated space freed: ~30-40 GB"
echo "Remaining local models: Kokoro TTS (~340MB), KWS (~5MB), YOLO (~35MB)"
df -h /
```

**Estimated space freed:**

| Component | Size |
|---|---|
| Riva Docker images (2x L4T images) | ~15-20 GB |
| Riva model artifacts + TRT engines (~riva_quickstart/) | ~5-10 GB |
| Ollama binary + models (qwen2.5:7b + qwen2.5vl:7b) | ~10-12 GB |
| Docker layer cache | ~2-5 GB |
| **Total freed** | **~30-40 GB** |

**Remaining local model footprint:** ~380 MB (Kokoro 340 MB + KWS 5 MB + YOLO 35 MB). All CPU.

---

### Step 12: Remove/archive unused files

| File | Action | Reason |
|---|---|---|
| `src/voice_control/server.py` | Delete or move to `old_code/` | ASR bridge server, replaced by dartmouth_stt.py |
| `scripts/setup_riva.sh` | Delete | Riva setup, no longer needed |
| `src/voice_control/asr_pb2.py` | Delete | Generated gRPC proto stubs |
| `src/voice_control/asr_pb2_grpc.py` | Delete | Generated gRPC proto stubs |
| `src/voice_control/asr.proto` | Delete or archive | Proto definition for removed ASR bridge |

**Keep:**
- `scripts/setup_kokoro.py` (still needed for Kokoro TTS model download)
- `scripts/setup_kws.py` (still needed for wake word model download)
- `src/voice_control/intent.py` (regex fallback, used with `--no-brain`)

---

### Step 13: Update documentation

| File | Changes |
|---|---|
| `docs/architecture/overview.md` | Update data flow diagram: Dartmouth API instead of Riva/Ollama |
| `docs/architecture/services.md` | Update resource table: remove GPU services, add API endpoints |
| `docs/getting-started/` | Add API key setup instructions, remove Riva/Ollama setup |
| `README.md` | Update quick-start section, prerequisites, architecture diagram |

---

## Verification Plan

| Test | Command / Action | Expected |
|---|---|---|
| **Auth** | `python -c "from voice_control.dartmouth_auth import get_auth; print(get_auth().get_jwt()[:20])"` | JWT string printed |
| **STT** | Record WAV, POST to `/api/ai/speech-recognition`, check response | Transcript returned |
| **LLM** | `ChatDartmouth(model_name="meta.llama-3-1-8b-instruct").invoke("Say hello")` | Response with `.content` |
| **LLM JSON** | Send full system prompt + "turn left 30 degrees" | Valid JSON with `actions` and `response` |
| **VLM** | Send test JPEG to cloud vision model via ChatDartmouth | Scene description returned |
| **Audio output** | `python run_voice_control.py --list-output-devices` | List of output devices |
| **Bluetooth** | `python run_voice_control.py --output-device "bluez_sink.XX"` | TTS plays on BT speaker |
| **End-to-end** | `python run_voice_control.py --no-wake-word`, speak "go to the kitchen" | Full pipeline: STT->LLM->TTS->dispatch |
| **Latency** | Measure end-of-speech to TTS start | Target: <2s on campus WiFi |
| **Cleanup** | Run `scripts/cleanup_local_inference.sh`, check `df -h` | ~30 GB freed |

---

## Open Questions / Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dartmouth STT request format undocumented (multipart? base64?) | Medium | Step 3 includes format probing; test with `curl` first (P4) |
| `meta.llama-3-1-8b-instruct` may not produce reliable JSON | Medium | Use `with_structured_output()` Pydantic schema; or regex JSON extraction as fallback |
| Cloud VLM daily token budget could be exhausted | Low | `describe` is infrequent; can swap to free on-prem `meta.llama-3-2-11b-vision-instruct` |
| Network latency adds ~200-500ms per API call | Expected | Acceptable for conversational robot. Safety commands (stop/estop) remain regex-only, zero latency |
| `grpcio` removal may affect Spot SDK | Low | Spot SDK wheels bundle their own grpcio; test imports after removal |
| JWT expiry timing unknown (assumed ~1 hour) | Low | Auto-refresh 5 min early; adjust if different |
| `langchain_dartmouth` adds transitive deps (langchain-core) | Low | Worth it for auth handling, message types, structured output. ~50 MB total. |

---

## Resource Impact Summary

| Component | Before (Local) | After (Dartmouth API) | Savings |
|---|---|---|---|
| Riva ASR Docker | ~3 GB VRAM, ~20-30 GB disk | 0 (HTTP POST) | 3 GB VRAM, 20-30 GB disk |
| Ollama qwen2.5:7b | ~5 GB VRAM, ~5 GB disk | 0 (HTTP POST) | 5 GB VRAM, 5 GB disk |
| Ollama qwen2.5vl:7b | ~6 GB VRAM, ~6 GB disk | 0 (HTTP POST) | 6 GB VRAM, 6 GB disk |
| Kokoro TTS | 0 VRAM, ~340 MB, CPU | **Unchanged** | 0 |
| Wake word KWS | 0 VRAM, ~5 MB, CPU | **Unchanged** | 0 |
| YOLO (visual nav) | 0 VRAM, ~35 MB, CPU | **Unchanged** | 0 |
| **Total** | ~14 GB VRAM, ~31-41 GB disk | ~380 MB disk, 0 VRAM | **~14 GB VRAM, ~30-40 GB disk** |

---

## New Files

| File | Purpose | Est. Lines |
|---|---|---|
| `src/voice_control/dartmouth_auth.py` | JWT auth helper + caching | ~60 |
| `src/voice_control/dartmouth_stt.py` | Dartmouth speech-recognition wrapper | ~90 |
| `.env.template` | Environment variable template | ~15 |
| `scripts/cleanup_local_inference.sh` | Remove Riva + Ollama from Jetson | ~30 |

## Modified Files

| File | Changes |
|---|---|
| `src/config.py` | Add ~10 Dartmouth/model env vars |
| `src/voice_control/llm_brain.py` | Replace Ollama with ChatDartmouth for LLM + VLM |
| `src/voice_control/client_mic.py` | Replace gRPC ASR with dartmouth_stt, add --output-device |
| `src/voice_control/audio_feedback.py` | Add `init_audio_feedback()` for deferred singleton |
| `scripts/run_voice_control.py` | Remove Riva/Ollama startup, add API check, forward --output-device |
| `requirements.txt` | Remove riva/grpc, add langchain-dartmouth |

## Deleted Files

| File | Reason |
|---|---|
| `src/voice_control/server.py` | ASR bridge server replaced by dartmouth_stt.py |
| `scripts/setup_riva.sh` | Riva no longer used |
| `src/voice_control/asr_pb2.py` | gRPC proto stubs no longer needed |
| `src/voice_control/asr_pb2_grpc.py` | gRPC proto stubs no longer needed |

---

## TODO Checklist

### Prerequisites
- [ ] P1: Obtain `DARTMOUTH_API_KEY` from developer.dartmouth.edu/keys
- [ ] P2: Obtain `DARTMOUTH_CHAT_API_KEY` from chat.dartmouth.edu
- [ ] P3: Test JWT exchange from Jetson (`curl -X POST https://api.dartmouth.edu/api/jwt -H "Authorization: KEY"`)
- [ ] P4: Test STT endpoint format (`curl` with WAV file to `/api/ai/speech-recognition`)

### Implementation
- [ ] S1: Extend `src/config.py` with Dartmouth env vars
- [ ] S2: Create `src/voice_control/dartmouth_auth.py`
- [ ] S3: Create `src/voice_control/dartmouth_stt.py`
- [ ] S4a: Refactor `llm_brain.py` -- replace Ollama text LLM with ChatDartmouth
- [ ] S4b: Refactor `llm_brain.py` -- replace Ollama VLM with ChatDartmouth cloud VLM
- [ ] S5a: Update `client_mic.py` -- replace `send_to_asr()` with `dartmouth_stt.transcribe()`
- [ ] S5b: Update `client_mic.py` -- add `--output-device` arg, wire to TTS and beep
- [ ] S6: Update `audio_feedback.py` -- add `init_audio_feedback()` for deferred singleton
- [ ] S7: Update `run_voice_control.py` -- remove Riva/Ollama startup, add API check, forward --output-device
- [ ] S8: Update `requirements.txt` (remove riva/grpc, add langchain-dartmouth)
- [ ] S9: Create `.env.template`
- [ ] S10: Create `scripts/cleanup_local_inference.sh`
- [ ] S11: Remove/archive `server.py`, `setup_riva.sh`, proto files
- [ ] S12: Update docs (overview, services, getting-started, README)

### Testing
- [ ] T1: Test STT end-to-end (mic -> Dartmouth API -> transcript)
- [ ] T2: Test LLM JSON output with full system prompt
- [ ] T3: Test VLM with camera JPEG
- [ ] T4: Test Bluetooth `--output-device` routing
- [ ] T5: Run cleanup script, verify disk space freed
- [ ] T6: Full integration test (voice command -> robot action)
- [ ] T7: Latency benchmark (target <2s end-to-end)
