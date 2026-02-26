# Feature Status

Current implementation status of all features.

---

## Complete

| Feature | Description | Key Files |
|---------|-------------|-----------|
| Voice Pipeline | Mic → VAD → ASR → LLM → Dispatch | `client_mic.py`, `server.py` |
| LLM Brain | Structured JSON command parsing via Ollama | `llm_brain.py` |
| Wake Word | "Hey Spot" detection via sherpa-onnx KWS | `wake_word.py` |
| Text-to-Speech | Spoken responses via Kokoro TTS | `spot_tts.py` |
| Audio Feedback | Beep/chime tones for status | `audio_feedback.py` |
| Movement | Walk, strafe, turn with parameters | `spot_dispatch.py` |
| Posture | Stand, sit, crouch, self-right, body height | `spot_dispatch.py` |
| Navigation | Go-to, tour, patrol, come-back via GraphNav | `spot_dispatch.py`, `graph_nav_utils.py` |
| Location Management | Save/load named waypoints | `location_manager.py` |
| Safety | Stop, freeze, e-stop (instant, bypasses LLM) | `client_mic.py` |
| Status Queries | Battery, robot status, power off | `spot_dispatch.py` |
| Visual Detection | YOLO object detection on camera feeds | `visual_nav.py` |
| Object Navigation | "Go to the red chair" via YOLO + approach | `visual_nav.py` |
| VLM Descriptions | "What do you see?" via qwen2.5vl | `llm_brain.py` |
| Web Panel | Browser-based E-Stop + control | `web_panel.py` |
| Command Chaining | "Go to kitchen then sit down" | `client_mic.py`, `llm_brain.py` |
| One-Breath Commands | "Hey Spot stand up" in one phrase | `client_mic.py` |
| Regex Fallback | 80+ patterns for --no-brain mode | `intent.py` |

---

## In Progress

| Feature | Description | Status | Next Step |
|---------|-------------|--------|-----------|
| Follow Mode | Person following | YOLO unreliable on fisheye | Switch to AprilTag fiducials via WorldObjectClient |
| Door Opening | Push-bar doors via AutoPush | Implemented, untested | Test with `--vlm --depth 0.5` |

---

## Planned

| Feature | Description | Notes |
|---------|-------------|-------|
| AprilTag Follow | Follow person wearing AprilTag | Zero Jetson CPU cost, 3D position from Spot |
| Stair Handling | Follow mode stair ascent/descent | Spot descends backwards, needs special logic |

---

## Not Planned

| Feature | Reason |
|---------|--------|
| PTZ Camera | Requires additional payload hardware |
| Cloud ASR | All inference is on-device by design |

---

## Latency Optimizations (Applied)

| Optimization | Change | Impact |
|-------------|--------|--------|
| System prompt compression | Cut 10 obvious examples, keep 5 | ~25% fewer prompt tokens per LLM call |
| History reduction | MAX_HISTORY 20 → 12 | ~500-800 fewer tokens after 6+ exchanges |
| YOLO pre-loading | Background model load at startup | Eliminates 2-3s first-use delay |
| Timing instrumentation | [Timing] prints for ASR, LLM, total | Enables data-driven optimization |
| GraphNav route recovery | RouteGenParams backtrack on stuck | Better stuck recovery during navigation |
