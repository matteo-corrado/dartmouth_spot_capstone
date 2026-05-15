# client_mic.py

`src/voice_control/client_mic.py` -- Main voice control loop.

Captures audio from the microphone, runs VAD to detect speech, sends audio to
ASR, and routes the transcript through the LLM brain (or regex fallback) for
action dispatch on Spot. Safety commands (stop/freeze/estop) bypass the LLM
for zero-latency execution.

## State Machine

```
VoiceState.WAKE_WORD  -->  VoiceState.LISTENING  -->  VoiceState.RECORDING
       |                         |                          |
  (wake word detected)    (VAD speech onset)        (silence timeout)
                                                     --> process_utterance()
```

- **WAKE_WORD**: Waiting for "Hey Spot". Only safety commands execute.
- **LISTENING**: Wake word heard, waiting for speech onset. Times out after
  `LISTENING_TIMEOUT` (300s) and returns to WAKE_WORD.
- **RECORDING**: Speech detected, accumulating audio frames until silence
  timeout or max duration.

## Key Functions

### `main()`

CLI entry point. Parses arguments, initializes all services (ASR, LLM brain,
TTS, wake word detector, YOLO), opens the audio stream, and runs the main
event loop.

```
Args (argparse):
    --list-devices     List audio devices and exit
    --device INT       Audio device index (default: 24, XVF3800)
    --no-brain         Disable LLM brain, use regex-only mode
    --no-tts           Disable text-to-speech
    --no-wake-word     Always listening, skip wake word requirement
    --debug-audio      Print audio levels each status tick
```

### `process_utterance(stub, speech_buffer, speech_float_buffer, brain, safety_only, tts, has_wake_detector)`

Core NLP pipeline: ASR -> safety check -> wake phrase strip -> LLM -> dispatch.

```python
def process_utterance(
    stub,                    # ASRStub gRPC channel
    speech_buffer: bytearray,       # Raw PCM16 bytes
    speech_float_buffer: list,      # List of float32 numpy arrays
    brain=None,              # SpotBrain instance (None = regex mode)
    safety_only=False,       # True when in WAKE_WORD state
    tts=None,                # SpotTTS instance
    has_wake_detector=False  # True if dedicated KWS is active
) -> str | None
```

Returns `"wake_detected"` if wake phrase found via ASR fallback, `None` otherwise.

### `send_to_asr(stub, pcm_bytes) -> str`

Stream PCM16 audio to the ASR gRPC server. Sends config first, then
audio chunks. Returns the final transcript string.

### `check_safety_command(text) -> dict | None`

Regex check for stop/freeze/estop. Returns intent dict on match, None otherwise.
Patterns: `stop|halt`, `freeze`, `emergency stop|e-stop`.

### `execute_on_spot(intent) -> bool`

Import and call `spot_dispatch.dispatch_intent`. Returns success boolean.

### `get_spot_state() -> dict`

Import and call `spot_dispatch.get_robot_state_dict`. Returns state dict for
LLM context (battery, posture, location, estop).

### `calibrate_noise_floor(duration_sec, device) -> float`

Record ambient audio for `duration_sec` seconds, compute RMS. Caps high noise
(>0.003, likely motor noise) to 0.001. Seeds the adaptive noise tracker.

### `audio_callback(indata, frames, time_info, status)`

sounddevice stream callback. Selects mono channel from stereo input
(`MIC_CHANNEL`, default 0 = left/beamformed), converts to PCM16, pushes
to `audio_queue`. Discards audio when `_mic_muted` is True (during TTS).

## Configuration Constants

| Name | Default | Description |
|------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `FRAME_MS` | 30 | Frame duration (ms) |
| `VAD_LEVEL` | 1 | webrtcvad aggressiveness (0-3) |
| `SILENCE_TIMEOUT_SHORT` | 0.8 | Silence timeout for short utterances (<2s) |
| `SILENCE_TIMEOUT_LONG` | 1.5 | Silence timeout for longer utterances (>2s) |
| `SILENCE_CROSSOVER` | 2.0 | Speech duration threshold to switch timeouts |
| `SPEECH_ONSET_FRAMES` | 3 | Consecutive VAD+energy frames to confirm onset (~90ms) |
| `NAV_ONSET_FRAMES` | 5 | Higher onset threshold during navigation (~150ms) |
| `PREROLL_FRAMES` | 5 | Pre-onset ring buffer size (~150ms) |
| `MAX_UTTERANCE_SECONDS` | 20 | Force-send after this duration |
| `ENERGY_THRESHOLD_MULTIPLIER` | 2.0 | Speech energy must exceed noise floor by this factor |
| `NAV_ENERGY_MULT` | 5.0 | Extra energy multiplier during navigation |
| `NOISE_EMA_ALPHA` | 0.03 | EMA smoothing for adaptive noise floor |
| `NOISE_FLOOR_MIN` | 0.001 | Minimum noise floor |
| `NOISE_FLOOR_MAX` | 0.05 | Maximum noise floor |
| `NOISE_CALIBRATION_SECONDS` | 2 | Initial calibration recording duration |
| `LISTENING_TIMEOUT` | 300.0 | Seconds idle before returning to WAKE_WORD |
| `MIN_SPEECH_DURATION` | 0.3 | Reject utterances shorter than this |

## Usage Example

```bash
# Default: LLM brain + TTS + wake word on device 24
python src/voice_control/client_mic.py --device 24

# Regex-only mode, no TTS, always listening
python src/voice_control/client_mic.py --no-brain --no-tts --no-wake-word

# Debug audio levels
python src/voice_control/client_mic.py --debug-audio
```

## Dependencies

**Project modules:** `intent`, `llm_brain`, `spot_tts`, `audio_feedback`,
`spot_dispatch`, `visual_nav`, `wake_word`, `asr_pb2`, `asr_pb2_grpc`

**External packages:** `sounddevice`, `numpy`, `webrtcvad`, `grpc`
