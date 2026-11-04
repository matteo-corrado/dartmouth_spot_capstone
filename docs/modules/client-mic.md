# client_mic (voice client)

> **Split into packages.** The old `client_mic.py` no longer owns any code. The voice client now lives in `src/spot/` (see [package-map.md](../architecture/package-map.md)). `src/voice_control/client_mic.py` is a stateless launch stub: it fixes `sys.path` and calls `src.spot.app.main.main()`. `scripts/run_voice_control.py` still launches that path, so the command line below is unchanged.

The voice client captures audio from the microphone, runs VAD to detect speech, sends audio to ASR, and routes the transcript through the LLM brain (or regex fallback) for action dispatch on Spot. Safety commands (stop/freeze/estop) bypass the LLM for zero-latency execution.

## Where the old client_mic code went

| Old `client_mic.py` job | Now in |
|---|---|
| `main()`, argument parsing, startup order, shutdown | `src/spot/app/main.py` (`main`, `_parse_args`, `_start_audio`, `_bring_up_robot`, `_ShutdownOnce`) |
| `audio_callback`, audio queue, mute interlock, `calibrate_noise_floor`, device lookup, endpointing | `src/spot/audio/capture.py` (`AudioCapture`, `CAPTURE`; `calibrate_floor`) |
| LED ring control | `src/spot/mic/ring_control.py` (`Ring`) |
| Safety keyword listener, `check_safety_command`-style rescue | `src/spot/safety/listener.py` (`SafetyListener`, `rescue_safety`) |
| State machine, `process_utterance`, `execute_on_spot`, `get_spot_state`, chaining waits, gaze worker | `src/spot/conversation/loop.py` (`Conversation`, `CONVERSATION`) |
| Speaker gate, conversation timeout helpers, filler | `src/spot/conversation/{gate,filler}.py` |
| Speaker lock, voiceprint, session state, logs | `src/spot/conversation/{speaker_lock,voiceprint,session_state,conversation_log,utterance_log}.py` |
| `send_to_asr` | `src/spot/stt/client.py` |
| Status snapshot and spotctl immediate commands | `src/spot/operator/control.py` (server in `control_server.py`) |
| `VoiceState` enum | `src/spot/core/voice_state.py` |

## State Machine

```
VoiceState.WAKE_WORD  -->  VoiceState.LISTENING  -->  VoiceState.RECORDING
       |                         |                          |
  (wake word detected)    (VAD speech onset)        (silence timeout)
                                                     --> process_utterance()
```

The loop is `Conversation.run()` in `src/spot/conversation/loop.py`.

- **WAKE_WORD**: Waiting for "Hey Spot". Only safety commands execute.
- **LISTENING**: Wake word heard, waiting for speech onset. The conversation expires after `Tuning.lock_timeout_s` (20 s default; `conversation_expired` in `conversation/gate.py`) and returns to WAKE_WORD.
- **RECORDING**: Speech detected, accumulating audio frames until the silence timeout or the maximum duration.

## Key Functions

### `main(argv=None)` -- `src/spot/app/main.py`

CLI entry point. Parses arguments, initializes all services (ASR, LLM brain, TTS, wake word detector, vision), opens the audio stream, and runs the conversation loop. The startup order is documented in [restructure-notes.md](../project/restructure-notes.md).

```
Args (argparse, see _parse_args):
    --list-devices         List audio devices and exit
    --device INT           Input device index
    --output-device INT    Output device index
    --no-brain             Disable LLM brain, use regex-only mode
    --no-tts               Disable text-to-speech
    --tts-backend NAME     kokoro | elevenlabs
    --no-wake-word         Always listening, skip wake word requirement
    --dry-run              No robot connection
    --map NAME             Open a GraphNav session with this map
    --no-arm / --enable-arm
    --mic-channel N, --volume, --no-transcript-log, --allow-no-safety-kws
    --debug-audio          Print audio levels each status tick
```

### `Conversation.process_utterance(stub, speech_buffer, speech_float_buffer, brain, ...)` -- `conversation/loop.py`

Core NLP pipeline: ASR -> safety check -> wake phrase strip -> LLM -> dispatch. Returns `"wake_detected"` if the wake phrase was found via the ASR fallback, `None` otherwise. Read the signature in the source; it has grown since the monolith.

### `send_to_asr(stub, pcm_bytes, timeout=None) -> str` -- `stt/client.py`

Stream PCM16 audio to the ASR gRPC server. Sends config first, then audio chunks. Returns the final transcript.

### Safety rescue -- `safety/listener.py`

`rescue_safety(stub, pcm_bytes, ...)` runs the safety regex on a transcript. The always-on safety keyword spotter (`safety_kws_loop`, `SafetyListener`) calls the halt directly, never through the event bus.

### `execute_on_spot(intent, brain=None, speak=None) -> bool` -- `conversation/loop.py`

Calls `src.spot.robot.dispatcher.dispatch_intent` (see [spot-dispatch.md](spot-dispatch.md)). Returns a success boolean.

### `get_spot_state() -> dict` -- `conversation/loop.py`

Wraps `robot/state.get_robot_state_dict` for LLM context (battery, posture, location, estop).

### `AudioCapture.calibrate_floor(duration_sec, device) -> float` -- `audio/capture.py`

Record ambient audio for `duration_sec` seconds and compute RMS to seed the adaptive noise floor.

### `AudioCapture.callback(indata, frames, time_info, status)` -- `audio/capture.py`

sounddevice stream callback. Selects the mono channel (`Tuning.mic_channel`, `--mic-channel`), converts to PCM16 and pushes to a bounded audio queue (`AUDIO_QUEUE_MAX = 33`; on overflow frames are dropped, never blocking PortAudio). Frames are discarded while the audio player is busy (TTS or chime), plus `Tuning.speaker_tail_s` afterwards.

## Configuration

Most knobs are now tuning keys in `config/tuning.yaml` (class `Tuning` in `src/spot/core/tuning.py`). The constants that remain are module-level.

| Name | Value | Where |
|------|---------|-------|
| `SAMPLE_RATE` | 16000 | `audio/capture.py` |
| `FRAME_MS` | 30 | `audio/capture.py` |
| `NOISE_CALIBRATION_SECONDS` | 2 | `audio/capture.py` |
| VAD | Silero ONNX (`SILERO_VAD_PATH`), no longer webrtcvad | `audio/capture.py` |
| `Tuning.onset_frames` | 3 | consecutive speech frames to confirm onset |
| `Tuning.end_silence_s_short` / `end_silence_s` | 1.0 / 1.2 | silence timeout for short / longer utterances |
| `SILENCE_CROSSOVER` | 2.0 | `conversation/loop.py`: speech length that switches timeouts |
| `PREROLL_FRAMES` | 25 (0.75 s) | `conversation/loop.py`: audio kept before onset |
| `Tuning.max_utterance_s` | 20 | force-send after this duration |
| `Tuning.min_speech_dbfs` | -50 | speech level floor; `NAV_SPEECH_DB_BOOST` (+6 dB) is added while the robot moves |
| `MIN_SPEECH_DURATION` | 0.3 | `conversation/loop.py`: reject shorter utterances |
| `Tuning.lock_timeout_s` | 20 | conversation timeout (replaced the fixed 300 s `LISTENING_TIMEOUT`) |
| `Tuning.speaker_tail_s` | 0.15 | mic mute grace period after playback |

The old webrtcvad energy multipliers and the EMA noise tracker constants were replaced by the Silero VAD, the floor tracker (`floor_*` tuning keys) and the `min_snr_db` gate; check `core/tuning.py` for current values.

## Usage Example

```bash
# Default: LLM brain + TTS + wake word (launch stub forwards to src/spot/app/main.py)
python src/voice_control/client_mic.py --device 24

# Regex-only mode, no TTS, always listening
python src/voice_control/client_mic.py --no-brain --no-tts --no-wake-word

# Debug audio levels
python src/voice_control/client_mic.py --debug-audio
```

## Dependencies

**Project packages:** `src.spot.audio`, `mic`, `safety`, `conversation`, `operator`, `stt`, `brain`, `tts`, `wake`, `robot`, `core`

**External packages:** `sounddevice`, `numpy`, `grpc` (Silero VAD runs through `src/spot/audio/frontend.py`)
