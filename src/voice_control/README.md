# Spot Voice Control System

This directory contains the voice I/O system for controlling Spot via natural language commands.

## Architecture

- **`server.py`**: gRPC ASR server using Faster Whisper for speech-to-text
- **`client_mic.py`**: Microphone client with VAD (Voice Activity Detection) that sends audio to ASR server
- **`intent.py`**: Intent parser that maps natural language to structured robot commands
- **`spot_dispatch.py`**: Command dispatcher that executes Spot actions via the SDK
- **`asr_pb2*.py`**: Generated protobuf files for gRPC communication

## Setup

1. **Install dependencies:**
   ```bash
   cd /Users/christian/Documents/ai_dev/spot/dartmouth_spot_capstone
   source spot-env-christian/bin/activate
   pip install -r requirements.txt
   ```

2. **Ensure E-Stop is running:**
   ```bash
   python scripts/estop_run.py
   ```
   Keep this running in a separate terminal!

## Usage

### Option 1: Integrated Script (Recommended)

```bash
python scripts/run_voice_control.py
```

This automatically starts both the ASR server and microphone client.

### Option 2: Manual (Two Terminals)

**Terminal 1 - ASR Server:**
```bash
cd src/voice_control
python server.py
```

**Terminal 2 - Voice Client:**
```bash
cd src/voice_control
python client_mic.py
```

## Supported Commands

The intent parser recognizes these voice commands:

- **"stop"** / **"halt"** / **"freeze"**: Emergency stop
- **"follow me"** / **"follow"**: Start following behavior (not yet implemented)
- **"come here"** / **"come to me"**: Walk towards speaker (relative position)
- **"turn left [N] degrees"**: Rotate left by N degrees (e.g., "turn left 45")
- **"turn right [N] degrees"**: Rotate right by N degrees (e.g., "turn right 90")
- **"look at me"** / **"look here"**: Aim PTZ camera at speaker (not yet implemented)

## How It Works

1. **Audio Capture**: Microphone captures audio at 16kHz, mono
2. **VAD**: WebRTC VAD detects when speech starts/stops
3. **ASR**: Audio chunks are sent to Faster Whisper server for transcription
4. **Intent Parsing**: Transcript is parsed to extract intent and parameters
5. **Spot Execution**: Intent is dispatched to Spot via `spot_dispatch.py`
6. **Session Management**: Spot session stays open for multiple commands

## Configuration

- **ASR Server Port**: Default `localhost:50055` (hardcoded in `client_mic.py`)
- **Whisper Model**: `large-v3-turbo` (configured in `server.py`)
- **Audio Settings**: 16kHz, 30ms frames, VAD level 2

## Troubleshooting

- **"ModuleNotFoundError"**: Make sure you're in the virtual environment
- **"Could not connect to robot"**: Check network, credentials in `.env`, and robot power
- **"E-Stop error"**: Ensure `scripts/estop_run.py` is running
- **"No speech detected"**: Check microphone permissions and audio levels
- **ASR server slow**: First run downloads model (~3GB). Subsequent runs are faster.

## Future Improvements

Future enhancements:
- Streaming/partial transcription for lower latency
- Person following behavior implementation
- PTZ camera control (requires payload)
- More voice commands (walk forward/back, specific distances)
- Text-to-speech feedback
- Confidence thresholds and error handling

