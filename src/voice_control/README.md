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

## Voice Commands

| Category | Command | Example Phrases |
|----------|---------|-----------------|
| **Safety** | Stop | "stop", "halt" |
| | Freeze | "freeze" |
| | E-Stop | "emergency stop", "e-stop" |
| **Posture** | Stand | "stand", "stand up", "get up" |
| | Sit | "sit", "sit down" |
| | Self-right | "self right", "recover" |
| **Movement** | Walk forward | "walk forward", "walk forward 2 meters" |
| | Walk backward | "walk back", "back up", "walk backward 1 meter" |
| | Strafe left | "strafe left", "step left", "move left 0.5 meters" |
| | Strafe right | "strafe right", "step right" |
| **Turning** | Turn left | "turn left", "turn left 45 degrees" |
| | Turn right | "turn right", "turn right 90" |
| | Turn around | "turn around", "spin around", "180" |
| **Body** | Crouch | "crouch", "lower body", "duck" |
| | Stand tall | "stand tall", "raise body", "max height" |
| | Normal height | "normal height" |
| **Speed** | Slow | "slow down", "walk slowly" |
| | Normal | "normal speed" |
| | Fast | "fast mode", "speed up" |
| **Navigation** | Go to | "go to kitchen", "navigate to lobby" |
| | Save location | "save location kitchen", "remember this as home" |
| | List locations | "list locations", "what locations" |
| **Status** | Battery | "battery", "battery status" |
| | Status | "status", "status check" |
| | Power off | "power off", "shut down" |

**Distance defaults:** Walk = 1m, Strafe = 0.5m, Turn = 90°

## How It Works

1. **Audio Capture**: Microphone captures audio at 16kHz, mono
2. **VAD**: WebRTC VAD detects when speech starts/stops
3. **ASR**: Audio chunks are sent to Faster Whisper server for transcription
4. **Intent Parsing**: Transcript is parsed to extract intent and parameters
5. **Spot Execution**: Intent is dispatched to Spot via `spot_dispatch.py`
6. **Session Management**: Spot session stays open for multiple commands

## Configuration

- **ASR Server Port**: Default `localhost:50055` (hardcoded in `client_mic.py`)
- **Whisper Model**: `small` or `large-v3-turbo` (configured in `server.py`)
- **Audio Settings**: 16kHz, 30ms frames, VAD level 2
- **LLM Intent Parser**: Optional Ollama with `qwen2.5:3b` for natural language understanding

## Troubleshooting

- **"ModuleNotFoundError"**: Make sure you're in the virtual environment
- **"Could not connect to robot"**: Check network, credentials in `.env`, and robot power
- **"E-Stop error"**: Ensure `scripts/estop_run.py` is running
- **"No speech detected"**: Check microphone permissions and audio levels
- **ASR server slow**: First run downloads model (~3GB). Subsequent runs are faster.

## LLM Intent Parser (Optional)

For better natural language understanding, install Ollama:

```bash
# Install Ollama on Jetson
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama

# Pull recommended model
ollama pull qwen2.5:3b

# Test the LLM parser
python src/voice_control/intent_llm.py
```

The system uses regex matching first (fast), then falls back to LLM for complex phrases.

## Future Improvements

- [x] More voice commands (walk, strafe, body height, speed control)
- [x] LLM-based natural language understanding
- [ ] Streaming/partial transcription for lower latency
- [ ] Person following behavior implementation
- [ ] PTZ camera control (requires payload)
- [ ] Text-to-speech feedback
- [ ] Confidence thresholds and error handling

