# dartmouth_spot_capstone

Voice I/O prototype and helpers for Dartmouth’s Spot work (25F): streaming ASR over gRPC, rules-based intent parsing, and stubs to dispatch intents to Spot. Mapping and other tracks may live alongside but this doc focuses on voice.

## What’s here
- `src/voice_control/server.py` — gRPC ASR server using Faster-Whisper. Expects 16 kHz mono PCM16; buffers a full utterance and returns one final transcript.
- `src/voice_control/client_mic.py` — Microphone capture + WebRTC VAD (30 ms frames) → streams PCM to server → receives transcript → `intent.parse_intent(...)`.
- `src/voice_control/intent.py` — Rules-based parser returning `{intent, params, raw}`; example: “turn left 30 degrees” → `{intent: 'turn', params: {dir: 'left', deg: 30}}` (deg clamped to [0,360]).
- `src/voice_control/spot_dispatch.py` — Connection helper and placeholders for mapping intents to Spot SDK commands.

## Requirements
Install the Boston Dynamics Spot SDK wheels first (from the sibling `spot-sdk/prebuilt` directory in this workspace), then the Python deps below.

Authentication: your venv `Activate.ps1` supplies credentials/tokens; do not embed user/password in commands or code. Scripts should rely on the environment and SDK token flow.

### Python dependencies (installed via requirements.txt)
- Faster Whisper stack: `faster-whisper`, `numpy`
- Audio/VAD: `sounddevice`, `webrtcvad`
- gRPC: `grpcio`, `protobuf`

See `requirements.txt` in this folder for an installable list.

## Quick start (Windows)
1) Activate your project venv.

2) Install Spot SDK wheels (from workspace root or adjust path):
	- Install all wheels under `spot-sdk/prebuilt/*.whl`.

3) Install Python deps:
	- `pip install -r requirements.txt`

4) Run the ASR server (in one terminal):
	- `python src/voice_control/server.py`

5) Run the mic client (in another terminal):
	- `python src/voice_control/client_mic.py`

Speak commands like “turn left 30 degrees”, “follow me”, “stop”. The client prints the transcript and parsed intent. Hook intent execution in `spot_dispatch.py`.

## Notes, contracts, and limits
- Audio contract: 16 kHz mono PCM16; VAD frames are 30 ms (webrtcvad supports 10/20/30 ms).
- ASR server is single-shot (returns one final result) and runs without TLS. For partials or security, extend later.
- SDK usage (when wiring Spot): time-sync before commands; use `LeaseKeepAlive` around blocking command sequences.

## Troubleshooting
- No transcript: verify ASR server is running and reachable at `localhost:50055`.
- “Robot estopped”: clear estop on tablet; ensure your venv auth is loaded by `Activate.ps1`.
- Audio device errors: confirm mic permissions and that the sample rate is 16 kHz.