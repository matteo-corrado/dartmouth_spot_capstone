# Dartmouth Spot Capstone

Voice-controlled Boston Dynamics Spot robot powered by on-device LLM inference on NVIDIA Jetson AGX Orin. Built by the Dartmouth 25F capstone team.

## What It Does

Say **"Hey Spot, stand up"** and the robot stands. Ask **"What do you see?"** and it describes its surroundings using a vision-language model. Tell it **"go to the lab"** and it navigates there autonomously using a pre-recorded map.

A local LLM (qwen2.5:7b via Ollama) acts as the robot's brain -- it interprets natural language, decides which actions to take, and generates spoken responses. Everything runs on-device with no cloud APIs.

## Key Features

- **Wake word activation** -- "Hey Spot" via sherpa-onnx keyword spotter
- **Natural language understanding** -- LLM interprets free-form speech, not fixed commands
- **Command chaining** -- "stand up, walk forward, then turn left" executes sequentially
- **Visual scene description** -- VLM analyzes camera feeds and describes what Spot sees
- **Visual object navigation** -- "go to the red chair" uses YOLO + camera servoing
- **Autonomous map navigation** -- "go to the lab" uses pre-recorded GraphNav maps
- **Tour and patrol modes** -- "visit all locations" or "patrol the hallway"
- **Text-to-speech responses** -- Spot talks back via Kokoro TTS
- **Safety-first design** -- "stop", "freeze", and "e-stop" bypass the LLM for instant execution
- **Web control panel** -- mobile-friendly E-Stop and pipeline management UI

## Quick Start

**Terminal 1** -- E-Stop (must stay running):

```bash
source spot-env/bin/activate
python scripts/estop_run.py
```

**Terminal 2** -- Voice control:

```bash
source spot-env/bin/activate
python scripts/run_voice_control.py
```

**Terminal 3** -- Talk to Spot:

Say **"Hey Spot, stand up"** into the microphone.

See [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) for a full walkthrough.

## Architecture

```
Mic (XVF3800) → WebRTC VAD → Wake Word (sherpa-onnx) → ASR (Riva) → LLM Brain (qwen2.5:7b) → Spot Dispatcher → Robot
                                                                          ↓
                                                                    TTS (Kokoro) → Speaker
```

The voice client captures audio, gates it through VAD and energy detection, checks for the wake word, transcribes via Riva ASR, then sends the transcript to the LLM brain. The brain returns structured JSON with actions and a spoken response. The dispatcher executes actions on Spot while TTS plays the response.

For the full architecture diagram and design decisions, see [docs/architecture/overview.md](docs/architecture/overview.md).

## Tech Stack

| Component | Technology | Runs On |
|-----------|-----------|---------|
| ASR (speech-to-text) | NVIDIA Riva (Canary-Qwen-2.5B) | GPU (Docker) |
| LLM (language model) | Ollama qwen2.5:7b | GPU |
| VLM (vision-language) | Ollama qwen2.5vl:7b | GPU |
| TTS (text-to-speech) | kokoro-onnx Kokoro v1.0 (fp16-gpu) | GPU |
| Wake Word | sherpa-onnx KWS (zipformer-gigaspeech-3.3M) | CPU |
| Object Detection | YOLOv8n + YOLO-World (ultralytics) | CPU |
| Robot SDK | Boston Dynamics SDK 5.0.1.1 | CPU |
| Platform | Jetson AGX Orin 64GB, JetPack 6.2.1 | -- |

## Project Structure

```
dartmouth_spot_capstone/
    scripts/                        # Entry points and setup
        run_voice_control.py            Main launcher (starts all services)
        estop_run.py                    E-Stop keepalive
        web_panel.py                    Web UI for E-Stop + pipeline control
        setup_riva.sh                   NVIDIA Riva ASR Docker setup
        setup_kokoro.py                 Download Kokoro TTS model
        setup_kws.py                    Download wake word model
        setup_map.py                    Upload GraphNav map to Spot
        download_map_from_spot.py       Download map from robot
        map_waypoints.py                List/label map waypoints
    src/                            # Core source code
        config.py                       Environment variables (.env)
        session.py                      Spot SDK connection manager
        estop.py                        E-Stop utilities
        graph_nav_utils.py              GraphNav map upload/localization
        location_manager.py             Named location persistence
        voice_control/                  Voice pipeline modules
            client_mic.py                   Mic listener + VAD + main loop
            server.py                       gRPC ASR bridge (Riva proxy)
            llm_brain.py                    LLM/VLM inference via Ollama
            spot_dispatch.py                Intent → robot action executor
            intent.py                       Regex intent parser (fallback)
            spot_tts.py                     Kokoro TTS playback
            wake_word.py                    sherpa-onnx keyword spotter
            audio_feedback.py               Status beep tones
            visual_nav.py                   YOLO object seeking + following
    docs/                           # Full documentation (see below)
    models/                         # Downloaded by setup scripts (gitignored)
    maps/                           # GraphNav maps (gitignored)
    locations.json                  # Saved named waypoints
    requirements.txt
```

## Documentation

Detailed guides and reference docs live in [`docs/`](docs/):

**Getting Started**
- [Hardware Setup](docs/getting-started/hardware.md) -- Jetson, Spot, microphone, network
- [Software Setup](docs/getting-started/software.md) -- Python env, Riva, Ollama, dependencies
- [Model Downloads](docs/getting-started/models.md) -- TTS, wake word, YOLO, LLM/VLM models
- [Quick Start](docs/getting-started/quickstart.md) -- 5-minute getting-started walkthrough
- [Daily Operations](docs/getting-started/daily-operations.md) -- Day-to-day usage
- [Tailscale](docs/getting-started/tailscale.md) -- Remote access to Jetson

**Architecture**
- [System Overview](docs/architecture/overview.md) -- Architecture diagram and design decisions
- [Voice Pipeline](docs/architecture/voice-pipeline.md) -- Audio processing deep dive
- [Services](docs/architecture/services.md) -- Service dependencies and startup
- [BD SDK Primer](docs/architecture/sdk-primer.md) -- Boston Dynamics SDK concepts

**Module Reference**
- [client_mic](docs/modules/client-mic.md) -- Voice client main loop
- [llm_brain](docs/modules/llm-brain.md) -- LLM/VLM integration
- [spot_dispatch](docs/modules/spot-dispatch.md) -- Command dispatcher
- [spot_tts](docs/modules/spot-tts.md) -- Text-to-speech
- [visual_nav](docs/modules/visual-nav.md) -- YOLO visual navigation
- [wake_word](docs/modules/wake-word.md) -- Wake word detection
- [audio_feedback](docs/modules/audio-feedback.md) -- Audio feedback tones
- [session](docs/modules/session.md) -- Spot connection manager
- [graph_nav](docs/modules/graph-nav.md) -- GraphNav utilities
- [location_manager](docs/modules/location-manager.md) -- Named locations
- [intent](docs/modules/intent.md) -- Regex intent parser

**Guides**
- [Command Reference](docs/commands/reference.md) -- All voice commands
- [Adding Commands](docs/guides/adding-commands.md) -- How to add new commands
- [Mapping](docs/guides/mapping.md) -- Recording and managing maps
- [Web Panel](docs/guides/web-panel.md) -- Browser-based control
- [Troubleshooting](docs/guides/troubleshooting.md) -- Common issues and fixes

**Project**
- [Feature Status](docs/project/status.md) -- What's done, in progress, and planned
- [Dartmouth Setup](docs/project/dartmouth-setup.md) -- Campus-specific configuration
- [Script Reference](docs/scripts/reference.md) -- All scripts and their usage

## Connecting to Jetson

The Jetson is accessible via Tailscale (see [Tailscale guide](docs/getting-started/tailscale.md)) or USB serial console:

```bash
# Find the serial device (macOS)
ls /dev/cu.usbmodem*

# Connect at 115200 baud
sudo screen /dev/cu.usbmodem<DEVICE_NUMBER> 115200

# Exit screen: Ctrl+A, then K, then Y
```
