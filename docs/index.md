# Dartmouth Spot Capstone

Voice-controlled Boston Dynamics Spot robot powered by on-device LLM inference on NVIDIA Jetson AGX Orin.

## What It Does

This system lets you control a Spot robot entirely by voice. Say "Hey Spot, stand up" and the robot stands. Ask "what do you see?" and it describes its surroundings using a vision-language model. Tell it "go to the lab" and it navigates there autonomously using a pre-recorded map.

A local LLM (qwen2.5:7b via Ollama) acts as the robot's "brain" -- it interprets natural language, decides which actions to take, and generates spoken responses. Everything runs on-device; no cloud APIs required.

## Key Features

- **Wake word activation** -- "Hey Spot" via sherpa-onnx keyword spotter (CPU, ~0ms latency)
- **Natural language understanding** -- LLM interprets free-form speech, not just fixed commands
- **Command chaining** -- "stand up, walk forward, then turn left" executes sequentially
- **Visual scene description** -- VLM analyzes camera feeds and describes what Spot sees
- **Visual object navigation** -- "go to the red chair" uses YOLO + camera servoing
- **Autonomous map navigation** -- "go to the lab" uses pre-recorded GraphNav maps
- **Tour and patrol modes** -- "visit all locations" or "patrol the hallway"
- **Text-to-speech responses** -- Spot talks back via Kokoro TTS
- **Safety-first design** -- "stop", "freeze", and "e-stop" bypass the LLM for zero-latency execution
- **Web control panel** -- mobile-friendly E-Stop and pipeline management UI
- **Adaptive noise handling** -- energy-gated VAD suppresses motor noise during movement

## Quick Start

Terminal 1 -- E-Stop (must stay running):

```bash
source spot-env/bin/activate
python scripts/estop_run.py
```

Terminal 2 -- Voice control:

```bash
source spot-env/bin/activate
python scripts/run_voice_control.py
```

Terminal 3 -- Talk to Spot:

Say **"Hey Spot, stand up"** into the microphone.

See the [Quick Start guide](getting-started/quickstart.md) for a full walkthrough.

## Documentation

| Section | Description |
|---------|-------------|
| [Hardware Setup](getting-started/hardware.md) | Jetson, Spot, microphone, network |
| [Software Setup](getting-started/software.md) | Python env, Riva, Ollama, dependencies |
| [Model Downloads](getting-started/models.md) | TTS, wake word, YOLO, LLM/VLM models |
| [Quick Start](getting-started/quickstart.md) | 5-minute getting-started walkthrough |

## Tech Stack

| Component | Technology | Runs On |
|-----------|-----------|---------|
| ASR (speech-to-text) | NVIDIA Riva (Canary-Qwen-2.5B) | GPU (Docker) |
| LLM (language model) | Ollama qwen2.5:7b | GPU |
| VLM (vision-language) | Ollama qwen2.5vl:7b | GPU |
| TTS (text-to-speech) | kokoro-onnx Kokoro v1.0 (fp16-gpu) | GPU |
| Wake Word | sherpa-onnx KWS (zipformer-gigaspeech-3.3M, int8) | CPU |
| Object Detection | YOLO (ultralytics) -- YOLOv8n + YOLO-World | CPU |
| Robot SDK | Boston Dynamics SDK 5.0.1.1 | CPU |
| Platform | Jetson AGX Orin 64GB, JetPack 6.2.1 | -- |

## Project Structure

```
dartmouth_spot_capstone/
    scripts/
        run_voice_control.py    # Main entry point -- starts everything
        estop_run.py            # E-Stop keepalive (run separately)
        web_panel.py            # Web UI for E-Stop + pipeline control
        setup_riva.sh           # NVIDIA Riva ASR Docker setup
        setup_kokoro.py         # Download Kokoro TTS model
        setup_kws.py            # Download wake word model
        setup_map.py            # Upload GraphNav map to Spot
        download_map_from_spot.py
        map_waypoints.py        # List/label map waypoints
    src/
        config.py               # Environment variables (.env)
        session.py              # Spot SDK connection manager
        estop.py                # E-Stop utilities
        graph_nav_utils.py      # GraphNav map upload/localization
        location_manager.py     # Named location persistence
        voice_control/
            client_mic.py       # Microphone listener + VAD + main loop
            server.py           # gRPC ASR bridge (Riva proxy)
            llm_brain.py        # LLM/VLM inference via Ollama
            spot_dispatch.py    # Intent-to-robot-action executor
            intent.py           # Regex intent parser (legacy fallback)
            spot_tts.py         # Kokoro TTS playback
            wake_word.py        # sherpa-onnx keyword spotter
            audio_feedback.py   # Beep tones for status feedback
            visual_nav.py       # YOLO object seeking + person following
    models/                     # Downloaded by setup scripts (gitignored)
        tts/kokoro-v1.0/
        kws/
    maps/                       # GraphNav maps (gitignored)
    locations.json              # Saved named locations
    requirements.txt
    mkdocs.yml                  # This documentation site config
    .env                        # Robot credentials (gitignored)
```
