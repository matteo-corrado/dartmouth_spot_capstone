# Quick Start

Get Spot responding to voice commands in 5 minutes. This assumes all [software](software.md) and [models](models.md) are already installed.

## Prerequisites Checklist

Before starting, confirm:

- [ ] Spot is powered on (fans running, not e-stopped from tablet)
- [ ] Jetson is connected to Spot's WiFi (`ping 192.168.80.3` succeeds)
- [ ] `.env` file exists with correct robot credentials
- [ ] Python venv is set up with all dependencies installed
- [ ] Riva Docker container has been initialized (`riva_init.sh` run at least once)
- [ ] Ollama models are pulled (`ollama list` shows qwen2.5:7b and qwen2.5vl:7b)
- [ ] TTS and KWS models are downloaded (`models/tts/` and `models/kws/` exist)
- [ ] Microphone (XVF3800) is plugged into Jetson USB

## Step 1: Start E-Stop

Open a terminal and start the E-Stop keepalive. This must stay running the entire time you use Spot.

```bash
cd ~/spot/dartmouth_spot_capstone
source spot-env/bin/activate
python scripts/estop_run.py
```

Expected output:

```
Starting E-Stop keepalive. Press Ctrl+C to stop (will issue STOP).
This will claim the E-Stop even if another client holds it.

[E-Stop] Claimed and active as 'dartmouth_estop'
```

!!! warning "Do not close this terminal"
    If the E-Stop process exits, the robot will stop all movement within 3 seconds. Keep this terminal open and running.

**Alternative**: Use the web panel for E-Stop control from your phone:

```bash
python scripts/web_panel.py
```

Then open `http://<jetson-ip>:8080` on your phone or laptop.

## Step 2: Start Voice Control

Open a second terminal:

```bash
cd ~/spot/dartmouth_spot_capstone
source spot-env/bin/activate
python scripts/run_voice_control.py
```

The startup sequence takes 30-60 seconds. Here is what happens:

1. **Riva check** -- verifies the ASR Docker container is running (starts it if stopped)
2. **Ollama check** -- verifies the LLM service is running
3. **ASR bridge** -- starts the gRPC proxy server on port 50055
4. **Noise calibration** -- records 2 seconds of silence to set the energy threshold
5. **LLM warm-up** -- sends a throwaway prompt to pre-load the model into VRAM (~10-15s on cold start)
6. **Wake word init** -- loads the sherpa-onnx keyword spotter
7. **YOLO preload** -- begins loading object detection models in the background

When you see this, the system is ready:

```
============================================================
LISTENING -- Mode: LLM Brain (qwen2.5:7b)
  Wake word: ON (sherpa-onnx keyword spotter)
  Safety commands (stop/freeze/estop) always instant
  Everything else goes through the LLM brain
============================================================
```

## Step 3: Talk to Spot

Say **"Hey Spot"** clearly into the microphone. You will hear a confirmation beep. Then give a command:

- **"Stand up"** -- Spot stands from sitting position
- **"Walk forward"** -- Spot walks forward 1 meter
- **"What do you see?"** -- Spot takes a photo and describes the scene
- **"Sit down"** -- Spot sits back down

You can also say the wake word and command in one breath: **"Hey Spot, stand up"**.

## Example Session

Here is what a typical session looks like in the terminal:

```
[Say 'Hey Spot' to activate...]
>>> Wake word detected!

>>> Speech detected...
    Recording: 0.5s

>>> Processing...
[Sending 0.8s to ASR...]
[Timing] ASR: 0.34s

============================================================
HEARD: "Stand up"
============================================================
[Brain] Actions: stand
[Timing] LLM: 0.82s

SPOT: "Getting up now!"
>>> SUCCESS
[Timing] Total: 1.43s

[Listening...]

>>> Speech detected...
    Recording: 1.2s

>>> Processing...
[Sending 1.4s to ASR...]
[Timing] ASR: 0.41s

============================================================
HEARD: "What do you see in front of you?"
============================================================
[Brain] Actions: describe
[Timing] LLM: 1.12s

SPOT: "Let me take a look..."
[Brain] Querying VLM (qwen2.5vl:7b)...
[Brain] VLM responded in 4.2s

SPOT (VLM): "I can see a long hallway with fluorescent lighting. There are
several doors on both sides and a whiteboard mounted on the wall to my left."
[Timing] Total: 6.15s
```

## Common Options

```bash
# Skip wake word (always listening, no "Hey Spot" needed)
python scripts/run_voice_control.py --no-wake-word

# Disable TTS (silent mode, text-only output)
python scripts/run_voice_control.py --no-tts

# Disable LLM (regex-only command parsing, no conversation)
python scripts/run_voice_control.py --no-brain

# Use a different microphone device
python scripts/run_voice_control.py --device 10

# Debug audio levels (shows RMS/threshold for mic troubleshooting)
python scripts/run_voice_control.py --debug-audio

# Skip auto-starting Riva/Ollama (if you manage them manually)
python scripts/run_voice_control.py --skip-services
```

## Stopping

1. Press **Ctrl+C** in the voice control terminal. Spot will sit down and power off.
2. Press **Ctrl+C** in the E-Stop terminal. The E-Stop will be released.

To emergency stop at any time, say **"stop"**, **"freeze"**, or **"e-stop"**. These safety commands bypass the LLM and execute immediately, even without the wake word.
