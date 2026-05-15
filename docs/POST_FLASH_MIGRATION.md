# Post-Flash Setup: JetPack 6.2.1 → Full Voice Control

> **Note (Stage 1, 2026-05-15):** This document is a historical migration record. Phase 3 (NVIDIA Riva ASR Docker setup) is superseded — Riva has been removed in Stage 1. For current setup, follow [Software Setup](getting-started/software.md) which reflects the current ASR backend (server.py wrapper; Parakeet swap pending Stage 2).

Complete step-by-step guide from flashing the Jetson AGX Orin to having full
voice-controlled Spot with navigation, TTS, and arm capabilities.

## Overview

| Component | Technology | Runs on |
|-----------|-----------|---------|
| ASR (speech-to-text) | server.py wrapper (Parakeet swap pending Stage 2) | CPU |
| LLM (intent + chat) | Ollama + qwen2.5:7b | GPU |
| VLM (vision) | Ollama + qwen2.5-vl:7b | GPU |
| TTS (text-to-speech) | Kokoro ONNX (82M params) | CPU |
| Voice pipeline | VAD → ASR → LLM Brain → Dispatch → TTS | Jetson |

## Why flash first?

| Blocker on JetPack 5.1.2 | Fixed by JetPack 6.2.1 |
|---------------------------|------------------------|
| CUDA 11.4 (Ollama needs 12) | CUDA 12.6 |
| glibc 2.31 (wheels need 2.35) | glibc 2.35 |
| Python 3.8 (EOL) | Python 3.10 |

---

## Phase 1: Flash the Jetson

### 1.1 Pre-flash backup

Only one file to save (everything else is on GitHub):

```bash
cat /home/spotdog/dartmouth_spot_capstone/.env
# Save the output somewhere safe — contains robot credentials
```

### 1.2 Flash JetPack 6.2.1

1. Install [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) on a Windows PC
   - SDK Manager auto-configures WSL2, Ubuntu, and USBIPD-Win
   - Install the APX driver when prompted
2. Connect Jetson USB-C port (next to 40-pin header) to Windows PC
3. Put Jetson in Force Recovery Mode (hold REC button + press power)
4. Flash **JetPack 6.2.1** via SDK Manager
5. Choose **Pre-Config** for OEM setup (set username: `spotdog`, password)
   - This enables headless boot — no monitor needed

### 1.3 Verify flash

```bash
nvcc --version        # Should show CUDA 12.x
python3 --version     # Should show 3.10.x
lsb_release -a        # Should show Ubuntu 22.04
```

---

## Phase 2: Clone and Set Up the Project

### 2.1 Clone the repo

```bash
cd /home/spotdog
git clone https://github.com/matteo-corrado/dartmouth_spot_capstone.git
cd dartmouth_spot_capstone
git checkout feature/llm-brain
```

### 2.2 Restore credentials

```bash
nano .env
```

Paste saved credentials (must contain these):

```
BOSDYN_CLIENT_USERNAME=<your-username>
BOSDYN_CLIENT_PASSWORD=<your-password>
BOSDYN_ROBOT_IP=<your-robot-ip>
```

### 2.3 Create Python virtualenv

```bash
python3 -m venv spot-env
source spot-env/bin/activate
pip install --upgrade pip
```

### 2.4 Install PyTorch (JetPack 6 NVIDIA wheel)

```bash
pip install numpy
pip install torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v62
```

### 2.5 Install project dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `bosdyn-client==5.0.1.1` (BD SDK, matched to robot firmware)
- `kokoro-onnx>=0.5.0` (Kokoro TTS)
- `sounddevice`, `webrtcvad`, `grpcio`, etc.

### 2.6 Verify Python packages

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import bosdyn.client; print('BD SDK OK')"
python3 -c "import kokoro_onnx; print('Kokoro OK')"
python3 -c "import sounddevice; print('sounddevice OK')"
```

---

## Phase 3: ASR Backend ~~(NVIDIA Riva — removed in Stage 1)~~

> **Superseded:** Riva has been removed in Stage 1. The ASR backend is now `src/voice_control/server.py` auto-started by `run_voice_control.py`. No manual setup required. This section is kept as a historical record of the original Riva-based setup.

---

## Phase 4: Set Up Ollama (LLM + VLM)

### 4.1 Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 4.2 Pull models

```bash
ollama pull qwen2.5:7b       # LLM brain (intent parsing + chat)
ollama pull qwen2.5-vl:7b    # VLM (visual descriptions from cameras)
```

### 4.3 Verify

```bash
ollama list
# Should show qwen2.5:7b and qwen2.5-vl:7b
```

---

## Phase 5: Set Up Kokoro TTS

### 5.1 Download model

```bash
python scripts/setup_kokoro.py
```

Downloads the Kokoro v1.0 fp16-gpu model (~170MB) and voice embeddings
(~27MB) to `models/tts/kokoro-v1.0/`. The fp16 weights run on the Jetson
AGX Orin's CUDA Execution Provider — no flags needed.

### 5.2 Test TTS

```bash
cd src/voice_control
python spot_tts.py
```

Should play "Hello! I am Spot..." through speakers. If no audio device is
available, it will print `[TTS] Not available` — that's fine for headless setup,
it'll work when a speaker is connected.

---

## Phase 6: Set Up the Microphone

### 6.1 ReSpeaker XVF3800 (recommended)

```bash
# Clone config tool (optional, for mic tuning)
cd /home/spotdog
git clone https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY.git
```

### 6.2 Find your audio device index

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Look for the XVF3800 device. Note its index number.

### 6.3 Configure device in client_mic.py (if needed)

The default device is hardcoded in `src/voice_control/client_mic.py`. If your
device index differs from `24`, update the `INPUT_DEVICE` constant.

---

## Phase 7: Connect to Spot and Upload Map

### 7.1 Connect Jetson to Spot's network

Connect the Jetson to Spot's WiFi or Ethernet. Default robot IP: `192.168.80.3`.

```bash
ping 192.168.80.3
```

### 7.2 Start E-Stop (required before any robot commands)

In a **dedicated terminal** that stays open:

```bash
cd /home/spotdog/dartmouth_spot_capstone
source spot-env/bin/activate
python scripts/estop_run.py
```

> **This must stay running the entire time you use Spot.** It provides the
> software emergency stop. If this process dies, the robot will e-stop.

### 7.3 Record a map on the tablet (if no map exists)

On the Spot tablet:
1. Drive Spot to the starting position
2. Open **Autowalk** > **Record**
3. Walk Spot through all areas you want in the map
4. Place fiducials (AprilTags) at key locations for localization
5. End recording and save

### 7.4 Download map from Spot

```bash
python scripts/download_map_from_spot.py
```

This saves the map to `maps/downloaded_map/` and auto-extracts waypoint names
to `locations.json`.

### 7.5 Name your locations

```bash
python scripts/map_waypoints.py
```

This lists all waypoints and lets you assign human-readable names (e.g.,
`kitchen`, `lobby`, `lab_door`). Names are saved to `locations.json`.

### 7.6 Upload map to Spot's GraphNav

```bash
python scripts/setup_map.py --map-path maps/lab_map2
```

For waypoint-based localization (robot must be at a known waypoint):

```bash
python scripts/setup_map.py --map-path maps/lab_map2 --waypoint-init
```

For fiducial-based localization (robot must see an AprilTag from the map):

```bash
python scripts/setup_map.py --map-path maps/lab_map2
```

---

## Phase 8: Run Voice Control

### 8.1 Pre-flight checklist

Before starting voice control, make sure:

- [ ] E-Stop is running (`python scripts/estop_run.py` in a separate terminal)
- [ ] Ollama is running (`sudo systemctl start ollama`)
- [ ] Map is uploaded and robot is localized (Phase 7)
- [ ] Microphone and speaker are connected

### 8.2 Start voice control

```bash
python scripts/run_voice_control.py
```

This automatically:
1. Checks Ollama is running
2. Starts the ASR bridge server (port 50055)
3. Starts the mic client with VAD, LLM brain, and TTS

### 8.3 Test voice commands

Say "Hey Spot" to wake, then try these commands:

**Basic movement:**
- "Stand up" — Spot stands
- "Sit down" — Spot sits
- "Walk forward 2 meters" — Spot walks forward
- "Turn left 90 degrees" — Spot turns
- "Crouch" — Spot lowers body

**Single-location navigation:**
- "Go to the kitchen" — navigates to saved location "kitchen"
- "Save this location as front desk" — saves current position

**Multi-location navigation:**
- "Loop the map" — visits all saved locations once in recording order
- "Go to kitchen then lab then lobby" — visits specific locations in sequence
- "Patrol the map" — continuously loops all locations until stopped
- "Come back" / "Go home" — returns to where Spot was before navigating

**Vision:**
- "What do you see?" — captures front camera, describes via VLM
- "What's behind you?" — captures back camera

**Safety (bypass LLM, instant execution):**
- "Stop" — stops all movement, cancels any navigation
- "Freeze" — holds position
- "Emergency stop" — immediate stop

**Status:**
- "Battery status" — reports battery level
- "List locations" — lists all saved locations

### 8.4 Manual start (two terminals)

If `run_voice_control.py` doesn't work, start components separately:

**Terminal 1 — ASR Server:**
```bash
cd src/voice_control
python server.py
```

**Terminal 2 — Voice Client:**
```bash
cd src/voice_control
python client_mic.py
```

---

## Phase 10: Test Components Individually

### Test LLM Brain (no robot needed)

```bash
python src/voice_control/llm_brain.py
```

Type commands and verify JSON output:
- "Loop the map" should return `tour` action with `{"locations": "all"}`
- "Go to kitchen then lab" should return `tour` with `{"locations": ["kitchen", "lab"]}`
- "Patrol the map" should return `patrol` action
- "Come back" should return `come_back` action
- "Stand up" should return `stand` action (no regression)

### Test TTS

```bash
python src/voice_control/spot_tts.py af_heart "Testing Kokoro TTS on Spot"
```

### Test ASR Bridge

```bash
python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('ASR Bridge: OK' if s.connect_ex(('127.0.0.1',50055))==0 else 'ASR Bridge: DOWN (auto-starts with run_voice_control.py)'); s.close()"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to Ollama` | Run `sudo systemctl start ollama` |
| `Kokoro model files not found` | Run `python scripts/setup_kokoro.py` |
| `E-Stop error` / robot won't move | Ensure `python scripts/estop_run.py` is running |
| `Not localized` | Run `python scripts/setup_map.py --waypoint-init` |
| `Location 'X' not found` | Run `python scripts/map_waypoints.py` to name waypoints |
| `No waypoints in map` | Upload a map: `python scripts/setup_map.py --map-path ...` |
| ASR gives empty results | Check mic is connected, device index is correct, ASR bridge running (port 50055) |
| "Stop" doesn't cancel navigation | Fixed in latest code, pull and redeploy |
| `ModuleNotFoundError` | Activate venv: `source spot-env/bin/activate` |
| Robot credentials error | Check `.env` has correct username and password |

## Port Reference

| Port | Service |
|------|---------|
| 50055 | ASR bridge server (server.py) |
| 11434 | Ollama LLM/VLM API |

## Boot Sequence (Quick Reference)

After rebooting the Jetson, run these in order:

```bash
cd /home/spotdog/dartmouth_spot_capstone
source spot-env/bin/activate

# 1. Start infrastructure
sudo systemctl start ollama            # Ollama LLM

# 2. Start E-Stop (separate terminal, keep open)
python scripts/estop_run.py

# 3. Upload map + localize (if not already loaded)
python scripts/setup_map.py --map-path maps/lab_map2 --waypoint-init

# 4. Start voice control
python scripts/run_voice_control.py
```
