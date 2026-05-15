# Software Setup

Complete software setup for a fresh Jetson. Follow these steps in order.

## 1. Docker and NVIDIA Container Toolkit (Optional)

Docker is not required for the current ASR backend (server.py wrapper). It is pre-installed on JetPack 6.2.1 if needed for other purposes. Skip this section unless you have a specific need for Docker.

## 2. Clone the Repository

```bash
cd ~/spot
git clone https://github.com/matteo-corrado/dartmouth_spot_capstone.git
cd dartmouth_spot_capstone
git checkout feature/llm-brain
```

## 3. Environment Variables

Create a `.env` file in the project root with your Spot robot credentials:

```bash
cat > .env << 'EOF'
BOSDYN_CLIENT_USERNAME=your_username_here
BOSDYN_CLIENT_PASSWORD=your_password_here
BOSDYN_ROBOT_IP=192.168.80.3
EOF
```

!!! warning "Do not commit `.env`"
    The `.env` file contains robot credentials and is listed in `.gitignore`. Never commit it. Ask a team member for the actual values.

The IP defaults to `192.168.80.3` if not set. Username and password must match what is configured in Spot's admin console (`https://192.168.80.3`).

## 4. Python Virtual Environment

```bash
python3 -m venv spot-env
source spot-env/bin/activate
```

You must activate this venv in every terminal session before running any project scripts.

## 5. Install PyTorch

PyTorch must be installed from NVIDIA's JetPack 6 wheel index (standard PyPI wheels do not work on aarch64 Jetson):

```bash
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v61
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output: a PyTorch version and `CUDA: True`.

!!! note "PyTorch version"
    The exact version depends on what NVIDIA publishes for your JetPack. As of JetPack 6.2.1, PyTorch 2.5.x is available. The project does not pin a specific PyTorch version -- any JP6-compatible build works.

!!! note "Install PyTorch before requirements.txt"
    PyTorch must be installed first because `ultralytics` (YOLO) depends on it. If you run `pip install -r requirements.txt` without PyTorch, pip will try to install a CPU-only PyTorch from PyPI which will not work on the Jetson.

## 6. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `bosdyn-client`, `bosdyn-mission`, `bosdyn-choreography-client` (v5.0.1.1) -- Spot SDK
- `sounddevice`, `webrtcvad` -- audio capture and voice activity detection
- `grpcio`, `grpcio-tools` -- gRPC for ASR bridge
- `sherpa-onnx` -- Kokoro TTS and wake word (bundles its own ONNX runtime for aarch64)
- `ultralytics` -- YOLO object detection
- `python-dotenv`, `numpy`, `requests` -- utilities

## 7. ASR Backend

The ASR backend runs as a gRPC server (`src/voice_control/server.py`) on port 50055. It is auto-started by `run_voice_control.py` — no manual setup required. A Parakeet-based replacement is planned for Stage 2.

## 8. Ollama (LLM / VLM)

Ollama serves the text LLM and vision-language model for Spot's brain.

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama installs as a systemd service and starts automatically. Enable it so it persists across reboots:

```bash
sudo systemctl enable ollama
```

### Pull Models

```bash
ollama pull qwen2.5:7b       # Text LLM (~4.7GB)
ollama pull qwen2.5vl:7b     # Vision-Language Model (~4.7GB)
```

### Verify Ollama

```bash
# Check service is running
systemctl status ollama

# Test inference
ollama run qwen2.5:7b "Say hello in one sentence"
```

!!! note "VRAM sharing"
    The text LLM and VLM share GPU memory. Ollama swaps models in and out as needed. The first request after a swap takes 5-15 seconds (model loading). Subsequent requests are fast (~1-2s).

## 9. Download Models

TTS and wake word models must be downloaded separately:

```bash
python scripts/setup_kokoro.py    # Kokoro TTS v1.0 GPU (~200MB)
python scripts/setup_kws.py       # Wake word KWS (~5MB)
```

See [Model Downloads](models.md) for details.

## Verify Everything

Run these checks to confirm the full stack is working:

```bash
# 1. Spot connection
python -c "
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD
from bosdyn.client import create_standard_sdk
sdk = create_standard_sdk('test')
robot = sdk.create_robot(BOSDYN_ROBOT_IP)
robot.authenticate(BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD)
print('Spot: OK')
"

# 2. ASR bridge
python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('ASR Bridge: OK' if s.connect_ex(('127.0.0.1',50055))==0 else 'ASR Bridge: DOWN (will auto-start with run_voice_control.py)'); s.close()"

# 3. Ollama
ollama list
# Should show qwen2.5:7b and qwen2.5vl:7b

# 4. TTS
python src/voice_control/spot_tts.py
# Should speak a test sentence

# 5. Wake word
python scripts/setup_kws.py --check
# Should report "KWS model OK"

# 6. Microphone
python src/voice_control/client_mic.py --list-devices
# Should list audio devices including XVF3800
```

If all checks pass, proceed to the [Quick Start](quickstart.md).
