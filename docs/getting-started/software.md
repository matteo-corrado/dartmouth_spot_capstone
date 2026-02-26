# Software Setup

Complete software setup for a fresh Jetson. Follow these steps in order.

## 1. Install Docker and NVIDIA Container Toolkit

Riva ASR runs as a Docker container with GPU access. JetPack 6.2.1 may include Docker pre-installed, but the NVIDIA container runtime often needs manual setup.

```bash
# Install Docker (skip if 'docker --version' already works)
sudo apt-get update
sudo apt-get install -y docker.io

# Install NVIDIA Container Toolkit (required for GPU access in Docker)
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Allow running Docker without sudo
sudo usermod -aG docker $USER
```

!!! warning "Log out and back in"
    The `usermod` group change requires a new login session. Log out and SSH back in before continuing. Verify with `groups | grep docker`.

Verify Docker can access the GPU:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

If you see GPU info, Docker is ready. If you get `could not select device driver "nvidia"`, the nvidia-container-toolkit is not installed correctly — re-run `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker`.

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
- `nvidia-riva-client` -- Riva ASR Python client
- `sounddevice`, `webrtcvad` -- audio capture and voice activity detection
- `grpcio`, `grpcio-tools` -- gRPC for ASR bridge
- `sherpa-onnx` -- Kokoro TTS and wake word (bundles its own ONNX runtime for aarch64)
- `ultralytics` -- YOLO object detection
- `python-dotenv`, `numpy`, `requests` -- utilities

## 7. NVIDIA Riva ASR (Docker)

Riva provides the speech-to-text engine. It runs as a Docker container with the Canary-Qwen-2.5B ASR model.

### First-Time Setup

```bash
chmod +x scripts/setup_riva.sh
./scripts/setup_riva.sh
```

This downloads the Riva quickstart scripts and configures ASR-only mode. Then initialize the models (downloads ~5GB and runs TensorRT optimization -- takes 15-30 minutes on first run):

```bash
cd ~/riva_quickstart && bash riva_init.sh
```

### Starting Riva

```bash
./scripts/setup_riva.sh start
```

Or manually:

```bash
cd ~/riva_quickstart && bash riva_start.sh
```

### Verify Riva

```bash
./scripts/setup_riva.sh test
```

Or check the Docker container:

```bash
docker ps | grep riva
# Should show riva-speech container running
```

!!! note "Riva starts automatically"
    `run_voice_control.py` checks if the Riva container is running and starts it automatically. You only need to do the first-time setup manually.

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
python scripts/setup_kokoro.py    # Kokoro TTS (~340MB)
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

# 2. Riva ASR
./scripts/setup_riva.sh test

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
