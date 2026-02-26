# Runtime Services

The voice control system depends on several services running on the Jetson
and on the Spot robot itself. This document lists each service, how to start
it, and how to verify it is running.

## Service Table

| Service | Port | Technology | Start Command |
|---|---|---|---|
| Riva ASR | 50051 | Docker (Canary-Qwen-2.5B) | `./scripts/setup_riva.sh start` |
| Ollama | 11434 | systemd service | `sudo systemctl start ollama` |
| ASR Bridge | 50055 | Python gRPC server | Auto-started by `run_voice_control.py` |
| Spot API | 192.168.80.3 | BD SDK (robot onboard) | Always on when Spot is powered |
| Web Panel | 8080 | Python HTTP (stdlib) | `python scripts/web_panel.py` |

### Riva ASR (port 50051)

NVIDIA Riva runs as a Docker container named `riva-speech`. It hosts the
Canary-Qwen-2.5B speech recognition model, which provides high-accuracy
English ASR with punctuation. The container uses the NVIDIA runtime for
GPU access.

First-time setup requires downloading and optimizing models (~15-30 minutes):

```bash
./scripts/setup_riva.sh          # install client + download quickstart
cd ~/riva_quickstart && bash riva_init.sh   # download + TensorRT optimize
./scripts/setup_riva.sh start    # start the server
```

### Ollama (port 11434)

Ollama serves the text LLM (`qwen2.5:7b`) and the VLM (`qwen2.5vl:7b`).
Both models share VRAM and swap in/out as needed. The text LLM is kept
alive for 30 minutes after each request; the VLM for 5 minutes.

```bash
sudo systemctl start ollama
sudo systemctl enable ollama  # persist across reboots
ollama pull qwen2.5:7b       # text LLM (~4.7GB)
ollama pull qwen2.5vl:7b     # vision LLM (~6GB)
```

!!! warning "Enable Ollama for boot persistence"
    Without `sudo systemctl enable ollama`, Ollama will not start after a Jetson reboot. You'd have to manually run `sudo systemctl start ollama` each time.

### ASR Bridge (port 50055)

The ASR bridge server (`src/voice_control/server.py`) translates the custom
`asr.proto` protocol to Riva's native API. It is auto-started by
`run_voice_control.py` unless `--no-server` is passed. The bridge handles
hallucination filtering and minimum duration rejection.

### Spot API (192.168.80.3)

The robot exposes its SDK API on a fixed IP (`192.168.80.3`) over a direct
Ethernet connection. No service management is needed -- the API is available
whenever Spot is powered on. Connection credentials are loaded from `.env`
or environment variables.

### Web Panel (port 8080)

A mobile-friendly web UI for E-Stop management and voice pipeline control.
No external dependencies (Python stdlib only). Access via the Jetson's
Tailscale or local network IP.

## Boot Sequence

Start services in this order:

```
1. Riva ASR          ./scripts/setup_riva.sh start
                     (wait for port 50051 to accept connections)

2. Ollama            sudo systemctl start ollama
                     (wait for port 11434)

3. E-Stop            python scripts/estop_run.py
                     (or: python scripts/web_panel.py for phone-based E-Stop)
                     Must run in a separate terminal. Robot cannot move
                     without an active E-Stop endpoint.

4. Map Upload        python scripts/setup_map.py
   (optional)        Only needed if using GraphNav navigation.
                     Robot must see a fiducial for localization.

5. Voice Control     python scripts/run_voice_control.py
                     Auto-starts Riva, Ollama, and ASR bridge if not running.
                     Calibrates noise floor, warms up LLM, opens mic.
```

`run_voice_control.py` automates steps 1-2 and the ASR bridge. The only
manual prerequisite is the E-Stop (step 3).

## Checking Service Status

### From the command line

```bash
# Riva ASR
docker ps --filter name=riva-speech
python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('OK' if s.connect_ex(('127.0.0.1',50051))==0 else 'DOWN'); s.close()"

# Ollama
systemctl is-active ollama
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# ASR Bridge
python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('OK' if s.connect_ex(('127.0.0.1',50055))==0 else 'DOWN'); s.close()"

# Spot API (from Jetson, Spot must be connected via Ethernet)
ping -c 1 192.168.80.3

# Web Panel
curl -s http://localhost:8080/status
```

### From the web panel

Open `http://<jetson-ip>:8080` in a browser. The status bar shows live
indicators for Riva, Ollama, E-Stop, and voice pipeline status, updated
every 2 seconds.

### From within the voice pipeline

`run_voice_control.py` checks all services on startup and reports status.
If Riva is not running, it attempts to start the Docker container. If
Ollama is not running, it attempts `sudo systemctl start ollama`. If
either fails, it prints instructions and exits (or falls back to
regex-only mode for Ollama).

## Resource Usage

| Service | GPU VRAM | CPU | RAM |
|---|---|---|---|
| Riva ASR (Canary-Qwen-2.5B) | ~3 GB | Low | ~1 GB |
| Ollama qwen2.5:7b | ~5 GB | Low | ~1 GB |
| Ollama qwen2.5vl:7b | ~6 GB | Low | ~1 GB |
| YOLO-World + YOLOv8n | 0 (CPU) | Moderate | ~500 MB |
| sherpa-onnx KWS | 0 (CPU) | Minimal | ~10 MB |
| sherpa-onnx Kokoro TTS | 0 (CPU) | Low | ~350 MB |
| Voice client (VAD, audio) | 0 | Low | ~50 MB |

Riva and Ollama share the GPU. The text LLM and VLM models swap in/out of
VRAM on demand (Ollama manages this automatically). Total peak VRAM usage
is approximately 8-9 GB when both Riva and one Ollama model are loaded.
