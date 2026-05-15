# Hardware Setup

## Jetson AGX Orin 64GB

The compute platform is an NVIDIA Jetson AGX Orin 64GB Developer Kit. It rides on Spot's back payload rail and runs all inference (ASR, LLM, VLM, TTS, wake word, YOLO) on-device.

Key specs:

- **GPU**: Ampere architecture, 2048 CUDA cores, 64 Tensor Cores
- **RAM**: 64GB unified LPDDR5 (shared between CPU and GPU)
- **Storage**: 64GB eMMC + NVMe SSD recommended for models
- **Power**: 15W-60W configurable (use MAXN for best inference speed)

## JetPack 6.2.1

The Jetson runs **JetPack 6.2.1** (L4T r36.4.3, Ubuntu 22.04). This version was chosen because it provides:

- **CUDA 12.6** -- required by Ollama
- **Python 3.10** -- compatible with Boston Dynamics SDK 5.0.1.1
- **glibc 2.35** -- required by sherpa-onnx aarch64 wheels
- **TensorRT 10.3** -- pre-installed, available for future ASR backends (Parakeet)
- **Docker + NVIDIA Container Toolkit** -- pre-installed

!!! warning "Do not upgrade to JetPack 6.3+ without testing"
    JetPack upgrades can break CUDA/TensorRT compatibility with Ollama and other on-device models. Stick with 6.2.1 unless you have a specific reason to upgrade and can re-validate the full stack.

## Flashing the Jetson

If you need to re-flash (new Jetson or recovery), use NVIDIA SDK Manager on a **Windows or Ubuntu x86 host machine**.

### Prerequisites

- USB-C cable (Jetson to host)
- Host machine with [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) installed
- Jetson power supply connected

### Steps

1. Put the Jetson into **Force Recovery Mode**:
    - Power off the Jetson
    - Hold the middle button (Force Recovery) while pressing the power button
    - Release both after 2 seconds
    - Verify on host: `lsusb | grep NVIDIA` should show the device

2. Open SDK Manager on the host, select:
    - **Target**: Jetson AGX Orin Developer Kit
    - **JetPack**: 6.2.1
    - Flash both OS and SDK components

3. Flash takes 15-30 minutes. The Jetson will reboot and show the Ubuntu setup wizard.

### Post-Flash Verification

After first boot, verify the key components:

```bash
# CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.6

# Python
python3 --version
# Expected: Python 3.10.x

# Docker
docker --version
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
# Should show GPU info

# Disk space
df -h /
# Check available space (need ~20GB for models)
```

## Spot Robot Connection

The Jetson connects to Spot over **WiFi**. Spot acts as a WiFi access point.

| Setting | Value |
|---------|-------|
| Spot WiFi SSID | Spot's network (check robot admin console) |
| Spot IP | `192.168.80.3` |
| Admin console | `https://192.168.80.3` (browser) |
| SDK port | 443 (gRPC over HTTPS) |

To connect:

1. Power on Spot (press the power button on the battery, wait for the fans)
2. Connect the Jetson to Spot's WiFi network
3. Verify connectivity:

```bash
ping 192.168.80.3
```

4. Verify SDK access:

```bash
source spot-env/bin/activate
python -c "
from bosdyn.client import create_standard_sdk
sdk = create_standard_sdk('test')
robot = sdk.create_robot('192.168.80.3')
robot.authenticate('USERNAME', 'PASSWORD')
print('Connected:', robot.get_id())
"
```

!!! note "Robot credentials"
    The robot username and password are set in the `.env` file. Ask a team member or check the robot admin console at `https://192.168.80.3`.

## Microphone

The system uses a **ReSpeaker XVF3800 USB 4-mic array** (appears as a 2-channel USB audio device).

- Channel 0 (left): beamformed audio -- used for ASR
- Channel 1 (right): reference/raw -- not used by default
- Sample rate: 16kHz (set in software)
- Device index: typically `24` on the Jetson (verify with `--list-devices`)

To find the device index:

```bash
source spot-env/bin/activate
python src/voice_control/client_mic.py --list-devices
```

Look for a device named `XVF3800` or similar. Pass its index with `--device N` if it is not 24.

## Audio Output

Any USB speaker or 3.5mm audio output works for TTS playback. The system uses `sounddevice` which routes to the default ALSA output device.

To test audio output:

```bash
speaker-test -t sine -f 440 -l 1
```

## Physical Mounting

The Jetson AGX Orin mounts on Spot's **top payload rail** (the flat area on the robot's back).

### Power

The Jetson can be powered in two ways:

| Method | Details |
|--------|---------|
| **Spot payload port** | Spot provides 24V through the payload connector. Use a 24V-to-19V DC-DC converter (barrel jack) to power the Jetson. This is the cleanest setup — no external batteries. |
| **External battery** | A USB-C PD power bank (65W+) or separate AC adapter. Useful for bench testing without Spot. |

### USB Connections

The Jetson has multiple USB ports. Typical setup:

| Port | Device |
|------|--------|
| USB-A | ReSpeaker XVF3800 microphone array |
| USB-A | USB speaker (or use 3.5mm audio jack) |
| USB-C | Power (if using external power) |

If you need more USB ports, use a powered USB hub.

### Cable Management

Route USB cables along the payload rail and secure with zip ties or velcro straps. Keep cables away from Spot's hip joints to avoid snagging during movement. The mic should face forward (toward Spot's head) for best voice pickup.

### Jetson Power Mode

For best inference performance, set the Jetson to MAXN power mode:

```bash
sudo nvpmodel -m 0       # MAXN (60W, all cores, full GPU)
sudo jetson_clocks        # Lock clocks to max frequency
```

Verify with:

```bash
sudo nvpmodel -q          # Should show "MAXN"
```

## Network Topology

```
[Spot Robot]
    |
    | WiFi (192.168.80.x)
    |
[Jetson AGX Orin]
    |--- Ollama LLM/VLM (localhost:11434)
    |--- ASR Bridge (localhost:50055)
    |--- Web Panel (port 8080, accessible via Tailscale or LAN)
    |
    | USB
    |
[ReSpeaker XVF3800 Mic]
[USB/3.5mm Speaker]
```

The Jetson and Spot must be on the same network. If the Jetson also needs internet access (for model downloads, SSH), connect it to a second network (Ethernet or Tailscale).
