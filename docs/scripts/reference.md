# Scripts Reference

All runnable scripts in the `scripts/` directory. Scripts assume the Python
virtual environment is activated (`source spot-env/bin/activate`) and the
working directory is the project root.

---

## run_voice_control.py

Main entry point for the voice control system. Auto-starts Riva, Ollama, and
the ASR bridge server, then launches the microphone client.

```bash
python scripts/run_voice_control.py
python scripts/run_voice_control.py --no-tts
python scripts/run_voice_control.py --skip-services --no-wake-word
python scripts/run_voice_control.py --server-only
python scripts/run_voice_control.py --device 24 --debug-audio
```

| Flag | Description |
|---|---|
| `--skip-services` | Do not auto-start Riva or Ollama (assume already running) |
| `--no-brain` | Disable LLM brain; use regex-only intent parsing (legacy mode) |
| `--no-tts` | Disable text-to-speech output |
| `--no-wake-word` | Always listening (skip "Hey Spot" requirement) |
| `--server-only` | Start only the ASR bridge server; do not start the mic client |
| `--device INDEX` | Audio input device index (default: 24 = XVF3800 beamforming mic) |
| `--debug-audio` | Print frame-level audio energy, noise floor, and threshold |

**Prerequisite:** E-Stop must be running in a separate terminal.

---

## estop_run.py

E-Stop keepalive. Must run in a separate terminal for the entire session.
Claims the E-Stop endpoint and sends heartbeats every 0.5s. If this script
exits, the robot's motors are cut within 3 seconds.

```bash
python scripts/estop_run.py
```

No flags. Press Ctrl+C to stop (triggers motor cutoff).

**Alternative:** Use `web_panel.py` for phone-based E-Stop control.

---

## setup_kws.py

Download the sherpa-onnx keyword spotting model for "Hey Spot" wake word
detection. Downloads ~5MB of int8-quantized ONNX files from HuggingFace and
creates the `keywords.txt` file.

```bash
python scripts/setup_kws.py          # download and setup
python scripts/setup_kws.py --check  # verify existing install
```

| Flag | Description |
|---|---|
| `--check` | Verify all required files exist without downloading |

Model: `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`
Target: `models/kws/`

---

## setup_kokoro.py

Download the Kokoro v1.0 TTS model files (~200MB total) for the
`kokoro-onnx` GPU pipeline.

```bash
python scripts/setup_kokoro.py
```

No flags. The script checks if the model files already exist and skips if so.
Requires `wget` to be installed.

Model: Kokoro v1.0 (54 voices, default `af_sarah`), CUDA Execution Provider
Target: `models/tts/kokoro-v1.0/`
Files: `kokoro-v1.0.fp16-gpu.onnx`, `voices-v1.0.bin`

---

## setup_map.py

Upload a GraphNav map to Spot and initialize localization. The robot must be
standing and able to see a fiducial marker from the map for localization.

```bash
python scripts/setup_map.py
python scripts/setup_map.py --map-path maps/my_custom_map
python scripts/setup_map.py --no-localize
python scripts/setup_map.py --waypoint-init
```

| Flag | Description |
|---|---|
| `--map-path PATH` | Path to map directory (default: `maps/lab_map2`) |
| `--no-localize` | Upload map but skip localization initialization |
| `--waypoint-init` | Use waypoint-based localization instead of fiducial-based |

**Prerequisite:** E-Stop must be running.

---

## setup_riva.sh

Manage the NVIDIA Riva ASR Docker container. Handles first-time setup,
model initialization, and server lifecycle.

```bash
./scripts/setup_riva.sh              # full setup (first time)
./scripts/setup_riva.sh init         # download + TensorRT optimize models
./scripts/setup_riva.sh start        # start Riva server container
./scripts/setup_riva.sh stop         # stop Riva server container
./scripts/setup_riva.sh restart      # restart Riva server
./scripts/setup_riva.sh test         # test ASR connection
```

| Subcommand | Description |
|---|---|
| `setup` | Install nvidia-riva-client + download Riva quickstart (default) |
| `init` | Download and optimize models with TensorRT (15-30 min first run) |
| `start` | Start the `riva-speech` Docker container |
| `stop` | Stop the `riva-speech` Docker container |
| `restart` | Stop then start the container |
| `test` | Send 1 second of silence to Riva and verify response |

---

## download_map_from_spot.py

Fetch the current GraphNav map from Spot's onboard storage and save it
locally. Also extracts named waypoints into `locations.json`.

```bash
python scripts/download_map_from_spot.py
python scripts/download_map_from_spot.py --output maps/my_new_map
```

| Flag | Description |
|---|---|
| `--output DIR` | Output directory (default: `maps/downloaded_map`) |

Downloads the graph, waypoint snapshots, and edge snapshots. Does not
require E-Stop or standing.

---

## map_waypoints.py

Interactive tool for listing GraphNav waypoints and mapping them to
human-readable location names. Saves mappings to `locations.json`.

```bash
python scripts/map_waypoints.py
```

No flags. The script connects to Spot (no standing required), lists all
waypoints in the uploaded map, and provides an interactive prompt:

```
> 3 kitchen          # map waypoint #3 to "kitchen"
> current lab        # save robot's current position as "lab"
> list               # show all existing mappings
> q                  # quit
```

---

## calibrate_door.py

Calibrate the push bar handle position for door opening. Three modes:

```bash
# Step 1: Capture images from Spot's cameras
python scripts/calibrate_door.py capture

# Step 2: Set handle coordinates from inspected images
python scripts/calibrate_door.py set \
    --handle-x 400 --handle-y 250 \
    --hinge left \
    --camera left

# All-in-one GUI mode (requires OpenCV display)
python scripts/calibrate_door.py interactive
```

**capture mode flags:** None. Saves images to `door_calibration/`.

**set mode flags:**

| Flag | Description |
|---|---|
| `--handle-x X` | Handle X pixel coordinate (required) |
| `--handle-y Y` | Handle Y pixel coordinate (required) |
| `--hinge left\|right` | Which side the door hinge is on (required) |
| `--camera left\|right` | Camera: left=frontleft, right=frontright (default: left) |
| `--from-sbs` | Coordinates are from the side-by-side image (auto-adjusts) |

**interactive mode flags:** None. Requires OpenCV with display support.

Saves calibration to `door_config.json`.

---

## test_door_standalone.py

Test push-bar door opening using the BD DoorService `AutoPushCommand`.
The robot must be standing ~1m from the door, facing it.

```bash
python scripts/test_door_standalone.py --vlm
python scripts/test_door_standalone.py --vlm --depth 0.5
python scripts/test_door_standalone.py --vlm --hinge right --camera left
python scripts/test_door_standalone.py --vision-xyz 1.65,2.68,0.81
```

| Flag | Description |
|---|---|
| `--vlm` | Use VLM to auto-detect push bar pixel location |
| `--vision-xyz X,Y,Z` | Direct 3D vision-frame coordinates (skip VLM) |
| `--hinge left\|right` | Door hinge side (default: left) |
| `--depth METERS` | Estimated depth to door for VLM mode (default: 0.5) |
| `--camera left\|right` | Which front camera for VLM mode (default: left) |

Must specify either `--vlm` or `--vision-xyz`.

**Prerequisites:** E-Stop running, Ollama running (for `--vlm`).

---

## test_visual_nav.py

Test YOLO detection on Spot's camera feed. Runs YOLO-World
(open-vocabulary) and YOLOv8n (person detection), prints bounding boxes,
confidence scores, and steering math.

```bash
python scripts/test_visual_nav.py                        # detect "person"
python scripts/test_visual_nav.py --target "red chair"   # detect specific object
python scripts/test_visual_nav.py --save                  # save annotated image
python scripts/test_visual_nav.py --no-spot               # use synthetic test image
```

| Flag | Description |
|---|---|
| `--target TEXT` | Object to search for with YOLO-World (default: "person") |
| `--save` | Save annotated image to `/tmp/visual_nav_test.jpg` |
| `--no-spot` | Use a synthetic gray test image instead of Spot's camera |

---

## web_panel.py

Mobile-friendly web control panel for E-Stop management and voice pipeline
start/stop. No external dependencies (stdlib HTTP server). Accessible from
any device on the same network or via Tailscale.

```bash
python scripts/web_panel.py
python scripts/web_panel.py --port 9090
python scripts/web_panel.py --no-spot
```

| Flag | Description |
|---|---|
| `--port PORT` | HTTP port (default: 8080) |
| `--no-spot` | UI-only mode: E-Stop buttons disabled, no Spot connection |

The web panel provides:
- **E-Stop controls:** Claim, E-Stop (cut motors), Release
- **Voice pipeline controls:** Start, Stop
- **Status indicators:** Riva, Ollama, E-Stop, Voice pipeline (polled every 2s)
