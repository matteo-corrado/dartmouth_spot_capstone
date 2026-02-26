# Web Panel

The web panel (`scripts/web_panel.py`) is a mobile-friendly control interface for managing Spot's E-Stop and voice pipeline from a phone or laptop browser. It uses no external dependencies — just Python's built-in HTTP server.

## Starting the Web Panel

```bash
cd ~/spot/dartmouth_spot_capstone
source spot-env/bin/activate
python scripts/web_panel.py
```

Output:

```
==================================================
  Spot Web Control Panel
==================================================

  Listening on port 8080
  Local:     http://localhost:8080
  Network:   http://192.168.80.2:8080

  Open this URL on your phone to control Spot.
  Press Ctrl+C to stop.
```

Open the **Network** URL on your phone or laptop. If using Tailscale, use `http://<tailscale-ip>:8080`.

## What It Does

The panel has two main sections:

### E-Stop Controls

| Button | Action |
|--------|--------|
| **Claim** | Connects to Spot and registers a software E-Stop endpoint. The robot cannot move until an E-Stop is claimed. |
| **E-STOP** (large red button) | Immediately cuts motor power. The robot will collapse to the ground. Use in emergencies. |
| **Release** | Re-enables motor power after an E-Stop. The robot can move again once released. |

Workflow: **Claim** first, then the robot can accept commands. Hit **E-STOP** if anything goes wrong. Hit **Release** to resume.

### Voice Pipeline Controls

| Button | Action |
|--------|--------|
| **Start** | Launches `run_voice_control.py` as a background process on the Jetson |
| **Stop** | Kills the voice control process |

This lets you start/stop the voice pipeline without an SSH terminal.

### Status Bar

Four live indicators update every 2 seconds:

| Indicator | Green | Red | Yellow |
|-----------|-------|-----|--------|
| **Riva** | ASR Docker container running (port 50051) | Container stopped | — |
| **Ollama** | LLM service running (port 11434) | Service stopped | — |
| **E-Stop** | E-Stop claimed and active | E-Stop triggered (motors cut) | Not claimed |
| **Voice** | Voice pipeline process running | Pipeline stopped | — |

### Log Panel

A scrollable log at the bottom shows all actions and their results (last 50 entries). Resets on page refresh.

## Options

```bash
# Custom port
python scripts/web_panel.py --port 9090

# UI-only mode (no Spot connection, for testing the UI)
python scripts/web_panel.py --no-spot
```

## Running as a Background Service

To keep the web panel running even after disconnecting SSH:

```bash
# Option 1: tmux
tmux new -s panel
python scripts/web_panel.py
# Ctrl+B, D to detach

# Option 2: nohup
nohup python scripts/web_panel.py > /dev/null 2>&1 &
```

## Using the Web Panel as Your Only E-Stop

The web panel can fully replace `estop_run.py`. Instead of keeping an SSH terminal open with `estop_run.py`, start the web panel and use the **Claim** button from your phone. The panel runs its own E-Stop keepalive in a background thread.

!!! warning "Keep the browser tab open"
    The E-Stop keepalive runs on the Jetson (server-side), not in the browser. The browser is just a control interface. If you close the browser tab, the E-Stop remains active until you either press E-STOP or kill the `web_panel.py` process. Killing the process will trigger E-Stop (motors cut).

## Security Considerations

The web panel has **no authentication**. Anyone who can reach port 8080 on the Jetson can control the robot. This is acceptable when:

- The Jetson is only accessible via Spot's private WiFi network
- Access is restricted to your Tailscale tailnet

Do **not** expose port 8080 to the public internet.

## Architecture

```
Phone/Laptop                    Jetson (web_panel.py)             Spot
┌──────────┐   HTTP GET /    ┌────────────────────┐
│ Browser   │ ──────────────►│ HTML page served   │
│           │                │                    │
│           │  POST /estop/* │ EStopManager       │◄── BD SDK ──► E-Stop service
│           │ ──────────────►│ (keepalive thread) │
│           │                │                    │
│           │ POST /voice/*  │ VoicePipelineManager│──► subprocess: run_voice_control.py
│           │ ──────────────►│                    │
│           │                │                    │
│           │  GET /status   │ Port checks +      │
│           │ ──────────────►│ manager status     │
└──────────┘   (every 2s)   └────────────────────┘
```
