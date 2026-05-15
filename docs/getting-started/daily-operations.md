# Daily Operations

A quick-reference checklist for every session. This assumes all [software](software.md) and [models](models.md) are already installed.

## Boot Sequence (After Jetson Reboot)

When the Jetson reboots, Tailscale reconnects automatically but the voice pipeline needs to be restarted. Ollama restarts automatically if you ran `sudo systemctl enable ollama` during setup.

```bash
# SSH into the Jetson
ssh spotdog@<tailscale-ip>

# Go to the project directory and activate venv
cd ~/spot/dartmouth_spot_capstone
source spot-env/bin/activate

# 1. Verify Ollama is running (should auto-start via systemd)
systemctl is-active ollama
# If "inactive": sudo systemctl start ollama

# 2. Start E-Stop (keep this terminal open, or use web panel)
python scripts/estop_run.py
```

!!! tip "Use the web panel instead of estop_run.py"
    If you prefer phone-based E-Stop, run `python scripts/web_panel.py` in a `tmux` session and open `http://<jetson-ip>:8080` on your phone. This lets you manage E-Stop without keeping an SSH terminal open.

## Starting Voice Control

Open a second terminal (or `tmux` pane):

```bash
cd ~/spot/dartmouth_spot_capstone
source spot-env/bin/activate
python scripts/run_voice_control.py
```

This auto-checks Ollama, starts the ASR bridge, calibrates the mic, warms up the LLM, and opens the microphone. When you see `LISTENING`, the system is ready.

## Pre-Flight Checklist

Before giving Spot commands, verify:

- [ ] Spot is powered on (fans audible)
- [ ] Jetson is on Spot's WiFi (`ping 192.168.80.3`)
- [ ] E-Stop is active (terminal or web panel)
- [ ] Microphone and speaker are plugged in
- [ ] Map is uploaded if using navigation (`python scripts/setup_map.py` — only needed once per power cycle)

## Uploading the Map (If Using Navigation)

GraphNav maps are cleared when Spot reboots. Re-upload after each Spot power cycle:

```bash
python scripts/setup_map.py --map-path maps/lab_map2
```

For localization, either:

- **Fiducial**: Position Spot where it can see an AprilTag from the map (auto-localizes)
- **Waypoint**: Drive Spot to a known location and use `--waypoint-init`

## Stopping

1. Say **"sit down"** to Spot (or "stop" if mid-motion)
2. **Ctrl+C** in the voice control terminal — Spot sits and powers off
3. **Ctrl+C** in the E-Stop terminal — releases E-Stop

### When launched with `wakespot`

The orchestrator owns both E-Stop and voice control in a single terminal,
so the shutdown is driven by `wakespot.py` itself. Three keys, three
behaviors:

| Key | Behavior |
|---|---|
| **Ctrl+C** (once) | **Graceful.** wakespot asks voice control to run `blocking_sit` then `power_off` motors, waits up to 45s, then releases the E-Stop endpoint **without** issuing a motor cut. |
| **Ctrl+C twice** | **Escalates to emergency** mid-graceful: issues an immediate E-Stop CUT and `SIGKILL`s voice control. |
| **Ctrl+\\** (`SIGQUIT`) | **Always-immediate emergency.** Same as double Ctrl+C but no first-press needed. Use this when Spot is doing something unsafe and you don't want to wait for the sit. |

After the first Ctrl+C, wakespot prints `...waiting for voice control` every
5 seconds so you can tell it's waiting on Spot, not hung. If voice control
doesn't exit within 45 seconds, wakespot escalates to the emergency cut on
its own — you don't have to press anything else.

!!! warning "The graceful path is graceful, the emergency path is not"
    The graceful path leaves Spot sitting on its hocks with motors cleanly
    powered off — what you want at the end of a normal session. The
    emergency cut de-energizes motors instantly; if Spot is standing or
    moving, it will collapse onto its joints. Only use Ctrl+\\ / double
    Ctrl+C when something is genuinely wrong.

## Using tmux for Persistent Sessions

If you're SSHing in, use `tmux` so processes survive SSH disconnects:

```bash
# Start a new tmux session
tmux new -s spot

# Split into panes: Ctrl+B then %  (vertical split)
# Switch panes: Ctrl+B then arrow keys

# Pane 1: E-Stop (or web panel)
python scripts/estop_run.py

# Pane 2: Voice control
python scripts/run_voice_control.py

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t spot
```

## Quick Service Health Check

```bash
# All-in-one status check
systemctl is-active ollama && echo "Ollama: OK" || echo "Ollama: DOWN"
python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('ASR Bridge: OK' if s.connect_ex(('127.0.0.1',50055))==0 else 'ASR Bridge: DOWN'); s.close()"
ping -c 1 -W 1 192.168.80.3 > /dev/null 2>&1 && echo "Spot: OK" || echo "Spot: DOWN"
```

Or open the web panel at `http://<jetson-ip>:8080` — the status bar shows live indicators for all services.
