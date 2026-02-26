# Dartmouth Setup

Campus-specific configuration for the Spot robot at Dartmouth College.

---

## Hardware

- **Robot**: Boston Dynamics Spot (firmware 5.0.1.1)
- **Compute**: NVIDIA Jetson AGX Orin 64GB (JetPack 6.2.1)
- **Microphone**: ReSpeaker XVF3800 USB 4-Mic Array
- **Username on Jetson**: `spotdog`

---

## Network

Spot and the Jetson must be on the same network.

- **Spot IP**: `192.168.80.3` (configured in `.env`)
- **Robot credentials**: Set in `.env` as `BOSDYN_CLIENT_USERNAME` and `BOSDYN_CLIENT_PASSWORD`

To verify connectivity:

```bash
ping 192.168.80.3
```

---

## Connecting to the Jetson

### Via Tailscale (Remote — Recommended)

See the [Remote Access (Tailscale)](../getting-started/tailscale.md) guide for full setup. Once configured:

```bash
ssh spotdog@<tailscale-ip>
```

This works from anywhere — on campus, at home, or on the go.

### Via Local SSH (On Spot's Network)

If you're connected to Spot's WiFi, the Jetson is reachable at its local IP:

```bash
ssh spotdog@<jetson-local-ip>
```

### Via Serial Console (USB — Last Resort)

For initial setup or when the network is down. Connect a USB-C cable from the Jetson to your laptop.

On macOS:

```bash
ls /dev/cu.usbmodem*
sudo screen /dev/cu.usbmodem<DEVICE_NUMBER> 115200
```

On Linux:

```bash
ls /dev/ttyACM*
sudo screen /dev/ttyACM0 115200
```

Exit: `Ctrl+A`, then `K`, then `Y`

---

## Map Locations

The lab map is stored in `maps/lab_map3/`. Named waypoints are in `locations.json`.

To add or rename locations:

```bash
python scripts/map_waypoints.py --list    # See all waypoints
python scripts/map_waypoints.py           # Interactive naming
```

Or via voice: "Save this location as reception desk"

---

## Audio Device

The ReSpeaker XVF3800 typically appears as device index 24. Verify with:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

If the index differs, pass it explicitly:

```bash
python scripts/run_voice_control.py --device <INDEX>
```

---

## Project Repository

```
https://github.com/matteo-corrado/dartmouth_spot_capstone
```

- **Main branch**: `main`
- **Feature branch**: `feature/llm-brain` (voice control + LLM)
- Push requires manual authentication (HTTPS, no credential helper on Jetson)
