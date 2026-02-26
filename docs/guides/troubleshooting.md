# Troubleshooting

Common issues and fixes.

---

## Services

| Problem | Fix |
|---------|-----|
| `Riva server NOT detected on port 50051` | Run `./scripts/setup_riva.sh start` |
| `Cannot connect to Ollama` | Run `sudo systemctl start ollama` |
| `Kokoro model files not found` | Run `python scripts/setup_kokoro.py` |
| `KWS model not found` | Run `python scripts/setup_kws.py` |
| `ModuleNotFoundError` | Activate venv: `source spot-env/bin/activate` |
| `docker: Error response from daemon: could not select device driver "nvidia"` | Install nvidia-container-toolkit: `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker` |
| Riva takes 60-90s to start | Normal — Docker container initialization + TensorRT model loading. Wait for port 50051 to accept connections. |
| Ollama not running after reboot | Enable persistence: `sudo systemctl enable ollama` |

---

## Robot Connection

| Problem | Fix |
|---------|-----|
| `Could not connect to robot` | Check network, verify `ping 192.168.80.3` |
| Robot credentials error | Check `.env` has correct BOSDYN_CLIENT_USERNAME and BOSDYN_CLIENT_PASSWORD |
| `E-Stop error` / robot won't move | Ensure `python scripts/estop_run.py` is running in a separate terminal |
| Robot stops unexpectedly | E-Stop process may have died — restart it |
| `KeepaliveMotorsOff` error | Stale keepalive policy from a crashed session. The code auto-clears these, but if persistent: restart Spot or use `remove_all_policies()` (see [SDK Primer](../architecture/sdk-primer.md)) |
| `Timeout waiting for time sync` | Network issue between Jetson and Spot. Verify `ping 192.168.80.3`, then retry |
| Lease contention (tablet vs Jetson) | The voice pipeline uses `take()` to override. If the tablet keeps stealing the lease, disconnect it from Spot's network |

---

## Navigation

| Problem | Fix |
|---------|-----|
| `Location 'X' not found` | Check `locations.json` for exact spelling, or run `python scripts/map_waypoints.py --list` |
| `No waypoints in map` | Upload a map: `python scripts/setup_map.py --map-path maps/YOUR_MAP` |
| `Not localized` | Run `python scripts/setup_map.py --waypoint-init` or localize via tablet |
| Robot is stuck during navigation | Clear obstacles, or the robot will retry up to 2 times then skip |
| Navigation takes a long path | GraphNav uses shortest path by default (fewest edges). Re-record map with more direct paths if needed. |

---

## Voice Recognition

| Problem | Fix |
|---------|-----|
| No speech detected | Check mic is connected, find device with `--list-devices` |
| ASR gives empty/wrong results | Verify Riva is running, check mic device index matches |
| Wake word not triggering | Try speaking louder/closer, or adjust KEYWORDS_THRESHOLD in wake_word.py |
| Wake word triggers too easily | Increase KEYWORDS_THRESHOLD (default 0.25, try 0.35) |
| "Stop" doesn't cancel navigation | Ensure you're on the latest code (safety commands bypass LLM) |
| Commands not understood | Check `[Brain] LLM responded` logs. If LLM is not running, system falls back to regex |

---

## Audio

| Problem | Fix |
|---------|-----|
| No TTS audio output | Check speaker is connected. Run `python src/voice_control/spot_tts.py` to test |
| TTS feedback loop (echo) | The mic should auto-mute during TTS playback. Check on_mute callbacks |
| ALSA errors in logs | These are suppressed by audio_feedback.py. If persistent, check audio device config |
| Audio device not found | Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices |

---

## LLM/VLM

| Problem | Fix |
|---------|-----|
| LLM very slow (>10s) | First request loads model (~17s). Subsequent should be 0.2-1s. Check `ollama list` |
| VLM slow after LLM command | Normal — VLM and LLM share VRAM, models swap (~2-3s). One-time per switch |
| LLM returns wrong action | Check system prompt in `llm_brain.py`. Verify saved_locations matches command |
| JSON parse error | Rare with `format="json"`. Check Ollama version is up to date |
| YOLO model downloading on first run | Normal — takes 30-60s to download weights (~6-25MB). Looks like a hang but is not. Subsequent runs use cached models. |

---

## Port Reference

| Port | Service | Check Command |
|------|---------|---------------|
| 50051 | Riva ASR (Docker) | `curl -s localhost:50051 && echo OK` |
| 50055 | ASR bridge (server.py) | Auto-started by run_voice_control.py |
| 11434 | Ollama LLM/VLM | `curl -s localhost:11434/api/tags` |
| 8080 | Web panel | `python scripts/web_panel.py` |

---

## Boot Sequence (After Reboot)

See [Daily Operations](../getting-started/daily-operations.md) for the full boot sequence checklist.

```bash
cd ~/spot/dartmouth_spot_capstone
source spot-env/bin/activate

# 1. Start infrastructure
./scripts/setup_riva.sh start
sudo systemctl start ollama    # skip if 'systemctl enable ollama' was run

# 2. Start E-Stop (separate terminal, keep open)
python scripts/estop_run.py

# 3. Upload map + localize (if not already loaded)
python scripts/setup_map.py --map-path maps/lab_map2 --waypoint-init

# 4. Start voice control
python scripts/run_voice_control.py
```
