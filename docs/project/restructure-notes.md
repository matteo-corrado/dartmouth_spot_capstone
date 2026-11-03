# Operator notes after the restructure

What changes for the person running the robot now that the stack lives under `src/spot/` (see [package-map.md](../architecture/package-map.md)). Launch commands are unchanged: `scripts/wakespot` and `scripts/run_voice_control.py` still work, and `src/voice_control/client_mic.py` and `server.py` are launch stubs that forward to `src/spot/app/main.py` and `src/spot/stt/server.py`.

## Startup order changed

`src/spot/app/main.py` starts the safety listener, mic, LED ring and wake word before the ASR connection, TTS and brain. Startup log lines therefore come in a different order than before: the `[SafetyKWS]` and `[WakeWord]` lines now appear before the ASR, TTS and `[Brain]` lines. The brain warm-up (up to 90 s waiting for llama-server) now happens with the safety listener already running.

## A feature that fails to load is fatal on the robot

Each enabled feature is loaded and started at step 4. On failure the launcher prints `[Features] FAILED: <name> (...)` with the traceback.

- Robot mode: startup stops with `[Features] refusing to start on the robot`. Fix the feature, or run with `--dry-run`.
- `--dry-run`: only a warning (`[Features] WARNING: --dry-run continues without the failed feature(s)`).

## "ROBOT STILL POWERED" refusal

If a startup step fails after the robot stood (for example the eager `--map` session or the arm start) and closing the half-up session also fails, the launcher prints `[Spot] ROBOT STILL POWERED -- refusing to continue in dry-run; use the e-stop or spotctl` and stops, instead of dropping to dry-run with a standing robot. Use the e-stop or `spotctl`, then restart.

## Ctrl-C during startup

Ctrl-C at any point after the features have started now runs cleanup exactly once: the robot is sat and powered off, and the mic, safety thread and control server are closed.

## Adding or removing a feature

- Add: create a folder `src/spot/features/<name>/` with a `FEATURE` declaration (`__init__.py`) and `handlers.py`, then add one line `"<name>",` to `ENABLED` in `src/spot/features/enabled.py`. The LLM grammar, prompt, regex table and spotctl help are generated from it. Worked example: [Adding Commands](../guides/adding-commands.md).
- Remove: delete its line from `src/spot/features/enabled.py`. Optionally delete the folder, and check that no other feature lists it in `requires`.

## Operator action pending: reinstall the llama-server unit

The installed `llama-server.service` still mounts `src/voice_control/grammar`, a symlink kept only for this reason. The unit in the repo already points at `src/spot/brain/grammar`. From the repo root, run:

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/ && sudo systemctl daemon-reload
```

Restart the service when it is safe to do so (this interrupts the brain, so not while the demo pipeline is live): `sudo systemctl restart llama-server`. After the new unit is running, the `src/voice_control/grammar` symlink can be deleted.
