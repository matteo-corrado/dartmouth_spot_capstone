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

## Exit code on a refused start

`src/voice_control/client_mic.py` now exits with `main()`'s return code: 1 when startup is refused (a feature failure on the robot, the safety-KWS gate, ROBOT STILL POWERED) or the mic stream cannot open, 0 otherwise. `wakespot` prints it as `Voice control exited (rc=1)` and returns it; neither launcher restarts on a non-zero code. Before, every refusal exited 0.

## Fixed beams cleared at mic open

The voice client turns the XVF3800 fixed beams off once when it opens the mic (a runtime write, nothing saved to the device), so beams left on by an interrupted `record_demo_clips.py --beam` no longer carry into a session (bug G10).

## Ctrl-C during startup

Ctrl-C at any point after the features have started now runs cleanup exactly once: the robot is sat and powered off, and the mic, safety thread and control server are closed.

## Adding or removing a feature

- Add: create a folder `src/spot/features/<name>/` with a `FEATURE` declaration (`__init__.py`) and `handlers.py`, then add one line `"<name>",` to `ENABLED` in `src/spot/features/enabled.py`. The LLM grammar, prompt and regex table are generated from it (spotctl help is still hand-written in `scripts/spotctl.py`). If it uses another feature, declare `requires=("<other>",)`; a missing requirement is a startup failure. Worked example: [Adding Commands](../guides/adding-commands.md).
- Remove: delete its line from `src/spot/features/enabled.py`. That turns it off, including `mouth` (no automatic arm fold or jaw; `stow` still stows) and `gaze` (no gaze toward the speaker). Check that no other feature lists it in `requires` (`follow` requires `visual_nav`, which requires `navigation`), or startup refuses on the robot. Optionally delete the folder. Steps: [Package Map](../architecture/package-map.md#how-to-remove-a-feature).

## Operator action pending: reinstall the llama-server unit

The repo unit no longer mounts any grammar folder: both the demo branch and the restructure send the grammar inside every request, and llama-server never reads a grammar file. Without the mount the service works the same on every branch (a mount of a folder missing on the checked-out branch makes Docker create a root-owned empty folder in the repo, which can block `git checkout`). From the repo root, run:

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/ && sudo systemctl daemon-reload
```

Restart the service when it is safe to do so (this interrupts the brain, so not while the demo pipeline is live): `sudo systemctl restart llama-server`. After the new unit is running, nothing uses the `src/voice_control/grammar` symlink and it can be deleted.
