# Restructure: test on the robot, then merge into the upgrade branch

Context: `restructure-modular` (tag `restructure-verified-2026-11-03`, checked out in the main folder). The voice/robot stack moved from `src/voice_control/` into `src/spot/` packages plus feature folders (`src/spot/features/`); map in [INDEX.md](../../../INDEX.md) and [package-map.md](../../architecture/package-map.md), operator-visible changes in [restructure-notes.md](../restructure-notes.md). Behaviour is meant to be identical to `demo-poster-2026-10-29`, including the ported safety-KWS fixes (`bd384b4`..`44125a7`) and voiceprint / beam tooling (`d16cb6d` -> `c3d4a4b`). **Never run on the robot yet.**

Bench evidence so far (2026-11-03): 936 tests pass incl. golden conversations; `tests/audio/replay_demo.py` output identical to the demo branch; `wakespot --dry-run` reaches LISTENING with `[SafetyKWS] ... 3 streams, thr 0.25` and shuts down cleanly on Ctrl+C; whole-branch safety review and `/code-review high` done, findings fixed or logged as bugs G1-G11.

Merge target: `tour_guide_upgrade_matteo` is an ancestor of `restructure-modular` (254 commits ahead), so the merge is a fast-forward. **Ask Matteo before merging or pushing.**

---

## 0. Before the robot run

1. **Reinstall the llama-server unit** (sudo, operator): `sudo cp systemd/llama-server.service /etc/systemd/system/ && sudo systemctl daemon-reload`, then `sudo systemctl restart llama-server` when nothing is live. Check: `curl -s localhost:11435/health` returns ok. Then the `src/voice_control/grammar` symlink can be deleted.
2. **Re-run the bench gates from the main folder** (it has the credentials file and models):
   ```
   spot-env/bin/python -m pytest -q
   spot-env/bin/python -m pytest tests/integration -q
   spot-env/bin/python tests/audio/replay_demo.py
   scripts/wakespot --dry-run      # Ctrl+C after LISTENING
   ```
   **Pass:** all tests pass; replay numbers equal `docs/project/demo-tuning-results.md` (wake hits, gate pass/fail, voiceprint sims, safety-KWS hits); dry run shows `LLM Brain (llamacpp)`, safety KWS ready, exits in a few seconds.
3. Spot on, e-stop in hand, clear 3 m area, demo pipeline not running.

## 1. On-robot checklist (in this order)

Start with `scripts/wakespot`. Log to keep: the full terminal output, saved under `logs/`.

| # | Check | Pass |
|---|---|---|
| 1.1 | Startup | No audio overflow lines; `scripts/spotctl status` shows `safety_kws_ok: true` and `mic_ok: true`; `[Features]` lists all 8 features, no `FAILED`; startup order is safety/mic/wake first, then ASR/TTS/brain (expected, see notes) |
| 1.2 | Stop, normal voice | "Hey Spot, walk forward", then "stop": motion ends within ~0.5 s |
| 1.3 | Stop, quiet voice | Same with a quiet "stop" and a quiet "freeze" (the bug the 3-stream fix targets) |
| 1.4 | Stop from 3 m | Same, speaking from ~3 m |
| 1.5 | `spotctl stop` | Stops from another terminal |
| 1.6 | Arm fold + mouth | After stand, arm folds to the talk pose; jaw moves while Spot speaks |
| 1.7 | Gaze | Talk from ~45 deg left and right: `[Spot] gaze: speaker ... -> shoulder yaw ...` log line, arm turns toward you, never past +/-60 |
| 1.8 | Walk with mouth | Walk command while talking: arm stows for the walk; "stop" stows |
| 1.9 | Quiet one-word commands | "sit", "stand", "stop" said quietly are heard |
| 1.10 | Noise | ~20 claps / taps / coughs: each gives `[Short] dropped`, no false commands |
| 1.11 | Dances | Each dance once at 1 m clearance; "stop" during one ends it and stows |
| 1.12 | Voiceprint sims | `[Voice] sim=` lines appear; own follow-ups score above threshold |
| 1.13 | Mic reconnect | Unplug the XVF3800 for ~5 s and plug it back: pipeline recovers, wake works again |
| 1.14 | Ctrl+C | Shows `[Voice] Ctrl+C received`, `[Spot] Sitting...`, `[Spot] Powering off motors...`, `Session closed.`; robot sits and powers off |
| 1.15 | Ctrl+C during startup | Start with `--map`, press Ctrl+C after the stand but before LISTENING: robot still sits and powers off |

Known and accepted (do not test): "come here" while the LLM is down walks to an odom point (G1, fixed with the vision port); stop is ignored for up to ~35 s during shutdown, use the e-stop (G6).

## 2. Scripts that need one robot or mic run

These had only import-path changes. One short run each; **pass** = starts, does its job, no `ImportError`:
`estop_run`, `xvf_probe`, `mic_verify`, `record_demo_clips`, `diag_av_leds`, `diag_charging`, `try_walk_pose`, `test_visual_nav`, `spike_gripper_rate`, `preflight_2c_capture`, `probe_image_sources`, `map_waypoints`, `setup_map`, `download_map_from_spot`, `web_panel`, `spotctl`.

Record results (date, pass/fail, log line for any failure) at the bottom of this file. Failures go to `docs/project/bugs.md`.

## 3. Merge into `tour_guide_upgrade_matteo`

Only after sections 1 and 2 pass, and **after Matteo says yes**:
```
git checkout tour_guide_upgrade_matteo
git merge --ff-only restructure-modular
spot-env/bin/python -m pytest -q
git push origin tour_guide_upgrade_matteo        # ask first
```
Then tag `restructure-robot-verified-<date>`, update `docs/project/status.md`, and delete the `restructure-modular` branch once nothing else needs it.

## 4. After the merge (separate todo files)

1. Vision transplant onto the merged branch as features ([vision-actions.md](vision-actions.md) section 3), which also fixes G1-G4.
2. Friendly Spot social mode on top of that ([friendly-spot-social-mode.md](friendly-spot-social-mode.md)).
3. Mic and voice validation and XVF3800 improvements ([mic-and-voice.md](mic-and-voice.md)); this code is already on the restructure.
4. Remaining small bugs G5-G9, G11 and the parked cleanup (unused ActionSpec fields, duplicate helpers, patrol vs tour, splitting `conversation/loop.py` and `features/visual_nav/navigator.py`).

---

## Results

| Date | Section | Result | Notes |
|---|---|---|---|
