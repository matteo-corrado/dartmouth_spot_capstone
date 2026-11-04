# Open work

One file per topic. Each file has: what is built but unvalidated, the exact steps and pass criteria to validate it, and ranked next improvements. Mark items done in place (strike through + commit hash); delete a file once everything in it is done or moved to `status.md`.

| File | Topic | Blocking? |
|---|---|---|
| [restructure-test-and-merge.md](restructure-test-and-merge.md) | Robot checklist for `restructure-modular`, script checks, fast-forward merge into `tour_guide_upgrade_matteo` | Yes: comes first; vision and mic work land on top of it |
| [mic-and-voice.md](mic-and-voice.md) | Voiceprint running enrolment, beam-steering A/B, XVF3800 improvements (AEC, per-beam lock, gain, HPF) | No; all new behaviour is off by default |
| [vision-actions.md](vision-actions.md) | Vision upgrade (describe, find, look at speaker, go to, follow on YOLO26-l TensorRT): pre-run setup, on-robot checks with pass criteria, transplant, ranked improvements | Yes for go-to/follow: needs the safety KWS merge and the latency check first; lives on `vision/perception-v2` |
| [friendly-spot-social-mode.md](friendly-spot-social-mode.md) | Friendly Spot `social_mode` (comfort model, gripper gaze, emotion + gesture models): on-robot checks, merge gates, ranked improvements | No; only runs when asked, lives on `vision/friendly-social` |
| [voice-commands.md](voice-commands.md) | Relative and obstacle-distance moves, lost short commands, direction-lock on/off and re-pointing | No; not started |
