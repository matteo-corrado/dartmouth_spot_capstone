---
name: spot-services-check
description: Probe estop, voice_control, and web_panel processes, Ollama model load state (qwen2.5 + qwen2.5vl), the Riva ASR container, audio devices, and Spot session reachability. Prints a one-screen health report so you know whether the stack is actually up before saying "Hey Spot".
disable-model-invocation: true
---

# Spot services check

Run a single end-to-end health probe across the Spot stack on this Jetson. Useful when:
- You just rebooted the Jetson and want to know if anything auto-started.
- "Hey Spot" isn't waking the robot and you need to find which link is broken.
- You're about to demo and want a green-light dashboard.

## What to do

Run the bundled script and present its output to the user, then offer next steps for anything that's red.

```
bash .claude/skills/spot-services-check/healthcheck.sh
```

If the script reports anything as DOWN or DEGRADED, do not "fix" it silently — surface the failure to the user and ask which they want to investigate. Common failure modes:

- **estop DOWN** → the robot will not move regardless. User must run `python scripts/estop_run.py` in a dedicated terminal.
- **Ollama models not loaded** → first request will cold-start (~17s for VLM). Suggest the user pre-warm with `ollama run qwen2.5:7b ''` and `ollama run qwen2.5vl:7b ''`.
- **Riva container not running** → ASR is dead. Suggest `bash scripts/setup_riva.sh` or `docker start <riva-container>`.
- **Audio device missing** → microphone (XVF3800 reSpeaker) is at a different ALSA index than expected. Suggest `arecord -l` to find the new index.

## What NOT to do

- Don't start, kill, or restart any service yourself. This skill is read-only diagnosis. The user controls the lifecycle of estop / voice_control / web_panel.
- Don't run `pip install` or modify `requirements.txt` if a Python import fails. Surface the error and let the user decide.
