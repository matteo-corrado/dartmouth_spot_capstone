# Daily Operations (Operator Runbook)

The day-of procedure for running Spot as a voice-controlled demo. It assumes [software](software.md) and [models](models.md) are installed. For *why* things behave as they do, read the [pipeline explainer](../architecture/pipeline-explainer.md).

!!! danger "The E-Stop is the only thing that is guaranteed"
    Keep the hardware E-Stop (or the tablet E-Stop) within reach for the whole event. Voice "stop" and `spotctl stop` are fast, but they only send a motion-stop command; they cannot help during robot bring-up (power-on / stand, about 30 s) and need the pipeline to be running. The real motor cut is the E-Stop, or Ctrl+\\ in the wakespot terminal.

---

## 1. Pre-flight checks (about 10 minutes, before visitors)

Work through these in order on the Jetson (`ssh spotdog@<tailscale-ip>`, then `cd ~/spot/dartmouth_spot_capstone`). Do all of it inside **tmux** (`tmux new -s spot`) so an ssh drop does not kill the demo.

| # | Check | Command / how | Good looks like |
|---|---|---|---|
| 1 | Brain server up | `systemctl is-active llama-server` then `curl -s localhost:11435/health` | `active`, and an `ok`-style 200 reply. A 503 means still loading: wait. Ollama is **not** needed on the default backend |
| 2 | Robot reachable at the right address | `ping -c 2 192.168.50.3` (wired LAN) and `nc -zv 192.168.50.3 443` | Replies. The `SPOT_IP` and `BOSDYN_ROBOT_IP` settings (in the project environment file, which also holds API keys: never display or paste it) must both be this address. Confirm with the launch log line `[Spot] <ip> reachable` |
| 3 | Mic plugged in | `arecord -l \| grep -i xvf` | The reSpeaker XVF3800 shows up. USB control also works: `spot-env/bin/python scripts/xvf_probe.py live` prints direction numbers while you talk (Ctrl+C) |
| 4 | Speaker plugged in | `aplay -l \| grep -i uac` | `UACDemoV1.0` listed |
| 5 | E-Stop reachable | Hardware button / tablet in hand | You can press it |
| 6 | Event knowledge filled in | Open `src/voice_control/knowledge/event_knowledge.md` | No `[FILL IN]` left on lines you care about (lines still containing it are silently dropped, which is safe: Spot says it does not know) |
| 7 | Battery and space | Robot battery above about 50 %; `df -h /mnt/ssd` has room | |
| 8 | Wi-Fi decision | Primary voice is ElevenLabs (cloud). Venue Wi-Fi bad? Plan to launch with `--tts-backend kokoro` | See section 6 |
| 9 | Arm pose | Decided 2026-10-28: the folded **joints** pose (`WALK_JOINTS` in `src/session.py`) is used for both talking and walking (`WALK_POSE = "joints"`). Re-test with `scripts/try_walk_pose.py` only if you change it (stop voice control first; it takes the lease) | Claw opens/closes without hitting the body |
| 10 | Clear floor | About 1.5 m of free floor around Spot if you want it to dance | Dance refuses otherwise |

Also keep the **map decision** in mind (section 2): with a map the robot powers on and stands *at launch*; without one it stands on the first command that needs it.

---

## 2. Launch

One command owns everything (E-Stop claim, ASR server, voice client, shutdown):

```bash
cd ~/spot/dartmouth_spot_capstone
scripts/wakespot <map>            # e.g. scripts/wakespot jackson : upload that map, localise, stand at launch
scripts/wakespot                  # same, with the last-used map
scripts/wakespot --no-map         # no map upload; robot stands on the first command that needs it
```

Common options (all optional; they are forwarded to the voice client):

| Option | Effect |
|---|---|
| `--no-map` | Skip the map upload. Use it when the demo is stationary |
| `--no-arm` | Opt out of the automatic arm fold + gripper "mouth" (default is on, see below). Same as `auto_arm: false` in `config/tuning.yaml`. "Enable mouth" by voice still works |
| `--enable-arm` | No-op, kept so old launch lines still work (the arm and mouth are now automatic) |
| `--no-transcript-log` | Privacy: logs and `spotctl status` store `transcript: null`; console still shows what was heard |
| `--latency off\|summary\|file\|all` | Timing telemetry. `file` writes `logs/latency-<timestamp>.jsonl`; `summary` prints a percentile table at shutdown |
| `--tts-backend kokoro\|elevenlabs` | Force the voice engine (Kokoro is local, works with no network) |
| `--no-tts` | Silent mode (chimes and ring still work) |
| `--no-wake-word` | Always listening, no wake phrase (not for a crowd) |
| `--volume 0.0-1.5` | Speaker gain for speech and chimes |
| `--mic-channel N` | 0 = beamformed (default), 1 = ASR beam |
| `--device N` / `--output-device N` | Override auto-detected audio devices |
| `--skip-services` | Do not try to start Ollama |
| `--debug-audio` | Print mic levels every 5 s |

Not forwarded by wakespot (run `scripts/run_voice_control.py` directly if you need them, with the E-Stop running separately): `--dry-run`, `--no-brain`, `--allow-no-safety-kws`, `--latency-out`.

Recommended for a poster demo: `scripts/wakespot --no-map --no-transcript-log --latency file`. (Add `--no-arm` if you do not want the gripper mouth.)

### Arm, mouth and gaze (automatic)

- **The arm only moves once Spot is standing with motors on.** Launch does not bring the robot up. After the first stand (`spotctl stand`, a voice "stand up", or the first command that brings the session up) the arm folds into the joints pose (slow, about 2.5 s) and the gripper mouth turns on. If the session came up during a walk, the fold happens at the start of the next turn.
- **One pose for talking and walking.** The arm stays folded and the mouth keeps moving while Spot walks; walking sends no arm command (a turned shoulder is re-centred first, without waiting).
- **Dances** close the claw, stow the arm, dance, then fold back and turn the mouth on again.
- **stop / freeze** stow the arm and turn the mouth off; the next turn that is allowed to move folds it back. **E-stop, sit, power off and shutdown** stow it and do not bring it back on their own; after an e-stop the arm folds again only after an explicit stand.
- **"Disable mouth"** (or `spotctl stow`) stows the arm and turns the automatic fold off for the rest of the run; "enable mouth" turns it back on.
- **Gaze:** while Spot stands still in the folded pose, the shoulder turns slowly toward the locked speaker, at most `gaze_max_deg` (default 45°) either side, only when the speaker's direction moved more than 15° and at most every 2 s. It re-centres when the lock is released. A stop always wins: no gaze follows a stop. Set `gaze_max_deg: 0` in `config/tuning.yaml` to turn gaze off; if Spot turns the wrong way, set `gaze_sh0_sign: -1`.

### What you should see (about 60 to 90 s from cold)

1. `[wakespot] E-Stop active.`
2. ASR server start-up lines, then `Voice client`.
3. `Calibrating noise floor ... please stay quiet` (2 s). **Stay quiet.**
4. `[Brain] Ready`, warm-up (15 to 20 s cold).
5. `[TTS] Ready`, `[WakeWord] keyword spotter ready`, **`[SafetyKWS] ready`**.
6. `[Spot] <ip> reachable` (or map load + `Connected and standing`).
7. `[MicArray] XVF3800 control ready (DoA poller 20 Hz)`.
8. `[SafetyKWS] safety thread running`, then the **rising power-on chime** and `LISTENING`. Ring: steady dim white.

### Launch problems

| You see | Meaning | Do |
|---|---|---|
| `Auto-enabling dry-run mode` | Robot unreachable; the pipeline talks but never moves the robot | Fix the address / cabling, restart. **Do not demo like this by accident** |
| `Refusing to start` (safety keyword spotter) | Safety "stop" model failed to load; startup is blocked on purpose | `spot-env/bin/python scripts/setup_safety_kws.py`, restart. Only as a last resort add `--allow-no-safety-kws` by running `run_voice_control.py` directly |
| `Another wakespot is already running` | A stale or live copy | `pgrep -af wakespot`; stop the old one, or delete `/tmp/wakespot.pid` if the process is gone |
| `Could not find mic device matching 'XVF3800'` | Mic not enumerated | Replug, `arecord -l`, relaunch |
| `[Brain] llama-server not available yet` | Brain down or loading; only simple commands work until it answers | `systemctl status llama-server`; it is re-checked on every utterance |
| `[Control] server not started (port 8770 ...)` | Another copy owns the control port; spotctl will not work | Find and stop the old process (`pgrep -af client_mic`) |
| `[Gate] ... High noise ... during calibration` | Room was loud while calibrating | Usually fine (the floor drops by itself); relaunch if the room was truly abnormal |

### Warm the robot up before visitors

If you launched with `--no-map`, **the first command that needs the robot stands it (about 30 s) and a "stop" during that window does nothing** (no session to stop). Do it yourself first:

```bash
ssh -t spotdog@<ip> ~/spot/dartmouth_spot_capstone/scripts/spotctl stand
```

Wait until `Connected and standing ready for voice commands` prints.

---

## 3. Operator control with spotctl (from your laptop)

`spotctl` talks to the running pipeline over a localhost-only port, so you run it **on the Jetson via ssh**:

```bash
ssh spotdog@<ip> ~/spot/dartmouth_spot_capstone/scripts/spotctl status       # one-shot, prints JSON
ssh -t spotdog@<ip> ~/spot/dartmouth_spot_capstone/scripts/spotctl          # hotkey mode (needs -t); q quits
```

Tip: add `alias spotctl='ssh -t spotdog@<ip> ~/spot/dartmouth_spot_capstone/scripts/spotctl'` on your laptop (use `ssh` without `-t` for one-shots in scripts).

Exit codes: 0 ok, 1 error reply, 2 cannot reach the server (pipeline not running), 3 no reply within 5 s (the command may still be executing).

### Commands

**Immediate** (act now, even while Spot is thinking or talking):

| Command | Hotkey | What it does |
|---|---|---|
| `stop` | `s` | Cuts speech, aborts any action chain, sends the robot motion-stop **first**, then stows the arm |
| `freeze` | `f` | Same as stop but holds a stand if the robot is already standing (never stands a sitting robot) |
| `sit` | none | Stops, stows the arm, then sits |
| `stand` | none | Stands the robot (brings the session up if needed), then folds the arm + turns the mouth on (unless `--no-arm` / "disable mouth") |
| `stow` | `w` | Stows the arm and turns the mouth (and the automatic fold) off; also cancels an arm-dependent routine |
| `hush` | `h` | Cuts Spot's speech now and aborts the action chain; does not touch the robot's legs |
| `mute` | `m` | Ignore all audio (except the safety stop word). Ring goes solid red |
| `unmute` | `u` | Resume listening |
| `status` | `?` | JSON snapshot (below) |

**Queued** (done by the main loop between frames, so a busy turn can delay them; the reply `{"queued": true}` means accepted, not done):

| Command | Hotkey | What it does |
|---|---|---|
| `listen` | `l` | Open a conversation **now with no speaker lock** (push-to-talk; accepts any direction) |
| `release` | `r` | Drop the speaker lock and return to idle (descending "released" cue). The next wake phrase starts fresh |
| `reset` | `x` | Release **and** clear the brain's conversation history |
| `say <text>` | none | Speak that text (in the current persona's voice) |
| `persona <name>` | none | Switch persona by name (see `config/personas.yaml`); unknown names are created on the fly (slow, up to about 30 s) |
| `crowd [on\|off]` | `c` | Crowd mode: after every reply go back to idle, so every turn needs a wake phrase. No argument toggles |
| `voiceprint [on\|off]` | `v` | **Status flag only**: voiceprint checking is not implemented on this build |

`status` fields: `state` (`WAKE_WORD`/`LISTENING`/`THINKING`/`RESPONDING`), `locked_deg` (speaker direction, body frame: 0 ahead, +90 left, null when unlocked), `floor_dbfs`, `last_transcript`, `last_latency_s`, `muted`, `crowd`, `voiceprint`, `safety_kws_ok` (must be `true`).

Voice "stop" and "freeze" are the always-on safety words (heard with no wake phrase). A transcribed "e-stop" still acts as a motion-stop, but it is not in the always-on listener. **None of these is the E-Stop.**

---

## 4. What the ring and chimes mean

| What visitors see/hear | Meaning |
|---|---|
| Ring steady dim white | Idle, waiting for a wake phrase |
| Rising two-note chime, ring green sweeping toward the speaker | Mic is open for that person: go ahead |
| Ring solid green | Mic open, direction unknown (anyone can speak) |
| Lower two-note rising chime, ring breathing amber | Heard you, working on it |
| Ring solid blue, soft tick | Spot is speaking (mic is closed; wait) |
| Ring flashes red for a moment | Someone outside the locked direction spoke and was ignored |
| Descending two-note chime, ring returns to white | Conversation ended (20 s of silence, or you ran `release`) |
| Ring solid red | Muted from spotctl |
| Double beep | A robot action was accepted |
| Low descending buzz | Something failed |
| Rising three-note power-on chime | Pipeline just finished starting up |

Wake phrases: **HEY SPOT, LISTEN UP SPOT, EXCUSE ME SPOT, HEY THERE SPOT.** After a reply there is a short settle (about 0.5 s) and a go-ahead chime; the speaker does not need to say the wake phrase again, but anyone else must. The lock accepts speech within +/-35 degrees of where the speaker was.

Spot never says "stop", "halt", "freeze" or "estop" aloud (its own voice would trigger the safety listener).

---

## 5. Recovery playbook

First move for anything odd: `spotctl status`. Look at `state`, `muted`, `locked_deg`, `floor_dbfs`, `safety_kws_ok`, and the console.

| Problem | Likely cause | Fix |
|---|---|---|
| **Spot not hearing** (no reaction to a person) | Muted (ring red) | `spotctl unmute` |
| | Person outside the lock (red flashes; console `[Lock] rejected`) | `spotctl release`, then they say a wake phrase facing the mic; or `spotctl listen` (no lock) |
| | Gate drop (console `[Gate] dropped: ... SNR ...`): too quiet or room too loud | Ask them to come closer/speak up; if the whole room is loud, lower `min_snr_db` in `config/tuning.yaml` and restart |
| | Wake phrase not recognised | Say it clearly, closer, no pause after "Spot"; use `spotctl listen` meanwhile |
| | `floor_dbfs` in `status` is frozen or console stopped printing `[Say 'Hey Spot'...]` | Mic audio gone (USB glitch / unplugged): replug, then restart wakespot |
| **Wrong person locked** | Two people at similar angles, or the previous talker's direction | `spotctl release`; the right person says a wake phrase. Another person can also just say the wake phrase to take over. `spotctl crowd on` forces a fresh wake every turn |
| **Spot stuck talking / droning** | Long reply or slow TTS | `spotctl hush` (or say "stop"; that also halts motion) |
| **Long silence after a question** | Brain slow (about 4 s: Spot says "Hmm, let me think."; longer: llama-server stalled) | Wait; if it never answers: `curl localhost:11435/health`, `systemctl restart llama-server`, then `spotctl reset` |
| **Brain answers nonsense / lost context** | Conversation history polluted | `spotctl reset` |
| **Robot not responding to commands** | Dry-run (launch said `Auto-enabling dry-run`) | Fix address, restart wakespot |
| | Command printed `>>> FAILED` | Read the `[Spot]` line above it. Common: not standing (`spotctl stand`), battery low, lease taken by the tablet (close the tablet's control), E-Stop engaged |
| | Robot sitting after a stop/sit | `spotctl stand` (or "stand up") |
| **Dance refused** | "I need a little more room" / "can't check my surroundings" | Clear 1.5 m around Spot; or skip it |
| **Spot walked / is walking and you want it halted** | | `spotctl stop` (or say "stop"). Still moving: **E-Stop** |
| **TTS Wi-Fi drop** | ElevenLabs unreachable | Automatic: after a 4.5 s wait each sentence falls back to the local Kokoro voice and stays there for 60 s. Expect a voice change, not silence. Console: `using fallback voice af_sarah`. To make it permanent: restart with `--tts-backend kokoro` |
| **Mic unplugged** | USB unplugged mid-run | Audio stops arriving. Replug, wait 5 s, restart wakespot (Ctrl+C once, wait for the sit). If only the *control* link is lost the console says `[MicArray] XVF3800 control lost`: voice keeps working but the ring is dark and the speaker lock accepts everyone |
| **Ring dark, voice fine** | Mic control link lost or LED write errors | Voice is unaffected; replug and restart when convenient |
| **`spotctl: cannot reach control server`** | Pipeline not running or port 8770 conflict | Check the wakespot terminal; relaunch |
| **Safety stop word not reacting** | `safety_kws_ok` false, or the person is very quiet/far | Use `spotctl stop`. If `safety_kws_ok` is false, restart wakespot (it fails closed at launch) |
| **Whole pipeline wedged** | | Ctrl+C once in the wakespot terminal (graceful, up to 45 s), wait; if still hung Ctrl+C again (emergency cut), restart |

Quick rule: **anything involving the robot's body that you do not like: `spotctl stop`, then E-Stop if it continues.**

---

## 6. Shutdown

In the wakespot terminal:

| Key | Effect |
|---|---|
| **Ctrl+C once** | **Graceful.** Asks voice control to stow the arm, sit and power off the motors; waits up to 45 s (prints `...waiting for voice control` every 5 s); then releases the E-Stop endpoint without cutting power |
| **Ctrl+C twice** | Escalates to an immediate E-Stop CUT and kills voice control |
| **Ctrl+\\** | Immediate E-Stop CUT, no first press needed |

!!! warning "Emergency cut collapses a standing robot"
    The emergency path de-energises the motors instantly. Use it only when something is genuinely wrong. Normal end of day: Ctrl+C **once** and wait.

Saying "shut down" / "power off" also powers the robot down. After shutdown, confirm the terminal printed `Released E-Stop and cleaned up` before unplugging anything.

Optional afterwards: copy `logs/utterances/<date>.jsonl`, `logs/conversations/<date>.jsonl` and any `logs/latency-*.jsonl` off for tuning (they contain transcripts unless you launched with `--no-transcript-log`).

---

## 7. Rollback

**Code.** The known-good pre-upgrade state is tagged `pre-demo-2026-10-29`:

```bash
cd ~/spot/dartmouth_spot_capstone
git status                       # commit or stash local edits first
git checkout pre-demo-2026-10-29
```

This is the old pipeline (no speaker lock, no ring signals, no spotctl, no emotes, ASR/brain/dispatcher as before the upgrade). Return to the demo branch with `git checkout demo-poster-2026-10-29`. A rehearsal-approved tag, if one has been created, is `demo-ready-2026-10-29`; check with `git tag`.

**Individual tuning.** Delete or comment lines in `config/tuning.yaml` to return to the defaults (then restart).

**Bosdyn SDK** (upgraded to 5.2.0 on 2026-10-27). Rollback venv: `/mnt/ssd/spot-backups/spot-env-5.1.4-backup`:

```bash
cd ~/spot/dartmouth_spot_capstone
rm -rf spot-env && cp -a /mnt/ssd/spot-backups/spot-env-5.1.4-backup spot-env
# then restore the 5.1.4 bosdyn pins in requirements.txt
```

Run the backup venv only from the `spot-env` path (its scripts hard-code it).

**Wake phrase list.** Backups of the keyword files are in `/mnt/ssd/spot-backups/kws-2026-10-28/`. To change phrases, edit `config/wake_phrases.txt` and run `spot-env/bin/python scripts/setup_kws.py`.

---

## Appendix: other tools

| Tool | Use |
|---|---|
| `scripts/estop_run.py` | Standalone E-Stop terminal (not needed when using wakespot, which holds the E-Stop itself) |
| `scripts/web_panel.py` | Phone-friendly E-Stop and control panel on port 8080 |
| `scripts/xvf_probe.py live\|clap\|index\|leds\|show` | Mic direction, calibration and ring preview (run with the pipeline *stopped*) |
| `scripts/try_walk_pose.py` | Pick the tucked-arm walking pose (pipeline stopped) |
| `scripts/record_demo_clips.py` | Guided recording of test clips in quiet/loud/overlap conditions |
| `scripts/setup_map.py` / `scripts/map_waypoints.py` | Map upload / labelling |

Quick service check:

```bash
systemctl is-active llama-server && echo "brain: OK" || echo "brain: DOWN"
python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('ASR bridge: OK' if s.connect_ex(('127.0.0.1',50055))==0 else 'ASR bridge: DOWN'); s.close()"
ping -c 1 -W 1 192.168.50.3 > /dev/null 2>&1 && echo "Spot: OK" || echo "Spot: DOWN"
```
