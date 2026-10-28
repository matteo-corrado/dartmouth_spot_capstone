# Poster-Demo Readiness — Design

**Date:** 2026-10-27 · **Demo:** Thursday 2026-10-29 · **Branch:** `tour_guide_upgrade_matteo`
Related: [status](../../project/status.md) · [roadmap](../../project/roadmap.md) (cut line) · [bug register](../../project/bugs.md) (IDs like A1, B3 below)

## 1. Intent

Spot is shown at a poster/networking event (not Thayer, no tour). Visitors chat with the operator, then turn to Spot and talk to it. The room is loud with other conversations, and people may stand at any angle around Spot.

Today the interaction falls apart in that setting:
- the wake word misfires or misses;
- people get cut off mid-sentence;
- crowd chatter leaks in;
- nobody can tell when they may speak;
- the operator has no quick way to recover.

**Success on Thursday:**
1. A visitor wakes Spot with a phrase, and Spot visibly turns its attention to them.
2. They can talk back and forth, pauses included, without re-waking.
3. Spot ignores the people around them.
4. It is always obvious whether Spot is listening, thinking, speaking, or idle.
5. The operator can fix any stuck state from a laptop with one command.

**Anti-goals** (a failure even if everything "works"):
- Spot answering a stranger's side conversation.
- The arm swinging near someone's face.
- Spot stuck in a state that needs a restart.
- Spot speaking garbled text or making things up about itself.

## 2. Conversation model (operator's decisions)

- The **wake phrase** starts a conversation and **locks Spot to that speaker**.
- After the lock, the conversation is **multi-turn with no re-waking**.
- **Someone else takes over only by saying the wake phrase.** That re-locks Spot to the new speaker.
- The lock ends on silence timeout, on an operator command, or on a new wake.
- The wake word must work on its own; **the laptop is a backup**, never required.

**Event setup (operator's answers, 2026-10-27; revised Wed 2026-10-28):**
- **Wake phrase:** not final. The operator wants a better phrase than "Hey Spot". Several candidates get recorded in the noise test, and the winner is picked by the numbers (fewest false fires, most hits).
- **Movement:** Spot stands, and **walking by voice is allowed**. Whenever Spot walks, the arm goes to its most tucked position first: stowed, or the tightest carry pose tested on the robot, chosen so the gripper can still open and close as a mouth. When the walk ends, the arm returns to its conversation pose.
- **Robot time:** it is now **Wednesday**, and the full plan is still the target. The operator records test clips in a **quiet room and a loud room, including overlapping voices**, with the mic in its Thursday mounting. Thresholds are tuned from those clips the same day.
- **Network:** venue Wi-Fi with Tailscale. `spotctl` runs over SSH. The voice is ElevenLabs, with a per-sentence Kokoro fallback.

## 3. Scope

### In (Thursday)
| # | Part | Bugs addressed |
|---|---|---|
| P1 | Operator control from the laptop (terminal) | A14 |
| P2 | Audio front-end rework, **mic-reopen timing**, measurement | A1–A3, A5–A8, A15–A17, A19 |
| P3 | Wake phrase + tuning on recorded crowd noise | A9, A18 |
| P4 | Speaker lock by mic direction (XVF3800), then voiceprint check (P4b) | A4 |
| P5 | Turn signalling: mic LED ring first, Spot LEDs second, chimes | A10–A12 |
| P6 | Conversation-quality fixes | A13, B1–B7, B11–B14, E1 |
| P7 | Arm safety + mouth refinement | C1–C4 |
| P8 | Arm gaze toward the locked speaker (direction only) | — |
| P9 | `emote` action: all built-in routines, natural-language triggers | — |
| P10 | Event knowledge pack (important) + venue map (last, if time) | — |
| P11 | Demo operations: rollback, logging, rehearsal, docs | E3, E4, E8 + pipeline explainer |

**First robot session (Wednesday):** quiet/loud/overlapping-voice recordings for P2/P3, the P4 clap-test calibration, the P5 LED limits check, and the P7 walking-pose test.

Priority if time runs short: P1 → P7 (safety parts, including the walking pose) → P2 → P5 → P3 → P4 → P6 → P10 (knowledge) → P11 → P9 → P8 → P4b (voiceprint) → P7 (mouth) → P10 (map). Anything unfinished by Wednesday night is cut, not rushed.

### Out (after the demo)
- Camera world model and person tracking.
- GPU YOLO and the vision overhaul (2C).
- VLFM hand-off.
- Custom-trained wake models (the training may run in parallel, but they are only swapped in if they win the noise test).
- Semantic end-of-turn model.
- Refactors (dispatcher split, client_mic split, double-import fix E2).
- Web-panel controls.

## 4. Parts

### P1. Operator control
- **What:** a command-line tool, `spotctl`, used from the laptop over SSH (Tailscale) into the Jetson.
  - It talks to the running voice process through a small control endpoint bound to localhost only.
  - Two modes: one-shot (`spotctl mute`) and interactive single-key mode (`spotctl` → hotkeys) for speed during the event.
- **Commands:**

  | Group | Commands |
  |---|---|
  | Mic | `mute`, `unmute` |
  | Turn-taking | `listen` (push-to-talk: open a turn, no wake needed), `release` (drop the speaker lock → idle), `reset` (release + clear conversation history) |
  | Speech | `hush` (stop speaking now: flush audio, close mouth), `say "<text>"` (Spot speaks the operator's text) |
  | Persona | `persona <name>` |
  | Robot | `stop`, `freeze`, `sit`, `stand`, `stow` (arm) |
  | Status | `status`: state, lock angle, noise floor dB, last transcript, last latency |
- **Safety:** commands act in **every** voice state, including mid-speech and mid-navigation. Robot commands go through the existing dispatcher safety path. The hard E-Stop stays with wakespot (double Ctrl+C).
- **My calls** (the builder may change these):
  - Control endpoint: HTTP on `127.0.0.1` from a stdlib thread inside the voice process. The web panel can reuse it later.
  - Interactive keys: m mute, u unmute, l listen, r release, x reset, h hush, s stop, f freeze, w stow, ? status.

### P2. Audio front-end rework
1. Feed **every** mic frame to Silero VAD. Replace the "2× adaptive floor" energy gate with:
   - an absolute minimum level (dBFS, configurable);
   - a per-utterance SNR check (speech level minus floor level ≥ a configurable dB, starting around 10 dB).
2. Keep the measured calibration (log it, never overwrite it). Track the noise floor as a slow low-percentile over a few seconds that falls quickly and rises slowly, so it can't ratchet up.
3. **Mic-reopen timing (A19). This is the operator's main dropped-audio problem.** After Spot finishes speaking, the mic stays shut for the drain, the 0.5 s settle and the 150 ms cooldown. Visitors who answer straight away lose their first words.
   - Measure the actual tail: how long Spot's own voice and echo stay audible after playback ends. Cut the reopen delay to just that.
   - Start buffering pre-roll the moment the gate releases.
   - Fire the "listening" signal (P5) **at the exact instant the mic reopens**, never before, so people learn to wait for it.
   - Spot's own TTS stays hard-muted while playing, because the mic can't cancel the separate speaker. Secondary: the wake chime also mutes ~400 ms (A5); fix it the same way, but it is lower priority.
4. Rework the end-of-speech timer:
   - count silence in frames;
   - end only when the VAD says silence **and** the level is below the floor;
   - default ~1.5 s inside a locked conversation;
   - all thresholds configurable.
5. Add a `--mic-channel {0,1}` flag (default 0, unless the noise test shows 1 is better).
6. Fix the wake-input gain clipping (A8) with a peak-safe gain.
7. Pop VAD segments (A15) and fill the pre-roll in every state (A16).
8. **Measurement:**
   - one JSON line per utterance: floor dBFS, speech dBFS (median and p90), SNR, endpoint reason, silence timeout used, duration, transcript, reached-LLM flag;
   - a 1 Hz idle line with floor and threshold.

   This is the "measurement level" the operator tunes from.

### P3. Wake phrase and tuning
- **Tuesday night:** record 3–4 candidate phrases, including `HEY SPOT` as the baseline, from the operator's shortlist. Pick the winner from the replay numbers. Put the chosen phrase(s) in `scripts/setup_kws.py` so they survive regeneration (A18).
- **Choosing candidates:** 3+ syllables, unusual sounds, not common in conversation ("spot" alone is: "spot on", "parking spot"), and easy for a visitor to read off a sign. Every candidate must tokenise with the KWS BPE model.
- **Noise test** (also used by P2):
  - Record both mic channels in a **quiet room** and a **loud room**, plus **overlapping voices** (a second person talking while the visitor speaks; and the visitor with babble from a laptop). Note the approximate dBA of each.
  - Clips per condition: about 10 wake+command clips per candidate phrase, commands with mid-sentence pauses, quick replies right after Spot stops talking (for A19), and a few minutes of no-wake background.
  - Replay the recordings through the real frame loop and report:
    - wake hits and false fires per minute;
    - cut-offs (turn ended before the last expected word);
    - 20 s caps;
    - ASR text.
  - Pick the KWS score/threshold/gain and the mic channel from those numbers. With overlapping voices, also report how often the lock (P4) picks the right direction.
- Extend `scripts/record_corpus.py` (stereo flag) and `tests/audio/eval_wake.py` (room parameter). Add a replay harness for the frame loop.

### P4. Speaker lock by direction
- **New module:** USB control of the XVF3800 (firmware 2.1.1).
  - A background poller at ~20 Hz keeps a short history of (time, azimuth, speech flag).
  - Protocol and parameters are in Appendix A. Reads use vendor control transfers (`0xC0`, `wValue = 0x80|cmd`, `wIndex = resid`, status byte first).
  - Never call `set_configuration` or detach drivers, because the audio stream must keep running.
- **Lock behaviour:**
  - **On wake:** lock angle = circular median of speech-flagged azimuths during the wake phrase.
  - **Each utterance while locked:** accept it only if its median azimuth is within ±35° of the lock (configurable). Otherwise ignore it, with a brief LED flicker and no reply.
  - Each accepted utterance nudges the lock angle toward the new reading, so a speaker who shifts slightly stays locked.
  - **Releases:** a configurable silence timeout (my call: 20 s), `spotctl release` / `reset`, or a new wake phrase (re-locks to the new speaker).
  - **If direction readings are unusable** (no speech-flagged samples): accept the utterance and log it. Never deadlock.
- **Crowd mode:** an operator toggle (`spotctl crowd on`) requires the wake phrase on every turn. This is the fallback if direction locking misbehaves.
- **Calibration:** a clap-test script prints the measured azimuth at known body angles. It stores the mic-to-body yaw offset and direction sign in config. P5 and P8 need body-frame angles.
- **Beam steering** (fixed beams pointed at the lock angle): implement behind a flag, default **off**. Turn on only if the noise test shows a gain.

### P4b. Voiceprint check (after the direction lock is tested)
- Start only once P4 works on the robot.
- **On wake:** compute a speaker embedding from the wake phrase plus the first utterance (WeSpeaker ResNet34-LM ONNX, verified on this Jetson at ~7 ms on CUDA; it needs an 80-mel Kaldi fbank front-end with mean normalisation).
- **Each later utterance:** compute the embedding and accept the utterance only if **both** checks pass: direction within the window **and** cosine similarity ≥ a threshold set from the quiet/loud recordings. If either signal is missing, the other decides.
- Embeddings live in memory only and are dropped on release. Never written to disk (privacy).
- `spotctl voiceprint on|off` toggle; default on only if the recordings show it rejects other voices without rejecting the real speaker.
- Model download: `Wespeaker/wespeaker-voxceleb-resnet34-LM` (`voxceleb_resnet34_LM.onnx`, 26 MB, CC-BY-4.0) into the models dir.

### P5. Turn signalling
Every state change goes through `_enter_state` (fixes A10, A11).

**The mic LED ring is the primary signal** for listening and speaking, because it is more visible than Spot's front LEDs. Spot's LEDs are secondary.

**Before building, measure each device's limits** and record them in the pipeline explainer:
- **XVF3800 ring:** how fast it accepts updates, colour accuracy, whether ring mode (per-LED) and effect changes interfere with direction tracking, and brightness readability in a lit room.
- **Spot AV LEDs:** whether this Spot has `has_audio_visual_system`, colours and patterns available, update latency, and visibility from the side and back.

The table below is the starting mapping; adjust it to the measured limits.

| State | XVF LED ring | Spot status LEDs (if fitted) | Sound |
|---|---|---|---|
| Idle (waiting for wake) | dim slow breath | idle | — |
| Locked / listening | LEDs light **toward the speaker** (ring mode, green) | green | "listening" chime on lock and **each time the mic reopens** after a reply |
| Thinking | amber breath | amber | thinking chime |
| Speaking | solid blue | blue | short "speaking" chime as speech starts (mouth moves) |
| Lock released | brief fade | idle | distinct descending "bye" chime |
| Muted by operator | red | red | — |

- Three distinct chimes, for **listening**, **thinking** and **speaking**, distinguishable from each other and from the power-on chime (A12). They are short and quiet enough not to annoy in a long session. The listening chime and ring change fire exactly when the mic opens (P2 item 3).
- **Unverified:** whether this Spot has the audio-visual LED hardware. If it doesn't, the existing code degrades gracefully, and the LED ring plus sounds carry the signal.

### P6. Conversation-quality fixes
- **A13:** the "stop" safety regex fires only for short command-like utterances (e.g. ≤4 words, or stop/halt as the first word). The safety keyword spotter and `spotctl stop` remain the guaranteed paths.
- **B1:** decode the LLM stream as UTF-8. **B2:** handle escape sequences properly before TTS.
- **B3 / B4:** speak the results of battery, status, obstacle and location queries, either fed back to the LLM or as templated speech (builder's choice). Remove the "battery is looking good" example.
- **B5 / B7:** put the large static prompt first, and the persona and changing state last, so llama.cpp reuses its cache. Drop `uptime_s` / `process_pid` from the prompt. **B6:** a persona switch takes effect for the reply that announces it.
- **B11:** ignore utterances that are only fillers or fewer than ~2 real words.
- **B12:** if no audio has started ~4 s after the end of speech, speak a short persona-appropriate filler ("hmm, let me think").
- **B13:** add `enable_mouth` / `disable_mouth` to the prompt. **B14:** rewrite robotic canned lines and never speak raw keys.
- **E1:** the brain availability check uses the active backend (llama.cpp), not Ollama.

### P7. Arm safety and mouth
- **Safety (C1–C3):**
  - stop / freeze / estop and shutdown also stow the arm, after closing the gripper;
  - any locomotion request first moves the arm to the **walking pose**: the most tucked pose, chosen on the robot between `arm_stow` and a tight carry pose, with the gripper still free so the mouth keeps working. Walking waits until the arm arrives, and the conversation pose returns after the walk;
  - walking speed is capped (configurable; starting value 0.5 m/s) and obstacle avoidance stays on;
  - arm deploy and stow are velocity-limited;
  - add an `arm_deployed` state flag.

  Review all of this with the `safety-reviewer` agent.
- **Mouth (C4):**
  - attack/release smoothing;
  - one normalisation level per whole reply, not per chunk;
  - opening cap ~0.6;
  - ease to a small rest opening between sentences, and close fully at the end of the reply;
  - a configurable lead time measured from a slow-motion video of audio vs claw.

### P8. Arm gaze toward the locked speaker
- On lock, and while the conversation lasts, point the gripper toward the speaker:
  - azimuth from P4, converted to body frame using the P4 calibration;
  - a point ~2 m out at head height;
  - `arm_gaze_command` with low velocity limits, re-issued only when the angle changes by more than ~15°.
- Only when the arm is in its conversation pose and Spot is not walking. Gaze pauses during walks and resumes after.
- For targets behind Spot (beyond ~±120°), don't twist the arm. Optionally yaw the body in place (my call: skip body yaw for Thursday).
- Must coexist with the mouth's gripper commands. Verify on the robot that gaze (arm) and claw (gripper) commands don't cancel each other.
- **Camera refinement of the aim is out of scope for Thursday.**

### P9. `emote` action
- New dispatcher action `emote(name)` that runs a built-in routine by name, confirmed working without a license on 2026-10-27:
  - dances: `hey_join_me`, `performance_bow`, `smooth_sailing`
  - emotes: `happy_gesture`, `tail_wag`, `upright_sit`, `wanna_play`
- **All seven are allowed.** The two mission signals (`signal_alert`, `signal_undock`) are excluded.
- **Natural-language triggers:** the visitor doesn't need the exact name. The system prompt lists each routine with a one-line description and example phrasings. The LLM picks the routine, and the grammar constrains `name` to the allowed set, so a near-miss can't produce an invalid call. Examples:
  - "dance", "show off", "do a trick" → one of the dances
  - "bow", "take a bow" → `performance_bow`
  - "are you happy?", "wag your tail" → the matching emote
  - "sit up nicely" → `upright_sit`
  - "want to play?" → `wanna_play`
  - Plain "dance" picks a dance at random, avoiding the one just done.
- Add the routine list to `scripts/` regression prompts so trigger accuracy can be checked offline.
- Add it to the GBNF grammar and the system prompt.
- While navigating it is refused. Otherwise it stows the arm first, runs the routine, then restores the conversation pose. It stands Spot first if needed.
- `stop` / `freeze` interrupt a routine.
- Big dances move a lot: before running one, check there is clear space (local obstacle grid) or require the operator's `spotctl` OK. My call: check the obstacle grid, and refuse with a spoken line if someone is within ~1.5 m.

### P10. Event knowledge pack (important) and venue map (last)
- **Knowledge, important:** a new knowledge file about the event, the poster, the team and the project. It is loaded with the existing packs and takes priority in the prompt over the Thayer tour material. Content comes from the operator. The builder supplies a template and wiring early so the operator can fill it in during the day.
- **Venue map, lowest priority:** the operator maps and labels the venue with the existing tools (`scripts/setup_map.py`, `scripts/map_waypoints.py`) only if time allows. Everything must work with no map loaded (`--no-map`).

### P11. Demo operations
- **Branching:** everything up to now is committed to `tour_guide_upgrade_matteo`, tagged `pre-demo-2026-10-29`. All demo work happens on a new branch, `demo-poster-2026-10-29`. It is merged back into `tour_guide_upgrade_matteo` only if testing on the robot passes. A second tag marks the version that passes rehearsal.
- **Launch:**
  - wakespot passes `--latency` through (E3);
  - the conversation log goes to `logs/` and not `src/` (E4);
  - tests stop writing to the real log.
- **Privacy:** voiceprints are in memory only (P4b). Visitor transcripts stay in local logs only (builder may add a `--no-transcript-log` flag).
- **TTS:** ElevenLabs (venue Wi-Fi). Build the per-sentence fallback: a failed ElevenLabs sentence is re-synthesised with Kokoro instead of going silent (B8). Kokoro stays warm at startup.
- **Rehearsal:** a full run with babble playing, every `spotctl` command exercised, and an emote run.
- **Docs:**
  - a pipeline explainer at `docs/architecture/pipeline-explainer.md`: every stage, its module and functions, data flow, settings, failure modes and how to debug each;
  - update `daily-operations.md` to the real launch;
  - update `status.md`, `bugs.md` and the roadmap as fixes land.

## 5. Constraints

- Jetson AGX Orin, Python 3.10, venv `spot-env`. bosdyn SDK 5.2.0 (newest on PyPI) against robot OS **5.2.1** (updated 2026-10-28).
- Run pin-guardian on any requirements change.
- Any new dispatcher action must be added to `grammar/spot_action.gbnf`.
- Never change the XVF3800's saved configuration (no `SAVE_CONFIGURATION`). Runtime settings only.
- The existing 104 tests stay green. New pure logic gets unit tests: lock/accept decision, voiceprint accept rule, floor tracker, endpointing, reopen timing, stop-regex rule, chunk escaping, envelope smoothing, emote name selection.
- Robot-touching changes get one on-robot check each, with the E-Stop live.

## 6. Left to the builder

- Module layout and names.
- Control endpoint details.
- Exact thresholds, which the noise test produces.
- How spoken query results reach the user (LLM round-trip vs template).
- Filler wording.
- Chime sound design.
- Order of work within the priority list.

## Appendix A — XVF3800 USB control (firmware 2.1.1, verified live 2026-10-27)

**Device:** USB `2886:001a`. The udev rule `/etc/udev/rules.d/99-respeaker.rules` grants non-root access; pyusb 1.3.1 is in spot-env.

**Transfers:**
- **Read:** `dev.ctrl_transfer(0xC0, 0, 0x80 | cmdid, resid, count*size + 1)`.
  - `resp[0]` is a status byte: 0 means ok, 64 means retry (sleep 10 ms, up to 100 tries).
  - The payload in `resp[1:]` is little-endian.
- **Write:** `dev.ctrl_transfer(0x40, 0, cmdid, resid, payload)`. Writes are runtime-only.
- **Never** send `SAVE_CONFIGURATION` (48/9), `REBOOT` (48/7), `set_configuration`, or a kernel-driver detach.

| Name | resid | cmd | count × type | R/W | Meaning / live value |
|---|---|---|---|---|---|
| VERSION | 48 | 0 | 3 × u8 | R | (2, 1, 1) |
| DOA_VALUE | 20 | 18 | 2 × u16 | R | [azimuth deg 0–359, **speech flag 0/1**]; live: (90, 0) in a quiet room |
| AUDIO_MGR_SELECTED_AZIMUTHS | 35 | 11 | 2 × f32 rad | R | [processed speaker DoA (NaN when no speech), auto-select DoA]; live: [nan, 1.57] |
| AEC_AZIMUTH_VALUES | 33 | 75 | 4 × f32 rad | R | focused beam 1, focused beam 2, free-running, auto-select |
| AEC_SPENERGY_VALUES | 33 | 80 | 4 × f32 | R | per-beam speech energy (>0 = speech), same order |
| AEC_MIC_ARRAY_GEO | 33 | 74 | 12 × f32 | R | mic XYZ in metres; live: square ±0.0333 m |
| AEC_FIXEDBEAMSONOFF | 33 | 37 | 1 × i32 | RW | fixed-beam mode (both beams fixed together) |
| AEC_FIXEDBEAMSAZIMUTH_VALUES | 33 | 81 | 2 × f32 rad | RW | fixed beam azimuths |
| AEC_FIXEDBEAMSGATING | 33 | 83 | 1 × u8 | RW | gate the inactive beam |
| AUDIO_MGR_OP_L / _R | 35 | 15 / 19 | 2 × u8 | RW | (category, source) per USB channel; live: L=(8,0) processed, R=(7,3) ASR beam |
| LED_EFFECT | 20 | 12 | 1 × u8 | RW | 0 off, 1 breath, 2 rainbow, 3 solid, 4 DoA, 5 ring (per-LED); live: 4 |
| LED_BRIGHTNESS / LED_SPEED | 20 | 13 / 15 | 1 × u8 | RW | |
| LED_COLOR | 20 | 16 | 1 × u32 | RW | 0xRRGGBB for breath/solid |
| LED_DOA_COLOR | 20 | 17 | 2 × u32 | RW | [base, direction] colours for DoA mode |
| LED_RING_COLOR | 20 | 19 | 12 × u32 | RW | per-LED colours (ring mode) |
| GPI / MUTE_FUNCTION_ENABLE | — | — | — | RW | new in 2.1.1: physical Mute button readable by host (ids in Seeed `xvf_host.py`) |

**Angle reference:** 0° and the direction of rotation relative to Spot's body are **not yet calibrated**. The P4 clap test settles them.

**Source:** github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY (`python_control/xvf_host.py`), and the XMOS XVF3800 user guide.
