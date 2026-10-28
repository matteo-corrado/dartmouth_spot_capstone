# Roadmap

Master plan, written 2026-10-27. Current system state is in [status.md](status.md).
Each workstream says what's planned, what's done, and what goes **before the Thursday 2026-10-29 demo** vs **after**.
Thursday items are a *proposal* until the operator approves the demo scope.

The demo: poster/networking event (not Thayer). Visitors talk to the operator, then turn to Spot. Spot mostly stays put.
Success = Spot answers the person talking to it, ignores the room, makes it obvious when you may talk, and the operator can recover anything from a laptop.

---

## Workstreams

### W1. Operator control (laptop)
- **Thursday:** terminal command set first, controlling the running pipeline: mute/unmute mic, release speaker lock, force listen (push-to-talk), stop speaking, reset conversation, set persona, stop/freeze robot, status readout. The safety commands must work in every voice state.
- **After:** the same controls as buttons on the web panel (`scripts/web_panel.py`).

### W2. Audio front-end and noise (operator item 5)
Findings: the energy gate starves the VAD, calibration is discarded, the floor can ratchet, the wake chime eats the command, and ch0's AGC defeats ratio thresholds.
- **Thursday:**
  1. Feed every frame to the VAD, and use a fixed absolute floor plus a per-utterance SNR check instead of the 2× ratio gate.
  2. Keep the measured calibration (log it, don't overwrite it).
  3. Stop muting the mic for the wake chime.
  4. Add per-utterance metrics (floor dBFS, speech dBFS, SNR, endpoint reason) to a log.
  5. A/B ch0 vs ch1 on recorded babble.
- **Test protocol:** record "hey spot" + commands with mid-sentence pauses, with laptop babble at off / ~50 / ~60 / ~70 dBA, both channels. Replay through the frame loop and count wake hits, false fires, cut-offs and 20 s caps. Tools: `scripts/record_corpus.py` (needs a stereo flag), `scripts/mic_verify.py`, `tests/audio/eval_wake.py` (needs a room parameter), `make eval-voice`.
- **After:** a semantic end-of-turn model (Pipecat Smart Turn v3.2, or Parakeet realtime-EOU) for "listening through pauses".

### W3. Wake word (operator item 3)
- **Thursday:**
  1. Pick a longer, distinctive phrase (or 2–3 alternatives) in the sherpa keyword file. Phrases go in `scripts/setup_kws.py`, which overwrites the file.
  2. Retune score, threshold and gain on recorded babble.
  3. Fix the ×8 gain clipping.
- **In parallel:** run the cluster training for custom livekit models (`scripts/training/wake/`). Switch only if they beat sherpa on the babble recordings.
- **After:** livekit models through the full evaluation (≥3 h of real negatives).

### W4. Conversation flow and speaker lock (operator items 4, 6)
The flow agreed with the operator:
- The wake phrase starts a conversation and **locks Spot to that speaker**.
- The conversation continues multi-turn without re-waking.
- Another person takes over only by saying the wake phrase.
- The lock releases on silence timeout or from the laptop.

**Thursday:**
1. Read the speaker's direction from the XVF3800 at wake time; accept follow-up speech only from within about ±30° of it.
2. Optionally steer the mic beams there.
3. Calibrate the mic's 0° against Spot's body with a clap test.
4. Route every state change through `_enter_state` so the LEDs always match the state.
5. Distinct chimes for "your turn", "lock released" and "thinking".
6. Fix "stop" firing on any sentence containing the word.

**Signalling:**

| State | Mic LED ring | Spot front LEDs | Gripper | Sound |
|---|---|---|---|---|
| Idle | dim | idle colour | stowed or rest | — |
| Locked / listening | points at the speaker | green | faces the speaker | "your turn" chime |
| Thinking | — | amber | — | thinking chime |
| Speaking | solid colour | — | mouth moves | — |
| Released | — | back to idle | — | "bye" chime |

**After:**
- A voiceprint check (WeSpeaker ResNet34-LM; verified on device at ~7 ms) on top of the direction lock.
- ReDimNet2 later.

### W5. People awareness and gaze (operator items 2, 11, 1)
- **Thursday, cheapest first:**
  1. Arm gaze along the mic direction, with no cameras.
  2. If time allows: YOLO on the gripper colour camera to re-centre on the person.
- **After:** a world model.
  1. GPU person detection on all body fisheyes.
  2. 3D positions in the `vision` frame from depth or the ground plane.
  3. Simple tracker with IDs.
  4. Gripper camera confirms "real person vs legs".
  5. `find_person_near_bearing(θ)` for speaker selection.
  6. Optionally publish to the world-object service for the tablet.

  Note: Spot does not detect people natively without the EAP2 lidar payload.

### W6. Vision behaviour (operator items 1, 11)
- **Thursday:**
  1. Rotate the image sent to the VLM upright.
  2. Use the gripper camera for "what do you see" when the arm is out.
  3. Speak "I don't see it" instead of guessing.
- **After (Stage 2C plan plus additions):**
  - per-camera rotation table used by capture, unrotate and depth (the 2C plan still has a 90°-only unrotate; fix that);
  - GPU YOLO26 / YOLOE-26 sidecar with a batched all-camera scan;
  - a camera priority policy per request type: describe → gripper / front pair; find → all cameras + VLM confirm; follow → all body cameras fused;
  - bearing-based turning;
  - spoken navigation outcomes;
  - borrow from VLFM: explore when the target isn't visible, project depth to 3D, confirm before walking.

### W7. Gripper mouth and choreography (operator item 8)
- **Thursday:**
  1. Stow the arm on stop / estop / shutdown.
  2. Cap the opening at ~0.6 and add attack/release smoothing.
  3. Ease to a rest position between sentences.
  4. Measure the sync offset on video and add a lead time.
  5. Choreography: add a `dance`/`emote` action that runs a built-in routine by name. Confirmed working without a license (see W12). Add it to `grammar/spot_action.gbnf` and the prompt, and block it while the arm is out or Spot is navigating.
- **After:** idle micro-motions, gaze while talking, velocity-limited deploy, no walking while the arm is deployed.

### W8. Venue, knowledge, content
- **Thursday:**
  1. Map and label the venue (`scripts/setup_map.py`, `scripts/map_waypoints.py`).
  2. Write an event / poster / team knowledge file.
  3. Remove the copied "battery is looking good" example from the prompt.
  4. Speak the battery / status / location-list results.
  5. Fix the Latin-1 decoding and spoken escape sequences.
  6. Add a proper-noun fix list for ASR.

### W9. Demo readiness
- **Thursday:**
  1. Tag a known-good rollback point before any change.
  2. Decide Kokoro vs ElevenLabs based on venue Wi-Fi.
  3. Keep Ollama running (or fix the hidden dependency).
  4. Check battery runtime.
  5. Privacy: keep voiceprints in memory only, and consider not logging visitor transcripts or posting a sign.
  6. Rehearse with babble playing.
  7. Update the launch doc.

### W10. Documentation (operator item 7)
- **Thursday:** a pipeline explainer (`docs/architecture/pipeline-explainer.md`) covering every stage, what each module does, how they connect, the settings, the failure modes and how to debug them. It is built from `status.md` plus the understand-anything graph.
- **After:** keep `status.md` and this roadmap updated with every merged change.

### W11. Code cleanup
- **Thursday (safe, small):**
  - delete dead modules and functions. **Keep `perception/camera_intrinsics.py`:** the operator wants Spot-reported intrinsics used. A live check on 2026-10-27 showed all 20 image sources return pinhole intrinsics (gripper colour f=552, fisheyes f≈256). Wire it into the W5/W6 3D projection and gaze instead of deleting it;
  - gitignore `graphify-out/`, `.understand-anything/` and `.claude/worktrees/`;
  - point tests at `voice_state.VoiceState`.
- **After:**
  1. Fix the double-import bug.
  2. Split `dispatch_intent` into one handler per command (write tests first).
  3. Split `client_mic.py` (audio I/O, robot bridge, loop).
  4. Centralise env vars.
  5. Single-source the safety regexes and camera constants.
  6. Merge the branch to `main` and reconcile `origin/main`.

### W12. SDK, OS and models (operator items 9, 10, model review)
- **SDK:** upgraded to **5.2.0** (newest on PyPI) on 2026-10-27.
  - Only the 6 bosdyn packages changed.
  - 104 tests pass, and the read-only robot probe works.
  - Rollback venv: `/mnt/ssd/spot-backups/spot-env-5.1.4-backup`. To restore, run `rm -rf spot-env && cp -a /mnt/ssd/spot-backups/spot-env-5.1.4-backup spot-env` from the repo root, then restore the 5.1.4 pins in `requirements.txt`. Only run the backup from the `spot-env` path, because its scripts hard-code that path.
  - No repo code catches `RetryableUnavailableError`, the error class 5.2 changed.
- **Robot OS:** 5.2.1 (updated 2026-10-28). License: Explorer, no choreography feature. The robot exposes a `live-signal-entity-detection` service; investigate whether it reports people.
- **Choreography:** 9 built-in routines are listable without a license:
  - dances: `hey_join_me`, `performance_bow`, `smooth_sailing`
  - emotes: `happy_gesture`, `tail_wag`, `upright_sit`, `wanna_play`
  - signals: `signal_alert`, `signal_undock`

  `list_all_moves` is permission-denied. **`execute_choreography` works without a license:** `tail_wag` ran on 2026-10-27, going WAITING_FOR_START_TIME → DANCING (~6 s) → COMPLETED_SEQUENCE. The client needs `create_standard_sdk(name, [ChoreographyClient])`. Test script: stand, execute by name, poll `get_choreography_status`, sit. A voice `emote`/`dance` action is therefore feasible for Thursday; add it to `grammar/spot_action.gbnf` and the prompt.
- **VLFM:** keep it as a separate docker process. A later `vlfm_find(goal)` voice action would hand over the lease, run the container, speak the result and take the lease back. Its first logged robot run (2026-08-26) detected nothing.
- **Models, as of 2026-10:** keep the stack. Nothing gets swapped before Thursday.

| Component | Verdict | Next step (after the demo) |
|---|---|---|
| Gemma 4 E4B on llama.cpp | Keep | Fixed prompt prefix for KV reuse; try Q5_K_M/Q4_K_M; Gemma 4 MTP drafter; benchmark Qwen3.5-4B |
| Parakeet ASR | Keep | NeMo GPU phrase boosting for proper nouns |
| Turn detection | Add | Smart Turn v3.2 (8 MB ONNX) after Silero |
| Silero VAD | Keep | A/B TEN VAD |
| Noise suppression | Don't add blindly | DeepFilterNet3 / GTCRN only if recordings show an ASR gain |
| sherpa KWS | Keep for demo | livekit custom models |
| ElevenLabs + Kokoro | Keep | Kokoro v1.0; try Qwen3-TTS 0.6B / Kyutai Pocket TTS |
| WeSpeaker | Keep | ReDimNet2 |
| YOLO26 / YOLOE-26 | Keep plan | TensorRT on GPU (YOLOE is AGPL) |
| Full-duplex speech models | Skip | Not practical on Orin, and no constrained robot actions |

---

## Thursday cut line (proposal)

Priority order. Stop wherever the time runs out; everything below the line moves to "after".

1. Rollback tag; laptop terminal controls; arm stow on stop/estop/shutdown (W1, W7, W9)
2. Audio front-end fixes + babble recordings + wake phrase retune (W2, W3)
3. Direction lock + state signalling + "stop" fix (W4)
4. Venue map + event knowledge + spoken status + text decoding fixes (W8)
5. Mouth smoothing; arm gaze toward the speaker's direction (W7, W5)
6. Pipeline explainer doc (W10)
7. Rehearsal with babble (W9)

Below the line: gripper-camera person re-centring, dance action, VLM rotation, cleanup deletions.
