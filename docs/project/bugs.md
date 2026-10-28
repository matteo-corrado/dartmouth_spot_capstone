# Bug Register

Bugs and fragile behaviour found in the 2026-10-27 audit. Each was confirmed against the code, and the ones marked ✔ were also checked by hand.
**Fix** says when: **Thu** = before the 2026-10-29 demo, **After** = post-demo. Line numbers are as of commit `020fa39`.
Strike an entry through (or delete it) when the fix lands, and note the commit.

Severity: **S1** breaks the demo or is unsafe · **S2** visible to visitors · **S3** internal/latent.

---

## A. Audio front-end, wake word, turn-taking

| ID | Sev | Bug | Where | Effect | Fix |
|---|---|---|---|---|---|
| A1 ✔ | S1 | Silero VAD is only fed frames above the energy gate; below it `is_speech=False` is forced | `client_mic.py:1047-1060` | Quiet syllables and pauses count as silence → people cut off mid-sentence; VAD state goes stale between utterances → false onsets | Thu |
| A2 ✔ | S1 | Startup calibration discards any measured floor above 0.003 RMS and substitutes 0.001 | `client_mic.py:372-374` | Real room level thrown away in every normal room; the gate starts wrong | Thu |
| A3 | S1 | Adaptive floor updates on every VAD-negative frame, clamped up to 0.05 (×5 while navigating) | `client_mic.py:1225-1229`, `:1041-1045` | Steady noise (fans, HVAC, crowd) ratchets the threshold up until speech can't clear it | Thu |
| A4 | S1 | Follow-up LISTENING window accepts any voice; babble keeps recording until the 20 s cap | `client_mic.py:1065-1121` | Crowd speech goes to ASR → LLM reply → new 15 s window → a self-sustaining loop | Thu |
| A5 | S2 | Wake chime plays through the player, and the mic callback drops all frames while it's busy plus 150 ms | `client_mic.py:1025`, `:228-232` | ~400 ms after "Hey Spot" is lost; one-breath commands lose their first word. The operator reports the reopen gap (A19) matters more | Thu (lower priority) |
| A19 | S1 | After Spot speaks, the mic stays shut for the drain + 0.5 s settle + 150 ms cooldown, and nothing signals the moment it reopens | `response_gate.py:14`, `client_mic.py:945-983`, `:206, 228-232` | **Operator's main dropped-audio problem:** visitors who answer straight away lose their first words | Thu |
| A6 | S2 | End of speech after 1.2 s silence (short utterances), counted in wall time | `client_mic.py:142-144`, `:1176-1188` | Natural pauses ("so… um…") end the turn | Thu (tune) / After (semantic end-of-turn model) |
| A7 | S2 | XVF3800 ch0 has AGC + noise suppression; the ratio gate (2× floor) is eaten by AGC | `client_mic.py:186`, `:151-154` | Speech-to-floor ratio collapses toward 0 dB in noise; gate either never or always opens | Thu (A/B ch1) |
| A8 | S2 | Wake KWS input gain ×8, hard-clipped | `wake/sherpa_onnx.py:33`, `:111-113`; `wake/safety_kws.py` | Close or loud talkers distort; babble boosted → misses and false fires | Thu |
| A9 | S2 | Wake tuning is loose (score 1.5, threshold 0.05), single short phrase | `wake/sherpa_onnx.py:30-31`, `models/kws/keywords.txt` | False fires on chatter | Thu |
| A10 | S2 | Wake detection doesn't call `_enter_state(LISTENING)`; LEDs stay in wake-word colour | `client_mic.py:1026` | No visual "I heard you" | Thu |
| A11 | S2 | Follow-up timeout drops to WAKE_WORD without `_enter_state`; LED stays green, no sound | `client_mic.py:1065-1071` | Visitors think Spot is still listening | Thu |
| A12 | S2 | The "your turn" chime is skipped when THINKING wasn't announced, and it's the same sound as power-on | `client_mic.py:978-983`, `audio_feedback.py` | Ambiguous turn signal | Thu |
| A13 | S2 | Safety regex `\b(stop\|halt)\b` matches anywhere in the transcript | `client_mic.py:545-549`, also duplicated in `intent.py:20-22` | "What's the next stop?" halts the robot and drops the question | Thu |
| A14 | S2 | Safety KWS (stop/freeze) only runs in WAKE_WORD state; mic is hard-gated while thinking/speaking | `client_mic.py:1004`, `:191-196` | Can't say "stop" while Spot talks or walks; needs the laptop | Thu (laptop control) |
| A15 | S3 | Sherpa VAD segments are never popped | `client_mic.py` VAD use | Memory grows over a long session | Thu (trivial) |
| A16 | S3 | Pre-roll buffer isn't filled in WAKE_WORD state | `client_mic.py:1086` vs `:1172` | Stale pre-roll after wake | Thu (with A5) |
| A17 | S3 | `MIC_CHANNEL` comment says "set from --channel arg", but no such flag exists | `client_mic.py:186` | Can't switch channel without editing code | Thu (with A7) |
| A18 | S3 | `setup_kws.py` overwrites `keywords.txt` on every run | `scripts/setup_kws.py:68-71` | Hand-edited wake phrases are lost | Thu (put phrases in the script) |

## B. Brain, speech output, dispatcher

| ID | Sev | Bug | Where | Effect | Fix |
|---|---|---|---|---|---|
| B1 ✔ | S2 | Streaming SSE decoded with `iter_lines(decode_unicode=True)`; no charset → requests uses Latin-1 | `brain/llamacpp_backend.py:89` | Em-dashes and other non-ASCII become junk ("departmentsâall") and get spoken | Thu |
| B2 | S2 | `TTSChunker` drops the backslash but keeps the rest of escape sequences | `spot_tts.py:326-330` | `\n` spoken as "n", `—` as "u2014" | Thu |
| B3 | S2 | Info actions (battery, status, check_obstacles, list_locations, list_maps) print results but never speak them; the LLM answers before the action runs | `spot_dispatch.py:1030-1052`, `:1171-1223` | Made-up answers ("battery is looking good") | Thu |
| B4 | S2 | Prompt example "Battery is looking good and I'm ready to help" is copied verbatim | `llm_brain.py:136` | False battery claims | Thu |
| B5 | S2 | Persona prefix sits before the 12K-token system prompt | `llm_brain.py:414-423` | Every persona switch drops the KV cache → ~15 s turns | Thu |
| B6 | S2 | Prompt says the reply "will be spoken in the NEW voice", but the voice changes next turn | `llm_brain.py:115`, `client_mic.py:1440-1444` | Persona's first line is in the old voice | Thu |
| B7 | S2 | State block includes changing `uptime_s` / `process_pid` | `spot_dispatch.py:103-106`, `llm_brain.py:414-423` | Weakens prompt caching | Thu (with B5) |
| B8 | S2 | ElevenLabs failure mid-run drops that sentence silently; Kokoro fallback only at init | `tts/__init__.py:56-90`, `audio_player.py:572-575` | Silent gaps if venue Wi-Fi drops | Thu (decide backend / add fallback) |
| B9 | S2 | Navigation arrival, stuck, lost, tour-complete and go-to failures are printed, not spoken | `spot_dispatch.py:381-500`, `:1386-1404`; `client_mic.py:1601-1604` | Silent failures; just an error buzz | Thu (spoken failures) / After (full nav narration) |
| B10 | S2 | `go_to_object` returns True as soon as its thread starts | `spot_dispatch.py:1170` | Spot says "heading there" even when nothing is found | After |
| B11 | S2 | Brain answers fillers ("uh I think my uh") with full replies | `client_mic.py:process_utterance` | Odd replies to noise | Thu (min-length / filler filter with A4) |
| B12 | S2 | No spoken ack during long waits (persona switch, add_persona up to ~30 s, first describe, 60 s LLM timeout) | `client_mic.py`, `brain/llamacpp_backend.py:58` | Visitor thinks it froze | Thu (filler line after ~4 s) |
| B13 | S3 | `enable_mouth`/`disable_mouth` are in the grammar but not in the system prompt | `grammar/spot_action.gbnf:14`, `llm_brain.py:~100-145` | Mouth can't be toggled by voice | Thu |
| B14 | S3 | Canned lines are robotic or speak raw keys ("tour_guide") | `client_mic.py:1475`, `:1507`; `llm_brain.py:609-613`; `spot_dispatch.py:734` | Awkward phrasing | Thu (cheap) |

## C. Robot safety, arm, mouth

| ID | Sev | Bug | Where | Effect | Fix |
|---|---|---|---|---|---|
| C1 | S1 | stop / freeze / estop close the gripper but never stow the arm; shutdown sits without stowing | `spot_dispatch.py:576, 795, 808, 1094`, `:255` | Arm left out near people | Thu |
| C2 | S1 | Navigation and follow allowed at full speed with the arm deployed; no `arm_deployed` state | `spot_dispatch.py`, `chatbot/session_state.py:10` | Unsafe in a crowd | Thu (tuck arm into walking pose before any walk; speed cap) |
| C3 | S2 | Arm-ready deploy has no velocity cap (the spec requires one) | `session.py:174-179` | Fast arm swing near faces | Thu |
| C4 | S2 | Mouth envelope normalised per chunk, no smoothing, no sync offset, snaps shut between sentences, no opening cap | `animation/mouth.py:31-52`, `spot_tts.py:203-237` | Jittery, over-wide, out-of-sync mouth | Thu |
| C5 | S3 | No watchdog closing the gripper if playback stalls | `animation/mouth.py` | Gripper could hang open | After |

## D. Vision

| ID | Sev | Bug | Where | Effect | Fix |
|---|---|---|---|---|---|
| D1 | S2 | `capture_frame` doesn't rotate; the VLM gets the front fisheye sideways | `spot_dispatch.py:183` | Worse "what do you see" answers | Thu |
| D2 | S2 | All cameras rotated 90° CW; side/back/right are actually 0°/0°/180° | `visual_nav.py:168, 323, 375, 856, 914`, `_unrotate_pixel :382`, `_depth_at_bbox :291-294` | YOLO runs on sideways side/back images; wrong left/right labels. The 2C plan repeats the 90°-only assumption | After (2C, fix plan too) |
| D3 | S2 | Front cameras are crossed (FL looks right); "centre in image" isn't a body direction | `visual_nav.py` follow + `detect_in_image` | Wrong steering / wrong side words | After |
| D4 | S2 | YOLO on CPU; YOLO-World not preloaded (`preload_models` skips it and is never called anyway) | `visual_nav.py:35`, `:129-141` | Slow first vision command, unreliable detection | After (GPU sidecar) |
| D5 | S2 | go_to_object turns a fixed 0/±90/180° instead of the real bearing; first camera hit wins | `visual_nav.py:625`, `:175-184` | Misses or turns to the wrong spot | After |
| D6 | S3 | describe has no gripper-camera option and no fallback camera | `spot_dispatch.py:40-47`, `client_mic.py:1544-1596` | Lower-quality images | Thu (gripper when arm out) |

## E. Infrastructure and code health

| ID | Sev | Bug | Where | Effect | Fix |
|---|---|---|---|---|---|
| E1 | S1 | `LLMBrain.is_available()` checks **Ollama** even on the llama.cpp backend; `describe_persona` calls Ollama directly | `llm_brain.py:284-306`, `:387` | If Ollama is down the brain silently falls back to regex-only | Thu (check the active backend) |
| E2 | S2 | `client_mic.py` imports `audio_player`, `spot_tts`, `audio_feedback`, `latency` by bare name while other modules use `src.voice_control.*` → two module copies | `client_mic.py:85-92` | Voice "set volume" doesn't change beeps; TTS latency marks never recorded; `_tts_instance` workaround | After (medium risk) |
| E3 | S3 | wakespot doesn't pass `--latency` / `--no-brain` through | `scripts/wakespot.py:157-187` | No timing data from demo runs | Thu |
| E4 | S3 | Conversation log path is relative to the cwd; tests write into the production log | `chatbot/conversation_log.py:12`, `tests/voice_control/test_brain_concurrency.py:39` | Logs land in `src/`; test junk in real logs | Thu (cheap) |
| E5 | S3 | `requirements.txt` lists `livekit-wakeword` unpinned (needs Python 3.11) | `requirements.txt:54` | `pip install -r` can break on this 3.10 box | Thu (move to training extras) |
| E6 | S3 | Tests import `VoiceState` from `client_mic` (runs its side effects) | `tests/voice_control/test_spot_leds.py`, `test_state_feedback.py` | Fragile tests | Thu (cheap) |
| E7 | S3 | Dead code: `startup_status.py`, `perception/raycast_helpers.py`, unused functions in `visual_nav.py` / `llm_brain.py`. **Keep** `perception/camera_intrinsics.py` (to be wired in) | various | Noise when reading the code | Thu (safe) |
| E8 | S3 | Stale docs: `daily-operations.md` (Ollama, two terminals), `server.py:4` says nemotron, `run_voice_control.py:280-282` Ollama comment | — | Misleading | Thu |
| E9 | S3 | Branch never merged; local `main` 126 ahead / 1 behind `origin/main` | git | Risky push / merge later | After |
