# Poster-Demo Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Spot's voice interaction survive a crowded poster session: reliable wake, speaker lock by direction (then voiceprint), no dropped first words, clear listening/thinking/speaking signals, laptop control, arm safety while walking, emotes/dances by natural phrasing.

**Architecture:** Pure, unit-tested logic lives in small new modules: `tuning`, `audio_frontend`, `speaker_lock`, `mic_array` decode helpers, `control_server` routing, `emotes`, `speaker_embed` accept rule. `client_mic.py` and `spot_dispatch.py` call into them, so the big files only gain thin wiring. All thresholds live in one tunables file (`config/tuning.yaml`), and the recordings decide the values.

**Tech Stack:** Python 3.10 (`spot-env`), sounddevice, sherpa-onnx (KWS, Silero VAD), pyusb 1.3.1 (XVF3800 control), bosdyn SDK 5.2.0 (robot OS 5.2.1), onnxruntime-gpu 1.23 (WeSpeaker), stdlib `http.server`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-poster-demo-design.md`. Bug IDs (A1, B3 …) refer to `docs/project/bugs.md`.

## Global Constraints

- Branch: `demo-poster-2026-10-29`. Rollback tag: `pre-demo-2026-10-29`. Merge into `tour_guide_upgrade_matteo` only after the robot rehearsal passes.
- Commit messages: **one line, no trailers** (repo convention).
- Run the tests with `spot-env/bin/python -m pytest -q -p no:cacheprovider tests/voice_control`. The 104 existing tests must stay green after every task.
- Any new dispatcher action is added to `src/voice_control/grammar/spot_action.gbnf`. Alternation stays on **one line**.
- Never send `SAVE_CONFIGURATION` (48/9) or `REBOOT` (48/7) to the XVF3800. Never call `set_configuration` and never detach kernel drivers. Runtime writes only.
- Any `requirements.txt` change goes through the `pin-guardian` agent.
- Robot-touching code (`spot_dispatch.py`, `session.py`, arm, emote, gaze) is reviewed by the `safety-reviewer` agent and gets one on-robot check with the E-Stop live.
- Spot wired IP `192.168.50.3`. Mic USB `2886:001a`, firmware 2.1.1, ch0 = processed, ch1 = ASR beam.
- Voiceprints are kept in memory only. They are never written to disk.
- Everything must work with no map loaded and with the XVF3800 unplugged (lock and ring LEDs degrade, the voice loop keeps running).

## Review Focus

1. **No usable direction during speech:** the speech flag is never set, or the azimuth is NaN. The utterance is still accepted and logged, never dropped. Test: `test_accept_with_no_readings_accepts` (Task 7).
2. **Angle wrap-around:** a lock at 355° and speech at 10° is a 15° difference, inside the window. Test: `test_circular_median_wraps` and `test_accept_across_zero` (Task 7).
3. **Operator command mid-speech or mid-walk:** `hush` and `stop` must act immediately while TTS plays or a chained nav blocks the loop. Test: `test_immediate_commands_bypass_queue` (Task 4).
4. **Mic array unplugged mid-session:** the poller dies quietly, `samples_between` returns `[]`, LED writes are no-ops, and the voice loop continues. Test: `test_poller_survives_usb_error` (Task 2).
5. **Chimes no longer mute the mic:** a chime alone must not start an utterance or reach ASR. Check: the replay harness (Task 9) feeds the three chime WAVs and expects 0 utterances. This is also an on-robot listen check in Task 10.

## File map

| File | Responsibility |
|---|---|
| `src/voice_control/tuning.py` (new) | `Tuning` dataclass of all tunables, loaded from `config/tuning.yaml` |
| `config/tuning.yaml` (new) | Operator-editable values, set from recordings |
| `src/voice_control/mic_array.py` (new) | XVF3800 USB control: DoA reads, poller history, LED ring, fixed beams |
| `scripts/xvf_probe.py` (new) | Live DoA print and clap-test calibration; LED limits test |
| `scripts/record_demo_clips.py` (new) | Guided stereo recording + DoA log per clip |
| `src/voice_control/control_server.py` (new) | Localhost HTTP command endpoint and routing |
| `scripts/spotctl` + `scripts/spotctl.py` (new) | Laptop CLI (one-shot + hotkeys) |
| `src/voice_control/audio_frontend.py` (new) | dBFS, floor tracker, frame endpointer, utterance stats / SNR gate |
| `src/voice_control/speaker_lock.py` (new) | Circular stats, lock/accept/nudge/timeout, mic→body angle |
| `tests/audio/replay_demo.py` (new) | Replay recordings through KWS + VAD + frontend + lock and report numbers |
| `src/voice_control/emotes.py` (new) | Routine catalogue, name validation, random dance pick |
| `src/voice_control/speaker_embed.py` (new) | WeSpeaker fbank + embedding + cosine accept rule |
| `src/voice_control/knowledge/event_knowledge.md` (new) | Event/poster/team pack (template, operator fills) |
| `src/voice_control/client_mic.py` | Wiring: frontend, reopen timing, lock, ring/chimes, control queue, metrics |
| `src/voice_control/audio_player.py` | `is_speech_busy()` so chimes don't mute the mic |
| `src/voice_control/audio_feedback.py`, `state_feedback.py` | New chimes, ring mapping |
| `src/voice_control/spot_dispatch.py`, `src/session.py` | Arm safety, walking pose, speed cap, emote, spoken info results, gaze |
| `src/voice_control/llm_brain.py`, `brain/llamacpp_backend.py`, `spot_tts.py`, `tts/__init__.py` | Conversation-quality fixes, TTS fallback |
| `scripts/setup_kws.py`, `wake/sherpa_onnx.py` | Wake phrase list, gain fix |

## Parallel execution and review gates

Work runs in **waves of parallel lanes**. Each lane is one subagent in its own git worktree (`isolation: "worktree"`), branched from `demo-poster-2026-10-29`.

**File ownership rules:**
- Lanes in the same wave own **disjoint files**, so merges are conflict-free.
- `client_mic.py` has exactly one owner (Lane A) and `spot_dispatch.py` + `session.py` have exactly one owner (Lane B). Everyone else only adds new modules and touches the files listed for their lane.
- Every lane commits only its own files (`git add <paths>`, never `git add -A`).

**Wave 1 — pure modules and tools, all in parallel (~8 lanes):**

| Lane | Tasks | Owns |
|---|---|---|
| W1-a | 1 Tuning | `tuning.py`, `config/tuning.yaml` |
| W1-b | 2 Mic array + probe | `mic_array.py`, `scripts/xvf_probe.py` |
| W1-c | 3 Recorder | `scripts/record_demo_clips.py`. Codes against the Task 2 `MicArray` interface with a fake; real device after merge |
| W1-d | 5 Audio front-end, 7 Speaker lock | `audio_frontend.py`, `speaker_lock.py` |
| W1-e | 4 control server + CLI only (no client_mic wiring) | `control_server.py`, `scripts/spotctl*` |
| W1-f | 6 Wake phrases | `wake/phrases.py`, `wake/sherpa_onnx.py`, `wake/safety_kws.py`, `scripts/setup_kws.py`, `config/wake_phrases.txt` |
| W1-g | 15 pure part, 19 pure part, 18 pure part | `emotes.py`; `gaze.py` (new: `gaze_target`, `should_regaze`); `speaker_embed.py` + `scripts/setup_wespeaker.py` |
| W1-h | 16 Knowledge, 17 Ops | `knowledge/event_knowledge.md`, `llm_brain.py:214-245` only; `scripts/wakespot.py`, `chatbot/conversation_log.py`, `test_brain_concurrency.py`, `requirements.txt` (pin-guardian) |

**Gate 1 (after Wave 1):** merge the lanes in table order. Run the full suite; it must be green. A **wave reviewer** (`feature-dev:code-reviewer`) reads the combined Wave 1 diff, checking against this plan's Interfaces blocks: names, signatures, return types. Then **operator session 1** (recordings, clap test, LED limits) starts on the merged tools, while Wave 2 begins.

**Wave 2 — integration, three owner lanes in parallel plus replay:**

| Lane | Tasks, in order within the lane | Owns |
|---|---|---|
| A (client_mic owner) | 4 wiring → 8 client_mic part → 10 → 11 client_mic part → 12 client_mic items (A13 call site, B6, B11, B12, B14 `:1475/:1500-1507`) → 6 strip wiring | `client_mic.py` |
| B (dispatch owner) | 14 arm safety → 15 dispatcher branch + grammar → 12 dispatcher items (B3 spoken results, `:103-106`, `:734`) | `spot_dispatch.py`, `session.py`, `chatbot/session_state.py`, `graph_nav_utils.py` (speed cap), `grammar/spot_action.gbnf` |
| C (audio/brain/tts owner) | 8 `audio_player` part → 12 brain/tts items (B1 backend, B2 chunker, B4/B5/B7/B13 prompt, E1, A13 patterns in `intent.py`) → 13 TTS fallback → 11 `audio_feedback`/`state_feedback` part | `audio_player.py`, `brain/*`, `llm_brain.py` (except 214-245), `spot_tts.py`, `tts/*`, `intent.py`, `audio_feedback.py`, `state_feedback.py` |
| D (replay) | 9, after operator session 1 delivers clips | `tests/audio/replay_demo.py`, results doc, tuned `config/*.yaml` |

Cross-lane interfaces in Wave 2 are fixed by this plan:
- Lane A calls `AudioPlayer.is_speech_busy()` and the new chime names from Lane C.
- Lane B's `dispatch_intent(..., speak=...)` kwarg is called by Lane A.

Each lane codes against those exact names and merges at Gate 2.

**Gate 2:** merge B, then C, then A, then D. Then:
- full suite green;
- a dry-run smoke test of the whole pipeline (no robot);
- `safety-reviewer` on the combined Lane B + Lane A diff;
- the wave reviewer on the whole Wave 2 diff.

Then **operator session 2** on the robot: Task 14 walking pose, and the on-robot checks for Tasks 10, 11, 14 and 15.

**Wave 3 — late features, parallel lanes:**
- Lane A: 18 wiring (voiceprint), then 19 wiring (gaze call sites).
- Lane B: 19 `gaze_at_bearing`.
- Lane C: 20 mouth (`mouth.py`, `spot_tts.py`).
- Lane E: 21 docs. The explainer can be **drafted from Gate 1 onward** in parallel, and is finalised here.

**Gate 3:**
- suite green;
- `safety-reviewer` (gaze, mouth);
- the robot checks for 18–20;
- the **rehearsal** (Task 21 Step 3);
- a **final whole-branch review** with `/code-review high` against `pre-demo-2026-10-29`. Fix findings, then tag and merge (Task 21 Step 5).

**Per-task review (inside every lane):** after each task's tests pass and before its commit, a fresh reviewer subagent checks two things:
1. spec compliance against this task's Interfaces and steps;
2. code quality: bugs, simplicity, no out-of-scope edits.

Robot-touching tasks add `safety-reviewer`. A task isn't done until its reviews are clean or the findings are fixed.

**Cut order if late:** 20 → 19 → 18 → map part of 16. Never cut the review gates. A lane that misses its gate is dropped from that merge, and the demo runs from the last green gate.

---

### Task 1: Tunables file

**Files:**
- Create: `src/voice_control/tuning.py`, `config/tuning.yaml`
- Test: `tests/voice_control/test_tuning.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Tuning` and `load_tuning(path: Path | None = None) -> Tuning`. Default path: `<repo>/config/tuning.yaml`. A missing file means defaults. Fields and defaults:
  - `mic_channel=0`
  - `min_speech_dbfs=-50.0`, `min_snr_db=10.0`
  - `onset_frames=3`, `end_silence_s=1.5`, `end_silence_s_short=1.2`, `max_utterance_s=20.0`
  - `floor_window_s=5.0`, `floor_percentile=10.0`, `floor_rise_db_per_s=1.0`
  - `reopen_delay_s=0.5`, `speaker_tail_s=0.15`
  - `kws_score=1.5`, `kws_threshold=0.05`, `kws_peak_target=0.5`
  - `lock_window_deg=35.0`, `lock_timeout_s=20.0`, `lock_nudge=0.3`, `mic_yaw_offset_deg=0.0`, `mic_yaw_sign=1`
  - `beam_steering=False`
  - `voiceprint_enabled=False`, `voiceprint_threshold=0.5`
  - `thinking_filler_s=4.0`, `walk_speed_cap=0.5`, `dance_clearance_m=1.5`
  - `mouth_lead_s=0.0`, `mouth_max_open=0.6`, `mouth_attack_s=0.03`, `mouth_release_s=0.10`, `mouth_rest_open=0.08`

  Also `get_tuning() -> Tuning`: a module-level cached singleton.

- [ ] **Step 1: Write failing tests**

```python
def test_defaults_when_file_missing(tmp_path):
    t = load_tuning(tmp_path / "nope.yaml")
    assert t.lock_window_deg == 35.0 and t.mic_channel == 0 and t.lock_timeout_s == 20.0

def test_yaml_overrides(tmp_path):
    p = tmp_path / "t.yaml"; p.write_text("mic_channel: 1\nmin_snr_db: 8\n")
    t = load_tuning(p)
    assert t.mic_channel == 1 and t.min_snr_db == 8.0 and t.onset_frames == 3

def test_unknown_key_raises(tmp_path):
    p = tmp_path / "t.yaml"; p.write_text("lock_windw_deg: 20\n")
    with pytest.raises(ValueError, match="lock_windw_deg"):
        load_tuning(p)
```

- [ ] **Step 2:** Run `spot-env/bin/python -m pytest -q tests/voice_control/test_tuning.py`. Expect FAIL (module missing).
- [ ] **Step 3:** Implement with `yaml.safe_load`. Cast each value to the field's type. Raise `ValueError` naming any unknown key. Write `config/tuning.yaml` with every field commented at its default.
- [ ] **Step 4:** Run the tests. Expect PASS. Run the full suite. Expect green.
- [ ] **Step 5:** `git commit -m "tuning: single tunables file with defaults + yaml overrides"`

### Task 2: XVF3800 control module

**Files:**
- Create: `src/voice_control/mic_array.py`, `scripts/xvf_probe.py`
- Test: `tests/voice_control/test_mic_array.py`

**Interfaces:**
- Produces:
  - `decode_read(resp: bytes, fmt: str, count: int) -> tuple | None`. Returns `None` when `resp[0] == 64` (retry). Raises `XvfError` for any other non-zero status.
  - `class MicArray`:
    - `MicArray.open() -> MicArray | None`: `None` if the device is absent or pyusb is missing; never raises.
    - `.read_doa() -> tuple[int, bool]`: (deg, speech).
    - `.read_selected_azimuth() -> float | None`: radians; `None` if NaN.
    - `.set_led_effect(effect: int)`, `.set_led_color(rgb: int)`, `.set_ring(colors: list[int])`: exactly 12 values.
    - `.set_fixed_beams(az_rad: float | None)`: `None` turns fixed beams off.
    - `.start_poller(hz: float = 20.0)`, `.samples_between(t0: float, t1: float) -> list[tuple[float, int, bool]]` (monotonic t, deg, speech), `.close()`.
    - Every write is a no-op after a USB error. The poller thread catches every exception, marks the device dead and exits.
  - Constants taken verbatim from spec Appendix A. Reads use `ctrl_transfer(0xC0, 0, 0x80|cmd, resid, n*size+1)` and retry up to 100 times with a 10 ms sleep. Writes use `ctrl_transfer(0x40, 0, cmd, resid, payload)`.
- [ ] **Step 1: Write failing tests** with a `FakeDev` whose `ctrl_transfer` returns scripted byte arrays:

```python
def test_decode_ok():
    assert decode_read(bytes([0]) + struct.pack("<HH", 90, 1), "H", 2) == (90, 1)

def test_decode_retry_returns_none():
    assert decode_read(bytes([64, 0, 0, 0, 0]), "H", 2) is None

def test_read_doa_retries_then_succeeds():
    dev = FakeDev([bytes([64]) + b"\0" * 4, bytes([0]) + struct.pack("<HH", 270, 1)])
    assert MicArray(dev).read_doa() == (270, True)

def test_samples_between_filters_by_time():
    m = MicArray(FakeDev([])); m._history.extend([(1.0, 10, True), (2.0, 20, True), (3.0, 30, False)])
    assert m.samples_between(1.5, 3.5) == [(2.0, 20, True), (3.0, 30, False)]

def test_poller_survives_usb_error():
    m = MicArray(FakeDev(raise_after=2)); m.start_poller(hz=200); time.sleep(0.1)
    assert m.dead is True and m.samples_between(0, 1e12) is not None
    m.set_led_effect(3)  # no exception after death
```

- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement. The history is a `collections.deque(maxlen=400)` guarded by a lock. `MicArray.__init__(dev)` takes an injected device for tests. `open()` uses `usb.core.find(idVendor=0x2886, idProduct=0x001a)`.
- [ ] **Step 4:** `scripts/xvf_probe.py` with three subcommands:
  - `live` prints deg/speech at 10 Hz.
  - `clap` prompts "clap at FRONT / LEFT / BACK / RIGHT of Spot". For each, it records the median deg of speech-flagged samples over 3 s, then prints the suggested `mic_yaw_offset_deg` (front reading) and `mic_yaw_sign` (+1 if LEFT reads front+90 mod 360, else -1).
  - `leds` cycles effects 1/3/4/5, times 50 consecutive `set_ring` writes, and prints writes/s. This measures the LED limits the spec asks for.
- [ ] **Step 5:** Run the tests (PASS), then `spot-env/bin/python scripts/xvf_probe.py live` for 5 s on the device (prints values).
- [ ] **Step 6:** `git commit -m "mic_array: XVF3800 usb control, doa poller, led ring, probe/clap/led-limits script"`

### Task 3: Guided demo-clip recorder

**Files:**
- Create: `scripts/record_demo_clips.py`
- Test: `tests/voice_control/test_record_demo_clips.py`

**Interfaces:**
- Consumes: `MicArray` (Task 2), which is optional. If it is absent, clips are recorded without a DoA log.
- Produces:
  - Files at `tests/audio/demo/<condition>/<clip>.wav`: stereo, 16 kHz, int16, both channels.
  - `<clip>.doa.csv` with columns `t_rel_s,deg,speech`.
  - `tests/audio/demo/<condition>/manifest.json`: a list of `{clip, kind, phrase, expected_text, angle_deg}`.
  - `build_prompts(phrases: list[str]) -> list[Prompt]` (pure). `Prompt` fields: `clip`, `kind`, `say`, `angle_deg`, `duration_s`. Kinds: `wake`, `wake_pause`, `quick_reply`, `overlap`, `background`.
- **Prompt set** for each phrase P (slug `p`):
  - `wake_{p}_{front,left,back,right}`: "P, what is this poster about?" (4 s, angle 0/90/180/270)
  - `pause_{p}`: "P, can you tell me… [pause 1 s] …about the project" (6 s)
- Shared clips:
  - `quick_reply_{1..3}`: the script plays the "ready" chime through Spot's speaker, then the visitor immediately says "what's your name" (4 s)
  - `overlap_{1..3}`: a second person talks continuously from another side while the visitor says a wake+question (6 s; angle = visitor's)
  - `background`: no wake, 120 s
- `--condition` is one of `quiet`, `loud`, `overlap`, or any label. `--phrases` takes a `|`-separated list.

- [ ] **Step 1: Failing test**

```python
def test_build_prompts_counts_and_names():
    ps = build_prompts(["Hey Spot", "Hello Spotty"])
    names = [p.clip for p in ps]
    assert "wake_hey_spot_left" in names and "pause_hello_spotty" in names
    assert sum(p.kind == "wake" for p in ps) == 8
    assert sum(p.kind == "background" for p in ps) == 1
    assert {p.angle_deg for p in ps if p.kind == "wake"} == {0, 90, 180, 270}
```

- [ ] **Step 2:** Run. Expect FAIL. **Step 3:** Implement. Recording reuses the `find_device` pattern from `scripts/record_corpus.py` with `channels=2`. The DoA poller runs during each clip, and samples are written relative to clip start. Clips are skippable, redoable and quit-able like `record_corpus.py`. **Step 4:** Run. Expect PASS.
- [ ] **Step 5:** `git commit -m "record_demo_clips: guided stereo + doa recording for quiet/loud/overlap conditions"`

> **OPERATOR SESSION 1:** the mic stays mounted where it will be on Thursday, with Spot powered on (fans running).
> 1. `spot-env/bin/python scripts/xvf_probe.py clap`. Put the printed offset and sign into `config/tuning.yaml`.
> 2. `scripts/record_demo_clips.py --condition quiet --phrases "Hey Spot|<cand2>|<cand3>"`
> 3. The same with `--condition loud`, then `--condition overlap`.
> 4. `scripts/xvf_probe.py leds`, plus `scripts/diag_av_leds.py`. Note writes/s, and how visible each indicator is from the front, side and back.

### Task 4: Laptop control (`spotctl`) — priority P1

**Files:**
- Create: `src/voice_control/control_server.py`, `scripts/spotctl.py`, `scripts/spotctl` (bash shim like `scripts/wakespot`)
- Modify: `src/voice_control/client_mic.py`: start the server in `main()`, drain its queue once per loop iteration.
- Test: `tests/voice_control/test_control_server.py`

**Interfaces:**
- Produces:
  - `COMMANDS: dict[str, Command]`.
  - `Command(name: str, immediate: bool, takes_arg: bool)`. Names: `mute unmute listen release reset hush say persona stop freeze sit stand stow status crowd voiceprint`.
  - `parse_command(path: str, body: dict) -> tuple[str, str | None]`. Raises `ValueError` for an unknown name or a missing arg.
  - `class ControlServer(port: int = 8770, on_immediate: Callable[[str, str|None], dict], queue: queue.Queue)`, with `.start()` / `.stop()`. It binds **127.0.0.1 only**. `POST /cmd/<name>` takes JSON `{"arg": ...}`. Immediate commands call `on_immediate` on the HTTP thread and return its dict. Others are put on `queue` and return `{"queued": true}`. `GET /status` calls `on_immediate("status", None)`.
- **Immediate set:** `hush stop freeze sit stand stow status mute unmute`. They're safe off-thread because they touch the player, mouth and dispatcher only. In client_mic:
  - `hush` = `_audio_player.force_reset()` + `_close_mouth()`.
  - `stop`, `freeze`, `sit`, `stand` and `stow` use `dispatch_intent({"intent": "<name>"})`, the same as voice.
  - `mute` sets a new `_operator_muted` flag that `audio_callback` honors.
- **Queued set:** `listen release reset say persona crowd voiceprint`. The main loop handles these between frames, because they change `state` / lock / history.
- `spotctl`:
  - One-shot form: `spotctl <name> [arg]`.
  - Run with no arguments, it is interactive (raw tty): `m` mute, `u` unmute, `l` listen, `r` release, `x` reset, `h` hush, `s` stop, `f` freeze, `w` stow, `c` crowd toggle, `v` voiceprint toggle, `?` status, `q` quit.
  - It talks to `http://127.0.0.1:8770`. Used over `ssh jetson spotctl …` via Tailscale.

- [ ] **Step 1: Failing tests**

```python
def test_parse_known_and_arg():
    assert parse_command("/cmd/say", {"arg": "hello"}) == ("say", "hello")

def test_parse_unknown_raises():
    with pytest.raises(ValueError): parse_command("/cmd/dance", {})

def test_immediate_commands_bypass_queue():
    q = queue.Queue(); calls = []
    srv = ControlServer(port=0, on_immediate=lambda n, a: calls.append(n) or {"ok": True}, queue=q); srv.start()
    _post(srv.port, "hush"); _post(srv.port, "listen")
    assert calls == ["hush"] and q.get_nowait() == ("listen", None); srv.stop()

def test_binds_localhost_only():
    srv = ControlServer(port=0, on_immediate=lambda n, a: {}, queue=queue.Queue()); srv.start()
    assert srv.address[0] == "127.0.0.1"; srv.stop()
```

- [ ] **Step 2:** Run. Expect FAIL. **Step 3:** Implement the server (`ThreadingHTTPServer` on a daemon thread, `port=0` lets tests pick a free port and exposes `.port`). Implement the CLI.
- [ ] **Step 4:** Wire into client_mic:
  - Implement `_on_immediate(name, arg)` and the queued handler.
  - `listen` sets `state = LISTENING` with no lock and calls `_enter_state(LISTENING)`.
  - `release` drops the lock, sets `state = WAKE_WORD` and plays the released chime (Task 11).
  - `reset` = release + `brain.history.clear()`.
  - `say` → `tts.speak(arg)`.
  - `persona` → `do_set_persona({"name": arg}, brain)`.
  - `crowd` and `voiceprint` toggle flags.
  - `status` returns `{state, locked_deg, floor_dbfs, last_transcript, last_latency_s, muted, crowd, voiceprint}`.
- [ ] **Step 5:** Tests PASS and the full suite is green. Manual check: `spot-env/bin/python src/voice_control/client_mic.py --dry-run` (no robot), then from another shell `scripts/spotctl status` and `scripts/spotctl hush`. Both return JSON.
- [ ] **Step 6:** `safety-reviewer` on the wiring of stop/freeze/stow. `git commit -m "spotctl: localhost control endpoint + laptop cli (immediate safety cmds, queued turn cmds)"`

### Task 5: Audio front-end logic (pure)

**Files:**
- Create: `src/voice_control/audio_frontend.py`
- Test: `tests/voice_control/test_audio_frontend.py`

**Interfaces:**
- Consumes: `Tuning` (Task 1).
- Produces:
  - `frame_dbfs(pcm16: bytes) -> float`: −120 for silence.
  - `class FloorTracker(window_s, percentile, rise_db_per_s, frame_s=0.03)`:
    - `.update(db: float, is_speech: bool) -> None`: ignores speech frames.
    - `.floor_db -> float`: the low percentile of the window. It may fall immediately but rises no faster than `rise_db_per_s`.
    - `.seed(db)`: from calibration.
  - `class Endpointer(onset_frames, end_silence_frames, max_frames)`:
    - `.feed(is_speech: bool, db: float, floor_db: float) -> str | None` returns `"start"`, `"end"`, `"cap"` or `None`.
    - Silence counts only when `not is_speech and db < floor_db + 6`.
    - `.set_end_silence_frames(n)`.
  - `UtteranceStats` collects frames. `.speech_db_median`, `.speech_db_p90`, `.snr_db(floor_db)` and `.duration_s` are computed over speech-flagged frames.
  - `passes_gate(stats, floor_db, min_speech_dbfs, min_snr_db) -> bool`.
- [ ] **Step 1: Failing tests**

```python
def test_floor_cannot_ratchet_fast():
    f = FloorTracker(5.0, 10.0, rise_db_per_s=1.0); f.seed(-60)
    for _ in range(100): f.update(-30, is_speech=False)   # 3 s of loud steady noise
    assert f.floor_db <= -60 + 3.5

def test_floor_falls_immediately():
    f = FloorTracker(5.0, 10.0, 1.0); f.seed(-40); f.update(-70, False)
    assert f.floor_db == pytest.approx(-70, abs=0.5)

def test_pause_inside_speech_does_not_end():
    e = Endpointer(3, end_silence_frames=50, max_frames=666)
    events = [e.feed(True, -30, -60) for _ in range(10)] + [e.feed(False, -62, -60) for _ in range(33)] + [e.feed(True, -30, -60)]
    assert "start" in events and "end" not in events          # 1 s pause < 1.5 s

def test_quiet_syllable_counts_as_speech_not_silence():
    e = Endpointer(3, 50, 666); [e.feed(True, -30, -60) for _ in range(5)]
    assert all(e.feed(False, -45, -60) is None for _ in range(60))  # above floor+6 → not silence

def test_cap():
    e = Endpointer(3, 50, max_frames=20)
    assert "cap" in [e.feed(True, -30, -60) for _ in range(25)]

def test_snr_gate():
    s = UtteranceStats(); [s.add(-35, True) for _ in range(20)]
    assert passes_gate(s, floor_db=-50, min_speech_dbfs=-50, min_snr_db=10) is True
    assert passes_gate(s, floor_db=-40, min_speech_dbfs=-50, min_snr_db=10) is False
```

- [ ] **Step 2:** Run (FAIL). **Step 3:** Implement. **Step 4:** Run (PASS).
- [ ] **Step 5:** `git commit -m "audio_frontend: dbfs, non-ratcheting floor tracker, frame endpointer, snr gate"`

### Task 6: Wake phrase list and KWS gain

**Files:**
- Modify: `scripts/setup_kws.py:33,68-71`, `src/voice_control/wake/sherpa_onnx.py:30-33,98-130`, `src/voice_control/client_mic.py:564-567` (`WAKE_PHRASE_PATTERN`)
- Create: `src/voice_control/wake/phrases.py`
- Test: `tests/voice_control/wake/test_phrases.py`

**Interfaces:**
- Produces:
  - `WAKE_PHRASES: list[str]`, read from `config/wake_phrases.txt` (one per line, default `HEY SPOT`).
  - `strip_wake_phrase(text: str, phrases: list[str]) -> str`: removes a leading wake phrase, tolerating ASR variants, i.e. optional "hey/hi/a/say" and one fuzzy word (difflib ratio ≥ 0.75).
  - `peak_safe_gain(frame: np.ndarray, target_peak: float) -> float`: returns `min(8.0, target_peak / max(peak, 1e-6))`.
- `setup_kws.py --phrases-file config/wake_phrases.txt` writes one `keywords.txt` line per phrase with `:<kws_score> #<kws_threshold> @<PHRASE_SLUG>`, tokenised with `sherpa_onnx.text2token` and the model's `bpe.model`.
- [ ] **Step 1: Failing tests**

```python
def test_strip_exact():
    assert strip_wake_phrase("hey spot stand up", ["HEY SPOT"]) == "stand up"

def test_strip_asr_variant():
    assert strip_wake_phrase("hello spotty, what is this", ["HELLO SPOTTY"]) == "what is this"
    assert strip_wake_phrase("hello spotted what is this", ["HELLO SPOTTY"]) == "what is this"

def test_no_strip_mid_sentence():
    assert strip_wake_phrase("is the spot free", ["HEY SPOT"]) == "is the spot free"

def test_peak_safe_gain_caps_loud_frames():
    assert peak_safe_gain(np.full(480, 0.4, np.float32), 0.5) == pytest.approx(1.25)
    assert peak_safe_gain(np.full(480, 0.01, np.float32), 0.5) == 8.0
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. Replace the fixed ×8 in `sherpa_onnx.py` (and `safety_kws.py`) with `peak_safe_gain`. Replace the `WAKE_PHRASE_PATTERN` uses in client_mic with `strip_wake_phrase`. **Step 4:** PASS and the suite is green.
- [ ] **Step 5:** `git commit -m "wake: phrase list file, fuzzy wake-phrase strip, peak-safe kws gain"`

### Task 7: Speaker lock logic (pure)

**Files:**
- Create: `src/voice_control/speaker_lock.py`
- Test: `tests/voice_control/test_speaker_lock.py`

**Interfaces:**
- Produces:
  - `circular_median_deg(degs: list[float]) -> float | None`: the sample minimising the summed circular distance; `None` if empty.
  - `ang_diff(a, b) -> float`: in [0, 180].
  - `mic_to_body_deg(mic_deg, offset_deg, sign) -> float`: in [0, 360).
  - `class SpeakerLock(window_deg, timeout_s, nudge)`:
    - `.lock(degs: list[float], now: float) -> float | None`
    - `.accept(degs: list[float], now: float) -> bool`
    - `.touch(now)`, `.expired(now) -> bool`, `.release()`
    - `.angle -> float | None`, `.locked -> bool`
- `degs` are the speech-flagged azimuths over the window.
- `accept` rules:
  - Not locked → True.
  - Empty `degs` → True and logs "no direction" (Review Focus 1).
  - Within the window → True: nudge `angle += nudge * signed_diff` and touch.
  - Otherwise → False.
- `lock` with empty `degs` leaves the lock **unset**, so the conversation still runs without direction gating.
- [ ] **Step 1: Failing tests**

```python
def test_circular_median_wraps():
    assert ang_diff(circular_median_deg([350, 355, 5, 10, 0]), 0) <= 5

def test_accept_across_zero():
    L = SpeakerLock(35, 20, 0.3); L.lock([355], now=0)
    assert L.accept([10], now=1) is True

def test_reject_other_side():
    L = SpeakerLock(35, 20, 0.3); L.lock([90], 0)
    assert L.accept([250, 255], 1) is False

def test_accept_with_no_readings_accepts():
    L = SpeakerLock(35, 20, 0.3); L.lock([90], 0)
    assert L.accept([], 1) is True

def test_nudge_and_timeout():
    L = SpeakerLock(35, 20, 0.5); L.lock([0], 0); L.accept([20], 5)
    assert L.angle == pytest.approx(10, abs=0.5)
    assert L.expired(24) is False and L.expired(25.1) is True

def test_lock_with_no_readings_stays_unlocked():
    L = SpeakerLock(35, 20, 0.3)
    assert L.lock([], 0) is None and L.locked is False

def test_mic_to_body():
    assert mic_to_body_deg(100, offset_deg=90, sign=1) == pytest.approx(10)
    assert mic_to_body_deg(80, offset_deg=90, sign=-1) == pytest.approx(10)
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** `git commit -m "speaker_lock: circular stats, lock/accept/nudge/timeout, mic-to-body angle"`

### Task 8: Chimes that don't mute the mic + reopen timing (A19, A5)

**Files:**
- Modify:
  - `src/voice_control/audio_player.py` (`_Task`, `enqueue_raw`, `_worker_loop`, add `is_speech_busy`)
  - `src/voice_control/client_mic.py:216-233` (`audio_callback`), `:961-984` (reopen block)
  - `src/voice_control/response_gate.py`
- Test: `tests/voice_control/test_audio_player_busy.py`, `tests/voice_control/test_response_gate.py`

**Interfaces:**
- Produces: `AudioPlayer.is_speech_busy() -> bool`. True while any task whose label does **not** start with `"beep:"` is queued or playing. Keep a separate `_speech_inflight` counter under the existing `_inflight_lock`. `is_busy()` is unchanged.
- `audio_callback` discards frames only when `_mic_gated or _operator_muted or is_speech_busy()`. The tail cooldown uses `Tuning.speaker_tail_s` and is measured from the last **speech** busy frame.
- Reopen block:
  - The settle comes from `Tuning.reopen_delay_s`.
  - On reopen, **always** call `_enter_state(LISTENING)`, not only when `gate_announced`.
  - The drain at reopen keeps no audio, but pre-roll capture starts on the very next frame.
- [ ] **Step 1: Failing tests**

```python
def test_beep_does_not_count_as_speech(fake_stream_player):
    p = fake_stream_player; p.enqueue_raw(np.zeros(2400, np.float32), 24000, label="beep:listening")
    assert p.is_busy() is True and p.is_speech_busy() is False

def test_tts_counts_as_speech(fake_stream_player):
    p = fake_stream_player; p.enqueue_raw(np.zeros(2400, np.float32), 24000, label="tts:hello")
    assert p.is_speech_busy() is True

def test_reopen_uses_tuned_delay():
    assert should_reopen_mic(True, reopen_at=10.0 + 0.2, now=10.25) is True
```

  (The `fake_stream_player` fixture builds an `AudioPlayer` whose `_open_stream` is monkeypatched to a no-op writer that blocks on an event, so tasks stay in flight.)
- [ ] **Step 2:** FAIL. **Step 3:** Implement. Check that TTS enqueues in `spot_tts.py` use a non-`beep:` label; set `label="tts:..."` if they are empty. **Step 4:** PASS and the suite is green.
- [ ] **Step 5:** `git commit -m "mic reopen: chimes no longer mute mic, tuned settle, listening cue fires at the instant mic opens"`

### Task 9: Replay harness + tune from recordings

**Files:**
- Create: `tests/audio/replay_demo.py`
- Modify: `config/tuning.yaml`, `config/wake_phrases.txt` (values from the results)

**Interfaces:**
- Consumes:
  - Tasks 1, 5, 6, 7.
  - The recordings from operator session 1.
  - `WakeWordDetector` from `src/voice_control/wake/sherpa_onnx.py`.
  - sherpa Silero VAD with the same config as `client_mic.py:763-774`.
  - Optional ASR via the gRPC server `server.py` on port 50055.
- Produces: the CLI `spot-env/bin/python tests/audio/replay_demo.py --conditions quiet loud overlap --channel {0,1} [--asr]`. It prints one table per condition × channel × phrase:
  - wake hit rate (wake clips), false fires/min (background)
  - utterances started; cut-offs (endpoint before `expected_text`'s last word appears in ASR, or before 80% of clip speech when there's no ASR)
  - caps, SNR median
  - lock correctness: overlap/wake clips where `circular_median` of speech-flagged DoA is within the window of `angle_deg` after `mic_to_body_deg`
  - chime self-trigger count: each chime from `audio_feedback` rendered, resampled to 16 k and replayed → must be 0 utterances

  `--json out.json` saves the results.
- [ ] **Step 1:** Implement the harness. It reuses Task 5 and Task 7 classes exactly as client_mic will.
- [ ] **Step 2:** Run on all conditions for both channels. Paste the tables into `docs/project/demo-tuning-results.md`.
- [ ] **Step 3:** Choose the values:
  - the phrase with the best hit rate whose false fires/min are ≤ 0.2
  - the channel with fewer cut-offs at equal hit rate
  - `min_snr_db` and `end_silence_s` at the lowest values that give 0 cut-offs on the pause clips
  - `kws_threshold`
  - `beam_steering` stays False unless overlap lock correctness improves with it (re-record 1 overlap set with beams on via `xvf_probe` if time allows)

  Write the chosen values into `config/tuning.yaml` and `config/wake_phrases.txt`. Re-run `scripts/setup_kws.py --phrases-file config/wake_phrases.txt`.
- [ ] **Step 4:** `git commit -m "replay harness + demo tuning results; tuned thresholds, channel and wake phrase"`

### Task 10: Wire front-end + lock into client_mic

**Files:**
- Modify: `src/voice_control/client_mic.py`:
  - constants `:130-154`
  - `calibrate_noise_floor :340-380`
  - main loop `:1000-1230`
  - argparse `:593-623`
- Test: the full suite plus replay parity (below).

**Interfaces:**
- Consumes: Tasks 1, 2, 5, 6, 7, 8.
- Changes:
  1. **Calibration:** `calibrate_noise_floor` returns the measured RMS always, with a warning when high. The loop seeds `FloorTracker` with its dBFS (A2).
  2. **VAD on every frame:** every frame goes to `vad.accept_waveform`, then `is_speech = vad.is_speech_detected()`. Then `Endpointer.feed(is_speech, db, floor_db)` drives start, end and cap, replacing the energy gate and the wall-clock silence timer (A1, A3, A6). Remove `energy_threshold` / `NAV_ENERGY_MULT`. While the robot moves, add +6 dB to `min_speech_dbfs`.
  3. **VAD housekeeping:** pop VAD segments after each check (`while not vad.empty(): vad.pop()`, A15). The pre-roll deque is appended in every state (A16).
  4. **Utterance gate:** at end or cap, `passes_gate(...)` must be True, otherwise the utterance is dropped with a log line (A4, A7).
  5. **Mic channel:** `--mic-channel` defaults to `Tuning.mic_channel` (A17).
  6. **Lock on wake:**
     - Call `mic.samples_between(t_wake - 1.5, t_wake)` on the speech-flagged degrees, then `lock.lock(...)`.
     - Then `_enter_state(LISTENING)` (A10).
     - If `Tuning.beam_steering`, call `mic.set_fixed_beams(radians(lock.angle))`.
  7. **Each utterance:** `lock.accept(samples over the utterance span)`. A rejection calls `_signal_reject()` (a ring flicker, Task 11), with no ASR and no reply.
  8. **Conversation timeout:** `lock.expired(now)` replaces `LISTENING_TIMEOUT`. On expiry: `lock.release()`, `mic.set_fixed_beams(None)`, `state = WAKE_WORD`, `_enter_state(WAKE_WORD)` plus the released chime (A11).
  9. **New wake while locked:** re-lock to the new speaker.
  10. **Crowd mode:** after each reply go to `WAKE_WORD`, not `LISTENING`.
  11. **Metrics:** one JSON line per utterance to `logs/utterances/<date>.jsonl` with keys `ts, floor_dbfs, speech_dbfs_median, speech_dbfs_p90, snr_db, endpoint, end_silence_s, duration_s, lock_deg, utt_deg, accepted, transcript, reached_llm`. A 1 Hz idle line goes in the same file with `kind:"idle"`.
- [ ] **Step 1:** Make changes 1–5. Run the suite (green) and a dry run (`--dry-run --no-wake-word`) speaking 3 sentences with a 1 s pause. None are cut off; metrics lines appear.
- [ ] **Step 2:** Make changes 6–11. Dry run with the mic attached. Wake from the front, then talk from the back: rejected (log). Talk from the front: accepted. After 20 s of silence the released chime plays.
- [ ] **Step 3:** `git commit -m "client_mic: vad-every-frame endpointing, snr gate, speaker lock, conversation timeout, utterance metrics"`

### Task 11: Turn signalling (ring first, chimes)

**Files:**
- Modify:
  - `src/voice_control/state_feedback.py`
  - `src/voice_control/audio_feedback.py` (new chime methods)
  - `src/voice_control/client_mic.py:570-585` (`_enter_state`)
- Test: `tests/voice_control/test_state_feedback.py`, `tests/voice_control/test_audio_feedback.py`

**Interfaces:**
- Produces:
  - **New `AudioFeedback` methods:**
    - `listen_cue()`: rising two-note, E5→A5, 140 ms, vol 0.2
    - `think_cue()`: the existing `thinking()` tones
    - `speak_cue()`: a single soft C6, 70 ms, vol 0.15
    - `released_cue()`: descending G5→C5, 200 ms
    - `reject_cue()`: none (visual only)

    The power-on `ready()` is unchanged and no longer used for turn-taking.
  - `chime_for(state)` mapping: LISTENING→`listen_cue`, THINKING→`think_cue`, RESPONDING→`speak_cue`, WAKE_WORD→None. `released_cue` is played explicitly on lock release.
  - `ring_for(state, lock_deg: float | None, muted: bool) -> RingSpec`:
    - `RingSpec(effect: int, color: int | None, ring: list[int] | None, brightness: int)`.
    - WAKE_WORD → breath (1), dim white `0x202020`.
    - LISTENING → ring mode (5) with `ring_toward(lock_deg, 0x00C000)`, or solid green (3) if `lock_deg` is None.
    - THINKING → breath, amber `0xC06000`.
    - RESPONDING → solid (3), blue `0x0040FF`.
    - muted → solid red `0xC00000`, overriding all of the above.
  - `ring_toward(deg: float, color: int, width: int = 3) -> list[int]`: 12 values, lighting the `width` LEDs centred on `round(deg / 30) % 12` and 0 elsewhere.
  - `_enter_state` additionally calls `mic.set_led_effect`, `set_led_color` or `set_ring` from `ring_for`, each in its own try/except like the existing chime and LED blocks. `_signal_reject()` flashes the ring red for 200 ms, then restores the current spec.
- [ ] **Step 1: Failing tests**

```python
def test_ring_toward_front_and_wrap():
    r = ring_toward(0, 0x00C000); assert r[0] and r[1] and r[11] and not r[6]
    r = ring_toward(345, 0x00C000); assert r[0] and r[11] and r[10]

def test_ring_for_states():
    assert ring_for(VoiceState.RESPONDING, None, False).color == 0x0040FF
    assert ring_for(VoiceState.LISTENING, 90, False).effect == 5
    assert ring_for(VoiceState.THINKING, 90, True).color == 0xC00000

def test_turn_chimes_distinct_from_ready():
    names = {chime_for(s) for s in (VoiceState.LISTENING, VoiceState.THINKING, VoiceState.RESPONDING)}
    assert names == {"listen_cue", "think_cue", "speak_cue"} and "ready" not in names
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. The LED ring LED-index order is calibrated with `xvf_probe leds`: if LED 0 is not at mic 0°, add `ring_offset_leds` to `Tuning`. Apply the measured LED limits: throttle ring writes to the measured safe rate. **Step 4:** PASS.
- [ ] **Step 5:** On-device check: run a dry run and watch the ring through wake → think → speak → listening → timeout. Listen to the three cues. Confirm the reopen cue plays **as** the mic opens. Check with a quick reply that the first word is captured (see the metrics transcript).
- [ ] **Step 6:** `git commit -m "signalling: mic ring primary (points at speaker), distinct listen/think/speak/released cues"`

### Task 12: Conversation-quality fixes

**Files:**
- Modify:
  - `src/voice_control/client_mic.py:543-561` (safety), `:1475`, `:1500-1507`
  - `src/voice_control/intent.py:20-22`
  - `src/voice_control/brain/llamacpp_backend.py:89`
  - `src/voice_control/spot_tts.py:310-345`
  - `src/voice_control/llm_brain.py:100-145`, `:284-306`, `:387`, `:410-425`, `:609-619`
  - `src/voice_control/spot_dispatch.py:103-106`, `:734`
- Test: `tests/voice_control/test_safety_regex.py`, `tests/voice_control/test_chunker_escapes.py`, `tests/voice_control/test_prompt_layout.py`, `tests/voice_control/test_filler.py`

**Interfaces:**
- Produces:
  - `check_safety_command(text) -> dict | None`. Fires only if the utterance has ≤ 4 words **or** its first word is a safety word. Move the patterns to `intent.py` as the single source and import them in client_mic.
  - `is_filler_only(text: str) -> bool`: True if fewer than 2 words remain after removing {uh, um, er, ah, hmm, like, so, okay}.
  - `TTSChunker` decodes JSON string escapes (`\n`, `\"`, `\\`, `\uXXXX`) before splitting.
  - `LLMBrain._build_messages(...)` order: `SYSTEM_PROMPT + knowledge` (static), then the persona prefix, then the state block. No `uptime_s` / `process_pid`.
- [ ] **Step 1: Failing tests**

```python
def test_stop_inside_question_not_safety():
    assert check_safety_command("what's the next stop on the tour") is None
def test_short_stop_is_safety():
    assert check_safety_command("stop")["intent"] == "stop"
    assert check_safety_command("spot stop now")["intent"] == "stop"
def test_leading_stop_long_is_safety():
    assert check_safety_command("stop walking towards the table please")["intent"] == "stop"
def test_estop_wins():
    assert check_safety_command("e-stop")["intent"] == "estop"

def test_chunker_decodes_escapes():
    out = []; c = TTSChunker(out.append); c.feed('Hi\\nthere \\u2014 friend. '); c.flush()
    assert "u2014" not in " ".join(out) and "\\" not in " ".join(out)

def test_static_prompt_prefix_stable_across_personas(brain):
    a = brain._build_messages("hi", state={})[0]["content"]; brain.session_state.current_persona = "pirate"
    b = brain._build_messages("hi", state={})[0]["content"]
    assert a[:4000] == b[:4000] and "uptime_s" not in b

def test_filler_only():
    assert is_filler_only("uh I think my uh") is False   # "I think my" = 3 words
    assert is_filler_only("um uh hmm") is True
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement each item:
  - **B1:** `r.encoding = "utf-8"` before `iter_lines`.
  - **B2:** chunker escapes.
  - **A13:** safety rule; also update `test_safety` callers.
  - **B5/B7:** prompt order; remove volatile keys from `get_robot_state_dict` output used in the prompt.
  - **B6:** apply the persona voice switch before TTS of the same reply (move the switch at `client_mic.py:1440-1444` ahead of the speak call).
  - **B4:** delete the "Battery is looking good" example sentence (`llm_brain.py:136`).
  - **B13:** add `enable_mouth`/`disable_mouth` lines to the prompt's action list.
  - **B14:** rewrite the canned lines in the files listed (no raw keys, e.g. `persona.replace("_", " ")`).
  - **E1:** `is_available()` delegates to the active backend (`_backend().is_available()`). `describe_persona` uses `_backend()`.
  - **B11:** `is_filler_only` check before the LLM call.
- [ ] **Step 4:** **B3 spoken info results:** dispatcher handlers `battery_status`, `status`, `check_obstacles`, `list_locations`, `list_maps` return a short sentence string via a new `dispatch_intent(..., speak=tts.speak)` kwarg. The handler calls `speak(sentence)` itself after the data arrives, e.g. `f"My battery is at {pct} percent."`. In the system prompt, tell the LLM to reply only "Let me check." for these actions.
- [ ] **Step 5:** **B12 thinking filler:** in `process_utterance`, a `threading.Timer(Tuning.thinking_filler_s)` speaks one persona-neutral line ("Hmm, let me think.") if no TTS has started. It is cancelled on the first chunk.
- [ ] **Step 6:** Tests PASS and the suite is green. Dry run: ask "what's your battery" (spoken number with a robot; a graceful line without one), "what's the next stop" (no halt), and a persona switch (new voice on the announcing reply; check the latency is < 5 s in metrics).
- [ ] **Step 7:** `git commit -m "conversation fixes: stop-rule, utf-8 stream, escape-safe tts, cache-friendly prompt, spoken info results, filler + thinking cue, backend availability"`

### Task 13: TTS per-sentence fallback (B8)

**Files:**
- Modify: `src/voice_control/tts/__init__.py:56-90`, `src/voice_control/spot_tts.py` (render path), `src/voice_control/audio_player.py:572-575`
- Test: `tests/voice_control/tts/test_fallback.py`

**Interfaces:**
- Produces: `synthesize_with_fallback(text, voice_id, primary: TTSBackend, fallback: TTSBackend, fallback_voice: str) -> tuple[bytes, str]`. Returns the audio and the name of the backend used. It catches any primary exception or empty audio and logs once per minute. The Kokoro backend is constructed at startup regardless of the primary.
- [ ] **Step 1: Failing test**

```python
def test_fallback_on_primary_error():
    class Bad:  # primary
        def synthesize(self, t, v): raise ConnectionError("wifi")
    class Good:
        def synthesize(self, t, v): return b"\x01\x02"
    audio, used = synthesize_with_fallback("hi", "x", Bad(), Good(), "af_sarah")
    assert audio == b"\x01\x02" and used == "fallback"
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement and use it in the SpotTTS render path. **Step 4:** PASS. Manual check: run with ElevenLabs and pull the network for a moment; speech continues in the Kokoro voice.
- [ ] **Step 5:** `git commit -m "tts: per-sentence kokoro fallback when elevenlabs fails"`

### Task 14: Arm safety and walking pose (C1–C3)

**Files:**
- Modify:
  - `src/session.py:174-187` (`deploy_arm_safe`, `stow_arm`)
  - `src/voice_control/spot_dispatch.py`:
    - `_close_mouth :576`, `_handle_enable_mouth :592`, `_handle_disable_mouth :629`
    - stop `:792`, freeze `:805`, estop `:1091`
    - locomotion branches walk_to/turn/walk/strafe/go_to/tour/patrol/come_back/go_to_object/follow_me
    - `close_spot_session :255`
  - `src/voice_control/chatbot/session_state.py`
- Test: `tests/voice_control/test_arm_policy.py`

**Interfaces:**
- Produces:
  - `SessionState.arm_pose: str`: one of `"stowed" | "talk" | "walk"`.
  - `arm_policy_for(intent_name: str, current_pose: str) -> tuple[str, str]` (pure): (pose before, pose after). Locomotion intents → (`"walk"`, previous pose); `stop`/`freeze`/`estop`/shutdown → (`"stowed"`, `"stowed"`); `emote` → (`"stowed"`, previous); everything else → (current, current).
  - `move_arm_to(cmd_client, pose: str, timeout_s=5.0) -> bool` in `session.py`:
    - `"stowed"` = close gripper, then `arm_stow_command`.
    - `"talk"` = `arm_ready_command`.
    - `"walk"` = the tucked pose chosen in operator session 2.
    - All three use `ArmCartesianCommand`/joint limits with `max_linear_velocity` 0.3 m/s and `max_angular_velocity` 1.0 rad/s via `RobotCommandBuilder` params, where the SDK exposes them; otherwise `arm_joint_move_helper` with a ≥ 2 s trajectory.
  - The locomotion velocity cap is `Tuning.walk_speed_cap`, applied to every `synchro_velocity_command` and `RouteGenParams`/travel params in `graph_nav_utils.py`.
- [ ] **Step 1: Failing tests**

```python
def test_locomotion_tucks_then_restores():
    assert arm_policy_for("walk", "talk") == ("walk", "talk")
    assert arm_policy_for("go_to", "talk") == ("walk", "talk")
def test_safety_stows():
    for n in ("stop", "freeze", "estop"): assert arm_policy_for(n, "talk") == ("stowed", "stowed")
def test_chat_untouched():
    assert arm_policy_for("set_persona", "talk") == ("talk", "talk")
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement the policy. In `dispatch_intent`, wrap the existing branches: compute (before, after), call `move_arm_to(before)` if it differs from the current pose and the arm is deployed, run the action, then restore. Locomotion waits on `move_arm_to` returning True before moving. Stop/freeze/estop first send their motion stop (unchanged), then stow, so stowing never delays the stop.
- [ ] **Step 4: OPERATOR SESSION 2 (robot, E-Stop live).** Try `"walk"` candidates with a scratch script that calls `move_arm_to`:
  - (a) `arm_stow_command` with the gripper opening and closing while stowed;
  - (b) a tight carry from `arm_carry_command`;
  - (c) a custom joint pose (sh0 = 0, sh1 = −2.6, el0 = 2.6, el1 = 0, wr0 = 0, wr1 = 0).

  Pick the tightest one where the claw can still open and close without hitting the body. Record the choice in `move_arm_to`.
- [ ] **Step 5:** Tests PASS. `safety-reviewer` on the diff. On-robot check: talk pose → "walk forward one meter" (tucks, walks ≤ 0.5 m/s, restores) → "stop" mid-walk (stops, then stows).
- [ ] **Step 6:** `git commit -m "arm safety: stow on stop/freeze/estop/shutdown, tucked walking pose, velocity-limited arm moves, walk speed cap"`

### Task 15: Emotes and dances by natural phrasing

**Files:**
- Create: `src/voice_control/emotes.py`
- Modify: `src/voice_control/spot_dispatch.py` (new `emote` branch), `src/voice_control/grammar/spot_action.gbnf:14` (add `"\"emote\""`), `src/voice_control/llm_brain.py` (prompt section)
- Test: `tests/voice_control/test_emotes.py`

**Interfaces:**
- Produces:
  - `ROUTINES: dict[str, str]`: name → one-line description. The keys are exactly `hey_join_me`, `performance_bow`, `smooth_sailing`, `happy_gesture`, `tail_wag`, `upright_sit`, `wanna_play`.
  - `DANCES = ("hey_join_me", "performance_bow", "smooth_sailing")`.
  - `resolve_routine(name: str | None, last: str | None, rng=random) -> str`:
    - an exact key → that key;
    - `None`, `"dance"` or `"random"` → a random dance ≠ `last`;
    - an unknown name → raises `ValueError`.
  - `prompt_section() -> str`: lists each routine with its description and 2–3 trigger phrasings ("dance", "show off", "do a trick", "take a bow", "wag your tail", "are you happy", "sit up nicely", "want to play").
  - Dispatcher `emote` with params `{"name": str}`:
    1. Refuse while navigating, with a spoken line.
    2. `check_obstacles`: if the nearest obstacle is < `Tuning.dance_clearance_m` (1.5 m), refuse with the spoken line "I need a little more room to dance."
    3. Stand if needed.
    4. `move_arm_to("stowed")` (Task 14 policy).
    5. `ChoreographyClient` (SDK created with `[ChoreographyClient]`) `execute_choreography(name, client_start_time=time.time() + 1.0, choreography_starting_slice=0)`, then poll `get_choreography_status` until it is not `STATUS_DANCING` (timeout 60 s).
    6. Restore the arm pose.
  - `stop`/`freeze` during a routine stop it: the existing stop command overrides the choreography, so verify this on the robot.
- [ ] **Step 1: Failing tests**

```python
def test_resolve_exact_and_random():
    assert resolve_routine("tail_wag", None) == "tail_wag"
    r = resolve_routine("dance", last="hey_join_me", rng=random.Random(0))
    assert r in DANCES and r != "hey_join_me"
def test_unknown_raises():
    with pytest.raises(ValueError): resolve_routine("signal_alert", None)
def test_prompt_lists_all():
    s = prompt_section(); assert all(n in s for n in ROUTINES) and "show off" in s
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement and add the grammar entry. **Step 4:** PASS. Run offline trigger checks: send 10 phrasings through the brain with the robot in dry-run ("can you dance", "show off for us", "take a bow", "wag your tail", "are you happy", "sit up nicely", "want to play", "do a trick", "dance again", "bust a move") and confirm each yields an `emote` action with a valid name. Record the hit count in `docs/project/demo-tuning-results.md`.
- [ ] **Step 5:** `safety-reviewer`. On-robot check: run `performance_bow` and `smooth_sailing` with clear space. Check that "stop" mid-dance stops it, and that the obstacle check refuses when someone stands within 1.5 m.
- [ ] **Step 6:** `git commit -m "emote action: all built-in dances/emotes by natural phrasing, clearance check, arm stowed during routine"`

### Task 16: Event knowledge pack (important) + map (last)

**Files:**
- Create: `src/voice_control/knowledge/event_knowledge.md`
- Modify: `src/voice_control/llm_brain.py:214-245` (`_KNOWLEDGE_FILES` order and framing)
- Test: `tests/voice_control/test_knowledge_order.py`

**Interfaces:**
- `_KNOWLEDGE_FILES = ["event_knowledge.md", "thayer_knowledge.md", "tour_route.md"]`. The event pack is framed as "PRIMARY CONTEXT: you are at <event>; prefer this over tour material".
- The template headings are: Event (name, date, place), Our poster (title, one-paragraph pitch, three key results), Team (names, roles), How Spot works (plain-language pipeline), FAQs (10 Q/A slots), Things not to claim. The operator fills it in.
- [ ] **Step 1: Failing test**

```python
def test_event_pack_first():
    from src.voice_control import llm_brain
    assert llm_brain._KNOWLEDGE_FILES[0] == "event_knowledge.md"
    assert llm_brain._KNOWLEDGE_TEXT.index("event_knowledge.md") < llm_brain._KNOWLEDGE_TEXT.index("thayer_knowledge.md")
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement and write the template. **Step 4:** PASS. Ask the operator to fill it in. **Step 5:** `git commit -m "event knowledge pack template, loaded first as primary context"`
- [ ] **Step 6 (last, only if time):** the operator maps the venue (`scripts/setup_map.py`, `scripts/map_waypoints.py`). Confirm the launch works with `--no-map` regardless.

### Task 17: Ops: launch flags, logs, privacy

**Files:**
- Modify:
  - `scripts/wakespot.py:157-187` (pass `--latency`, `--mic-channel`, `--no-transcript-log`)
  - `src/voice_control/chatbot/conversation_log.py:12` (path = `<repo>/logs/conversations`)
  - `tests/voice_control/test_brain_concurrency.py:39` (tmp log dir via monkeypatch)
  - `requirements.txt:54` (move `livekit-wakeword` to a comment under "training-only extras"; run pin-guardian)
- Test: the existing `test_conversation_log` plus a new assertion that tests don't touch `logs/`.
- [ ] **Step 1:** Implement. `--no-transcript-log` makes the conversation and utterance logs store `transcript: null`.
- [ ] **Step 2:** Suite green. `scripts/wakespot --help` shows the new flags.
- [ ] **Step 3:** `git commit -m "ops: wakespot passes latency/channel/no-transcript flags, logs under logs/, tests isolated from real logs"`

### Task 18: Voiceprint check (P4b) — after Task 10 passes on the robot

**Files:**
- Create: `src/voice_control/speaker_embed.py`, `scripts/setup_wespeaker.py`
- Modify: `src/voice_control/client_mic.py` (lock-accept call site from Task 10)
- Test: `tests/voice_control/test_speaker_embed.py`

**Interfaces:**
- Produces:
  - `fbank(pcm16: bytes, sr=16000) -> np.ndarray`: shape (T, 80) float32, Kaldi-compatible 25 ms / 10 ms, mean-normalised per utterance.
  - `class SpeakerEmbedder(model_path)` with `.embed(pcm16: bytes) -> np.ndarray`: 256-d, L2-normalised; onnxruntime with CUDA EP, falling back to CPU.
  - `cosine(a, b) -> float`.
  - `accept_combined(dir_ok: bool | None, sim: float | None, threshold: float) -> bool`:
    - `None` means the signal is unavailable.
    - Both present → `dir_ok and sim >= threshold`.
    - Only one present → that one decides.
    - Neither → True.
- The enrolment embedding = mean of embed(wake audio) and embed(first accepted utterance). It is kept in memory and cleared on `lock.release()`.
- `setup_wespeaker.py` downloads `Wespeaker/wespeaker-voxceleb-resnet34-LM` → `voxceleb_resnet34_LM.onnx` (26 MB) into `models/speaker/`. That directory is gitignored.
- [ ] **Step 1: Failing tests**

```python
def test_accept_combined_truth_table():
    assert accept_combined(True, 0.7, 0.5) is True
    assert accept_combined(True, 0.3, 0.5) is False
    assert accept_combined(None, 0.6, 0.5) is True
    assert accept_combined(False, None, 0.5) is False
    assert accept_combined(None, None, 0.5) is True

def test_fbank_shape():
    pcm = (np.random.default_rng(0).standard_normal(16000) * 3000).astype(np.int16).tobytes()
    f = fbank(pcm); assert f.shape[1] == 80 and 95 <= f.shape[0] <= 100 and abs(float(f.mean())) < 1e-3
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. Use `torchaudio.compliance.kaldi.fbank` if torchaudio is importable in spot-env, otherwise a numpy implementation (povey window, preemph 0.97, 80 mel, 512 FFT). **Step 4:** PASS.
- [ ] **Step 5:** Threshold: extend `replay_demo.py --voiceprint`. For overlap clips, embed the visitor's wake clip versus (a) their own later speech and (b) background/second-voice segments. Set `voiceprint_threshold` midway between the two medians and record it in the results doc. Enable it in `config/tuning.yaml` only if the true-accept rate is ≥ 90 % and the false-accept rate is ≤ 20 %.
- [ ] **Step 6:** Wire into the Task 10 accept call: `accept_combined(lock.accept(...) if direction available else None, sim, thr)`. Add `spotctl voiceprint on|off`. Dry run with two people.
- [ ] **Step 7:** `git commit -m "voiceprint: wespeaker embedding gate combined with direction lock, memory-only enrolment"`

### Task 19: Arm gaze toward the locked speaker (P8)

**Files:**
- Create: `src/voice_control/gaze.py` (pure functions; Wave 1)
- Modify: `src/voice_control/spot_dispatch.py` (new `gaze_at_bearing(body_deg: float | None)`; Wave 3, Lane B), `src/voice_control/client_mic.py` (call on lock, on nudge > 15°, on release; Wave 3, Lane A)
- Test: `tests/voice_control/test_gaze.py`

**Interfaces:**
- Produces:
  - In `gaze.py`: `gaze_target(body_deg: float, dist_m=2.0, height_m=0.9) -> tuple[float, float, float] | None` (pure): a body-frame point. Returns `None` if `|body_deg|` (wrapped to [−180, 180]) is > 120.
  - `should_regaze(prev_deg, new_deg, min_change=15.0) -> bool`.
  - `gaze_at_bearing`:
    - Only when `arm_pose == "talk"`, Spot is not walking and no emote is running.
    - Sends `RobotCommandBuilder.arm_gaze_command(x, y, z, "body", max_linear_vel=0.3, max_angular_vel=1.0)` (5.2.0 signature).
    - `None` → `move_arm_to("talk")`.
- [ ] **Step 1: Failing tests**

```python
def test_gaze_front_and_side():
    x, y, z = gaze_target(0); assert x == pytest.approx(2.0) and abs(y) < 1e-6
    x, y, z = gaze_target(90); assert abs(x) < 1e-6 and y == pytest.approx(2.0)
def test_gaze_behind_none():
    assert gaze_target(170) is None and gaze_target(-150) is None
def test_regaze_threshold():
    assert should_regaze(0, 10) is False and should_regaze(0, 20) is True and should_regaze(350, 10) is True
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. `safety-reviewer`. On-robot check:
  - Wake from the front-left: the gripper turns toward you slowly.
  - While it's speaking: the mouth still moves. If the gaze and claw commands cancel each other, send the claw fraction inside the same synchro command as the gaze, or re-issue the gaze after each mouth frame batch.
  - Release: back to the talk pose.
- [ ] **Step 5:** `git commit -m "gaze: point gripper toward locked speaker bearing (front ±120 deg), slow, re-aim on >15 deg change"`

### Task 20: Mouth refinement (C4)

**Files:**
- Modify: `src/voice_control/animation/mouth.py:31-52,91-142`, `src/voice_control/spot_tts.py:203-237`
- Test: `tests/voice_control/animation/test_mouth_envelope.py`

**Interfaces:**
- Produces:
  - `envelope(samples, rate, fps=MOUTH_FPS, gate=0.06, intensity=1.0, ref_peak: float | None = None, max_open=0.6, attack_s=0.03, release_s=0.10) -> np.ndarray`:
    - normalises by `ref_peak` when given, otherwise the chunk peak;
    - applies one-pole attack/release smoothing;
    - caps at `max_open`.
  - `MouthDriver.play(fractions, t0)` schedules at `t0 + Tuning.mouth_lead_s`.
  - `MouthDriver.rest()` eases to `Tuning.mouth_rest_open` between sentences. `close()` runs only at the end of the reply.
  - `SpotTTS` computes `ref_peak` once per reply from the first rendered chunk, keeps it for the remaining chunks, and calls `rest()` between chunks and `close()` at the end.
- [ ] **Step 1: Failing tests**

```python
def test_cap_and_ref_peak():
    s = np.concatenate([np.zeros(2400), 0.2 * np.ones(4800)]).astype(np.float32)
    e = envelope(s, 24000, ref_peak=0.4, max_open=0.6)
    assert e.max() <= 0.6 + 1e-6 and e.max() == pytest.approx(0.3, abs=0.05)
def test_attack_release_smooths_step():
    s = np.concatenate([0.5 * np.ones(4800), np.zeros(4800)]).astype(np.float32)
    e = envelope(s, 24000, attack_s=0.03, release_s=0.10)
    drop = e[len(e)//2 : len(e)//2 + 3]; assert drop[0] > drop[1] > drop[2] > 0   # gradual release
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. On-robot check: record a slow-motion phone video of the claw and the speaker. Set `mouth_lead_s` to the measured offset (claw lag → positive lead).
- [ ] **Step 5:** `git commit -m "mouth: per-reply normalisation, attack/release smoothing, 0.6 cap, rest between sentences, tuned lead"`

### Task 21: Docs, rehearsal, merge

**Files:**
- Create: `docs/architecture/pipeline-explainer.md`
- Modify: `docs/getting-started/daily-operations.md`, `docs/project/status.md`, `docs/project/bugs.md` (strike fixed IDs with commit hashes), `docs/project/roadmap.md`, `mkdocs.yml` (add the explainer)
- [ ] **Step 1: Pipeline explainer.** One section per stage, in order:
  1. launch (wakespot → run_voice_control → server/client_mic, services)
  2. mic & XVF3800 (channels, DoA, ring)
  3. front-end (VAD, floor, endpointer, SNR gate)
  4. wake (phrases, KWS)
  5. speaker lock + voiceprint
  6. ASR
  7. safety path
  8. brain (prompt layout, knowledge, grammar)
  9. dispatcher (actions, arm policy, emotes)
  10. TTS + fallback + mouth
  11. signalling
  12. spotctl
  13. logs & metrics

  Each section covers: what it does, modules and key functions (`file:line`), its inputs and outputs, the tunables in `config/tuning.yaml`, failure modes, and how to debug each (which log line or command). Include the measured LED limits from session 1. Use `.understand-anything/knowledge-graph.json` layers as the skeleton.
- [ ] **Step 2:** `daily-operations.md`: the real launch. Check `llama-server` (and `ollama` until E1 lands), `scripts/wakespot <map>|--no-map [--enable-arm]`, `ssh` + `scripts/spotctl`, the rollback tag.
- [ ] **Step 3: Rehearsal** on the robot with babble playing from a laptop:
  - 10 visitor turns from different angles, including a second talker;
  - every `spotctl` command;
  - a walk with arm tuck;
  - two emotes;
  - one network drop (TTS fallback).

  Record pass/fail per item in `docs/project/demo-tuning-results.md`.
- [ ] **Step 4:** Update status/bugs/roadmap. `git commit -m "docs: pipeline explainer, daily ops, status/bugs/roadmap after demo prep"`
- [ ] **Step 5:** If the rehearsal passes: `git tag demo-ready-2026-10-29`, `git checkout tour_guide_upgrade_matteo && git merge --no-ff demo-poster-2026-10-29 -m "merge demo-poster-2026-10-29"`. If not, the demo runs from whichever tag last passed.
