"""The conversation loop: the voice main loop, the per-utterance turn
(ASR -> safety -> wake -> brain/regex -> dispatch + reply) and the robot
bridge it uses (dispatch, state, gaze, talk-pose resume, auto-arm retry).

Moved from the old client_mic.py (Task 10), logic unchanged. The
closures inside the old main() are Conversation methods; process_utterance
is a method; run() is the old main-loop body. The conversation state that
used to be client_mic module globals lives on CONVERSATION, created once at
import and reached through this module at call time.
"""
import math
import os
import queue
import random
import re
import threading
import time
from collections import deque
from contextlib import nullcontext

import numpy as np

import src.spot.audio.capture as _cap
import src.spot.core.events as _ev
import src.spot.conversation.filler as _filler
import src.spot.conversation.gate as _gate
import src.spot.conversation.utterance_log as _ul
import src.spot.mic.ring_control as _ring
import src.spot.operator.control as _ctl
import src.spot.safety.listener as _sl
import src.spot.stt.client as _stt
from src.spot.audio.barge_in import should_reset_after_player
from src.spot.audio.chimes import beep
from src.spot.audio.frontend import frame_dbfs, UtteranceStats, passes_gate
from src.spot.audio.response_gate import should_arm_reopen, should_reopen_mic
from src.spot.brain.intent import parse_intent, is_filler_only, short_utterance_ok
from src.spot.brain.persona import get_persona, voice_id_for
from src.spot.conversation.speaker_lock import SpeakerLock, circular_median_deg
from src.spot.conversation.voiceprint import VoiceprintGate
from src.spot.core.core_intents import intent_names
from src.spot.core.latency import get_recorder
from src.spot.core.tuning import get_tuning
from src.spot.core.voice_state import VoiceState
from src.spot.mic.state_feedback import chime_for
# Wake phrases (config/wake_phrases.txt). strip_wake_phrase removes a
# LEADING wake phrase, tolerating ASR variants; unchanged text = no match.
from src.spot.wake.phrases import WAKE_PHRASES, strip_wake_phrase

_tuning = get_tuning()

# The conversation timeout is Tuning.lock_timeout_s (20 s default; was a fixed
# 15 s LISTENING_TIMEOUT before the speaker lock).

# End silence: Tuning.end_silence_s_short until the utterance is this long,
# then Tuning.end_silence_s (chained commands pause mid-sentence).
SILENCE_CROSSOVER = 2.0
# Frames kept before onset (25 = 0.75 s): covers listen_cue (140 ms) + chime
# tail + Silero's onset lag with margin, so "Hey Spot stand up" said in one
# breath keeps the "stand" onset that arrived under the chime.
PREROLL_FRAMES = 25
TRAILING_KEEP_FRAMES = 10        # silence kept after the last speech frame for ASR (~300ms)
NAV_SPEECH_DB_BOOST = 6.0        # added to Tuning.min_speech_dbfs while the robot moves
LOCK_ACCEPT_MAX_SAMPLES = 100    # cap DoA samples per lock.accept (medoid is O(n^2))
WAKE_AUDIO_S = 1.5               # wake-phrase audio kept for the voiceprint (= DoA lock lookback)

# Inter-action waits when chaining LLM actions (e.g. "walk forward then sit").
# Empirical values — give Spot enough time for the previous command to finish
# before issuing the next, otherwise the second command races against the first.
CHAIN_MOVEMENT_WAIT_S = 3.0  # walk / strafe / turn (timed locomotion)
CHAIN_POSTURE_WAIT_S = 4.0   # sit / stand / selfright / body_height (slower body changes)

# Stock acknowledgments spoken before a VLM query. The LLM's own "response"
# field for a describe action is often a hallucinated answer (it can't see
# the camera), so we override it with one of these to fill the ~2-5s gap
# while the VLM actually runs. See process_utterance() for the override.
_DESCRIBE_ACKS = (
    "Let me take a look...",
    "One sec, checking now...",
    "Hold on, looking around...",
    "Taking a look...",
    "Let me see what's in front of me...",
    "Checking my camera...",
)

# Intent sets generated from core_intents + the enabled features, read at call
# time (Task 16):
# - "persona": set_persona / add_persona, run before the response is spoken.
# - "motion_free": still allowed in a turn whose utterance overlapped a safety
#   halt (final-review I1): nothing that moves the body or the arm.
# - "state_only": see execute_on_spot.

MIN_SPEECH_DURATION = 0.3  # Reject utterances shorter than this (catches noise bursts, real words are 0.3s+)


# ============================================================================
# Intent Execution
# ============================================================================
# State-only intents mutate brain/TTS state and do NOT touch the robot;
# dispatch_intent handles them BEFORE ensure_spot_session(), so they must run
# even in dry-run mode (Spot off). Otherwise "switch to pirate" no-ops and
# every subsequent voice_id resolves to the prior persona's voice.


def execute_on_spot(intent: dict, brain=None, speak=None) -> bool:
    """Execute parsed intent on Spot robot.

    `brain` is forwarded to dispatch_intent so handlers needing across-turn
    state (set_persona today, follow-ups later) can mutate it. Safety + regex
    fallback paths leave it None — those intents never touch brain state.
    """
    intent_name = intent.get("intent", "")
    if os.environ.get("SPOT_DRY_RUN") == "1" and intent_name not in intent_names("state_only"):
        shown = intent if CONVERSATION._store_transcript else {k: v for k, v in intent.items() if k != "raw"}
        print(f"[dry-run] would dispatch: {shown}")
        return True
    try:
        from src.spot.robot.dispatcher import dispatch_intent
        return dispatch_intent(intent, brain=brain, speak=speak)
    except Exception as e:
        print(f"[Spot error: {e}]")
        return False


def get_spot_state() -> dict:
    """Get current robot state for LLM context. Short-circuits in dry-run
    mode so the auto-degrade path stays robot-free (no spot_dispatch
    import, no disk I/O for locations.json / maps dir) per turn."""
    if os.environ.get("SPOT_DRY_RUN") == "1":
        return {}
    try:
        from src.spot.robot.state import get_robot_state_dict
        return get_robot_state_dict()
    except Exception as e:
        print(f"[State error: {e}]")
        return {}


def _is_robot_moving() -> bool:
    """Check if robot is actively navigating (motor noise expected)."""
    if os.environ.get("SPOT_DRY_RUN") == "1":
        return False  # no robot; don't import spot_dispatch (bosdyn) per utterance
    try:
        from src.spot.nav.runner import is_navigating
        return is_navigating()
    except Exception:
        return False


# Chain waits (between actions of one LLM reply). Movement and posture have
# no generated flag, so they stay literal; the nav wait is every generated
# locomotion intent that is not timed movement, so a new locomotion feature
# action gets the nav wait. walk_to is locomotion but had no wait in base:
# it stays excluded (G1 replaces walk_to with the vision port; its chain
# timing is not changed before that). Read at call time: a disabled
# feature's intents drop out.
_CHAIN_MOVEMENT = frozenset({"walk", "strafe", "turn"})
_CHAIN_POSTURE = frozenset({"sit", "stand", "selfright", "body_height"})
_CHAIN_NO_WAIT_LOCOMOTION = frozenset({"walk_to"})


def chain_wait_kind(cmd: str):
    """"nav", "movement", "posture" or None: the wait after `cmd` in a chain."""
    if cmd in _CHAIN_MOVEMENT:
        return "movement"
    if cmd in _CHAIN_POSTURE:
        return "posture"
    nav = (intent_names("locomotion") - _CHAIN_MOVEMENT - _CHAIN_NO_WAIT_LOCOMOTION) | {"emote"}
    return "nav" if cmd in nav else None


def _wait_for_nav_complete(timeout: float = 120.0):
    """Block until current navigation finishes (for command chaining)."""
    start = time.time()
    while _is_robot_moving() and (time.time() - start) < timeout:
        if CONVERSATION._abort_chain.wait(0.5):
            return  # operator stop — don't keep the chain waiting


def _enter_state(new_state):
    """Play the entry chime + set LEDs for a state transition. Best-effort —
    a chime or LED failure must NEVER crash the voice loop, so each indicator
    is isolated in its own try/except (the loop only catches KeyboardInterrupt)."""
    try:
        name = chime_for(new_state)
        if name is not None:
            getattr(beep, name)()
    except Exception as e:
        print(f"[Indicator] chime failed: {type(e).__name__}: {e}")
    if CONVERSATION._state_leds is not None:
        try:
            CONVERSATION._state_leds.set_state(new_state)
        except Exception as e:
            print(f"[Indicator] LED failed: {type(e).__name__}: {e}")
    _ring.RING._ring_state = new_state
    CONVERSATION._status["state"] = new_state.name
    _ring.RING.update()
    _ev.publish("voice_state_changed", state=new_state)


def _heard(text: str) -> str:
    """User words for console logs; redacted under --no-transcript-log."""
    if CONVERSATION._store_transcript:
        return f'"{text}"'
    return f"<heard {len((text or '').split())} words>"


def _safety_gen_now():
    """spot_dispatch's safety generation (None in dry-run / without a robot)."""
    if os.environ.get("SPOT_DRY_RUN") == "1":
        return None
    try:
        from src.spot.robot.dispatcher import current_safety_gen
        return current_safety_gen()
    except Exception:
        return None


def _auto_arm_at_turn(gen) -> None:
    """Retry an owed automatic fold at the start of a motion-allowed turn
    (e.g. the session came up mid-walk). Background thread; never raises."""
    if gen is None or os.environ.get("SPOT_DRY_RUN") == "1":
        return
    try:
        from src.spot.features.mouth.handlers import start_auto_arm
        start_auto_arm(gen)
    except Exception as e:
        print(f"[Spot] automatic arm fold skipped: {type(e).__name__}: {e}")


_gaze_q = queue.Queue(maxsize=1)  # latest gaze request only: (body deg or None, safety gen)
_gaze_worker = None
_gaze_stop = threading.Event()        # set by stop_gaze_worker (main's _shutdown_runtime)
_gaze_spawn_lock = threading.Lock()   # one gaze worker at a time


def _gaze_loop():
    while not _gaze_stop.is_set():
        try:
            body, gen = _gaze_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if _gaze_stop.is_set():
            break
        try:
            # Imported here: with the gaze folder deleted this fails per
            # request (printed), it never kills the worker.
            from src.spot.features.gaze.handlers import gaze_at_bearing
            gaze_at_bearing(body, gen)  # gen taken at request time: a stop since wins
        except Exception as e:  # noqa: BLE001 - gaze is cosmetic
            print(f"[Spot] gaze skipped: {type(e).__name__}: {e}")


def stop_gaze_worker() -> None:
    """Shutdown: the gaze worker exits within ~0.5 s and none is started
    again. Non-blocking (no join); never raises."""
    _gaze_stop.set()
    try:
        _gaze_q.get_nowait()  # drop a pending request
    except queue.Empty:
        pass


def allow_gaze_worker() -> None:
    """main() start: clear a previous run's stop (one main per process on the
    robot; tests run several). Any still-alive worker is reused, never doubled."""
    _gaze_stop.clear()


def _gaze_feature_loaded() -> bool:
    try:
        from src.spot.features import REGISTRY
        return REGISTRY.has("gaze")
    except Exception:  # noqa: BLE001
        return False


def _gaze_at_speaker(body_deg) -> None:
    """Ask the folded arm to turn toward the locked speaker (body-frame deg,
    None = centre). Off the audio loop on one worker; only the newest request
    is kept; the dispatcher clamps/rate-limits and checks pose/stop. Never
    raises; no-op in dry-run, after stop_gaze_worker, or when the gaze
    feature is not loaded."""
    global _gaze_worker
    if os.environ.get("SPOT_DRY_RUN") == "1":
        return
    try:
        if _gaze_stop.is_set() or not _gaze_feature_loaded():
            return  # shutting down, or gaze removed from enabled.py
        with _gaze_spawn_lock:
            if _gaze_worker is None or not _gaze_worker.is_alive():
                _gaze_worker = threading.Thread(target=_gaze_loop, name="gaze", daemon=True)
                _gaze_worker.start()
        try:
            _gaze_q.get_nowait()  # drop a stale request
        except queue.Empty:
            pass
        gen = _safety_gen_now()  # at request time, never at pickup
        if gen is None:
            return  # no robot / dry run: no gaze
        _gaze_q.put_nowait((body_deg, gen))
    except Exception as e:  # noqa: BLE001
        print(f"[Spot] gaze request failed: {type(e).__name__}: {e}")


def _resume_talk_after_halt(gen) -> None:
    """Bring the arm/mouth back to "talk" if a safety halt stowed them
    (final-review I1). `gen` was taken at the start of the turn, so a stop
    landing since then wins. Best-effort; never raises; no-op without a robot."""
    if gen is None or os.environ.get("SPOT_DRY_RUN") == "1":
        return
    try:
        from src.spot.robot.dispatcher import resume_talk_after_halt
        resume_talk_after_halt(gen)
    except Exception as e:
        print(f"[Spot] talk-pose resume skipped: {type(e).__name__}: {e}")


def _cue_speaking() -> None:
    hook, CONVERSATION._speaking_hook = CONVERSATION._speaking_hook, None
    if hook is not None:
        try:
            hook()
        except Exception as e:
            print(f"[Indicator] speaking cue failed: {type(e).__name__}: {e}")


def _say(tts, text, voice=None, cue=True) -> None:
    """tts.speak unless an operator hush/stop aborted this turn. cue=False for
    lines that are not the reply (the thinking filler)."""
    if tts is not None and text and not CONVERSATION._abort_chain.is_set():
        if cue:
            _cue_speaking()
        tts.speak(text, voice=voice)


def _brain_ready(brain) -> bool:
    """Positive availability is cached; a not-yet-ready llama-server (503
    while loading after a reboot) is re-checked on each utterance instead
    of disabling the brain for the whole session."""
    if CONVERSATION._brain_ok:
        return True
    try:
        CONVERSATION._brain_ok = bool(brain.is_available())
    except Exception:
        CONVERSATION._brain_ok = False
    if CONVERSATION._brain_ok:
        print("[Brain] llama-server is up — LLM brain active")
        try:
            brain.warm_up_async()  # first turn after a late start shouldn't be cold
        except Exception:
            pass
    else:
        print("[Brain] llama-server still not ready — regex fallback for this utterance")
    return CONVERSATION._brain_ok


def _resolve_voice(brain):
    """Voice ID for the brain's current persona on the active TTS backend."""
    from src.spot.tts import get_active_backend_name
    persona = get_persona(brain.session_state.current_persona, brain._persona_registry)
    return voice_id_for(persona, get_active_backend_name()) or None


class Conversation:
    """Owner of the conversation state (was client_mic module globals) and
    the voice loop (was the body of client_mic.main())."""

    def __init__(self):
        # Operator control (spotctl, Task 4). Plain attributes: the HTTP threads
        # read/flip them without locks so a slow request can never delay a stop.
        self._operator_muted = False
        self._crowd_mode = False        # after each reply drop back to WAKE_WORD
        self._voiceprint_mode = _tuning.voiceprint_enabled  # spotctl voiceprint on|off toggles live
        self._voiceprint_threshold = _tuning.voiceprint_threshold  # spotctl voiceprint threshold <x> sets live
        self._voiceprint_update_threshold = _tuning.voiceprint_update_threshold  # running-enrolment bar
        # Set by an operator stop/freeze/sit/stand/stow/hush: the chained-action loop
        # in process_utterance checks it before each step and waits on it instead of
        # sleeping, so an operator stop also cancels the rest of a chain.
        self._abort_chain = threading.Event()
        # Snapshot the main loop writes and `status` reads (no locks, never blocks).
        self._status = {
            "state": "WAKE_WORD", "locked_deg": None, "floor_dbfs": None,
            "last_transcript": None, "last_latency_s": None,
        }
        self._state_leds = None     # StateLeds instance (set in main); None = LEDs off
        # Speaker lock, MIC frame degrees (body frame only for status/metrics/gaze).
        self._lock = SpeakerLock(_tuning.lock_window_deg, _tuning.lock_timeout_s, _tuning.lock_nudge)
        # Voiceprint (Task 18): memory-only enrolment for the current conversation.
        # The embedder is assigned by a background warm-up thread (None until then:
        # fail open to direction-only). Only the main loop mutates the enrolment.
        self._voice = VoiceprintGate()
        self._brain_ok = False
        self._store_transcript = True  # False under --no-transcript-log: no user words on stdout
        # Per-turn "Spot starts talking" hook (final-review I2), set by
        # process_utterance. Fired once, just before the first reply line is queued,
        # so the RESPONDING ring + speak cue land before the speech (same FIFO player).
        self._speaking_hook = None

    def announce_thinking(self):
        # Passed into process_utterance; fired the moment a real transcript is
        # dispatched to the brain, so THINKING never chimes on noise/empty STT.
        self.gate_announced = True
        _enter_state(VoiceState.THINKING)

    def on_wake(self, t_wake, wake_pcm=None):
        """Wake fired (KWS frame at t_wake, or ASR fallback): lock onto the
        speaker's direction, then enter LISTENING (A10). A wake while already
        locked re-locks to the new speaker and restarts the voiceprint
        enrolment from this wake's audio."""
        now_m = time.monotonic()
        if wake_pcm is None:
            wake_pcm = b"".join(f for _, f in self.wake_audio)
        try:
            self._voice.set_wake(wake_pcm)  # clears the previous speaker's enrolment
        except Exception as e:  # noqa: BLE001 - a voiceprint failure must never stop the loop
            print(f"[Voice] set_wake failed ({e}); direction-only")
        degs = _cap._mic_speech_degs(t_wake - 1.5, t_wake)
        ang = self._lock.lock(degs, now_m)
        if ang is not None:
            print(f"[Lock] speaker at {ang:.0f} deg (mic), "
                  f"{_ul._body(ang):.0f} deg (body)")
            if _tuning.beam_steering and _cap._mic_usable():
                _cap.CAPTURE._mic.set_fixed_beams(math.radians(ang))
        else:
            print("[Lock] no direction readings — accepting all directions")
        self._status["locked_deg"] = _ul._body(ang)
        if ang is not None:
            _gaze_at_speaker(_ul._body(ang))  # gaze stays a direct call (C5; see features/gaze)
            _ev.publish("speaker_locked", body_deg=_ul._body(ang), gen=_safety_gen_now())
        self.state = VoiceState.LISTENING
        self.listening_since = now_m
        _enter_state(VoiceState.LISTENING)

    def release(self, cue: bool):
        """Drop the speaker lock + fixed beams and go back to WAKE_WORD."""
        self.ep = _cap._new_endpointer()  # drop any half-recorded utterance
        self.in_utt = False
        self.utt = []
        self._lock.release()
        self._voice.clear()
        self._status["locked_deg"] = None
        _gaze_at_speaker(None)  # back to centre
        _ev.publish("speaker_released")
        if _cap._mic_usable():
            _cap.CAPTURE._mic.set_fixed_beams(None)
        self.state = VoiceState.WAKE_WORD
        if self.wake_detector:
            self.wake_detector.reset()
        _enter_state(VoiceState.WAKE_WORD)
        if cue:
            try:
                beep.released_cue()
            except Exception as e:
                print(f"[Indicator] released cue failed: {type(e).__name__}: {e}")

    def finish_utterance(self, endpoint):
        """Endpoint reached: SNR gate, lock check, then ASR/brain."""
        floor_db = self.floor.floor_db
        t0 = self.utt[0][0]
        t1 = _gate.last_speech_t(self.utt)
        degs = _cap._mic_speech_degs(t0, t1)
        record = dict(
            endpoint=endpoint, end_silence_s=self.end_sil_s,
            duration_s=len(self.utt) * _cap.FRAME_S, floor_db=floor_db, stats=self.stats,
            lock_mic_deg=self._lock.angle, utt_mic_deg=circular_median_deg(degs))

        min_db = _tuning.min_speech_dbfs + (NAV_SPEECH_DB_BOOST if _is_robot_moving() else 0.0)
        # Short utterances (one-word "sit"/"stop") aren't judged by loudness:
        # a gate-failing short one still goes to ASR, but only a safety
        # command / direct command / mid-conversation answer gets through
        # (short_filter in process_utterance). Long ones keep the gate.
        short = _gate.is_short_utterance(self.stats, _tuning.short_utt_s)
        short_filter = False
        if not passes_gate(self.stats, floor_db, min_db, _tuning.min_snr_db):
            if not short:
                print(f"[Gate] dropped: speech {self.stats.speech_db_median:.1f} dBFS, "
                      f"SNR(peak) {self.stats.snr_db(floor_db):.1f} dB (need >= {min_db:.0f} dBFS, "
                      f">= {_tuning.min_snr_db:.1f} dB)")
                self.utt_log.write(_ul.build_utterance_record(**record, accepted=False,
                                                              drop_reason="gate"))
                return
            short_filter = True
            print(f"[Gate] short utterance ({self.stats.duration_s:.2f} s speech, "
                  f"SNR(peak) {self.stats.snr_db(floor_db):.1f} dB) -> ASR, commands only")
        record["short_path"] = short_filter
        # Direction verdict: None when unlocked / no readings (signal unavailable).
        prev_angle = self._lock.angle
        dir_ok = (self._lock.accept(degs[-LOCK_ACCEPT_MAX_SAMPLES:], time.monotonic(), touch=False)
                  if (self._lock.locked and degs) else None)
        utt_frames = _gate.trim_trailing(self.utt, TRAILING_KEEP_FRAMES)
        accepted, sim = dir_ok is not False, None
        if self.state == VoiceState.LISTENING:
            # Voiceprint on the utterance path (main loop; never the audio
            # callback or safety thread). OFF: direction-only, sim only logged.
            try:
                accepted, sim = self._voice.decide(b"".join(f for _, f, _, _ in utt_frames),
                                                   dir_ok, self._voiceprint_mode,
                                                   self._voiceprint_threshold,
                                                   self._voiceprint_update_threshold)
            except Exception as e:  # noqa: BLE001 - fail open to direction-only
                print(f"[Voice] decide failed ({e}); direction-only")
            print(f"[Voice] sim={'n/a' if sim is None else f'{sim:.2f}'} "
                  f"thr={self._voiceprint_threshold:.2f} "
                  f"gate={'on' if self._voiceprint_mode else 'off'} "
                  f"enrolled={f'yes(n={self._voice.n_enrol})' if self._voice.enrolled else 'no'}")
        record["voice_sim"] = None if sim is None else round(sim, 3)
        if not accepted:
            if dir_ok is False:
                print(f"[Lock] rejected utterance from {record['utt_mic_deg']:.0f} deg "
                      f"(locked {self._lock.angle:.0f} deg)")
                reason = "lock"
            else:
                self._lock.angle = prev_angle  # undo accept()'s nudge toward a rejected voice
                print(f"[Voice] rejected utterance: sim {sim:.2f} "
                      f"< {self._voiceprint_threshold:.2f}")
                reason = "voice"
            _ring.RING.signal_reject()
            if short:
                # Safety rescue: a short rejected utterance may still be a
                # quiet stop/freeze. ASR it; only a safety command executes.
                text, done = _sl.rescue_safety(self.stub, b"".join(f for _, f, _, _ in utt_frames))
                record.update(rescue_transcript=(text if self.store_transcript else "<not logged>")
                              if text else None, rescued=done)
                if done:
                    print("[Rescue] safety command from a rejected utterance")
            self.utt_log.write(_ul.build_utterance_record(**record, accepted=False,
                                                          drop_reason=reason))
            return
        _ring.RING.update()  # accept() may have nudged the lock angle
        self._status["locked_deg"] = _ul._body(self._lock.angle)
        if self._lock.locked:
            _gaze_at_speaker(_ul._body(self._lock.angle))

        print(f"\n>>> Processing ({endpoint})...")
        safety_only = (self.state == VoiceState.WAKE_WORD)
        if not safety_only:
            # Commit: mute the mic now (gate at STT start), but defer the
            # THINKING indicator until process_utterance confirms real words
            # bound for the LLM (via on_thinking).
            _cap.CAPTURE._mic_gated = True
            self.gate_announced = False
        info = {}

        def _announce_speaking():
            # Final-review I2: blue ring + speak tick as the reply starts,
            # not after every action in the chain has run.
            info["spoke"] = True
            _enter_state(VoiceState.RESPONDING)

        t_proc = time.monotonic()
        audio = [_cap.pcm16_to_float32(f) for _, f, _, _ in utt_frames]
        try:
            result = self.process_utterance(self.stub, None, audio, self.brain,
                                            safety_only=safety_only, tts=self.tts,
                                            has_wake_detector=bool(self.wake_detector),
                                            on_thinking=(None if safety_only else self.announce_thinking),
                                            info=info, utt_t0=t0,
                                            on_speaking=(None if safety_only else _announce_speaking),
                                            short_filter=short_filter,
                                            in_conversation=(self.state == VoiceState.LISTENING))
        except Exception as e:
            # A per-utterance failure (ASR/LLM/TTS) must drop the turn, not
            # crash voice control. result=None routes to the idle-based gate
            # release so the mic reopens.
            print(f"[Error] utterance processing failed: {type(e).__name__}: {e}")
            result = None
        latency = time.monotonic() - t_proc
        self.listening_since = _gate.refresh_conversation(self._lock, self.listening_since, time.monotonic(),
                                                          bool(info.get("reached_llm")))
        self._status["last_latency_s"] = round(latency, 2)
        if info.get("transcript"):
            self._status["last_transcript"] = info["transcript"] if self.store_transcript else "<not logged>"
        self.utt_log.write(_ul.build_utterance_record(
            **record, accepted=True, transcript=info.get("transcript"),
            reached_llm=bool(info.get("reached_llm")), latency_s=latency))

        if result != "wake_detected" and not safety_only and self.gate_announced:
            self.response_pending = True
            if not info.get("spoke"):
                _enter_state(VoiceState.RESPONDING)
            self.state = VoiceState.RESPONDING
        elif result == "wake_detected" and self.state == VoiceState.WAKE_WORD:
            print(">>> Now listening for commands...")
            self.on_wake(t1, b"".join(f for _, f, _, _ in utt_frames))
            _cap.CAPTURE.drain()
        elif self.use_wake_word:
            # safety-only / ignored utterance: keep prior behavior
            _cap.CAPTURE.drain()

    def on_queued(self, name, arg):
        if name == "listen":
            # Open the conversation with no direction lock (accept all).
            self._lock.release()
            _ev.publish("speaker_released")
            self._voice.clear()  # no stale enrolment; the first utterance enrols
            self._status["locked_deg"] = None
            if _cap._mic_usable():
                _cap.CAPTURE._mic.set_fixed_beams(None)
            self.state = VoiceState.LISTENING
            self.listening_since = time.monotonic()
            _enter_state(VoiceState.LISTENING)
        elif name in ("release", "reset"):
            self.release(cue=True)
            if name == "reset" and self.brain is not None:
                self.brain.history.clear()
        elif name == "say":
            if self.tts is not None and arg:
                self.tts.speak(arg)
        elif name == "persona":
            from src.spot.features import REGISTRY
            if not REGISTRY.has("settings"):
                print("[Control] persona: settings feature not enabled")
            elif self.brain is None:
                print("[Control] persona: no brain")
            else:
                from src.spot.features.settings.handlers import do_set_persona
                print(f"[Control] persona -> {do_set_persona({'name': arg}, self.brain)}")
        elif name == "crowd":
            self._crowd_mode = _ctl._toggle(self._crowd_mode, arg)
            print(f"[Control] crowd mode {'ON' if self._crowd_mode else 'OFF'}")
        elif name == "voiceprint":
            try:
                self._voiceprint_mode, self._voiceprint_threshold = _ctl.apply_voiceprint_arg(
                    arg, self._voiceprint_mode, self._voiceprint_threshold)
            except ValueError as e:
                print(f"[Control] voiceprint: {e}")
                return
            print(f"[Control] voiceprint gating {'ON' if self._voiceprint_mode else 'OFF'} "
                  f"threshold {self._voiceprint_threshold:.2f} "
                  f"(enrolled={'yes' if self._voice.enrolled else 'no'}, "
                  f"model={'ready' if self._voice.embedder is not None else 'not loaded'})")

    def run(self, *, args, stub, brain, tts, vad, wake_detector, use_wake_word,
            floor, utt_log, store_transcript, stream, shutdown, ctl_q=None, ctl_srv=None):
        """The voice main loop (was the body of client_mic.main() after
        startup). Returns when the loop exits (KeyboardInterrupt), after
        stopping the control server, the stream and the runtime."""
        self.stub, self.brain, self.tts = stub, brain, tts
        self.wake_detector, self.use_wake_word = wake_detector, use_wake_word
        self.floor, self.utt_log, self.store_transcript = floor, utt_log, store_transcript

        # ========================================================================
        # Main Loop State
        # ========================================================================
        window = b""
        window_t = time.monotonic()
        frame_count = 0
        last_vad_error_log_time = 0.0  # Token-bucket guard for VAD failure logs (one/min max)
        # Pre-roll: appended on EVERY frame in every state (A16). Each entry is
        # (t_monotonic, pcm16, dbfs, is_speech). Sized to cover the onset frames
        # plus Silero's min_speech_duration lag so first words aren't clipped.
        preroll_buffer = deque(maxlen=PREROLL_FRAMES + _tuning.onset_frames)
        # Last WAKE_AUDIO_S of frames: the wake phrase for the voiceprint enrolment.
        self.wake_audio = deque(maxlen=_cap._frames(WAKE_AUDIO_S))

        # Front-end (Task 10): VAD on every frame -> Endpointer; FloorTracker
        # seeded ONCE from calibration; SNR gate at utterance end.
        self.ep = _cap._new_endpointer()
        chime_mask = _cap.ChimeMask(_tuning.speaker_tail_s)
        self.in_utt = False
        self.utt = []                       # entries as in preroll_buffer
        self.stats = None
        self.end_sil_s = _tuning.end_silence_s_short
        last_idle_log = 0.0

        # Stage 2F P0a: barge-in reset bookkeeping. `player_was_busy` tracks the
        # AudioPlayer busy edge across iterations; `response_pending` is armed only
        # after a processed utterance (TTS response), never after the wake beep,
        # so the in-breath command audio kept on a wake fire is not drained.
        player_was_busy = False
        self.response_pending = False

        # Stage 2F P6: True once the current turn has announced THINKING — i.e. a
        # real transcript reached the brain (past STT + the safety/wake early-returns).
        # Gates the RESPONDING indicator so a noise/empty-STT turn stays silent.
        self.gate_announced = False

        # Wake word state
        self.state = VoiceState.WAKE_WORD if use_wake_word else VoiceState.LISTENING
        # Conversation clock (monotonic): fallback timeout when no direction lock.
        self.listening_since = time.monotonic()
        _enter_state(self.state)

        # Operator control server (spotctl). Immediate commands run on its HTTP
        # threads via _on_immediate; queued ones are drained here between frames.
        from src.spot.operator.control_server import drain as _drain_control
        if ctl_q is None:  # main starts it (step 8); standalone callers get one here
            ctl_q = queue.Queue()
            ctl_srv = _ctl._start_control_server(ctl_q)

        try:
            while True:
                _drain_control(ctl_q, self.on_queued)
                _cap._check_mic_alive()
                _ring.RING.flush()
                if self._operator_muted and self.in_utt:
                    # Muted mid-utterance: drop the half-recorded utterance and
                    # every pre-mute frame still queued.
                    self.ep = _cap._new_endpointer()
                    self.in_utt = False
                    self.utt = []
                    preroll_buffer.clear()
                    _cap.CAPTURE.drain(keep_tail=False)
                # Conversation timeout (A11) on every loop tick, so it still fires
                # while muted / no frames arrive. lock.expired replaces the old
                # fixed LISTENING_TIMEOUT; wall-clock fallback when unlocked.
                if (self.state == VoiceState.LISTENING and not self.in_utt and use_wake_word
                        and not _cap.CAPTURE._mic_gated and _gate.conversation_expired(
                            self._lock, self.listening_since, time.monotonic(),
                            _tuning.lock_timeout_s)):
                    print("[Conversation timeout — say 'Hey Spot' to activate]")
                    self.release(cue=True)

                # Stage 2F P0a: on the player busy->idle edge after a TTS response,
                # reset streaming-detector state so the speaker tail + stale VAD/KWS
                # decisions can't corrupt the next detection. Never touches `state`.
                busy = _cap._speech_busy()  # chimes don't hold the gate or count as a response
                if should_reset_after_player(self.response_pending, player_was_busy, busy):
                    vad.reset()
                    if wake_detector:
                        wake_detector.reset()
                    preroll_buffer.clear()
                    self.ep = _cap._new_endpointer()
                    self.in_utt = False
                    self.utt = []
                    _cap.CAPTURE.drain(keep_tail=False)  # full drain: no late phantom commands
                    self.response_pending = False
                    print("[Mic] detector reset after TTS response")
                player_was_busy = busy

                # Stage 2F P6 gate release (IDLE-based, not TTS-edge-based, so a turn
                # that plays no audio — e.g. a noise burst with an empty ASR result —
                # can't wedge the gate shut). Once gated and the player is idle, start
                # the settle; reopen when it elapses.
                if _cap.CAPTURE._mic_gated:
                    now_m = time.monotonic()
                    if busy:
                        _cap.CAPTURE._mic_reopen_at = None  # still speaking — hold the gate
                    elif should_arm_reopen(_cap.CAPTURE._mic_gated, _cap.CAPTURE._mic_reopen_at, busy):
                        _cap.CAPTURE._mic_reopen_at = now_m + _tuning.reopen_delay_s
                    elif should_reopen_mic(_cap.CAPTURE._mic_gated, _cap.CAPTURE._mic_reopen_at, now_m):
                        # Reset detectors + drop the settle-window audio BEFORE the
                        # gate lifts; pre-roll restarts on the very next frame so
                        # a quick reply's first word is kept.
                        vad.reset()
                        if wake_detector:
                            wake_detector.reset()
                        _cap.CAPTURE.drain(keep_tail=False)
                        preroll_buffer.clear()
                        self.ep = _cap._new_endpointer()
                        self.in_utt = False
                        self.utt = []
                        _cap.CAPTURE._mic_reopen_at = None
                        self.response_pending = False
                        turn_reached = self.gate_announced  # True iff the turn reached the LLM
                        self.gate_announced = False
                        _cap.CAPTURE._mic_gated = False
                        if self.state == VoiceState.WAKE_WORD:
                            # Operator released/reset during the reply: stay idle.
                            print(">>> Released during reply — say the wake word")
                        elif self._crowd_mode and use_wake_word:
                            # Crowd mode: every turn needs a fresh wake (A11/10).
                            self.release(cue=False)
                            print(">>> Crowd mode — say the wake word for the next turn")
                        else:
                            self.state = VoiceState.LISTENING
                            # Clock restarts at reopen only after a real turn (M1).
                            self.listening_since = _gate.refresh_conversation(
                                self._lock, self.listening_since, now_m, turn_reached)
                            # Always cue: the listen chime fires as the mic opens
                            # (beeps no longer mute the mic, so nothing is lost).
                            _enter_state(VoiceState.LISTENING)
                            print(">>> Listening for follow-up (no wake needed)...")
                # Get audio from queue. Short timeout so we still cycle (and stay
                # responsive to SIGTERM / KeyboardInterrupt / spotctl) even when the
                # audio callback has stopped putting frames in the queue.
                try:
                    window_t, pcm = _cap.CAPTURE.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                window += pcm

                # Process complete frames
                while len(window) >= _cap.BYTES_PER_FRAME:
                    frame = window[:_cap.BYTES_PER_FRAME]
                    window = window[_cap.BYTES_PER_FRAME:]
                    t_frame = window_t  # callback blocksize == 1 frame, so exact
                    frame_count += 1

                    # ---- per-frame front end: dBFS + VAD on EVERY frame ----
                    db = frame_dbfs(frame)
                    try:
                        vad.accept_waveform(_cap.pcm16_to_float32(frame))
                        is_speech = vad.is_speech_detected()
                        while not vad.empty():  # A15: don't let segments pile up
                            vad.pop()
                    except Exception as e:
                        if time.time() - last_vad_error_log_time > 60.0:
                            print(f"[VAD] accept_waveform failed: {type(e).__name__}: {e}")
                            last_vad_error_log_time = time.time()
                        is_speech = False
                    # A chime (beep, not speech) in flight + speaker tail: never
                    # let it start/extend an utterance (chime-only turn loop).
                    # Frames still go to pre-roll so speech right after is kept.
                    masked, chime_ended = chime_mask.update(_cap._chime_busy(), t_frame, is_speech)
                    if masked:
                        is_speech = False
                    elif chime_mask.should_reset_vad(chime_ended, self.in_utt):
                        vad.reset()  # clear Silero's hangover from the chime
                    else:
                        floor.update(db, is_speech)
                    self._status["floor_dbfs"] = round(floor.floor_db, 1)
                    preroll_buffer.append((t_frame, frame, db, is_speech))
                    self.wake_audio.append((t_frame, frame))

                    # Safety KWS runs on its own thread (safety_kws_loop) in every
                    # state; nothing to do here.

                    # Wake word (dedicated detector). Runs in LISTENING too, so a
                    # new speaker's wake re-locks onto them (Task 10 change 9).
                    if wake_detector and self.state in (VoiceState.WAKE_WORD, VoiceState.LISTENING):
                        if wake_detector.process_frame(frame):
                            print(">>> Wake word detected!")
                            was_idle = self.state == VoiceState.WAKE_WORD
                            self.on_wake(t_frame)
                            if was_idle:
                                # Fresh endpointer; the in-breath command ("Hey Spot
                                # stand up") is still arriving and pre-roll has the tail.
                                self.ep = _cap._new_endpointer()
                                self.in_utt = False
                                self.utt = []

                    # In WAKE_WORD mode with a dedicated detector, don't record —
                    # the keyword spotter handles wake at frame level (no ASR on
                    # crowd noise).
                    if self.state == VoiceState.WAKE_WORD and wake_detector:
                        continue
                    if self.state not in (VoiceState.WAKE_WORD, VoiceState.LISTENING):
                        continue

                    ev = self.ep.feed(is_speech, db, floor.floor_db)
                    if ev == "start":
                        self.in_utt = True
                        self.utt = list(preroll_buffer)  # includes this frame
                        self.stats = UtteranceStats(frame_s=_cap.FRAME_S,
                                                    window_frames=_cap._frames(_tuning.gate_window_ms / 1000.0))
                        for _, _, d, s in self.utt:
                            self.stats.add(d, s)
                        self.end_sil_s = _tuning.end_silence_s_short
                        self.ep.set_end_silence_frames(_cap._frames(self.end_sil_s))
                        if self.state == VoiceState.WAKE_WORD:
                            print("\n>>> Speech detected (wake word mode — safety only)...")
                        else:
                            print("\n>>> Speech detected...")
                    elif self.in_utt:
                        self.utt.append((t_frame, frame, db, is_speech))
                        self.stats.add(db, is_speech)
                        dur = len(self.utt) * _cap.FRAME_S
                        if dur >= SILENCE_CROSSOVER and self.end_sil_s != _tuning.end_silence_s:
                            # Longer utterances (chains) get the longer end silence.
                            self.end_sil_s = _tuning.end_silence_s
                            self.ep.set_end_silence_frames(_cap._frames(self.end_sil_s))
                        if len(self.utt) % 17 == 0:  # ~every 0.5s of audio
                            print(f"    Recording: {dur:.1f}s")

                    if ev in ("end", "cap") and self.in_utt:
                        self.in_utt = False
                        if ev == "cap":
                            print(f"\n>>> Max duration reached ({len(self.utt) * _cap.FRAME_S:.1f}s)")
                        self.finish_utterance(ev)
                        self.utt = []
                        self.stats = None

                    # Idle metrics line (1 Hz) + periodic console status
                    now_m = time.monotonic()
                    if not self.in_utt and now_m - last_idle_log >= 1.0:
                        last_idle_log = now_m
                        utt_log.write(_ul.build_idle_record(self.state.name, floor.floor_db, db,
                                                            self._lock.angle))
                    if not self.in_utt and frame_count % 166 == 0:  # ~5 seconds
                        if args.debug_audio:
                            print(f"[{self.state.name} dbfs={db:.1f} floor={floor.floor_db:.1f}]")
                        elif self.state == VoiceState.WAKE_WORD:
                            print("[Say 'Hey Spot' to activate...]")
                        else:
                            print("[Listening...]")

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            if ctl_srv is not None:
                try:
                    ctl_srv.stop()
                except Exception:
                    pass
            try:
                stream.stop()
                time.sleep(0.1)  # let PortAudio drain buffers before close
                stream.close()
            except Exception:
                pass
            shutdown()

    def process_utterance(self, stub, speech_buffer: bytearray, speech_float_buffer: list,
                          brain=None, safety_only=False, tts=None,
                          has_wake_detector=False, on_thinking=None, info=None, utt_t0=None,
                          on_speaking=None, short_filter=False, in_conversation=False):
        """Process recorded speech through ASR then LLM brain (or regex fallback).

        short_filter: a short utterance that failed the loudness gate; after the
                      safety check only a direct command (or a one-word answer when
                      in_conversation) continues, anything else is dropped (no LLM).

        Args:
            safety_only: If True, only execute safety commands (stop/freeze/estop).
                         Non-safety speech is discarded. Used when wake word not detected.
            tts: SpotTTS instance for spoken responses (None = no speech output).
            has_wake_detector: If True, dedicated wake word detector is active —
                              skip ASR-based wake phrase detection (detector handles it).
            info: optional dict filled with "transcript" and "reached_llm" for
                  the utterance metrics line.
            utt_t0: monotonic time of the utterance's first frame; motion is not
                  dispatched if a safety halt happened at/after it.
            on_speaking: called once, just before the first reply line is queued.

        Returns:
            "wake_detected" if a wake phrase was found (ASR fallback only), None otherwise.

        Flow:
            Audio → Whisper ASR → transcript
            transcript → safety check (instant regex for stop/estop/freeze)
            transcript → wake phrase check (ASR fallback, only if no dedicated detector)
            transcript → LLM brain (state + history → action + response)
            OR (legacy) → regex parser → intent → dispatch
        """
        if not speech_float_buffer:
            return None

        # New turn: forget earlier operator stops. A stop arriving from here on
        # (during ASR, LLM or the chain itself) aborts this turn and hushes it.
        self._abort_chain.clear()
        # A halt that landed while the user was still speaking (KWS false fire on
        # "next stop", a bystander, ...) already stopped the robot. It blocks this
        # turn's motion only; the spoken reply still plays (final-review I1).
        motion_blocked = utt_t0 is not None and _sl.LISTENER._last_safety_at >= utt_t0
        turn_gen = _safety_gen_now()  # any stop after this point cancels a talk-pose resume
        t_utterance_start = time.time()

        # Combine audio
        audio_float = np.concatenate(speech_float_buffer)
        duration = len(audio_float) / _cap.SAMPLE_RATE

        # Reject very short clips (noise bursts, not speech)
        if duration < MIN_SPEECH_DURATION:
            print(f"[Too short ({duration:.2f}s) — skipping]")
            return None

        # Convert to PCM16 and send to ASR
        pcm_bytes = _cap.float32_to_pcm16(audio_float)
        print(f"[Sending {duration:.1f}s to ASR...]")

        t_asr_start = time.time()
        transcript = _stt.send_to_asr(stub, pcm_bytes)
        t_asr_end = time.time()
        print(f"[Timing] ASR: {t_asr_end - t_asr_start:.2f}s")

        if info is None:
            info = {}
        if not transcript:
            print("[No speech recognized]")
            return None
        info["transcript"] = transcript

        print("\n" + "=" * 60)
        print(f"HEARD: {_heard(transcript)}")
        print("=" * 60)

        # Clean transcript (remove Whisper punctuation artifacts)
        clean = re.sub(r'\.\s*', ' ', transcript).strip()
        clean = re.sub(r'\s+', ' ', clean)
        if clean != transcript:
            print(f"CLEAN: {_heard(clean)}")

        # Reject empty or punctuation-only transcripts (Whisper hallucination on noise)
        if not clean or len(clean) < 2:
            print("[Empty transcript — likely noise, skipping]")
            return None

        # ------------------------------------------------------------------
        # 1. Safety fast-path: stop/freeze/estop bypass LLM (zero latency)
        #    Safety commands ALWAYS execute, regardless of wake word state.
        # ------------------------------------------------------------------
        if _sl._run_safety(transcript) or _sl._run_safety(clean):
            return None

        # ------------------------------------------------------------------
        # 1b. Wake phrase detection (ASR fallback — only when no dedicated detector)
        # ------------------------------------------------------------------
        wake_activated = False
        if safety_only:
            if has_wake_detector:
                # Dedicated detector handles wake word at frame level.
                # If we're here with safety_only=True, it means VAD triggered
                # but it wasn't a safety command — just discard.
                print(f"[Ignored {_heard(clean)} — waiting for 'Hey Spot' (detector)]")
                return None
            # ASR-based fallback: check transcript for a leading wake phrase
            remainder = strip_wake_phrase(clean, WAKE_PHRASES)
            if remainder != clean:
                remainder = remainder.strip()
                print(f"[Wake] Detected in: {_heard(clean)}")
                if remainder and len(remainder) >= 2:
                    # Command follows wake phrase — process it now ("Hey Spot stand")
                    print(f"[Wake] Command: {_heard(remainder)}")
                    clean = remainder
                    if _sl._run_safety(clean):  # "Hey Spot, stop ..." — re-check after strip
                        return None
                    safety_only = False  # Allow full processing below
                    wake_activated = True
                else:
                    # Just the wake phrase, no command — wait for next utterance
                    print("[Wake] Activated — waiting for command...")
                    return "wake_detected"
            else:
                # No wake phrase, no safety command — discard
                print(f"[Ignored {_heard(clean)} — say 'Hey Spot' first]")
                return None

        # ------------------------------------------------------------------
        # 1c. Strip wake phrase from transcript (dedicated detector mode)
        #     Handles "Hey Spot stand up" → "stand up" when said in one breath
        # ------------------------------------------------------------------
        if has_wake_detector and not safety_only:
            remainder = strip_wake_phrase(clean, WAKE_PHRASES)
            if remainder != clean:
                remainder = remainder.strip()
                if remainder and len(remainder) >= 2:
                    print(f"[Wake strip] {_heard(clean)} → {_heard(remainder)}")
                    clean = remainder
                    if _sl._run_safety(clean):  # re-check after strip, before LLM/regex
                        return None
                else:
                    # Just the wake phrase with no command — ignore
                    print(f"[Wake phrase only — no command after stripping]")
                    return None

        # Short, quiet utterance (failed the loudness gate): commands only.
        if short_filter and (safety_only or not short_utterance_ok(clean, in_conversation)):
            print(f"[Short] dropped {_heard(clean)} — not a command")
            return None

        # Stage 2F P6: the utterance survived STT (real words, not a safety command,
        # not a bare wake phrase) and is about to be dispatched to the brain — signal
        # THINKING now (not before STT) so noise/empty transcripts never chime.
        # B11: "uh um hmm" — nothing worth a turn.
        if is_filler_only(clean, mid_conversation=not safety_only and not wake_activated):
            print(f"[Filler only {_heard(clean)} — skipping]")
            return "wake_detected" if wake_activated else None

        info["reached_llm"] = True
        if on_thinking is not None:
            on_thinking()
        if not motion_blocked and not self._abort_chain.is_set():
            _resume_talk_after_halt(turn_gen)
            _auto_arm_at_turn(turn_gen)

        self._speaking_hook = on_speaking
        rec = get_recorder()
        trace = rec.begin_utterance() if rec else None
        try:
            # ------------------------------------------------------------------
            # 2. LLM Brain mode (primary)
            # ------------------------------------------------------------------
            if brain is not None and _brain_ready(brain):
                # Collect current robot state for context
                state = get_spot_state()

                # Stage 2E.1: resolve the persona-mapped voice ID for this turn.
                # B6: if the reply switches persona, the streamed speech is held
                # back (see _TokenWatch), the switch runs first, and the reply is
                # spoken in the NEW voice.
                voice_id = _resolve_voice(brain)

                # Streaming TTS: chunker emits response sentences as they arrive.
                # Suppresses itself if a describe action appears in the action list
                # (GBNF emits actions before response), so the LLM's hallucinated
                # camera description never reaches the speaker — client_mic.py
                # speaks a stock ack instead (see describe_action branch below).
                chunker = tts.enqueue_streaming(voice=voice_id) if tts and tts.is_available() else None
                filler = None
                if chunker is not None:
                    # B12: one neutral line if nothing has been said after
                    # Tuning.thinking_filler_s; cancelled when the reply starts.
                    filler = _filler.ThinkingFiller(
                        lambda: _say(tts, _filler.THINKING_FILLER_LINE, voice_id, cue=False),
                        _tuning.thinking_filler_s, _cap._speech_busy)
                    brain.on_token_callback = _filler._TokenWatch(chunker, filler)
                    filler.start()

                t_llm_start = time.time()
                try:
                    result = brain.process(clean, state)
                except Exception as e:
                    # A failed LLM call (server crash/restart → ConnectionError,
                    # slot contention → ReadTimeout, 5xx → HTTPError) must NOT kill
                    # the voice process. Bail this utterance and keep listening.
                    import traceback
                    print(f"[Brain] process() failed: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    try:
                        beep.error()
                    except Exception:
                        pass
                    if tts is not None:
                        try:
                            _say(tts, "Sorry, I lost my train of thought. Could you say that again?")
                        except Exception:
                            pass
                    return None
                finally:
                    if filler is not None:
                        filler.cancel()
                    if chunker is not None:
                        if self._abort_chain.is_set():
                            chunker.suppressed = True  # hushed: don't speak the tail
                        chunker.flush()
                        brain.on_token_callback = None
                t_llm_end = time.time()
                print(f"[Timing] LLM: {t_llm_end - t_llm_start:.2f}s")

                response = result.get("response", "")
                actions = result.get("actions", [])

                # Backward compat: old single "action" field
                if not actions and result.get("action"):
                    actions = [result["action"]]

                # If a describe action is queued, the LLM's "response" field is
                # often a hallucinated answer (the model can't actually see the
                # camera). Replace it with a stock acknowledgment so we don't
                # speak a fake description before the VLM provides the real one.
                # Note: this only suppresses what gets spoken — the raw LLM JSON
                # is still stored in brain.history, so the model can still see
                # its own hallucination on the next turn (acceptable footnote).
                describe_action = next(
                    (a for a in actions if a.get("intent") == "describe"),
                    None,
                )
                if describe_action:
                    query = str(describe_action.get("params", {}).get("query", "")).replace("_", " ").strip()
                    response = (
                        f"Let me take a look for the {query}..."
                        if query
                        else random.choice(_DESCRIBE_ACKS)
                    )

                # B6: run persona switches BEFORE speaking so the announcing reply
                # uses the new voice (the chunker held it back — _TokenWatch).
                persona_names = intent_names("persona")
                persona_actions = [a for a in actions if a.get("intent") in persona_names]
                if persona_actions:
                    for a in persona_actions:
                        execute_on_spot(a, brain=brain)
                    actions = [a for a in actions if a.get("intent") not in persona_names]
                    voice_id = _resolve_voice(brain)
                    if tts and response:
                        _say(tts, response, voice_id)
                        response = ""  # spoken; don't repeat below

                # Show what the robot "says"
                if response:
                    print(f"\nSPOT (LLM): \"{response}\"")

                # Track which models actually handled this utterance for the
                # end-of-turn [Models] summary line. vlm_used flips inside the
                # describe-action branch below.
                vlm_used = False
                vlm_error = False

                if actions:
                    beep.command_ok()

                    # Streaming chunker already spoke response sentence-by-sentence.
                    # Only fall back to tts.speak when chunker suppressed (describe
                    # path: `response` is now a stock ack) or chunker absent.
                    chunker_played = chunker is not None and not chunker.suppressed
                    if tts and response and not chunker_played:
                        _say(tts, response, voice_id)

                    if trace:
                        trace.mark("intent_dispatch")
                    for i, intent in enumerate(actions):
                        if self._abort_chain.is_set():
                            print("[Chain] aborted by operator stop/hush")
                            break
                        cmd = intent["intent"]
                        params = intent.get("params", {})
                        if motion_blocked and cmd not in intent_names("motion_free"):
                            print(f"[Dispatch] skipped {cmd}: safety halt during this utterance")
                            continue

                        if len(actions) > 1:
                            print(f"[Chain] Executing {i+1}/{len(actions)}: {cmd}")
                            if i > 0:
                                beep.chain_next()

                        # Special VLM flow for "describe" action
                        if cmd == "describe":
                            vlm_used = True
                            camera = params.get("camera", "front")
                            query = params.get("query", "")
                            try:
                                from src.spot.robot.state import capture_frame
                                image_bytes = capture_frame(camera)
                                if image_bytes:
                                    # If asking about a specific object, run YOLO first
                                    yolo_hint = ""
                                    if query:
                                        try:
                                            from src.spot.features.visual_nav.navigator import detect_in_image
                                            det = detect_in_image(image_bytes, query)
                                            if det["found"]:
                                                yolo_hint = (
                                                    f"IMPORTANT: Object detection confirms a '{query}' "
                                                    f"IS visible in this image ({det['position']}, "
                                                    f"confidence {det['confidence']:.0%}). "
                                                    f"Describe it and its surroundings."
                                                )
                                                print(f"[YOLO] Found '{query}' — {det['position']}, "
                                                      f"conf={det['confidence']:.2f}")
                                            else:
                                                yolo_hint = (
                                                    f"NOTE: Object detection did NOT find '{query}' "
                                                    f"in this image. If you also don't see it, say so."
                                                )
                                                print(f"[YOLO] '{query}' not detected")
                                        except ImportError:
                                            pass
                                    with (trace.span("vlm") if trace else nullcontext()):
                                        vlm_response = brain.query_vlm(image_bytes, clean, yolo_hint=yolo_hint)
                                    print(f"\nSPOT (VLM): \"{vlm_response}\"")
                                    if tts:
                                        tts.wait()
                                        _say(tts, vlm_response, voice_id)
                                else:
                                    beep.error()
                                    vlm_error = True
                                    fallback = "Sorry, I couldn't capture an image right now."
                                    print(f"\nSPOT (VLM, error): \"{fallback}\"")
                                    if tts:
                                        tts.wait()
                                        _say(tts, fallback, voice_id)
                            except Exception as e:
                                beep.error()
                                vlm_error = True
                                print(f"[VLM] Error: {e}")
                                fallback = "Sorry, my vision system isn't working right now."
                                print(f"\nSPOT (VLM, error): \"{fallback}\"")
                                if tts:
                                    tts.wait()
                                    _say(tts, fallback, voice_id)
                        else:
                            if execute_on_spot(intent, brain=brain,
                                               speak=((lambda t: _say(tts, t, voice_id)) if tts else None)):
                                print(f">>> SUCCESS" if len(actions) == 1 else f">>> {cmd} SUCCESS")
                            else:
                                beep.error()
                                print(f">>> FAILED" if len(actions) == 1 else f">>> {cmd} FAILED — stopping chain")
                                break

                            # Wait for movement/posture to complete before next action in chain
                            if len(actions) > 1 and i < len(actions) - 1:
                                wait_kind = chain_wait_kind(cmd)
                                if wait_kind == "nav":
                                    _wait_for_nav_complete()
                                elif wait_kind == "movement":
                                    self._abort_chain.wait(CHAIN_MOVEMENT_WAIT_S)
                                elif wait_kind == "posture":
                                    self._abort_chain.wait(CHAIN_POSTURE_WAIT_S)
                    if trace:
                        trace.mark("dispatch_complete")
                else:
                    # Conversation only — chunker already streamed sentences; speak
                    # only if no chunker (legacy) or chunker suppressed.
                    chunker_played = chunker is not None and not chunker.suppressed
                    if tts and response and not chunker_played:
                        _say(tts, response, voice_id)
                    print("(No physical action — conversation only)")

                # One-line summary of which model(s) actually handled this utterance.
                # Makes it obvious when a "describe surroundings" prompt got
                # hallucinated by the LLM instead of routed to the VLM.
                if vlm_used and not vlm_error:
                    models_tag = "LLM + VLM"
                elif vlm_used and vlm_error:
                    models_tag = "LLM + VLM (failed → fell back)"
                elif actions:
                    models_tag = f"LLM only (action: {actions[0]['intent']})"
                else:
                    models_tag = "LLM only (conversation)"
                print(f"[Models] {models_tag}")

                print(f"[Timing] Total: {time.time() - t_utterance_start:.2f}s")
                return "wake_detected" if wake_activated else None

            # ------------------------------------------------------------------
            # 3. Legacy regex-only fallback (--no-brain mode)
            # ------------------------------------------------------------------
            intent = parse_intent(clean)

            if intent:
                cmd = intent['intent']
                params = intent.get('params', {})
                print(f"Command: {cmd}" + (f" {params}" if params else ""))

                if motion_blocked or self._abort_chain.is_set():
                    print("[Dispatch] skipped — safety halt during this utterance")
                elif execute_on_spot(intent):
                    print(">>> SUCCESS")
                else:
                    print(">>> FAILED")
            else:
                print("(Not recognized — try: stand, sit, stop, turn left/right, go to [location])")

            print(f"[Timing] Total: {time.time() - t_utterance_start:.2f}s")
            return "wake_detected" if wake_activated else None
        finally:
            self._speaking_hook = None
            if rec is not None and trace is not None:
                if tts is not None:
                    try:
                        tts.wait()  # drain TTS worker so render/play marks land before serialize
                    except Exception:
                        pass
                rec.complete(trace)


CONVERSATION = Conversation()
process_utterance = CONVERSATION.process_utterance
