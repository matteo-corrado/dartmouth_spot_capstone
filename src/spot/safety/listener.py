"""Always-on spoken stop listener (stop/freeze/estop) + transcript safety path.

Moved from src/voice_control/client_mic.py (Task 10), logic unchanged.

The audio callback feeds the listener's queue with every frame before any
mute/gate check; a daemon thread is the ONLY owner of the safety detector
(the sherpa stream is never shared). A hit runs SafetyListener.on_keyword on
that thread: mark (abort any chain + timestamp), start the hush on its own
daemon thread, then the halt — a direct call, never through the event bus.

The listener state that used to be client_mic module globals lives on the
SafetyListener owner, LISTENER, created once at import and reached through
this module at call time.
"""
import os
import queue
import re
import threading
import time

import src.spot.core.events as _ev
import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
import src.spot.stt.client as _stt
from src.spot.audio.chimes import beep
# Safety commands (stop/freeze/estop) bypass the LLM. The rule + patterns live
# in intent.py (single source); re-exported by client_mic because tests and
# the barge-in eval import check_safety_command from there.
from src.spot.brain.intent import check_safety_command

SAFETY_QUEUE_MAX = 66            # ~2 s of 30 ms frames
SAFETY_REFRACTORY_S = 1.5        # ignore audio this long after a hit
SAFETY_ERROR_WARN_N = 50         # consecutive detector errors before "unhealthy"
_SAFETY_LABELS = {"stop": "stop", "halt": "stop", "freeze": "freeze",
                  "estop": "estop", "e stop": "estop", "e-stop": "estop",
                  "emergency stop": "estop"}
SAFETY_DROP_LOG_S = 5.0
RESCUE_ASR_TIMEOUT_S = 3.0  # rescue ASR deadline; a dead ASR server can't stall the loop


def safety_kws_loop(detector, q, stop_event, on_hit,
                    refractory_s=SAFETY_REFRACTORY_S, clock=time.monotonic, health=None):
    """Feed frames from `q` to `detector`; call on_hit(kw) per detection.
    Frames during the refractory window after a hit are dropped. Returns
    within ~0.1 s of stop_event being set. `health["ok"]` goes False after
    SAFETY_ERROR_WARN_N consecutive detector errors or if the loop dies."""
    health = health if health is not None else {}
    health["ok"] = True
    errors = 0
    last_hit = None
    try:
        while not stop_event.is_set():
            try:
                pcm = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if last_hit is not None and clock() - last_hit < refractory_s:
                continue
            try:
                kw = detector.process_frame(pcm)
                if errors >= SAFETY_ERROR_WARN_N:
                    print("[SafetyKWS] detector recovered")
                errors = 0
                health["ok"] = True
            except Exception as e:
                errors += 1
                if errors == SAFETY_ERROR_WARN_N:
                    health["ok"] = False
                    print(f"[SafetyKWS] !!! UNHEALTHY: {errors} consecutive detector errors "
                          f"({e}) — spoken stop may not work; use spotctl stop / e-stop")
                try:
                    detector.reset()  # fresh stream: recover from a wedged decoder
                except Exception:
                    pass
                kw = ""
            if not kw:
                continue
            last_hit = clock()
            try:
                on_hit(kw)
            except Exception as e:
                print(f"[SafetyKWS] handler error: {type(e).__name__}: {e}")
            try:
                detector.reset()
            except Exception:
                pass
    finally:
        health["ok"] = False


def _safety_intent_for(kw: str):
    """Intent for a detector label: intent.py's rule, then the fixed label map."""
    intent = check_safety_command(kw)
    if intent:
        return intent
    name = _SAFETY_LABELS.get(kw.strip().lower())
    return {"intent": name, "params": {}, "raw": kw} if name else None


class SafetyListener:
    """Owner of the safety-listener state (was client_mic module globals).

    halt(intent) -> bool sends the halt (direct call, lock-free);
    hush() cuts current speech and always runs on its own daemon thread.
    The bus is never on the stop path: safety_halt is published only after
    the halt call has returned (a notice; publish never raises).
    """

    def __init__(self, halt, hush):
        self._halt = halt
        self._hush = hush
        self._q = queue.Queue(maxsize=SAFETY_QUEUE_MAX)
        self._stop = threading.Event()
        self._active = False           # True once the safety thread is running
        self._thread = None
        self._health = {"ok": False}   # safety_kws_ok in spotctl status
        # Monotonic time of the last safety halt (KWS, transcript or operator). Motion
        # dispatch for an utterance that started before it is skipped (see
        # process_utterance) so _abort_chain.clear() can't erase a hit.
        self._last_safety_at = float("-inf")
        self._drops = {"total": 0, "pending": 0, "last_log": float("-inf")}

    def on_keyword(self, kw: str) -> None:
        """Safety-KWS hit (runs on the safety thread): abort any chain, cut speech,
        dispatch the halt. Lock-free; works while the main loop is blocked."""
        intent = _safety_intent_for(kw)
        if not intent:
            print(f"[SAFETY-KWS] unmapped label {kw!r} — ignored")
            return
        print(f"[SAFETY-KWS] '{kw}' — executing {intent['intent']} immediately")
        self.mark()
        # force_reset can take seconds (worker join + stream reopen), so the hush
        # runs on its own daemon thread; starting it first costs the halt nothing
        # and neither waits on the other.
        self.start_hush()
        self._halt(intent)
        _ev.publish("safety_halt", intent=intent["intent"])  # notice only, after the halt returned

    def start_hush(self) -> None:
        """Cut current speech on a daemon thread (never blocks or breaks the caller).

        Must never raise: callers send the halt right after this."""
        try:
            threading.Thread(target=self._hush, name="safety-hush", daemon=True).start()
        except Exception as e:  # noqa: BLE001 - a failed hush must not lose the halt
            print(f"[Safety] hush thread failed to start: {e}")

    def mark(self) -> None:
        """Record a safety halt: abort any chain + timestamp it (see process_utterance)."""
        self._last_safety_at = time.monotonic()
        _loop.CONVERSATION._abort_chain.set()

    def start(self, detector):
        """Start the safety-KWS daemon thread; returns it."""
        self._stop.clear()
        t = threading.Thread(target=safety_kws_loop,
                             args=(detector, self._q, self._stop, self.on_keyword),
                             kwargs=dict(health=self._health),
                             name="safety-kws", daemon=True)
        t.start()
        self._thread = t
        self._active = True
        return t

    def stop(self) -> None:
        """Signal the safety thread and wait briefly for it (shutdown)."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=0.5)

    def health(self) -> bool:
        """safety_kws_ok for spotctl status: running, healthy and alive."""
        return bool(self._active and self._health.get("ok")
                    and self._thread is not None and self._thread.is_alive())


def _start_hush() -> None:
    LISTENER.start_hush()


def _mark_safety() -> None:
    LISTENER.mark()


def _hush_player() -> None:
    _audio_player = _cap.CAPTURE._audio_player
    if _audio_player is not None:
        try:
            if not _audio_player.is_busy():
                return  # nothing playing (e.g. a hit while the user talks): no stream rebuild
            _audio_player.force_reset()  # hush: stop talking now
        except Exception as e:
            print(f"[SAFETY-KWS] hush failed: {e}")


def _dispatch_safety(intent) -> bool:
    """Execute a safety intent; dry-run says so plainly (no success beep)."""
    if os.environ.get("SPOT_DRY_RUN") == "1":
        print(f"[dry-run] SAFETY would dispatch: {intent['intent']}")
        return True
    if _loop.execute_on_spot(intent):
        beep.command_ok()
        print(">>> SAFETY COMMAND EXECUTED")
        return True
    beep.error()
    print(">>> SAFETY COMMAND FAILED")
    return False


def safety_startup_ok(have_detector: bool, dry_run: bool, allow_missing: bool) -> bool:
    """Fail closed: on the robot the safety KWS is required unless the
    operator explicitly passes --allow-no-safety-kws. Dry-run only warns."""
    if have_detector or dry_run:
        return True
    if allow_missing:
        print("[SafetyKWS] WARNING: running on the robot WITHOUT the always-on safety "
              "keyword spotter (--allow-no-safety-kws). Spoken stop needs a wake first.")
        return True
    print("ERROR: the always-on safety keyword spotter (stop/freeze/estop) failed "
          "to load, and the robot is reachable. Refusing to start.")
    print("  Fix: run scripts/setup_safety_kws.py, or pass --allow-no-safety-kws "
          "to run without it (operator's call), or --dry-run.")
    return False


def _start_safety_thread(detector):
    return LISTENER.start(detector)


def _stop_safety_thread() -> None:
    LISTENER.stop()


def _put_drop_oldest(q, item) -> bool:
    """Non-blocking put; on overflow discard the OLDEST item (safety audio
    must stay current). Returns True if a frame was dropped."""
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass
        return True


def _note_safety_drop(now: float) -> None:
    """Count a dropped safety frame; print at most once per SAFETY_DROP_LOG_S."""
    d = LISTENER._drops
    d["total"] += 1
    d["pending"] += 1
    if now - d["last_log"] >= SAFETY_DROP_LOG_S:
        print(f"[SafetyKWS] !!! queue overflow: dropped {d['pending']} frames (safety listener behind)")
        d["pending"], d["last_log"] = 0, now


def _run_safety(text: str) -> bool:
    """If `text` is a safety command, execute it now and return True.
    Also aborts any action chain in progress."""
    safety = check_safety_command(text)
    if not safety:
        return False
    print(f"[SAFETY] {safety['intent']} — executing immediately")
    _mark_safety()
    _start_hush()  # cut speech on its own thread; never delays the halt
    _dispatch_safety(safety)
    return True


def rescue_safety(stub, pcm_bytes: bytes, asr=None, run_safety=None) -> tuple:
    """Safety-only rescue for a short utterance that was rejected (lock /
    voiceprint): ASR it and run the transcript safety path. Anything that is
    not stop/freeze/estop is discarded (no LLM, no reply). Returns
    (transcript, executed). The always-on safety KWS thread is independent of
    this and never waits on it."""
    asr = asr or (lambda st, pcm: _stt.send_to_asr(st, pcm, timeout=RESCUE_ASR_TIMEOUT_S))
    run_safety = run_safety or _run_safety
    if not pcm_bytes:
        return "", False
    try:
        transcript = asr(stub, pcm_bytes) or ""
    except Exception as e:  # noqa: BLE001 - rescue must never break the loop
        print(f"[Rescue] ASR failed: {type(e).__name__}: {e}")
        return "", False
    if not transcript:
        return "", False
    clean = re.sub(r'\s+', ' ', re.sub(r'\.\s*', ' ', transcript)).strip()
    executed = bool(run_safety(transcript) or (clean and run_safety(clean)))
    return transcript, executed


def _hush():
    """Cut speech now: flush the player and latch the gripper mouth shut."""
    _audio_player = _cap.CAPTURE._audio_player
    if _audio_player is not None:
        try:
            _audio_player.force_reset()
        except Exception as e:
            print(f"[Control] hush: player reset failed: {e}")
    if os.environ.get("SPOT_DRY_RUN") != "1":
        try:
            from src.voice_control.spot_dispatch import _close_mouth
            _close_mouth()
        except Exception as e:
            print(f"[Control] hush: close mouth failed: {e}")


# halt/hush are looked up at call time so a patch of _dispatch_safety /
# _hush_player (tests, golden harness) reaches the KWS path too.
LISTENER = SafetyListener(halt=lambda intent: _dispatch_safety(intent),
                          hush=lambda: _hush_player())
_on_safety_kw = LISTENER.on_keyword
