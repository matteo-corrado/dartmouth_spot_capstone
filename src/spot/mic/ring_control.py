"""XVF3800 LED ring turn signal (Task 11 of the voice work): points at the
locked speaker while LISTENING. Writes are throttled to Tuning.ring_max_hz; a
throttled update is marked pending and flushed from the main loop. Every
device call is isolated — a missing/dead mic only means no ring.

Moved from the old client_mic.py (Task 10). The ring state that
used to be client_mic module globals lives on the Ring owner, RING, created
once at import and reached through this module at call time.
"""
import threading
import time

import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
from src.spot.core.tuning import get_tuning
from src.spot.core.voice_state import VoiceState
from src.spot.mic.state_feedback import ring_for as _ring_for

_tuning = get_tuning()

REJECT_FLASH_S = 0.2
REJECT_COLOR = 0xC00000
LED_SOLID = 3


class RingThrottle:
    """Pure rate limiter for ring writes (min interval = 1 / max_hz)."""

    def __init__(self, max_hz: float):
        self.min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self._last = None

    def allow(self, now: float) -> bool:
        if self._last is not None and now - self._last < self.min_interval:
            return False
        self._last = now
        return True


def _make_ring_animator(mic):
    """RingAnimator(mic, max_hz=Tuning.ring_max_hz); None -> direct writes."""
    from src.spot.mic.ring_animator import RingAnimator
    try:
        return RingAnimator(mic, max_hz=_tuning.ring_max_hz)
    except Exception as e:
        print(f"[Indicator] ring animator failed: {e} — direct ring writes")
        return None


class Ring:
    """Owner of the ring state (was client_mic module globals)."""

    def __init__(self):
        self._ring_state = None          # VoiceState the ring currently shows
        self._ring_last_write = 0.0
        self._ring_pending = False
        self._reject_until = 0.0
        self._ring_throttle = RingThrottle(_tuning.ring_max_hz)
        self._anim = None  # RingAnimator (owns ring writes + throttling) when available

    def current_spec(self):
        if self._ring_state is None:
            return None
        # Lock angle is MIC frame, which is what ring_for/ring_toward expect.
        return _ring_for(self._ring_state, _loop.CONVERSATION._lock.angle,
                         _loop.CONVERSATION._operator_muted,
                         ring_offset_leds=_tuning.ring_offset_leds,
                         reverse=_tuning.ring_reverse)

    def show(self, spec) -> None:
        """The ONE place turn-signal ring writes happen. With a RingAnimator it
        hands the spec over (non-blocking); otherwise writes .effect/.color/.ring
        directly, skipping None fields, each device write isolated."""
        if self._anim is not None:
            try:
                self._anim.show(spec)
            except Exception as e:
                print(f"[Indicator] ring animator show failed: {e}")
            return
        _mic = _cap.CAPTURE._mic
        for fn, val in ((_mic.set_led_effect, spec.effect), (_mic.set_led_color, spec.color),
                        (_mic.set_ring, spec.ring)):
            if val is None:
                continue
            try:
                fn(val)
            except Exception as e:
                print(f"[Indicator] ring write failed: {type(e).__name__}: {e}")

    def update(self) -> None:
        """Render the ring for (state, lock, muted) now, or mark it pending if
        throttled / mid reject-flash. Never raises, never sleeps."""
        if not _cap._mic_usable():
            return
        try:
            now = time.monotonic()
            if now < self._reject_until or (self._anim is None and not self._ring_throttle.allow(now)):
                self._ring_pending = True
                return
            self._ring_pending = False
            spec = self.current_spec()
            if spec is not None:
                self.show(spec)
        except Exception as e:
            print(f"[Indicator] ring failed: {type(e).__name__}: {e}")

    def flush(self) -> None:
        """Main-loop hook: apply a throttled/pending ring update."""
        if self._ring_pending:
            self.update()

    def end_reject_flash(self) -> None:
        self._reject_until = 0.0
        self._ring_pending = True
        self.update()

    def signal_reject(self):
        """Visual-only cue that an utterance from outside the lock was ignored:
        the ring flashes red for 200 ms, then a Timer restores the current spec
        (never sleeps on the main loop)."""
        print("[Indicator] reject (off-lock speaker)")
        if not _cap._mic_usable():
            return
        self._reject_until = time.monotonic() + REJECT_FLASH_S
        try:  # the muted spec is solid red: a real RingSpec for the animator
            spec = _ring_for(self._ring_state or VoiceState.LISTENING, None, True)
        except Exception as e:
            print(f"[Indicator] reject spec failed: {e}")
            return
        self.show(spec)
        t = threading.Timer(REJECT_FLASH_S, self.end_reject_flash)
        t.daemon = True
        t.start()


RING = Ring()
_current_ring_spec = RING.current_spec
_show_ring = RING.show
_update_ring = RING.update
_flush_ring = RING.flush
_end_reject_flash = RING.end_reject_flash
_signal_reject = RING.signal_reject
