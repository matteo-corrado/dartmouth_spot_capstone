"""client_mic LED-ring signalling (Task 11) + the tuning fields it reads."""
from collections import namedtuple

import src.voice_control.client_mic as cm
from src.voice_control.tuning import load_tuning
from src.voice_control.voice_state import VoiceState

Spec = namedtuple("Spec", "effect color ring brightness")


class FakeMic:
    dead = False

    def __init__(self):
        self.calls = []

    def set_led_effect(self, e):
        self.calls.append(("effect", e))

    def set_led_color(self, c):
        self.calls.append(("color", c))

    def set_ring(self, r):
        self.calls.append(("ring", r))


def test_throttle():
    t = cm.RingThrottle(10.0)
    assert t.allow(0.0) and not t.allow(0.05) and t.allow(0.11)


def _setup(monkeypatch, mic, ring_for):
    monkeypatch.setattr(cm, "_mic", mic)
    monkeypatch.setattr(cm, "_ring_for", ring_for)
    monkeypatch.setattr(cm, "_ring_throttle", cm.RingThrottle(1000.0))
    monkeypatch.setattr(cm, "_reject_until", 0.0)
    monkeypatch.setattr(cm, "_state_leds", None)
    monkeypatch.setattr(cm, "chime_for", lambda s: None)


def test_enter_state_writes_ring_spec(monkeypatch):
    mic, seen = FakeMic(), []

    def ring_for(state, deg, muted, ring_offset_leds=0, reverse=False):
        seen.append((state, deg, muted, ring_offset_leds, reverse))
        return Spec(5, None, [1] * 12, 255)

    _setup(monkeypatch, mic, ring_for)
    cm._lock.release()
    cm._enter_state(VoiceState.LISTENING)
    assert seen[-1][0] == VoiceState.LISTENING and seen[-1][2] is False
    assert seen[-1][3:] == (cm._tuning.ring_offset_leds, cm._tuning.ring_reverse)
    assert mic.calls == [("effect", 5), ("ring", [1] * 12)]


def test_throttled_write_is_pending_then_flushed(monkeypatch):
    mic = FakeMic()
    _setup(monkeypatch, mic, lambda s, d, m, ring_offset_leds=0, reverse=False: Spec(3, 0x0040FF, None, 255))
    monkeypatch.setattr(cm, "_ring_throttle", cm.RingThrottle(0.001))  # ~1000 s interval
    cm._enter_state(VoiceState.RESPONDING)
    cm._enter_state(VoiceState.THINKING)
    assert cm._ring_pending is True
    assert len(mic.calls) == 2  # only the first write happened
    monkeypatch.setattr(cm, "_ring_throttle", cm.RingThrottle(1000.0))
    cm._flush_ring()
    assert cm._ring_pending is False and len(mic.calls) == 4


def test_ring_degrades_without_mic(monkeypatch):
    _setup(monkeypatch, None, lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    cm._enter_state(VoiceState.LISTENING)  # no mic -> no ring, no raise


def test_ring_write_error_never_raises(monkeypatch):
    class Boom(FakeMic):
        def set_led_effect(self, e):
            raise RuntimeError("usb gone")

    mic = Boom()
    _setup(monkeypatch, mic, lambda s, d, m, ring_offset_leds=0, reverse=False: Spec(3, 1, None, 255))
    cm._enter_state(VoiceState.LISTENING)
    assert ("color", 1) in mic.calls


def test_reject_flashes_red_then_restores(monkeypatch):
    mic = FakeMic()
    _setup(monkeypatch, mic, lambda s, d, m, ring_offset_leds=0, reverse=False:
           Spec(cm.LED_SOLID, cm.REJECT_COLOR, None, 255) if m else Spec(5, None, [7] * 12, 255))
    started = []

    class T:
        def __init__(self, delay, fn):
            self.fn = fn
            self.daemon = False

        def start(self):
            started.append(self.fn)

    monkeypatch.setattr(cm.threading, "Timer", T)
    monkeypatch.setattr(cm, "_ring_state", VoiceState.LISTENING)
    cm._signal_reject()
    assert mic.calls == [("effect", cm.LED_SOLID), ("color", cm.REJECT_COLOR)]
    cm._update_ring()  # mid-flash: deferred, not overwritten
    assert cm._ring_pending is True and len(mic.calls) == 2
    started[0]()       # timer fires
    assert mic.calls[-1] == ("ring", [7] * 12)


def test_tuning_ring_fields_and_yaw_sign_validation(tmp_path):
    t = load_tuning(tmp_path / "missing.yaml")
    assert (t.ring_offset_leds, t.ring_reverse, t.ring_max_hz) == (6, False, 30.0)
    p = tmp_path / "t.yaml"
    p.write_text("mic_yaw_sign: 3\nring_offset_leds: 2\n")
    t = load_tuning(p)
    assert t.mic_yaw_sign == 1 and t.ring_offset_leds == 2


def test_repo_tuning_has_measured_mic_yaw():
    t = load_tuning()
    assert t.mic_yaw_offset_deg == 179.0 and t.mic_yaw_sign == -1
