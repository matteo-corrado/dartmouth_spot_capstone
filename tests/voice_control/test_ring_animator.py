"""RingAnimator: fake mic, no hardware."""
import time

from src.voice_control.mic_array import XvfError
from src.voice_control.ring_animator import RingAnimator
from src.voice_control.state_feedback import RingSpec


class FakeMic:
    def __init__(self):
        self.dead = False
        self.effects, self.rings = [], []

    def set_led_effect(self, e):
        self.effects.append(e)

    def set_ring(self, r):
        self.rings.append(list(r))


def spec(frames=None, ring=None, dt=0.01):
    return RingSpec(5, None, ring, 255, frames, dt)


def wait_for(cond, t=1.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_frames_cycle_and_effect_set_once():
    m = FakeMic(); a = RingAnimator(m, max_hz=1000)
    f = [[1] * 12, [2] * 12, [3] * 12]
    a.show(spec(frames=f))
    assert wait_for(lambda: len(m.rings) >= 7)
    a.stop()
    assert m.rings[:6] == f + f
    assert m.effects == [5, 4]


def test_static_written_once():
    m = FakeMic(); a = RingAnimator(m)
    a.show(spec(ring=[9] * 12))
    assert wait_for(lambda: m.rings)
    time.sleep(0.1)
    a.stop()
    assert m.rings == [[9] * 12]


def test_show_swaps_within_a_frame():
    m = FakeMic(); a = RingAnimator(m, max_hz=1000)
    a.show(spec(frames=[[1] * 12, [2] * 12], dt=0.05))
    assert wait_for(lambda: m.rings)
    a.show(spec(ring=[7] * 12))
    assert wait_for(lambda: [7] * 12 in m.rings, 0.3)
    n = len(m.rings); time.sleep(0.15)
    a.stop()
    assert len(m.rings) == n


def test_max_hz_caps_rate():
    m = FakeMic(); a = RingAnimator(m, max_hz=20)
    a.show(spec(frames=[[1] * 12, [2] * 12], dt=0.001))
    time.sleep(0.3)
    a.stop()
    assert len(m.rings) <= 9


def test_none_and_dead_mic_safe():
    a = RingAnimator(None); a.show(spec(ring=[1] * 12)); a.stop()
    m = FakeMic(); m.dead = True
    a = RingAnimator(m); a.show(spec(ring=[1] * 12)); a.stop()
    assert m.rings == [] and m.effects == []


def test_errors_never_raise():
    class Bad(FakeMic):
        def set_ring(self, r):
            raise XvfError("boom")
    m = Bad(); a = RingAnimator(m)
    a.show(spec(frames=[[1] * 12], dt=0.01))
    time.sleep(0.05)
    a.stop()


def test_stop_is_fast():
    m = FakeMic(); a = RingAnimator(m)
    a.show(spec(frames=[[1] * 12], dt=5.0))
    time.sleep(0.05)
    t = time.time(); a.stop()
    assert time.time() - t < 0.5 and m.effects[-1] == 4
    a.show(spec(ring=[1] * 12))   # after stop: ignored


def test_slow_write_racing_stop_ends_with_effect_4():
    class Slow(FakeMic):
        def set_led_effect(self, e):
            time.sleep(0.15)
            super().set_led_effect(e)
    m = Slow(); a = RingAnimator(m)
    a.show(spec(frames=[[1] * 12], dt=0.01))
    time.sleep(0.05)       # worker is mid effect-5 write
    a.stop()
    time.sleep(0.3)
    assert m.effects[-1] == 4 and 5 in m.effects[:-1]
    n = len(m.rings); time.sleep(0.1)
    assert len(m.rings) == n
