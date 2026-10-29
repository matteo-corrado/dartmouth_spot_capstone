"""XVF3800 control-link auto-recovery (fake usb backend, no hardware)."""
import struct
import threading
import time

from src.voice_control.mic_array import (
    RECONNECT_BACKOFF_S, MicArray, XvfError, backoff_delay)
from src.voice_control.ring_animator import RingAnimator
from src.voice_control.state_feedback import RingSpec

OK_DOA = bytes([0]) + struct.pack("<HH", 90, 1)
OK_VER = bytes([0, 2, 1, 1])
FORBIDDEN_WRITES = {(9, 48), (7, 48)}  # (wValue=cmd, wIndex=resid): SAVE_CONFIGURATION, REBOOT


class FakeUsbDev:
    """pyusb Device stand-in that records every call, incl. ones we must never make."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []        # (method, args)
        self.writes = []       # (wValue, wIndex, payload)
        self.reads = []        # (wValue, wIndex)

    def ctrl_transfer(self, bmRequestType, bRequest, wValue=0, wIndex=0,
                      data_or_wLength=None, timeout=None):
        self.calls.append(("ctrl_transfer", (bmRequestType, wValue, wIndex, timeout)))
        if self.fail:
            raise IOError("[Errno 5] Input/Output Error")
        if bmRequestType == 0x40:
            self.writes.append((wValue, wIndex, bytes(data_or_wLength)))
            return len(data_or_wLength)
        self.reads.append((wValue, wIndex))
        return OK_VER if (wValue & 0x7F, wIndex) == (0, 48) else OK_DOA

    def __getattr__(self, name):  # set_configuration, detach_kernel_driver, reset, ...
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*a, **kw):
            self.calls.append((name, a))
        return f


class FakeUsbBus:
    """finder/disposer pair: returns None `misses` times, then fresh devices."""

    def __init__(self, misses=0, never=False):
        self.misses = misses
        self.never = never
        self.find_times = []
        self.disposed = []
        self.devices = []

    def find(self):
        self.find_times.append(time.monotonic())
        if self.never or len(self.find_times) <= self.misses:
            return None
        d = FakeUsbDev()
        self.devices.append(d)
        return d

    def dispose(self, dev):
        self.disposed.append(dev)


def wait_for(cond, t=2.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def make(misses=0, never=False, backoff=(0.02, 0.04, 0.08)):
    first = FakeUsbDev()
    bus = FakeUsbBus(misses=misses, never=never)
    m = MicArray(first, finder=bus.find, disposer=bus.dispose)
    m.backoff_s = backoff
    return m, first, bus


def assert_clean(dev):
    names = {c[0] for c in dev.calls}
    assert names <= {"ctrl_transfer"}, names  # no set_configuration / detach / reset
    assert not any((w[0], w[1]) in FORBIDDEN_WRITES for w in dev.writes)
    assert all(c[1][3] == 150 for c in dev.calls if c[0] == "ctrl_transfer")


def test_backoff_schedule_caps_at_5s():
    assert [backoff_delay(i) for i in range(7)] == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0]
    assert RECONNECT_BACKOFF_S[0] == 0.5


def test_error_dead_retries_with_backoff_then_restored():
    m, first, bus = make(misses=3)
    restored = threading.Event()
    m.add_restore_listener(restored.set)
    m.start_poller(hz=200)
    assert wait_for(lambda: m.samples_between(0, 1e12))
    first.fail = True                       # transient USB error
    assert wait_for(lambda: m.dead)
    assert m.ok is False
    assert restored.wait(2.0)
    assert m.dead is False and m.ok and m.reconnects == 1
    # 3 misses + 1 success; gaps follow the non-decreasing backoff schedule
    assert len(bus.find_times) == 4
    gaps = [b - a for a, b in zip(bus.find_times, bus.find_times[1:])]
    assert gaps[0] >= 0.035 and gaps[1] >= 0.07 and gaps[2] >= 0.07
    assert first in bus.disposed            # old handle released before re-find
    new = bus.devices[-1]
    n = len(new.reads)
    assert wait_for(lambda: len(new.reads) > n + 3)   # poller resumed on new handle
    assert m.poller_alive
    m.close()
    for d in [first] + bus.devices:
        assert_clean(d)


def test_failed_verify_disposes_and_keeps_retrying():
    m, first, bus = make()
    orig_find = bus.find

    def find_flaky():
        d = orig_find()
        if len(bus.find_times) == 1:
            d.fail = True                   # found, but VERSION read fails
        return d
    m._finder = find_flaky
    restored = threading.Event()
    m.add_restore_listener(restored.set)
    first.fail = True
    try:
        m.read_doa()
    except XvfError:
        pass
    assert restored.wait(2.0)
    assert bus.devices[0] in bus.disposed and m._dev is bus.devices[1]
    m.close()


def test_single_reconnect_thread_under_error_storm():
    m, first, bus = make(never=True, backoff=(0.05,))
    first.fail = True
    for _ in range(20):
        m.set_led_effect(3)
        m._mark_dead("storm")
    rcs = [t for t in threading.enumerate() if t.name == "xvf-reconnect" and t.is_alive()]
    assert len(rcs) == 1
    m.close()


def test_stale_handle_error_does_not_kill_new_handle():
    m, first, bus = make()
    gen = m._dev_gen
    m._dev_gen += 1                 # a swap happened since that transfer started
    m._mark_dead("late error on old handle", gen)
    assert m.dead is False
    m.close()


def test_stop_during_retry_exits_promptly():
    m, first, bus = make(never=True, backoff=(5.0,))
    first.fail = True
    m.start_poller(hz=50)
    assert wait_for(lambda: m._rc_thread is not None and m._rc_thread.is_alive())
    rc = m._rc_thread
    t = time.monotonic()
    m.close()
    assert time.monotonic() - t < 0.5
    assert not rc.is_alive() and not m.poller_alive


def test_no_reconnect_after_close():
    m, first, bus = make()
    m.close()
    first.fail = True
    m.set_led_effect(3)
    assert m.dead and m._rc_thread is None and bus.find_times == []


def test_no_finder_keeps_legacy_latch():
    m = MicArray(FakeUsbDev(fail=True))
    m.set_led_effect(3)
    assert m.dead and m._rc_thread is None


class FakeMic:
    """MicArray-shaped fake for the animator: dead flag + restore listeners."""

    def __init__(self):
        self.dead = False
        self.effects, self.rings = [], []
        self.listeners = []

    def add_restore_listener(self, fn):
        self.listeners.append(fn)

    def set_led_effect(self, e):
        if not self.dead:
            self.effects.append(e)

    def set_ring(self, r):
        if not self.dead:
            self.rings.append(list(r))

    def restore(self):
        self.dead = False
        for fn in self.listeners:
            fn()


def spec(ring=None, frames=None, dt=0.01):
    return RingSpec(5, None, ring, 255, frames, dt)


def test_ring_reapplies_effect_and_latest_spec_on_restore():
    m = FakeMic(); a = RingAnimator(m, max_hz=1000)
    a.show(spec(ring=[1] * 12))
    assert wait_for(lambda: m.rings == [[1] * 12])
    m.dead = True
    a.show(spec(ring=[2] * 12))     # state changed during the outage
    time.sleep(0.05)
    assert m.rings == [[1] * 12]
    m.restore()
    assert wait_for(lambda: m.rings[-1] == [2] * 12)
    assert m.effects == [5, 5]      # effect 5 re-set after restore
    a.stop()


def test_ring_thread_restarts_after_death():
    m = FakeMic(); a = RingAnimator(m, max_hz=1000)
    a.show(spec(ring=[1] * 12))
    assert wait_for(lambda: m.rings)
    with a._cond:                   # make the worker exit (simulated death)
        a._stopped = True
        a._cond.notify_all()
    a._thread.join(1.0)
    a._stopped = False
    assert not a._thread.is_alive()
    a.show(spec(ring=[3] * 12))
    assert wait_for(lambda: m.rings[-1] == [3] * 12)
    assert a.restarts == 1
    a.stop()


def test_ring_worker_survives_render_exception():
    class BadSpec:
        @property
        def frames(self):
            raise RuntimeError("bad spec")
    m = FakeMic(); a = RingAnimator(m, max_hz=1000)
    a.show(BadSpec())
    time.sleep(0.05)
    assert a._thread.is_alive()
    a.show(spec(ring=[4] * 12))
    assert wait_for(lambda: m.rings and m.rings[-1] == [4] * 12)
    a.stop()


def test_end_to_end_mic_and_ring_recover():
    m, first, bus = make(misses=1)
    m.start_poller(hz=100)
    a = RingAnimator(m, max_hz=1000)
    a.show(spec(ring=[0x101010] * 12))
    assert wait_for(lambda: any(w[:2] == (19, 20) for w in first.writes))
    first.fail = True
    assert wait_for(lambda: m.dead)
    assert wait_for(lambda: not m.dead and bus.devices)
    new = bus.devices[-1]
    assert wait_for(lambda: (12, 20, bytes([5])) in new.writes)          # effect 5 again
    assert wait_for(lambda: any(w[:2] == (19, 20) for w in new.writes))  # ring re-applied
    a.stop()
    m.close()
    for d in [first] + bus.devices:
        assert_clean(d)
