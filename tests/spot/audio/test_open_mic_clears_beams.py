"""G10: the live pipeline clears leftover fixed beams once when the mic opens."""
import sys
import types

from src.spot.audio import capture as _cap


class FakeMicArray:
    """Records every call; any device-config call would show up here."""

    def __init__(self, beams_raise=False, dead_after_write=False):
        self.calls = []
        self.dead = False
        self.listeners = []
        self._beams_raise = beams_raise
        self._dead_after_write = dead_after_write

    def add_restore_listener(self, fn):
        self.listeners.append(fn)

    def set_fixed_beams(self, az):
        self.calls.append(("set_fixed_beams", az))
        if self._beams_raise:
            raise RuntimeError("usb gone")
        if self._dead_after_write:  # MicArray._write swallows the error, marks dead
            self.dead = True

    def start_poller(self, hz=20.0):
        self.calls.append(("start_poller", hz))

    def __getattr__(self, name):  # save_configuration / reboot / reset ... must never be called
        raise AssertionError(f"unexpected MicArray call: {name}")


def _patch_open(monkeypatch, ret):
    mod = types.ModuleType("src.spot.mic.xvf3800")
    mod.MicArray = types.SimpleNamespace(open=lambda: ret)
    monkeypatch.setitem(sys.modules, "src.spot.mic.xvf3800", mod)


def test_open_clears_fixed_beams_once_before_poller(monkeypatch):
    fake = FakeMicArray()
    _patch_open(monkeypatch, fake)
    assert _cap._open_mic_array() is fake
    assert fake.calls == [("set_fixed_beams", None), ("start_poller", 20.0)]


def test_no_xvf_device_is_a_no_op(monkeypatch):
    _patch_open(monkeypatch, None)
    assert _cap._open_mic_array() is None


def test_beams_clear_failure_never_raises(monkeypatch):
    fake = FakeMicArray(beams_raise=True)
    _patch_open(monkeypatch, fake)
    assert _cap._open_mic_array() is fake
    assert ("start_poller", 20.0) in fake.calls


def test_dead_after_clear_is_loud(monkeypatch, capsys):
    """h1: _write swallows USB errors; a dead link after the write is reported."""
    fake = FakeMicArray(dead_after_write=True)
    _patch_open(monkeypatch, fake)
    _cap._open_mic_array()
    assert "[MicArray] ⚠️  fixed-beams clear failed" in capsys.readouterr().out


def test_reconnect_restore_clears_beams_again(monkeypatch):
    """h1: a restored link gets the same runtime beams-off write (the listener
    runs on the reconnect thread)."""
    fake = FakeMicArray()
    _patch_open(monkeypatch, fake)
    _cap._open_mic_array()
    assert len(fake.listeners) == 1
    fake.calls.clear()
    fake.listeners[0]()
    assert fake.calls == [("set_fixed_beams", None)]


def test_real_micarray_restore_runs_listener_off_audio_thread():
    """The real MicArray runs restore listeners on its xvf-reconnect thread."""
    import inspect
    from src.spot.mic import xvf3800
    src = inspect.getsource(xvf3800.MicArray._reconnect_loop)
    assert "_restore_listeners" in src
