"""G10: the live pipeline clears leftover fixed beams once when the mic opens."""
import sys
import types

from src.spot.audio import capture as _cap


class FakeMicArray:
    """Records every call; any device-config call would show up here."""

    def __init__(self, beams_raise=False):
        self.calls = []
        self._beams_raise = beams_raise

    def set_fixed_beams(self, az):
        self.calls.append(("set_fixed_beams", az))
        if self._beams_raise:
            raise RuntimeError("usb gone")

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
