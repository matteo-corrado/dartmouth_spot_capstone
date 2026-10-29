# tests/voice_control/test_mic_array.py
"""Unit tests for the XVF3800 USB control module (fake device only)."""
import struct
import time

import pytest

from src.voice_control.mic_array import MicArray, XvfError, decode_read

OK_DOA = bytes([0]) + struct.pack("<HH", 0, 0)


class FakeDev:
    """Scripted stand-in for a pyusb device. Never touches real USB."""

    def __init__(self, script=None, raise_after=None):
        self.script = list(script or [])
        self.raise_after = raise_after
        self.calls = 0
        self.writes = []

    def ctrl_transfer(self, bmRequestType, bRequest, wValue=0, wIndex=0,
                      data_or_wLength=None, timeout=None):
        self.calls += 1
        if self.raise_after is not None and self.calls > self.raise_after:
            raise IOError("fake usb error")
        if bmRequestType == 0x40:
            self.writes.append((wValue, wIndex, bytes(data_or_wLength)))
            return len(data_or_wLength)
        if self.script:
            return self.script.pop(0)
        return OK_DOA


def test_decode_ok():
    assert decode_read(bytes([0]) + struct.pack("<HH", 90, 1), "H", 2) == (90, 1)


def test_decode_retry_returns_none():
    assert decode_read(bytes([64, 0, 0, 0, 0]), "H", 2) is None


def test_decode_bad_status_raises():
    with pytest.raises(XvfError):
        decode_read(bytes([3, 0, 0, 0, 0]), "H", 2)


def test_read_doa_retries_then_succeeds():
    dev = FakeDev([bytes([64]) + b"\0" * 4, bytes([0]) + struct.pack("<HH", 270, 1)])
    assert MicArray(dev).read_doa() == (270, True)


def test_selected_azimuth_nan_is_none():
    dev = FakeDev([bytes([0]) + struct.pack("<2f", float("nan"), 1.57)])
    assert MicArray(dev).read_selected_azimuth() is None


def test_samples_between_filters_by_time():
    m = MicArray(FakeDev([]))
    m._history.extend([(1.0, 10, True), (2.0, 20, True), (3.0, 30, False)])
    assert m.samples_between(1.5, 3.5) == [(2.0, 20, True), (3.0, 30, False)]


def test_poller_survives_usb_error():
    m = MicArray(FakeDev(raise_after=2))
    m.start_poller(hz=200)
    time.sleep(0.1)
    assert m.dead is True and m.samples_between(0, 1e12) is not None
    m.set_led_effect(3)  # no exception after death


def test_set_ring_packs_twelve_u32():
    dev = FakeDev()
    m = MicArray(dev)
    m.set_ring(list(range(12)))
    assert dev.writes == [(19, 20, struct.pack("<12I", *range(12)))]
    with pytest.raises(ValueError):
        m.set_ring([0] * 11)


def test_fixed_beams_on_and_off():
    dev = FakeDev()
    m = MicArray(dev)
    m.set_fixed_beams(1.0)
    assert dev.writes[0] == (81, 33, struct.pack("<2f", 1.0, 1.0))
    assert dev.writes[-1] == (37, 33, struct.pack("<i", 1))
    dev.writes.clear()
    m.set_fixed_beams(None)
    assert dev.writes == [(37, 33, struct.pack("<i", 0))]


def test_read_after_usb_error_raises_xvf_and_marks_dead():
    dev = FakeDev(raise_after=0)
    m = MicArray(dev)
    with pytest.raises(XvfError):
        m.read_doa()
    assert m.dead is True
    with pytest.raises(XvfError):  # dead: no further transfer attempted
        m.read_selected_azimuth()
    assert dev.calls == 1
    m.set_led_effect(3)  # no-op, no exception
    assert dev.calls == 1 and dev.writes == []


def test_transfers_pass_short_timeout():
    seen = []

    class TimeoutDev(FakeDev):
        def ctrl_transfer(self, *a, **kw):
            seen.append(a[5] if len(a) > 5 else kw.get("timeout"))
            return super().ctrl_transfer(*a, **kw)

    m = MicArray(TimeoutDev())
    m.read_doa()
    m.set_led_color(0x123456)
    assert seen and all(t is not None and t <= 200 for t in seen)


def test_led_values_masked_not_struct_error():
    dev = FakeDev()
    m = MicArray(dev)
    m.set_led_effect(300)
    m.set_led_color(-1)
    assert dev.writes[0][2] == struct.pack("<B", 300 & 0xFF)
    assert dev.writes[1][2] == struct.pack("<I", 0xFFFFFF)


def test_poller_rejects_nonpositive_hz():
    with pytest.raises(ValueError):
        MicArray(FakeDev()).start_poller(hz=0)


def test_read_beams_reads_azimuths_then_energies():
    dev = FakeDev([bytes([0]) + struct.pack("<4f", 0.0, 1.0, 2.0, 3.0),
                   bytes([0]) + struct.pack("<4f", 0.5, 0.0, 0.25, 0.5)])
    az, en = MicArray(dev).read_beams()
    assert az == pytest.approx([0.0, 1.0, 2.0, 3.0]) and en == pytest.approx([0.5, 0.0, 0.25, 0.5])
