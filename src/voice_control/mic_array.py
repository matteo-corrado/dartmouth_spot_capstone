# src/voice_control/mic_array.py
"""USB control of the reSpeaker XVF3800 mic array (firmware 2.1.1).

Reads speaker direction (DoA) and drives the 12-LED ring via vendor control
transfers. Runtime writes only: this module never sends SAVE_CONFIGURATION
(48/9) or REBOOT (48/7), never calls set_configuration, and never detaches
kernel drivers (the USB audio stream must keep running).

Protocol: see Appendix A of docs/superpowers/specs/2026-10-27-poster-demo-design.md.
"""
import collections
import logging
import math
import struct
import threading
import time

try:
    import usb.core
except ImportError:  # pyusb missing: MicArray.open() returns None
    usb = None

log = logging.getLogger(__name__)

VENDOR_ID = 0x2886
PRODUCT_ID = 0x001A

# (resid, cmd) pairs, verbatim from spec Appendix A.
VERSION = (48, 0)                         # 3 x u8
DOA_VALUE = (20, 18)                      # 2 x u16: [deg 0-359, speech flag]
AUDIO_MGR_SELECTED_AZIMUTHS = (35, 11)    # 2 x f32 rad: [processed (NaN=no speech), auto-select]
AEC_FIXEDBEAMSONOFF = (33, 37)            # 1 x i32
AEC_FIXEDBEAMSAZIMUTH_VALUES = (33, 81)   # 2 x f32 rad
AEC_FIXEDBEAMSGATING = (33, 83)           # 1 x u8 (not written by this module)
LED_EFFECT = (20, 12)                     # 1 x u8: 0 off,1 breath,2 rainbow,3 solid,4 DoA,5 ring
LED_COLOR = (20, 16)                      # 1 x u32 0xRRGGBB
LED_RING_COLOR = (20, 19)                 # 12 x u32

STATUS_OK = 0
STATUS_RETRY = 64
READ_RETRIES = 100
RETRY_SLEEP_S = 0.01
RING_LEDS = 12


class XvfError(Exception):
    """Device returned a bad status or never became ready."""


def decode_read(resp, fmt, count):
    """Decode a read response. None on retry status, XvfError on other errors."""
    status = resp[0]
    if status == STATUS_RETRY:
        return None
    if status != STATUS_OK:
        raise XvfError(f"xvf3800 read status {status}")
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt * count, bytes(resp[1:1 + count * size]))


class MicArray:
    def __init__(self, dev):
        self._dev = dev
        self._dev_lock = threading.Lock()
        self._history = collections.deque(maxlen=400)
        self._hist_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.dead = False

    @classmethod
    def open(cls):
        """Find the XVF3800. Returns None if absent or pyusb missing; never raises."""
        if usb is None:
            return None
        try:
            dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        except Exception as e:
            log.warning("xvf3800 lookup failed: %s", e)
            return None
        return cls(dev) if dev is not None else None

    # -- raw transfers -------------------------------------------------
    def _read(self, param, fmt, count):
        resid, cmd = param
        length = count * struct.calcsize("<" + fmt) + 1
        for _ in range(READ_RETRIES):
            with self._dev_lock:
                resp = self._dev.ctrl_transfer(0xC0, 0, 0x80 | cmd, resid, length)
            out = decode_read(resp, fmt, count)
            if out is not None:
                return out
            time.sleep(RETRY_SLEEP_S)
        raise XvfError(f"xvf3800 read {param} not ready after {READ_RETRIES} tries")

    def _write(self, param, payload):
        if self.dead:
            return
        resid, cmd = param
        try:
            with self._dev_lock:
                self._dev.ctrl_transfer(0x40, 0, cmd, resid, payload)
        except Exception as e:
            log.warning("xvf3800 write %s failed, disabling: %s", param, e)
            self.dead = True

    # -- reads ---------------------------------------------------------
    def read_doa(self):
        deg, speech = self._read(DOA_VALUE, "H", 2)
        return int(deg), bool(speech)

    def read_selected_azimuth(self):
        az = self._read(AUDIO_MGR_SELECTED_AZIMUTHS, "f", 2)[0]
        return None if math.isnan(az) else az

    # -- writes --------------------------------------------------------
    def set_led_effect(self, effect):
        self._write(LED_EFFECT, struct.pack("<B", effect))

    def set_led_color(self, rgb):
        self._write(LED_COLOR, struct.pack("<I", rgb))

    def set_ring(self, colors):
        if len(colors) != RING_LEDS:
            raise ValueError(f"set_ring needs exactly {RING_LEDS} colours, got {len(colors)}")
        self._write(LED_RING_COLOR, struct.pack("<12I", *colors))

    def set_fixed_beams(self, az_rad):
        """Point both fixed beams at az_rad (radians); None turns fixed beams off."""
        if az_rad is None:
            self._write(AEC_FIXEDBEAMSONOFF, struct.pack("<i", 0))
            return
        # Azimuths first so beams don't snap to a stale angle when enabled.
        self._write(AEC_FIXEDBEAMSAZIMUTH_VALUES, struct.pack("<2f", az_rad, az_rad))
        self._write(AEC_FIXEDBEAMSONOFF, struct.pack("<i", 1))

    # -- poller --------------------------------------------------------
    def start_poller(self, hz=20.0):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll, args=(1.0 / hz,), name="xvf-doa-poller", daemon=True)
        self._thread.start()

    def _poll(self, period):
        try:
            while not self._stop.is_set() and not self.dead:
                deg, speech = self.read_doa()
                with self._hist_lock:
                    self._history.append((time.monotonic(), deg, speech))
                self._stop.wait(period)
        except Exception as e:
            log.warning("xvf3800 poller died: %s", e)
            self.dead = True

    def samples_between(self, t0, t1):
        """(monotonic t, deg, speech) samples with t0 <= t <= t1."""
        with self._hist_lock:
            return [s for s in self._history if t0 <= s[0] <= t1]

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
