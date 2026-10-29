# src/voice_control/mic_array.py
"""USB control of the reSpeaker XVF3800 mic array (firmware 2.1.1).

Reads speaker direction (DoA) and drives the 12-LED ring via vendor control
transfers. Runtime writes only: this module never sends SAVE_CONFIGURATION
(48/9) or REBOOT (48/7), never calls set_configuration, and never detaches
kernel drivers (the USB audio stream must keep running).

Auto-recovery: a MicArray made by open() survives transient control-link
errors. Any failure marks it ``dead`` (lock/ring degrade), then ONE
background thread retries with backoff: dispose the pyusb handle, re-find
2886:001a, read VERSION. On success ``dead`` clears, the DoA poller resumes
and restore listeners (RingAnimator) re-apply their state. All USB I/O stays
on the poller / reconnect / ring threads, never on the caller's.

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
    import usb.util
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
USB_TIMEOUT_MS = 150  # short, so a hung device can't stall LED writes behind _dev_lock
RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 5.0)  # last value repeats forever


class XvfError(Exception):
    """Device returned a bad status or never became ready."""


def backoff_delay(attempt, schedule=RECONNECT_BACKOFF_S):
    """Delay before reconnect attempt `attempt` (0-based); caps at schedule[-1]."""
    return schedule[min(attempt, len(schedule) - 1)]


def decode_read(resp, fmt, count):
    """Decode a read response. None on retry status, XvfError on other errors."""
    status = resp[0]
    if status == STATUS_RETRY:
        return None
    if status != STATUS_OK:
        raise XvfError(f"xvf3800 read status {status}")
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt * count, bytes(resp[1:1 + count * size]))


def _raw_read(dev, param, fmt, count, stop=None):
    """One parameter read on `dev` (retry-status loop). Raises on any failure."""
    resid, cmd = param
    length = count * struct.calcsize("<" + fmt) + 1
    for _ in range(READ_RETRIES):
        resp = dev.ctrl_transfer(0xC0, 0, 0x80 | cmd, resid, length, USB_TIMEOUT_MS)
        out = decode_read(resp, fmt, count)
        if out is not None:
            return out
        if stop is not None and stop.is_set():
            break
        time.sleep(RETRY_SLEEP_S)
    raise XvfError(f"xvf3800 read {param} not ready after {READ_RETRIES} tries")


class MicArray:
    def __init__(self, dev, finder=None, disposer=None):
        """`finder` (() -> dev or None) enables auto-reconnect; without it the
        first error latches `dead` for good. `disposer(dev)` releases a handle."""
        self._dev = dev
        self._dev_gen = 0               # bumped on every handle swap
        self._dev_lock = threading.Lock()
        self._finder = finder
        self._disposer = disposer
        self._history = collections.deque(maxlen=400)
        self._hist_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._poll_period = None
        self._rc_lock = threading.Lock()  # guards dead + _rc_thread + _dev_gen together
        self._rc_thread = None
        self._lost_at = None
        self._restore_listeners = []
        self.backoff_s = RECONNECT_BACKOFF_S
        self.reconnects = 0
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
        if dev is None:
            return None
        return cls(dev,
                   finder=lambda: usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID),
                   disposer=usb.util.dispose_resources)

    @property
    def ok(self):
        """Lock-free health flag (for status)."""
        return not self.dead

    def add_restore_listener(self, fn):
        """fn() runs on the reconnect thread after control is restored."""
        self._restore_listeners.append(fn)

    # -- failure / recovery --------------------------------------------
    def _mark_dead(self, why, gen=None):
        """Mark the control link dead; start the reconnect thread if enabled.
        A failure on a handle that has since been replaced (stale `gen`) is ignored."""
        with self._rc_lock:
            if gen is not None and gen != self._dev_gen:
                return
            self.dead = True
            if (self._finder is None or self._stop.is_set()
                    or self._rc_thread is not None):
                return
            self._lost_at = time.monotonic()
            self._rc_thread = threading.Thread(
                target=self._reconnect_loop, name="xvf-reconnect", daemon=True)
            self._rc_thread.start()
        log.warning("[MicArray] control lost — retrying (%s)", why)

    def _dispose(self, dev):
        if dev is None or self._disposer is None:
            return
        try:
            self._disposer(dev)
        except Exception as e:  # noqa: BLE001
            log.debug("xvf3800 dispose failed: %s", e)

    def _try_reconnect(self):
        """Release the old handle, re-find the device, verify with a VERSION read."""
        with self._dev_lock:  # no transfer in flight while the handle goes away
            self._dispose(self._dev)
        dev = self._finder()
        if dev is None:
            return False
        try:
            with self._dev_lock:
                _raw_read(dev, VERSION, "B", 3, self._stop)
        except Exception as e:  # noqa: BLE001
            log.debug("xvf3800 reconnect verify failed: %s", e)
            self._dispose(dev)
            return False
        with self._dev_lock:
            self._dev = dev
            with self._rc_lock:
                self._dev_gen += 1
        return True

    def _reconnect_loop(self):
        attempt = 0
        while True:
            if self._stop.wait(backoff_delay(attempt, self.backoff_s)):
                return
            attempt += 1
            try:
                ok = self._try_reconnect()
            except Exception as e:  # noqa: BLE001  (finder raised etc.)
                log.debug("xvf3800 reconnect attempt %d failed: %s", attempt, e)
                ok = False
            if self._stop.is_set():
                return
            if ok:
                break
        with self._rc_lock:
            self.dead = False
            self._rc_thread = None
            down = time.monotonic() - (self._lost_at or time.monotonic())
        self.reconnects += 1
        log.warning("[MicArray] control restored after %.1f s (%d attempts)", down, attempt)
        self._ensure_poller()
        for fn in list(self._restore_listeners):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.warning("xvf3800 restore listener failed: %s", e)

    # -- raw transfers -------------------------------------------------
    def _read(self, param, fmt, count):
        """Read a parameter. Any failure marks the device dead and raises XvfError."""
        if self.dead:
            raise XvfError("xvf3800 dead")
        resid, cmd = param
        length = count * struct.calcsize("<" + fmt) + 1
        gen = self._dev_gen
        for _ in range(READ_RETRIES):
            try:
                with self._dev_lock:
                    gen = self._dev_gen
                    resp = self._dev.ctrl_transfer(
                        0xC0, 0, 0x80 | cmd, resid, length, USB_TIMEOUT_MS)
                out = decode_read(resp, fmt, count)
            except Exception as e:
                self._mark_dead(f"read {param}: {e}", gen)
                raise XvfError(f"xvf3800 read {param} failed: {e}") from e
            if out is not None:
                return out
            time.sleep(RETRY_SLEEP_S)
        self._mark_dead(f"read {param} not ready", gen)
        raise XvfError(f"xvf3800 read {param} not ready after {READ_RETRIES} tries")

    def _write(self, param, payload):
        if self.dead:
            return
        resid, cmd = param
        gen = self._dev_gen
        try:
            with self._dev_lock:
                gen = self._dev_gen
                self._dev.ctrl_transfer(0x40, 0, cmd, resid, payload, USB_TIMEOUT_MS)
        except Exception as e:
            log.warning("xvf3800 write %s failed: %s", param, e)
            self._mark_dead(f"write {param}: {e}", gen)

    # -- reads ---------------------------------------------------------
    def read_doa(self):
        deg, speech = self._read(DOA_VALUE, "H", 2)
        return int(deg), bool(speech)

    def read_selected_azimuth(self):
        az = self._read(AUDIO_MGR_SELECTED_AZIMUTHS, "f", 2)[0]
        return None if math.isnan(az) else az

    # -- writes --------------------------------------------------------
    def set_led_effect(self, effect):
        self._write(LED_EFFECT, struct.pack("<B", int(effect) & 0xFF))

    def set_led_color(self, rgb):
        self._write(LED_COLOR, struct.pack("<I", int(rgb) & 0xFFFFFF))

    def set_ring(self, colors):
        if len(colors) != RING_LEDS:
            raise ValueError(f"set_ring needs exactly {RING_LEDS} colours, got {len(colors)}")
        self._write(LED_RING_COLOR, struct.pack("<12I", *(int(c) & 0xFFFFFF for c in colors)))

    def set_fixed_beams(self, az_rad):
        """Point both fixed beams at az_rad (radians); None turns fixed beams off."""
        if az_rad is None:
            self._write(AEC_FIXEDBEAMSONOFF, struct.pack("<i", 0))
            return
        # Azimuths first so beams don't snap to a stale angle when enabled.
        az = float(az_rad)
        self._write(AEC_FIXEDBEAMSAZIMUTH_VALUES, struct.pack("<2f", az, az))
        self._write(AEC_FIXEDBEAMSONOFF, struct.pack("<i", 1))

    # -- poller --------------------------------------------------------
    def start_poller(self, hz=20.0):
        if hz <= 0:
            raise ValueError(f"poller hz must be > 0, got {hz}")
        self._poll_period = 1.0 / hz
        self._ensure_poller()

    @property
    def poller_alive(self):
        t = self._thread
        return t is not None and t.is_alive()

    def _ensure_poller(self):
        if self._poll_period is None or self._stop.is_set() or self.poller_alive:
            return
        self._thread = threading.Thread(
            target=self._poll, args=(self._poll_period,), name="xvf-doa-poller", daemon=True)
        self._thread.start()

    def _poll(self, period):
        """Poll DoA. While dead: idle until restored (exit if reconnect is off)."""
        while not self._stop.is_set():
            if self.dead:
                if self._finder is None:
                    return
                self._stop.wait(period)
                continue
            try:
                t = time.monotonic()  # stamp before the (possibly retried) read
                deg, speech = self.read_doa()
                with self._hist_lock:
                    self._history.append((t, deg, speech))
            except XvfError:
                pass  # _read already marked dead (and logged)
            except Exception as e:  # noqa: BLE001
                self._mark_dead(f"poller: {e}")
            self._stop.wait(period)

    def samples_between(self, t0, t1):
        """(monotonic t, deg, speech) samples with t0 <= t <= t1."""
        with self._hist_lock:
            return [s for s in self._history if t0 <= s[0] <= t1]

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        rc = self._rc_thread
        if rc is not None:
            rc.join(timeout=1.0)
