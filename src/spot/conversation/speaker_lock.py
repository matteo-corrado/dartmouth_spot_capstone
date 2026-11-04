"""Pure direction-based speaker lock (angles in degrees, circular)."""
import logging

log = logging.getLogger(__name__)


def ang_diff(a, b) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _signed_diff(frm, to) -> float:
    return ((to - frm + 540.0) % 360.0) - 180.0


def circular_median_deg(degs):
    if not degs:
        return None
    return min(degs, key=lambda c: sum(ang_diff(c, d) for d in degs)) % 360.0


def mic_to_body_deg(mic_deg, offset_deg, sign) -> float:
    return (sign * (mic_deg - offset_deg)) % 360.0


class SpeakerLock:
    def __init__(self, window_deg, timeout_s, nudge):
        self.window_deg = window_deg
        self.timeout_s = timeout_s
        self.nudge = nudge
        self.angle = None
        self._last = 0.0

    @property
    def locked(self) -> bool:
        return self.angle is not None

    def lock(self, degs, now):
        m = circular_median_deg(degs)
        if m is None:
            self.angle = None
            return None
        self.angle = m
        self._last = now
        return m

    def accept(self, degs, now, touch=True) -> bool:
        """touch=False: accept/nudge without restarting the conversation clock
        (the caller touches once the turn is known to be real)."""
        if not self.locked:
            return True
        m = circular_median_deg(degs)
        if m is None:
            log.info("speaker lock: no direction readings, accepting utterance")
            return True
        if ang_diff(m, self.angle) <= self.window_deg:
            self.angle = (self.angle + self.nudge * _signed_diff(self.angle, m)) % 360.0
            if touch:
                self._last = now
            return True
        return False

    def touch(self, now):
        self._last = now

    def expired(self, now) -> bool:
        return self.locked and (now - self._last) > self.timeout_s

    def release(self):
        self.angle = None
