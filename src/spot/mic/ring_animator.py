# src/voice_control/ring_animator.py
"""Single owner of all XVF3800 LED-ring writes.

`show(spec)` never blocks: it swaps the current RingSpec under a lock and
wakes one daemon thread, which sets LED effect 5 once, then either loops
``spec.frames`` at ``spec.frame_s`` (rate-capped by ``max_hz``) or writes the
static ``spec.ring`` once. A None mic makes everything a no-op, and nothing
here ever raises. Runtime writes only (no save-config / reboot).

While the mic is ``dead`` the latest spec is still kept (writes are skipped);
when MicArray restores control it calls ``_on_restore``, which re-sets effect
5 and re-renders that spec. A worker thread that dies is restarted by the
next show() / restore.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

EFFECT_RING = 5
EFFECT_DOA = 4   # LED effect restored on stop()
ERROR_BACKOFF_S = 0.1  # pause after an unexpected worker error (no hot loop)


class RingAnimator:
    def __init__(self, mic, max_hz: float = 30.0):
        self._mic = mic
        self._min_dt = 1.0 / max_hz if max_hz and max_hz > 0 else 0.0
        self._cond = threading.Condition()
        self._spec = None
        self._gen = 0
        self._stopped = False
        self._thread = None
        self._effect_set = False
        self._warned = False
        self._write_lock = threading.Lock()
        self.restarts = 0
        add = getattr(mic, "add_restore_listener", None)
        if callable(add):
            try:
                add(self._on_restore)
            except Exception as e:  # noqa: BLE001
                log.warning("ring animator: restore hook failed: %s", e)

    def _usable(self) -> bool:
        return self._mic is not None and not getattr(self._mic, "dead", False)

    def _ensure_thread(self) -> None:
        """Start (or restart) the worker. Caller holds self._cond."""
        t = self._thread
        if t is not None and t.is_alive():
            return
        if t is not None:
            self.restarts += 1
            log.warning("ring animator: worker thread died — restarting")
        self._thread = threading.Thread(target=self._run, daemon=True, name="ring-animator")
        self._thread.start()

    def show(self, spec) -> None:
        if spec is None or self._stopped or self._mic is None:
            return
        with self._cond:
            self._spec = spec
            self._gen += 1
            self._ensure_thread()
            self._cond.notify_all()

    def _on_restore(self) -> None:
        """MicArray restored control: effect 5 again, then re-render the spec."""
        with self._write_lock:
            self._effect_set = False
        with self._cond:
            if self._stopped or self._spec is None:
                return
            self._gen += 1
            self._ensure_thread()
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
            t = self._thread
        if t is not None:
            t.join(0.4)
        # Worker re-checks _stopped under _write_lock, so no write follows this.
        got = self._write_lock.acquire(timeout=0.5)
        try:
            if self._usable() and (self._effect_set or t is not None):
                try:
                    self._mic.set_led_effect(EFFECT_DOA)
                except Exception as e:  # noqa: BLE001
                    self._warn(e)
        finally:
            if got:
                self._write_lock.release()

    def _warn(self, e) -> None:
        if not self._warned:
            self._warned = True
            log.warning("ring animator: LED write failed (%s); further errors suppressed", e)

    def _write(self, ring) -> None:
        with self._write_lock:
            if self._stopped or not self._usable():
                return
            try:
                if not self._effect_set:
                    self._mic.set_led_effect(EFFECT_RING)
                    # A failed write marks the mic dead: retry effect 5 on restore.
                    self._effect_set = self._usable()
                if ring is not None:
                    self._mic.set_ring(ring)
            except Exception as e:  # noqa: BLE001
                self._warn(e)

    def _wait(self, gen, timeout=None) -> bool:
        """Wait up to timeout (None = forever) for a new spec or stop.
        True if the caller should abandon the current spec."""
        with self._cond:
            self._cond.wait_for(lambda: self._stopped or self._gen != gen, timeout)
            return self._stopped or self._gen != gen

    def _render(self, spec, gen) -> None:
        """Render one spec until it is replaced or stop()."""
        frames = spec.frames
        if not frames:
            self._write(spec.ring)
            self._wait(gen)
            return
        dt = max(spec.frame_s or 0.0, self._min_dt, 0.005)
        while True:
            for fr in frames:
                self._write(fr)
                if self._wait(gen, dt):
                    return

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stopped:
                    return
                spec, gen = self._spec, self._gen
            try:
                self._render(spec, gen)
            except Exception as e:  # noqa: BLE001  keep the worker alive
                self._warn(e)
                self._wait(gen, ERROR_BACKOFF_S)
