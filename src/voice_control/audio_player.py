"""Single-stream audio playback service for the voice pipeline.

Owns one persistent ``sounddevice.OutputStream`` and one worker thread that
pulls playback tasks from a queue. Both ``SpotTTS`` and ``AudioFeedback``
submit work here instead of calling ``sd.play()`` / ``sd.wait()`` directly.

Why this exists
---------------
``sd.play()`` / ``sd.stop()`` / ``sd.wait()`` operate on a *module-global*
``_last_callback`` inside sounddevice. They are not thread-safe. The previous
design spawned a new thread for every ``tts.speak()`` call AND played beeps
from yet another thread, which meant three concurrent producers fighting over
that single global. The race manifested as a permanent deadlock when
``capture_frame`` failed: a TTS thread would never reach its ``finally`` block
to clear ``_mic_muted``, and the input pipeline would block on an empty
``audio_queue`` forever.

The fix is to centralize all audio playback through one stream and one worker.
The worker plays items in submission order; ``is_busy()`` is the single source
of truth for "should the mic be muted right now".

Concurrency model
-----------------
Three threads touch this object:

1. **Main / caller threads**: ``enqueue_raw``, ``enqueue_render``, ``wait``,
   ``is_busy``, ``shutdown``.
2. **Worker thread** (``_worker_loop``): pulls from the queue, runs render
   functions, writes audio to the stream in 100ms chunks.
3. **Watchdog thread** (``_watchdog_loop``): polls the current task age every
   second; if a single task has been processing longer than ``WATCHDOG_TIMEOUT_S``
   it calls ``force_reset()`` to recover from a wedged worker.

Two locks coordinate them:

- ``_inflight_lock``: protects ``_inflight`` (the source of truth for
  ``is_busy()``), ``_stop`` set/check, and ``_current_task_started_at``.
  Held for nanoseconds at a time. Acquired by every enqueue, every is_busy
  read (~33/sec from the audio callback), and every task start/finish.
- ``_reset_lock``: serializes ``force_reset`` and ``shutdown`` so concurrent
  recovery attempts don't double-up streams or workers. Held for up to ~2s
  (the worker join timeout).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


# A render function returns (samples, source_rate). It runs INSIDE the worker
# thread, so callers don't pay rendering latency on their main thread.
RenderFn = Callable[[], tuple[np.ndarray, int]]

# How long a single task can be processing before the watchdog assumes the
# worker is wedged and calls force_reset(). Generous enough that no real
# Kokoro inference + chunked playback exceeds it (worst case ~10s for very
# long utterances on this hardware).
WATCHDOG_TIMEOUT_S = 30.0

# How often the watchdog wakes up to check the current task age.
WATCHDOG_POLL_INTERVAL_S = 1.0


def _safe_cb(cb: Optional[Callable[[], None]]) -> None:
    """Invoke an optional latency-tracing callback, swallowing exceptions.

    Used by the worker loop to fire on_render_start / on_render_end /
    on_play_start / on_play_end hooks for the latency module. Catches
    Exception only — KeyboardInterrupt still propagates so Ctrl-C works.
    A buggy callback can never wedge the worker.
    """
    if cb is None:
        return
    try:
        cb()
    except Exception as e:
        print(f"[Player] callback error: {e}")


@dataclass
class _Task:
    """One unit of playback work.

    Exactly one of ``samples`` or ``render_fn`` is set. ``rate`` is meaningful
    only for the ``samples`` case (render functions return their own rate).

    The four optional ``on_*`` callbacks are latency-tracing hooks fired by
    the worker thread inside ``_safe_cb``. They are GIL-atomic single
    attribute writes on the latency module's Trace object — see
    ``src/voice_control/latency.py`` for the consumer side.
    """
    label: str
    samples: Optional[np.ndarray] = None
    rate: Optional[int] = None
    render_fn: Optional[RenderFn] = None
    on_render_start: Optional[Callable[[], None]] = None
    on_render_end: Optional[Callable[[], None]] = None
    on_play_start: Optional[Callable[[], None]] = None
    on_play_end: Optional[Callable[[], None]] = None


class AudioPlayer:
    """Centralized audio playback through a single OutputStream + worker thread.

    Lifecycle:
        player = AudioPlayer(output_device=3)
        player.enqueue_raw(samples, 48000, label="beep:wake")
        player.enqueue_render(lambda: kokoro.create("hello"), label="tts:hello")
        player.wait()                    # block until everything plays out
        player.is_busy()                 # True if anything queued or playing
        player.force_reset()             # recovery: rebuild stream + worker
        player.shutdown()                # graceful exit
    """

    # Chunked write size — ~100ms of audio at the stream rate. The worker
    # checks the captured ``stop`` event between chunks so force_reset() can
    # interrupt a long utterance within ~100ms instead of being wedged inside
    # one giant blocking ``stream.write()`` call. Smaller chunks = faster
    # reaction but more PortAudio overhead; 100ms is well below human-
    # perceivable latency for the recovery path.
    _CHUNK_MS = 100

    def __init__(self, output_device=None, sample_rate: Optional[int] = None):
        """Open the output stream and start the worker + watchdog threads.

        Args:
            output_device: sounddevice device index or None for system default.
            sample_rate: Force a specific stream rate. If None, query the device
                and use its default — incoming samples are resampled to match.
        """
        self._available = False  # set to True only after full init succeeds

        if sd is None:
            print("[Player] sounddevice not installed — playback disabled")
            return

        self.output_device = output_device

        # Pick the stream sample rate. We resample everything to this rate so
        # the OutputStream stays at a single fixed rate for its lifetime
        # (changing it requires re-opening the stream, which is what we're
        # trying to avoid).
        if sample_rate is None:
            try:
                info = sd.query_devices(output_device) if output_device is not None \
                    else sd.query_devices(kind="output")
                sample_rate = int(info["default_samplerate"])
            except Exception:
                sample_rate = 48000  # safe default for most modern devices
        self.sample_rate = sample_rate

        # Try mono first; fall back to stereo by duplicating the mono signal
        # if the device refuses mono. The Spot speaker (UACDemoV1.0) accepts
        # mono so this fallback is defensive — never fires on current hardware.
        self._channels = 1
        self._stream = self._open_stream(channels=1)
        if self._stream is None:
            print("[Player] mono open failed — trying stereo fallback")
            self._stream = self._open_stream(channels=2)
            if self._stream is None:
                print("[Player] stereo fallback also failed — playback disabled")
                return
            self._channels = 2

        # Concurrency primitives.
        self._inflight: int = 0          # source of truth for is_busy()
        self._inflight_lock = threading.Lock()
        self._reset_lock = threading.Lock()  # serializes force_reset / shutdown
        self._stop = threading.Event()       # signals the worker to exit
        self._current_task_started_at: Optional[float] = None
        self._current_label: str = ""

        # Queue + worker.
        self._queue: "queue.Queue[_Task]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="AudioPlayer", daemon=True
        )
        self._worker.start()

        # Watchdog: polls the current task age, calls force_reset on wedge.
        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="AudioPlayer-Watchdog", daemon=True
        )
        self._watchdog.start()

        self._available = True
        print(f"[Player] Ready — device={output_device}, "
              f"rate={self.sample_rate}Hz, channels={self._channels}")

    def _open_stream(self, channels: int):
        """Open and start a new OutputStream. Returns the stream or None on failure."""
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=channels,
                dtype="float32",
                device=self.output_device,
            )
            stream.start()
            return stream
        except Exception as e:
            print(f"[Player] Failed to open OutputStream(channels={channels}): {e}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._available

    def is_busy(self) -> bool:
        """True if there is queued OR currently-playing audio.

        This is the single source of truth for the mic-mute interlock in
        ``client_mic.py``. The audio callback gates input frames on this.
        Atomically consistent — backed by an inflight counter under a lock,
        no race window between dequeue and worker-start.
        """
        if not self._available:
            return False
        with self._inflight_lock:
            return self._inflight > 0

    def enqueue_raw(
        self,
        samples: np.ndarray,
        rate: int,
        label: str = "",
        on_render_start: Optional[Callable[[], None]] = None,
        on_render_end: Optional[Callable[[], None]] = None,
        on_play_start: Optional[Callable[[], None]] = None,
        on_play_end: Optional[Callable[[], None]] = None,
    ) -> None:
        """Submit pre-rendered audio samples for playback.

        Optional ``on_*`` callbacks fire from the worker thread for latency
        tracing — see ``latency.py``. They run inside ``_safe_cb`` so a
        buggy callback can never wedge the worker.
        """
        if not self._available:
            return
        with self._inflight_lock:
            if self._stop.is_set():
                return  # reset/shutdown in progress — drop silently
            self._inflight += 1
            try:
                self._queue.put(_Task(
                    label=label, samples=samples, rate=rate,
                    on_render_start=on_render_start,
                    on_render_end=on_render_end,
                    on_play_start=on_play_start,
                    on_play_end=on_play_end,
                ))
            except Exception:
                self._inflight -= 1
                raise

    def enqueue_render(
        self,
        render_fn: RenderFn,
        label: str = "",
        on_render_start: Optional[Callable[[], None]] = None,
        on_render_end: Optional[Callable[[], None]] = None,
        on_play_start: Optional[Callable[[], None]] = None,
        on_play_end: Optional[Callable[[], None]] = None,
    ) -> None:
        """Submit a render function that will be called inside the worker.

        Use this for TTS so Kokoro inference happens in the worker thread,
        not on the caller's main thread. Optional ``on_*`` callbacks fire
        from the worker for latency tracing — see ``latency.py``.
        """
        if not self._available:
            return
        with self._inflight_lock:
            if self._stop.is_set():
                return
            self._inflight += 1
            try:
                self._queue.put(_Task(
                    label=label, render_fn=render_fn,
                    on_render_start=on_render_start,
                    on_render_end=on_render_end,
                    on_play_start=on_play_start,
                    on_play_end=on_play_end,
                ))
            except Exception:
                self._inflight -= 1
                raise

    def wait(self) -> None:
        """Block until the queue is fully drained AND the worker is idle."""
        if not self._available:
            return
        self._queue.join()

    def shutdown(self) -> None:
        """Graceful exit: stop the watchdog, drain the queue, join the worker,
        close the stream. Idempotent — safe to call from cleanup paths.

        Called from ``cleanup_spot()`` in ``client_mic.py`` so we don't leak
        the worker as an abruptly-killed daemon thread (which can corrupt the
        heap if it dies inside ``stream.write()``).
        """
        if not self._available:
            return
        with self._reset_lock:
            if not self._available:  # double-checked under the lock
                return
            print("[Player] shutdown — draining queue and joining worker")
            self._available = False
            self._watchdog_stop.set()

            # Signal worker to exit and drain the queue. Done under the
            # inflight lock so concurrent enqueues drop their tasks.
            with self._inflight_lock:
                self._stop.set()
                while True:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        break
                self._inflight = 0

            # Wait for the worker to exit cleanly. The chunked write design
            # means it should notice _stop within ~100ms.
            self._worker.join(timeout=2.0)
            worker_exited = not self._worker.is_alive()

            if worker_exited:
                # Safe to close the stream — nobody is touching it anymore.
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
            else:
                # Worker is wedged inside a render call (kokoro / CUDA).
                # Closing the stream from this thread while the worker might
                # still be inside stream.write() corrupts the heap (observed
                # in development). Leak the stream as the lesser evil.
                print("[Player] shutdown: worker did not exit within 2s — "
                      "leaking the stream rather than risking heap corruption")

            # Watchdog is a daemon and will exit on its next wake (it checks
            # _watchdog_stop). We don't join it because we already returned
            # _available = False, so any future is_busy() reads return False
            # immediately and the watchdog has nothing more to do.
            print("[Player] shutdown complete")

    def force_reset(self) -> None:
        """Recovery path: drain the queue, signal the worker to exit, and
        rebuild the stream + worker if possible.

        Called by the internal watchdog when the current task has been
        processing longer than ``WATCHDOG_TIMEOUT_S``. The goal is to
        recover the mic-mute interlock without a process restart.

        Safety notes
        ------------
        We do NOT call ``stream.abort()`` or ``stream.close()`` from this
        thread while the worker might still be inside ``stream.write()`` —
        PortAudio's ALSA backend corrupts the heap when that happens
        (observed during development as ``malloc(): unaligned tcache chunk
        detected``). Instead we:

          1. Set the ``_stop`` event so the worker exits between chunks.
          2. Drain the queue (also calls task_done so any caller blocked
             in ``wait()`` unblocks).
          3. Join the worker briefly. If it exits cleanly we own the stream
             and can close it. If it doesn't (genuinely deadlocked inside
             a render call), we abandon it as a leaked daemon thread —
             undesirable but bounded, and far better than corrupting the
             heap or leaving the mic muted forever.
          4. Try to open a fresh stream + worker. If the device is still
             held by the abandoned old stream, this fails and we mark the
             player unavailable so callers degrade gracefully.

        Serialized by ``_reset_lock`` so concurrent calls (e.g., watchdog
        firing twice in quick succession) don't double-rebuild.
        """
        if not self._available:
            return
        with self._reset_lock:
            if not self._available:
                return
            print("[Player] force_reset — draining queue and signaling worker to exit")

            old_worker = self._worker
            old_stream = self._stream

            # Block enqueues, drain pending work, reset the inflight counter.
            with self._inflight_lock:
                self._stop.set()
                while True:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        break
                self._inflight = 0
                self._current_task_started_at = None
                self._current_label = ""

            # Give the chunked worker loop time to notice _stop and exit.
            old_worker.join(timeout=2.0)
            worker_exited = not old_worker.is_alive()

            if worker_exited:
                # Safe to close the old stream — nobody is touching it anymore.
                try:
                    old_stream.stop()
                except Exception:
                    pass
                try:
                    old_stream.close()
                except Exception:
                    pass
            else:
                print("[Player] force_reset: worker did not exit within 2s — "
                      "abandoning it as a leaked daemon thread "
                      "(the underlying render is likely wedged in CUDA / Kokoro)")
                # Note: we leak the old stream too. The new open() below may
                # fail if the OS/ALSA still holds the device — that's why
                # we try anyway and gracefully mark unavailable on failure.

            # Replace _stop BEFORE spawning the new worker so it captures the
            # fresh event. The new worker captures whatever _stop points to at
            # the moment it starts (see _worker_loop's local capture).
            self._stop = threading.Event()

            # Try to rebuild the stream. If the abandoned worker is still
            # holding the device this will fail; mark unavailable so callers
            # degrade gracefully instead of crashing on every enqueue.
            self._stream = self._open_stream(channels=self._channels)
            if self._stream is None:
                print("[Player] force_reset: failed to reopen stream")
                print("[Player] AudioPlayer is now unavailable — restart the process to recover")
                self._available = False
                return

            self._worker = threading.Thread(
                target=self._worker_loop, name="AudioPlayer", daemon=True
            )
            self._worker.start()
            print("[Player] force_reset complete — worker restarted")

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        """Pull tasks from the queue and play them in order.

        Each iteration handles one task. The inflight counter is decremented
        in ``finally`` so a render or playback exception cannot leave
        ``is_busy()`` stuck or the queue permanently un-drained.

        Captures ``self._stop`` once at start so a ``force_reset`` that
        replaces the event still terminates this worker correctly.
        """
        stop = self._stop  # local capture — survives force_reset replacing self._stop
        chunk_samples = max(self.sample_rate * self._CHUNK_MS // 1000, 1)
        while not stop.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print(f"[DBG-worker] picked task={task.label} qsize={self._queue.qsize()} inflight={self._inflight}")
            played_first_chunk = False
            try:
                with self._inflight_lock:
                    self._current_task_started_at = time.monotonic()
                    self._current_label = task.label

                _safe_cb(task.on_render_start)
                samples, rate = self._materialize(task)
                if samples is None:
                    print(f"[DBG-worker] {task.label}: samples=None (render failed) -> skip")
                    _safe_cb(task.on_render_end)
                    continue
                _safe_cb(task.on_render_end)
                samples = self._resample(samples, rate)
                samples = self._format_for_stream(samples)

                # Chunked write so the worker can react to ``stop`` (set by
                # force_reset/shutdown) within ~100ms instead of blocking for
                # the full duration of the utterance.
                n_chunks_written = 0
                for offset in range(0, len(samples), chunk_samples):
                    if stop.is_set():
                        break
                    if not played_first_chunk:
                        _safe_cb(task.on_play_start)
                        played_first_chunk = True
                    self._stream.write(samples[offset:offset + chunk_samples])
                    n_chunks_written += 1
                print(f"[DBG-worker] {task.label}: wrote {n_chunks_written} chunks ({n_chunks_written * self._CHUNK_MS}ms), stop={stop.is_set()}, samples_total={len(samples)}")
                _safe_cb(task.on_play_end)
            except Exception as e:
                print(f"[Player] {task.label}: playback error: {e}")
                # Stage 2E.2: if playback already started, on_play_end (gripper
                # mouth close) MUST still fire on a write error — otherwise the
                # gripper is left commanding open after on_play_start ran.
                if played_first_chunk:
                    _safe_cb(task.on_play_end)
            finally:
                with self._inflight_lock:
                    self._inflight = max(0, self._inflight - 1)
                    self._current_task_started_at = None
                    self._current_label = ""
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    def _watchdog_loop(self) -> None:
        """Daemon thread that detects wedged tasks and triggers force_reset.

        Polls the current task age every WATCHDOG_POLL_INTERVAL_S. The age is
        per-task (not wall-clock busy time), so a long sequence of legitimate
        utterances does not trigger false positives — only a single task that
        has been processing for too long.

        Runs independently of the main loop, so it can recover from a wedge
        that happens *inside* ``process_utterance`` (the original failure
        shape that motivated this whole refactor).
        """
        while not self._watchdog_stop.wait(timeout=WATCHDOG_POLL_INTERVAL_S):
            if not self._available:
                return
            # Read both fields atomically so the age calculation can't race
            # with the worker clearing _current_task_started_at.
            with self._inflight_lock:
                started_at = self._current_task_started_at
                label = self._current_label
            if started_at is None:
                continue
            age = time.monotonic() - started_at
            if age > WATCHDOG_TIMEOUT_S:
                print(f"[Player-Watchdog] Task '{label}' age={age:.1f}s "
                      f"exceeds {WATCHDOG_TIMEOUT_S:.0f}s — calling force_reset")
                self.force_reset()
                # After force_reset the watchdog continues running against
                # the new worker; the next iteration will see a fresh
                # _current_task_started_at (or None if idle).

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _materialize(self, task: _Task) -> tuple[Optional[np.ndarray], int]:
        """Resolve a task into (samples, rate). Renders if needed."""
        if task.samples is not None:
            return task.samples, task.rate or self.sample_rate
        if task.render_fn is not None:
            try:
                samples, rate = task.render_fn()
            except Exception as e:
                print(f"[Player] {task.label}: render error: {e}")
                return None, 0
            return np.asarray(samples, dtype=np.float32), int(rate)
        return None, 0

    def _resample(self, samples: np.ndarray, rate: int) -> np.ndarray:
        """Linear-interpolation resample to the stream's fixed rate.

        Linear interp is good enough for speech and tones; the artifacts are
        well below what a USB speaker reproduces. We always upsample on this
        hardware (24kHz Kokoro → 48kHz device), so aliasing is not a concern.
        """
        if rate == self.sample_rate:
            return samples.astype(np.float32, copy=False)
        ratio = self.sample_rate / rate
        n_out = int(len(samples) * ratio)
        if n_out <= 0:
            return samples.astype(np.float32, copy=False)
        indices = np.arange(n_out) / ratio
        return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    def _format_for_stream(self, samples: np.ndarray) -> np.ndarray:
        """Reshape mono samples to match the stream's channel count.

        For stereo streams (only used as a fallback when mono open failed),
        duplicate the mono signal across both channels via ``column_stack``.
        """
        if self._channels == 1:
            return samples
        return np.column_stack([samples, samples]).astype(np.float32, copy=False)
