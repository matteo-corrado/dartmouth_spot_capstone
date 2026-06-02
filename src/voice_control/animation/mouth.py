"""Stage 2E.2 — gripper puppet mouth.

`envelope()` turns a TTS chunk's PCM into a per-frame gripper_open_fraction
array. `MouthDriver` paces those fractions to the arm in sync with playback.

MOUTH_FPS is set from the Task 0 feasibility spike — the proven smooth
gripper command rate on our Spot. Default 20; lower it if the spike found a
lower ceiling.

Threading model: all gripper I/O happens on ONE persistent daemon worker
thread fed by a queue. `play()`/`close()` only enqueue (and signal the
current job to cancel) — they NEVER block on a gRPC call or a join. This
keeps the AudioPlayer worker (which calls these from the TTS play callbacks)
and the voice-loop/safety thread (which calls close() on stop/estop) from
ever stalling on the arm command channel.
"""
import queue
import threading
import time as _time

import numpy as np

try:
    from bosdyn.client.robot_command import RobotCommandBuilder
except Exception:  # bosdyn not importable in unit-test env
    RobotCommandBuilder = None

MOUTH_FPS = 20  # set from spike_gripper_rate.py measurement


def envelope(samples: np.ndarray, rate: int, fps: int = MOUTH_FPS,
             gate: float = 0.06, intensity: float = 1.0) -> np.ndarray:
    """PCM float32 in [-1,1] -> per-frame gripper_open_fraction in [0,1].

    rectify -> window-max downsample to fps -> normalize by peak
    -> noise gate -> intensity scale + clamp.
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    fps = max(1, int(fps))
    rect = np.abs(samples.astype(np.float32))
    hop = max(1, int(round(rate / fps)))
    n_frames = int(np.ceil(rect.size / hop))
    pad_len = n_frames * hop
    if rect.size < pad_len:  # zero-pad the trailing partial window
        rect = np.concatenate([rect, np.zeros(pad_len - rect.size, dtype=np.float32)])
    frames = rect.reshape(n_frames, hop).max(axis=1)
    peak = float(frames.max())
    if peak > 0:
        frames = frames / peak
    frames[frames < gate] = 0.0
    return np.clip(frames * intensity, 0.0, 1.0).astype(np.float32)


class MouthDriver:
    """Paces gripper_open_fraction commands to the arm in sync with playback.

    Disabled by default (arm-safety). `enable()` is called by the
    enable_mouth dispatch handler AFTER the arm is deployed. `dry_run`
    records fractions to `self.log` instead of commanding the robot.

    All gripper commands run on a single daemon worker thread; public
    methods are non-blocking.
    """

    def __init__(self, cmd_client=None, fps: int = MOUTH_FPS,
                 dry_run: bool = False):
        self._cmd = cmd_client
        self.fps = max(1, int(fps))
        self.dry_run = dry_run
        self.enabled = False
        self.intensity = 1.0
        self.log = []  # list[(monotonic_ts, fraction)] — dry_run only
        self._lock = threading.Lock()
        self._q = queue.Queue()
        self._cur_stop = threading.Event()  # cancels the in-progress play
        self._worker = threading.Thread(target=self._serve, daemon=True)
        self._worker.start()

    def enable(self):
        with self._lock:
            self.enabled = True

    def disable(self):
        """Stop animating AND latch off, so a queued/next play() cannot
        re-open the gripper (used on the stop/estop safety path)."""
        with self._lock:
            self.enabled = False
        self.close()

    def play(self, fractions, t0: float):
        """Enqueue a paced open/close sequence. Non-blocking. No-op if
        disabled."""
        with self._lock:
            if not self.enabled:
                return
        self._cur_stop.set()  # supersede whatever is animating now
        self._q.put(("play", list(fractions), t0))

    def close(self):
        """Force the gripper closed. Non-blocking: cancels the current play,
        drops any queued plays so the close takes priority, and asks the
        worker to command 0.0."""
        self._cur_stop.set()
        self._drain_queue()
        self._q.put(("close",))

    def _drain_queue(self):
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                return

    def _serve(self):
        while True:
            job = self._q.get()
            try:
                kind = job[0]
                if kind == "close":
                    self._send(0.0)
                elif kind == "play":
                    _, fractions, t0 = job
                    stop = threading.Event()
                    with self._lock:
                        self._cur_stop = stop
                    self._run(fractions, t0, stop)
            finally:
                self._q.task_done()

    def _run(self, fractions, t0, stop):
        for i, frac in enumerate(fractions):
            if stop.is_set():
                return
            target = t0 + i / self.fps
            now = _time.monotonic()
            if target > now:
                stop.wait(target - now)
            if stop.is_set():
                return
            self._send(frac)

    def _send(self, frac: float):
        frac = float(max(0.0, min(1.0, frac)))
        if self.dry_run:
            self.log.append((_time.monotonic(), frac))
            return
        if self._cmd is None or RobotCommandBuilder is None:
            return
        try:
            self._cmd.robot_command(
                RobotCommandBuilder.claw_gripper_open_fraction_command(frac))
        except Exception as e:
            # Never let an arm/comms error kill the worker thread.
            print(f"[MouthDriver] gripper command failed: {e}")

    def wait(self):
        """Block until all queued work has drained (test/diagnostic helper)."""
        self._q.join()
