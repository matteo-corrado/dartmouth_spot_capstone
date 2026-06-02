"""Stage 2E.2 — gripper puppet mouth.

`envelope()` turns a TTS chunk's PCM into a per-frame gripper_open_fraction
array. `MouthDriver` paces those fractions to the arm in sync with playback.

MOUTH_FPS is set from the Task 0 feasibility spike — the proven smooth
gripper command rate on our Spot. Default 20; lower it if the spike found a
lower ceiling.
"""
import numpy as np

MOUTH_FPS = 20  # set from spike_gripper_rate.py measurement


def envelope(samples: np.ndarray, rate: int, fps: int = MOUTH_FPS,
             gate: float = 0.06, intensity: float = 1.0) -> np.ndarray:
    """PCM float32 in [-1,1] -> per-frame gripper_open_fraction in [0,1].

    rectify -> window-max downsample to fps -> normalize by peak
    -> noise gate -> intensity scale + clamp.
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    rect = np.abs(samples.astype(np.float32))
    hop = max(1, int(round(rate / fps)))
    n_frames = int(np.ceil(rect.size / hop))
    frames = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frames[i] = rect[i * hop:(i + 1) * hop].max()
    peak = float(frames.max())
    if peak > 0:
        frames = frames / peak
    frames[frames < gate] = 0.0
    return np.clip(frames * intensity, 0.0, 1.0).astype(np.float32)


import threading
import time as _time

try:
    from bosdyn.client.robot_command import RobotCommandBuilder
except Exception:  # bosdyn not importable in unit-test env
    RobotCommandBuilder = None


class MouthDriver:
    """Paces gripper_open_fraction commands to the arm in sync with playback.

    Disabled by default (arm-safety). `enable()` is called by the
    enable_mouth dispatch handler AFTER the arm is deployed. `dry_run`
    records fractions to `self.log` instead of commanding the robot.
    """

    def __init__(self, cmd_client=None, fps: int = MOUTH_FPS,
                 dry_run: bool = False):
        self._cmd = cmd_client
        self.fps = fps
        self.dry_run = dry_run
        self.enabled = False
        self.intensity = 1.0
        self.log = []  # list[(monotonic_ts, fraction)] — dry_run only
        self._stop = threading.Event()
        self._thread = None
        self._gen = 0  # bumped by play()/close() to supersede stale threads
        self._lock = threading.Lock()

    def enable(self):
        with self._lock:
            self.enabled = True

    def disable(self):
        with self._lock:
            self.enabled = False
        self.close()

    def _send(self, frac: float, gen=None):
        # Drop sends from a superseded generation so a stale paced thread
        # cannot re-open the gripper after a close()/new play() — even if its
        # join timed out. gen=None (close's 0.0 command) always sends.
        if gen is not None:
            with self._lock:
                if gen != self._gen:
                    return
        frac = float(max(0.0, min(1.0, frac)))
        if self.dry_run:
            self.log.append((_time.monotonic(), frac))
            return
        if self._cmd is None or RobotCommandBuilder is None:
            return
        self._cmd.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(frac))

    def play(self, fractions, t0: float):
        # Whole supersede-and-start sequence under the lock so a concurrent
        # close()/disable() from a stop/estop handler thread cannot interleave
        # and leave the gripper open.
        with self._lock:
            if not self.enabled:
                return
            self._gen += 1
            gen = self._gen
            self._stop.set()              # signal any prior thread to quit
            self._stop = threading.Event()
            stop = self._stop
            t = threading.Thread(
                target=self._run, args=(list(fractions), t0, stop, gen),
                daemon=True)
            self._thread = t
            t.start()

    def _run(self, fractions, t0, stop, gen):
        for i, frac in enumerate(fractions):
            if stop.is_set():
                return
            target = t0 + i / self.fps
            now = _time.monotonic()
            if target > now:
                stop.wait(target - now)
            if stop.is_set():
                return
            self._send(frac, gen)

    def wait(self):
        with self._lock:
            t = self._thread
        if t:
            t.join(timeout=5.0)

    def close(self):
        """Force gripper closed + supersede any in-flight playback. The gen
        bump guarantees a stale thread's late frame is dropped even if its
        join times out."""
        with self._lock:
            self._gen += 1
            self._stop.set()
            old = self._thread
            self._thread = None
        if old and old.is_alive():
            old.join(timeout=1.0)
        self._send(0.0)
