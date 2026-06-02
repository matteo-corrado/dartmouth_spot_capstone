"""Stage 2E.2 — gripper puppet mouth.

`envelope()` turns a TTS chunk's PCM into a per-frame gripper_open_fraction
array. `MouthDriver` paces those fractions to the arm in sync with playback.

MOUTH_FPS is set from the Task 0 feasibility spike — the proven smooth
gripper command rate on our Spot. Default 20; lower it if the spike found a
lower ceiling.
"""
import numpy as np
from scipy.signal import butter, lfilter

MOUTH_FPS = 20  # set from spike_gripper_rate.py measurement


def envelope(samples: np.ndarray, rate: int, fps: int = MOUTH_FPS,
             gate: float = 0.06, intensity: float = 1.0,
             cutoff_hz: float = 50.0) -> np.ndarray:
    """PCM float32 in [-1,1] -> per-frame gripper_open_fraction in [0,1].

    rectify -> lowpass -> window-max downsample to fps -> normalize by peak
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
        self._lock = threading.Lock()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.close()

    def _send(self, frac: float):
        frac = float(max(0.0, min(1.0, frac)))
        if self.dry_run:
            self.log.append((_time.monotonic(), frac))
            return
        if self._cmd is None or RobotCommandBuilder is None:
            return
        self._cmd.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(frac))

    def play(self, fractions, t0: float):
        if not self.enabled:
            return
        self._cancel()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(list(fractions), t0, self._stop),
            daemon=True)
        self._thread.start()

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

    def _cancel(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._stop.set()
                self._thread.join(timeout=1.0)

    def wait(self):
        if self._thread:
            self._thread.join(timeout=5.0)

    def close(self):
        """Force gripper closed and cancel any in-flight playback."""
        self._cancel()
        self._send(0.0)
