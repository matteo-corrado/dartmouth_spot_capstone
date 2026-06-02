# tests/voice_control/animation/test_mouth.py
import numpy as np
from src.voice_control.animation.mouth import envelope, MOUTH_FPS


def _tone(freq, secs, rate=24000, amp=0.8):
    t = np.arange(int(secs * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_silence_keeps_gripper_closed():
    env = envelope(np.zeros(24000, dtype=np.float32), rate=24000, fps=20)
    assert env.shape[0] == 20  # 1s at 20 fps
    assert float(env.max()) == 0.0


def test_tone_opens_gripper():
    env = envelope(_tone(200, 1.0), rate=24000, fps=20)
    assert float(env.max()) > 0.9          # normalized to peak
    assert float(env.max()) <= 1.0


def test_gate_cuts_inter_word_silence():
    voiced = _tone(200, 0.5)
    silent = np.zeros(int(0.5 * 24000), dtype=np.float32)
    env = envelope(np.concatenate([voiced, silent]), rate=24000, fps=20)
    assert float(env[:10].max()) > 0.5     # first 0.5s voiced
    assert float(env[10:].max()) == 0.0    # last 0.5s gated to closed


def test_intensity_scales_and_clamps():
    base = envelope(_tone(200, 1.0), rate=24000, fps=20, intensity=1.0)
    hot = envelope(_tone(200, 1.0), rate=24000, fps=20, intensity=2.0)
    assert float(hot.max()) <= 1.0         # clamped
    # a mid-range frame should scale up under higher intensity
    mid_i = int(np.argmin(np.abs(base - 0.4)))
    assert hot[mid_i] >= base[mid_i]


import time
from src.voice_control.animation.mouth import MouthDriver


def test_driver_dry_run_emits_all_frames_in_order():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    fractions = [0.0, 0.5, 1.0, 0.5, 0.0]
    d.play(fractions, t0=time.monotonic())
    d.wait()
    assert [round(f, 3) for _, f in d.log] == fractions


def test_driver_dry_run_paces_to_clock():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    t0 = time.monotonic()
    d.play([0.0, 0.5, 1.0], t0=t0)
    d.wait()
    for i, (t_emit, _) in enumerate(d.log):
        assert abs((t_emit - t0) - i / 10.0) < 0.05


def test_disabled_driver_is_noop():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)  # starts disabled
    d.play([1.0, 1.0, 1.0], t0=time.monotonic())
    d.wait()
    assert d.log == []


def test_close_forces_zero():
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    d.close()
    d.wait()  # close() is async (worker thread issues the 0.0)
    assert d.log[-1][1] == 0.0


def test_close_supersedes_running_play():
    # A close() mid-play must drop remaining frames and end on 0.0 — the
    # safety path (stop/estop) relies on this.
    d = MouthDriver(cmd_client=None, fps=10, dry_run=True)
    d.enable()
    d.play([1.0] * 50, t0=time.monotonic())  # 5s of frames
    time.sleep(0.05)
    d.close()
    d.wait()
    assert d.log[-1][1] == 0.0
    assert len(d.log) < 50  # superseded, not all frames sent
