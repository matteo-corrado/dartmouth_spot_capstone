"""WakeWordDetector gain modes, using stubs (no model)."""
import numpy as np

from src.voice_control.wake.sherpa_onnx import WakeWordDetector


class _Stream:
    def __init__(self):
        self.peaks = []

    def accept_waveform(self, rate, samples):
        self.peaks.append(float(np.max(np.abs(samples))))


class _Spotter:
    def is_ready(self, stream):
        return False

    def get_result(self, stream):
        return ""


def _detector(tmp_path, **kw):
    det = WakeWordDetector(model_dir=tmp_path, **kw)  # empty dir -> unavailable
    det._available = True
    det._spotter = _Spotter()
    det._stream = _Stream()
    return det


def _frame(peak):
    return (np.full(160, peak) * 32767).astype(np.int16).tobytes()


def test_default_is_peak_safe(tmp_path):
    det = _detector(tmp_path)
    det.process_frame(_frame(0.2))
    assert abs(det._stream.peaks[0] - 0.9) < 0.01  # 0.2 * 4.5, under the x8 cap


def test_custom_peak_target(tmp_path):
    det = _detector(tmp_path, peak_target=0.5)
    det.process_frame(_frame(0.2))
    assert abs(det._stream.peaks[0] - 0.5) < 0.01


def test_legacy_fixed_gain(tmp_path):
    det = _detector(tmp_path, peak_target=0)
    det.process_frame(_frame(0.05))
    assert abs(det._stream.peaks[0] - 0.4) < 0.01  # fixed x8
    det.process_frame(_frame(0.2))
    assert det._stream.peaks[1] == 1.0  # clipped, not peak-limited
