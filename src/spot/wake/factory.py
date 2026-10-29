"""WakeDetector factory: SPOT_WAKE_BACKEND exactly as today."""
import os


def make_wake(tuning=None):
    """Wake backend from SPOT_WAKE_BACKEND; the sherpa-onnx spotter gets the
    peak-safe gain target from Tuning.kws_peak_target."""
    if tuning is None:
        from src.spot.core.tuning import get_tuning
        tuning = get_tuning()
    if os.environ.get("SPOT_WAKE_BACKEND", "sherpa_onnx").lower() == "sherpa_onnx":
        from src.voice_control.wake.sherpa_onnx import WakeWordDetector
        return WakeWordDetector(peak_target=tuning.kws_peak_target)
    from src.voice_control.wake import make_wake_detector
    return make_wake_detector()
