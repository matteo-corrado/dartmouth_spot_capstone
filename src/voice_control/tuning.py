"""Single source of truth for demo tunables. Overrides live in config/tuning.yaml."""
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "tuning.yaml"


@dataclass(frozen=True)
class Tuning:
    mic_channel: int = 0
    min_speech_dbfs: float = -50.0
    min_snr_db: float = 10.0
    onset_frames: int = 3
    end_silence_s: float = 1.5
    end_silence_s_short: float = 1.2
    max_utterance_s: float = 20.0
    floor_window_s: float = 5.0
    floor_percentile: float = 10.0
    floor_rise_db_per_s: float = 1.0
    reopen_delay_s: float = 0.5
    speaker_tail_s: float = 0.15
    kws_score: float = 1.5
    kws_threshold: float = 0.05
    kws_peak_target: float = 0.9  # = wake/sherpa_onnx TARGET_PEAK (authoritative)
    lock_window_deg: float = 35.0
    lock_timeout_s: float = 20.0
    lock_nudge: float = 0.3
    mic_yaw_offset_deg: float = 179.0  # measured on the robot 2026-10-28
    mic_yaw_sign: int = -1             # measured on the robot 2026-10-28
    ring_offset_leds: int = 6          # measured on the device 2026-10-28 (LED0 ~12:30 from front)
    ring_reverse: bool = False
    ring_max_hz: float = 30.0          # LED ring write cap (RingAnimator + direct writes)
    beam_steering: bool = False
    voiceprint_enabled: bool = False
    voiceprint_threshold: float = 0.5
    thinking_filler_s: float = 4.0
    walk_speed_cap: float = 0.5
    dance_clearance_m: float = 1.5
    mouth_lead_s: float = 0.0
    mouth_max_open: float = 0.6
    mouth_attack_s: float = 0.03
    mouth_release_s: float = 0.10
    mouth_rest_open: float = 0.08


def _cast(value, typ):
    if typ is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return typ(value)


def load_tuning(path: Path | None = None) -> Tuning:
    path = Path(path) if path is not None else DEFAULT_PATH
    if not path.exists():
        return Tuning()
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        log.warning("tuning file %s is not a mapping; using defaults", path)
        return Tuning()
    types = {f.name: f.default.__class__ for f in fields(Tuning)}
    kwargs = {}
    for key, value in data.items():
        if key not in types:
            log.warning("tuning: unknown key %r in %s ignored", key, path)
            continue
        kwargs[key] = _cast(value, types[key])
    if kwargs.get("mic_yaw_sign", 1) not in (-1, 1):
        log.warning("tuning: mic_yaw_sign must be -1 or 1, got %r; using 1", kwargs["mic_yaw_sign"])
        kwargs["mic_yaw_sign"] = 1
    return Tuning(**kwargs)


_cached: Tuning | None = None


def get_tuning() -> Tuning:
    global _cached
    if _cached is None:
        _cached = load_tuning()
    return _cached
