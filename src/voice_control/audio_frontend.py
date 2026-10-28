"""Pure audio front-end logic: dBFS, non-ratcheting noise floor, endpointer, SNR gate."""
from collections import deque

import numpy as np


def frame_dbfs(pcm16: bytes) -> float:
    """RMS level of an int16 frame in dBFS; -120 for silence/empty."""
    if not pcm16:
        return -120.0
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float64) / 32768.0
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x * x)))
    if rms <= 1e-6:
        return -120.0
    return max(-120.0, 20.0 * float(np.log10(rms)))


class FloorTracker:
    """Low-percentile noise floor over non-speech frames; falls instantly, rises rate-limited."""

    def __init__(self, window_s, percentile, rise_db_per_s, frame_s=0.03):
        self._buf = deque(maxlen=max(1, int(round(window_s / frame_s))))
        self._pct = percentile
        self._max_rise = rise_db_per_s * frame_s
        self._floor = -120.0

    def seed(self, db):
        self._floor = float(db)
        self._buf.clear()

    @property
    def floor_db(self) -> float:
        return self._floor

    def update(self, db: float, is_speech: bool) -> None:
        if is_speech:
            return
        self._buf.append(float(db))
        target = float(np.percentile(self._buf, self._pct))
        if target <= self._floor:
            self._floor = target
        else:
            self._floor = min(target, self._floor + self._max_rise)


class Endpointer:
    """Frame-level utterance start/end/cap detection."""

    def __init__(self, onset_frames, end_silence_frames, max_frames):
        self._onset = onset_frames
        self._end_sil = end_silence_frames
        self._max = max_frames
        self._reset()

    def _reset(self):
        self._in_speech = False
        self._onset_count = 0
        self._sil = 0
        self._n = 0

    def set_end_silence_frames(self, n):
        self._end_sil = n

    def feed(self, is_speech: bool, db: float, floor_db: float):
        if not self._in_speech:
            self._onset_count = self._onset_count + 1 if is_speech else 0
            if self._onset_count >= self._onset:
                self._in_speech = True
                self._sil = 0
                self._n = self._onset_count
                return "start"
            return None
        self._n += 1
        if not is_speech and db < floor_db + 6:
            self._sil += 1
        else:
            self._sil = 0
        if self._sil >= self._end_sil:
            self._reset()
            return "end"
        if self._n >= self._max:
            self._reset()
            return "cap"
        return None


class UtteranceStats:
    """Collects frames; statistics are over speech-flagged frames only."""

    def __init__(self, frame_s=0.03):
        self._frame_s = frame_s
        self._speech = []

    def add(self, db, is_speech):
        if is_speech:
            self._speech.append(float(db))

    @property
    def speech_db_median(self) -> float:
        return float(np.median(self._speech)) if self._speech else -120.0

    @property
    def speech_db_p90(self) -> float:
        return float(np.percentile(self._speech, 90)) if self._speech else -120.0

    def snr_db(self, floor_db) -> float:
        return self.speech_db_median - floor_db

    @property
    def duration_s(self) -> float:
        return len(self._speech) * self._frame_s


def passes_gate(stats, floor_db, min_speech_dbfs, min_snr_db) -> bool:
    if not stats._speech:
        return False
    return (stats.speech_db_median >= min_speech_dbfs
            and stats.snr_db(floor_db) >= min_snr_db)
