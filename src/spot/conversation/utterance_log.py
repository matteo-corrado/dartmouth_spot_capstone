"""Per-utterance / idle metrics records and the append-only JSONL log
(logs/utterances/<date>.jsonl).

Moved from the old client_mic.py (Task 10), unchanged.
"""
import json
import pathlib
import time

from src.spot.conversation.speaker_lock import mic_to_body_deg
from src.spot.core.tuning import get_tuning

_tuning = get_tuning()

project_root = pathlib.Path(__file__).resolve().parents[3]
UTTERANCE_LOG_DIR = project_root / "logs" / "utterances"


def _body(deg):
    if deg is None:
        return None
    return round(mic_to_body_deg(deg, _tuning.mic_yaw_offset_deg, _tuning.mic_yaw_sign), 1)


def _rnd(x):
    return None if x is None else round(float(x), 1)


def build_utterance_record(endpoint, end_silence_s, duration_s, floor_db, stats,
                           lock_mic_deg, utt_mic_deg, accepted, transcript=None,
                           reached_llm=False, latency_s=None, drop_reason=None,
                           voice_sim=None, short_path=False, rescued=None,
                           rescue_transcript=None) -> dict:
    """One metrics line per utterance (logs/utterances/<date>.jsonl)."""
    return {
        "kind": "utterance",
        "ts": time.time(),
        "floor_dbfs": _rnd(floor_db),
        "speech_dbfs_median": _rnd(stats.speech_db_median),
        "speech_dbfs_p90": _rnd(stats.speech_db_p90),
        "speech_dbfs_peak": _rnd(stats.speech_db_peak),
        "snr_db": _rnd(stats.snr_db(floor_db)),
        "speech_s": round(stats.duration_s, 2),
        "short_path": short_path,
        "rescued": rescued,
        "rescue_transcript": rescue_transcript,
        "endpoint": endpoint,
        "end_silence_s": end_silence_s,
        "duration_s": round(duration_s, 2),
        "lock_mic_deg": _rnd(lock_mic_deg),
        "utt_mic_deg": _rnd(utt_mic_deg),
        "lock_body_deg": _body(lock_mic_deg),
        "utt_body_deg": _body(utt_mic_deg),
        "accepted": accepted,
        "drop_reason": drop_reason,
        "transcript": transcript,
        "reached_llm": reached_llm,
        "latency_s": None if latency_s is None else round(latency_s, 2),
        "voice_sim": voice_sim,  # cosine score only; never the embedding
    }


def build_idle_record(state_name, floor_db, frame_db, lock_mic_deg) -> dict:
    return {"kind": "idle", "ts": time.time(), "state": state_name,
            "floor_dbfs": _rnd(floor_db), "frame_dbfs": _rnd(frame_db),
            "lock_mic_deg": _rnd(lock_mic_deg), "lock_body_deg": _body(lock_mic_deg)}


class UtteranceLog:
    """Append-only JSONL metrics. Opens lazily; an I/O error is logged once
    and never reaches the voice loop. store_transcript=False -> null."""

    def __init__(self, log_dir, store_transcript: bool = True):
        self._dir = pathlib.Path(log_dir)
        self.store_transcript = store_transcript
        self._failed = False

    def write(self, record: dict) -> None:
        if self._failed:
            return
        if not self.store_transcript and "transcript" in record:
            record = dict(record, transcript=None)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
            with open(path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError) as e:
            self._failed = True
            print(f"[Metrics] utterance log disabled: {type(e).__name__}: {e}")
