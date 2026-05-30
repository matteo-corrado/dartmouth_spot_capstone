# tests/audio/eval_wake_det.py
"""Stage 2F P3 wake DET sweep over the real lab corpus.

For each wav, take the MAX livekit classifier score across its 80ms-step
windows, then sweep thresholds: TPR over positives, false-accepts/hour over the
negative audio. Prints the highest-recall threshold with FAR CI-upper <= target.
Positives: hey_spot_*_lab.wav. Negatives: adv_*_lab.wav + idle_*_lab.wav.
"""
import glob
import json
import math
import pathlib
import sys
import wave

import numpy as np

from src.voice_control.wake.livekit import WINDOW_SAMPLES

CORPUS = pathlib.Path(__file__).resolve().parent
POSITIVE_GLOB = "hey_spot_*_lab.wav"
NEGATIVE_GLOBS = ["adv_*_lab.wav", "idle_*_lab.wav"]
SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * 30 // 1000 * 2     # 30 ms PCM16 frames
TARGET_FAR_PER_HOUR = 1.0


def _read_pcm16(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: expected 16kHz"
        return w.readframes(w.getnframes())


def _frames(pcm):
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


def _max_score(det, pcm):
    det.reset()
    best = 0.0
    # Prepend 2 s+ of silence so even sub-2s clips fill the detector's window and
    # the clip is scored within a full window (matches how the live loop fills).
    padded = bytes(WINDOW_SAMPLES * 2) + pcm
    for f in _frames(padded):
        det.process_frame(f)          # updates det.last_score (reset() cleared it)
        best = max(best, det.last_score)
    return best


def _ci_upper_per_hour(false_accepts, hours):
    if hours <= 0:
        return float("inf")
    if false_accepts == 0:
        return 3.0 / hours
    # Conservative closed-form upper bound (no scipy dep); deliberately looser
    # than a normal approx at low counts so the FAR<=1/hr gate is not passed on
    # a single borderline false-accept.
    return (false_accepts + 1 + 1.96 * math.sqrt(false_accepts + 1)) / hours


def main():
    from src.voice_control.wake.livekit import LiveKitWakeWord
    det = LiveKitWakeWord()
    if not det.is_available():
        raise SystemExit("[eval_wake_det] livekit detector unavailable — deploy the 3 ONNX first")

    pos = sorted(glob.glob(str(CORPUS / POSITIVE_GLOB)))
    neg = sorted(sum([glob.glob(str(CORPUS / g)) for g in NEGATIVE_GLOBS], []))
    pos_scores = [_max_score(det, _read_pcm16(p)) for p in pos]
    neg_seconds, neg_scores = 0.0, []
    for n in neg:
        pcm = _read_pcm16(n)
        neg_seconds += len(pcm) / 2 / SAMPLE_RATE
        neg_scores.append(_max_score(det, pcm))
    hours = neg_seconds / 3600.0

    best = None
    for thr in [round(t, 3) for t in np.linspace(0.05, 0.95, 181)]:
        tpr = sum(s >= thr for s in pos_scores) / len(pos_scores) if pos_scores else 0.0
        fa = sum(s >= thr for s in neg_scores)
        far_ci = _ci_upper_per_hour(fa, hours)
        if far_ci <= TARGET_FAR_PER_HOUR and (best is None or tpr > best["tpr"]):
            best = {"threshold": thr, "tpr": tpr, "false_accepts": fa, "far_ci_upper": round(far_ci, 3)}

    out = {
        "positives": len(pos_scores),
        "negative_hours": round(hours, 4),
        "best_at_far_target": best,
        "note": "negative_hours < 3 => FAR<=1/hr UNMEASURABLE (spec C3); record more idle audio",
    }
    print(json.dumps(out, indent=2))
    if hours < 3.0:
        print(f"[eval_wake_det] WARNING: only {hours:.3f}h negatives — FAR<=1/hr not yet provable.")


if __name__ == "__main__":
    sys.exit(main())
