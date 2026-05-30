# tests/audio/eval_safety_recall.py
"""Stage 2F P2: safety-phrase recall of the always-on safety KWS.

Positives: stop_lab.wav (+ any *stop*_lab clips). Reports recall and any
false-fires on the idle/adversarial negatives (a safety KWS that fires on
'hot spot' would be a nuisance, but missing 'stop' is the real hazard).
"""
import glob
import json
import pathlib
import sys
import wave

CORPUS = pathlib.Path(__file__).resolve().parent
SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * 30 // 1000 * 2


def _read_pcm16(path):
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def _frames(pcm):
    for i in range(0, len(pcm) - FRAME_BYTES, FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


def main():
    from src.voice_control.wake.safety_kws import make_safety_detector
    det = make_safety_detector()
    if not det.is_available():
        raise SystemExit("[eval_safety_recall] safety KWS unavailable — "
                         "run scripts/setup_safety_kws.py")

    positives = sorted(glob.glob(str(CORPUS / "stop_lab.wav")))
    hits = 0
    for p in positives:
        det.reset()
        if any(det.process_frame(f) for f in _frames(_read_pcm16(p))):
            hits += 1

    negatives = sorted(glob.glob(str(CORPUS / "idle_*_lab.wav")))
    false_fires = 0
    for n in negatives:
        det.reset()
        if any(det.process_frame(f) for f in _frames(_read_pcm16(n))):
            false_fires += 1

    result = {
        "safety_positives": len(positives),
        "safety_recall": (hits / len(positives)) if positives else None,
        "idle_false_fires": false_fires,
        "note": "recall MUST be 1.0 before the over-fire wake ships (spec C1/P3 gate)",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
