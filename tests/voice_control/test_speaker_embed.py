"""Unit tests for the voiceprint module (Task 18). Model tests skip if absent."""
from pathlib import Path

import numpy as np
import pytest

from src.voice_control.speaker_embed import SpeakerEmbedder, accept_combined, cosine, fbank

MODEL = Path(__file__).resolve().parents[2] / "models" / "speaker" / "voxceleb_resnet34_LM.onnx"


def _noise(n=16000, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 3000).astype(np.int16).tobytes()


def test_accept_combined_truth_table():
    assert accept_combined(True, 0.7, 0.5) is True
    assert accept_combined(True, 0.3, 0.5) is False
    assert accept_combined(None, 0.6, 0.5) is True
    assert accept_combined(False, None, 0.5) is False
    assert accept_combined(None, None, 0.5) is True
    assert accept_combined(False, 0.9, 0.5) is False
    assert accept_combined(True, None, 0.5) is True
    assert accept_combined(None, 0.4, 0.5) is False


def test_fbank_shape():
    f = fbank(_noise())
    assert f.shape[1] == 80 and 95 <= f.shape[0] <= 100 and abs(float(f.mean())) < 1e-3
    assert f.dtype == np.float32


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 2]) == pytest.approx(0.0)
    assert cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)


class _FakeSession:
    def __init__(self):
        self.fed = None

    def run(self, _outputs, feeds):
        self.fed = feeds
        return [np.full((1, 256), 3.0, dtype=np.float32)]


def test_embed_with_fake_session_normalises():
    emb = SpeakerEmbedder.__new__(SpeakerEmbedder)  # bypass onnxruntime/model load
    emb._sess, emb._input = _FakeSession(), "feats"
    e = emb.embed(_noise())
    assert e.shape == (256,) and np.linalg.norm(e) == pytest.approx(1.0)
    feats = emb._sess.fed["feats"]
    assert feats.ndim == 3 and feats.shape[0] == 1 and feats.shape[2] == 80


@pytest.mark.skipif(not MODEL.exists(), reason="wespeaker model not set up (run scripts/setup_wespeaker.py)")
def test_embed_shape_and_norm():
    e = SpeakerEmbedder(MODEL).embed(_noise(32000))
    assert e.shape == (256,) and np.linalg.norm(e) == pytest.approx(1.0, abs=1e-4)
