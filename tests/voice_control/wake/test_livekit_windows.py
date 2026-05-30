import numpy as np
from src.voice_control.wake.livekit import embedding_windows, EMBEDDING_WINDOW, EMBEDDING_STRIDE


def test_window_count_and_shape():
    # floor((T-76)/8)+1 windows of (76,32), count derived from the constants
    T = 200
    mel = np.zeros((T, 32), dtype=np.float32)
    win = embedding_windows(mel)
    expected = (T - EMBEDDING_WINDOW) // EMBEDDING_STRIDE + 1
    assert win.shape == (expected, EMBEDDING_WINDOW, 32)


def test_too_few_frames_returns_empty():
    mel = np.zeros((EMBEDDING_WINDOW - 1, 32), dtype=np.float32)
    assert embedding_windows(mel).shape[0] == 0


def test_windows_advance_by_stride():
    mel = np.arange(200 * 32, dtype=np.float32).reshape(200, 32)
    win = embedding_windows(mel)
    # second window starts EMBEDDING_STRIDE frames after the first
    assert np.array_equal(win[1, 0], mel[EMBEDDING_STRIDE])
