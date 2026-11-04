"""Pure-DSP loudness envelope for the gripper puppet mouth.

`envelope()` turns a TTS chunk's PCM into a per-frame gripper_open_fraction
array. It holds no mouth or arm state, so it lives in tts/: the speaker
computes frames here and hands them to whatever mouth driver is attached
(features/mouth/driver.py). Deleting the mouth feature folder must not break
TTS import.
"""
import numpy as np

ENVELOPE_FPS = 20    # default frame rate; features/mouth/driver.MOUTH_FPS matches
ENV_RANGE_DB = 12.0  # loudness range (below the chunk's loud level) mapped onto 0..1 opening
ENV_CURVE = 2.0      # >1 keeps the claw mostly shut, opening fully only on loud syllables


def envelope(samples: np.ndarray, rate: int, fps: int = ENVELOPE_FPS,
             gate: float = 0.2, intensity: float = 1.0) -> np.ndarray:
    """PCM float32 in [-1,1] -> per-frame gripper_open_fraction in [0,1].

    Per-frame RMS in dB relative to the chunk's loud level (95th percentile),
    mapped over ENV_RANGE_DB, shaped by ENV_CURVE, noise gate, intensity.
    (Operator 2026-10-29: the old peak-sample envelope kept the claw open for
    most of the sentence; on Kokoro speech this is closed ~50% vs ~18% before.)
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    fps = max(1, int(fps))
    hop = max(1, int(round(rate / fps)))
    n_frames = int(np.ceil(samples.size / hop))
    x = samples.astype(np.float32)
    if x.size < n_frames * hop:  # zero-pad the trailing partial window
        x = np.concatenate([x, np.zeros(n_frames * hop - x.size, dtype=np.float32)])
    rms = np.sqrt((x.reshape(n_frames, hop) ** 2).mean(axis=1))
    voiced = rms > 1e-4
    if not voiced.any():
        return np.zeros(n_frames, dtype=np.float32)
    db = 20.0 * np.log10(rms + 1e-9)
    ref = float(np.percentile(db[voiced], 95))
    frames = np.clip((db - (ref - ENV_RANGE_DB)) / ENV_RANGE_DB, 0.0, 1.0) ** ENV_CURVE
    frames[frames < gate] = 0.0
    return np.clip(frames * intensity, 0.0, 1.0).astype(np.float32)
