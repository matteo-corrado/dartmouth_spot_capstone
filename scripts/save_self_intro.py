#!/usr/bin/env python3
"""Render Spot's "who are you?" self-description to WAV files (no playback).

Two voices:
  - Kokoro  af_sarah            (local, default persona)
  - ElevenLabs narrator voice   (cloud, narrator persona — needs ELEVENLABS_API_KEY)

Both backends emit 24 kHz signed-16 mono PCM, so we wrap each in a WAV header
with the stdlib `wave` module. Files land in audio_out/.

Run: ./spot-env/bin/python scripts/save_self_intro.py
"""
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

from voice_control.tts.kokoro import KokoroBackend
from voice_control.tts.elevenlabs import ElevenLabsBackend

SAMPLE_RATE = 24000
OUT_DIR = REPO_ROOT / "audio_out"

# What Spot answers to "who are you?" in each persona's voice/style.
KOKORO_VOICE = "af_sarah"
KOKORO_TEXT = (
    "I'm Spot, a Boston Dynamics quadruped robot at Dartmouth's Thayer School "
    "of Engineering, your voice-controlled four-legged assistant who can walk, "
    "navigate, see through my cameras, and lend a hand."
)

ELEVEN_VOICE = "9FB0Xik2befSVRv9JsZ1"  # narrator persona, config/personas.yaml
ELEVEN_TEXT = (
    "I am Spot, a four-legged Boston Dynamics robot roaming the halls of "
    "Dartmouth's Thayer School of Engineering, voice-controlled and ever-ready "
    "to walk, to see, and to assist."
)


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    rc = 0

    # Kokoro (local) — should always work.
    try:
        pcm = KokoroBackend().synthesize(KOKORO_TEXT, KOKORO_VOICE)
        out = OUT_DIR / "spot_intro_kokoro_af_sarah.wav"
        write_wav(out, pcm)
        secs = len(pcm) / 2 / SAMPLE_RATE
        print(f"[kokoro]     OK  {out}  ({secs:.1f}s, {len(pcm)} bytes PCM)")
    except Exception as e:
        rc = 1
        print(f"[kokoro]     FAIL  {type(e).__name__}: {e}")

    # ElevenLabs (cloud) — needs ELEVENLABS_API_KEY + network + quota.
    try:
        pcm = ElevenLabsBackend().synthesize(ELEVEN_TEXT, ELEVEN_VOICE)
        out = OUT_DIR / "spot_intro_elevenlabs_narrator.wav"
        write_wav(out, pcm)
        secs = len(pcm) / 2 / SAMPLE_RATE
        print(f"[elevenlabs] OK  {out}  ({secs:.1f}s, {len(pcm)} bytes PCM)")
    except Exception as e:
        rc = 1
        print(f"[elevenlabs] FAIL  {type(e).__name__}: {e}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
