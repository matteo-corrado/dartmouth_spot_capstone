"""Speech-to-text via the Dartmouth speech-recognition REST API.

Replaces the former Riva gRPC ASR bridge (server.py).

Usage:
    from dartmouth_stt import transcribe
    transcript = transcribe(pcm_bytes, sample_rate=16000)
"""
import io
import wave
import time
from collections import Counter

import requests

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import DARTMOUTH_STT_URL
from dartmouth_auth import get_auth


# ============================================================================
# Post-processing filters (carried over from the former server.py)
# ============================================================================
HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end", ".", "...",
}

MIN_AUDIO_DURATION = 0.15  # seconds — reject clicks / bumps


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    """True if one word makes up > *threshold* of all words."""
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count / len(words) > threshold


# ============================================================================
# Core transcription
# ============================================================================
def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 mono audio in a WAV container (in-memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Send PCM16 audio to Dartmouth STT and return the transcript.

    Returns an empty string on failure or if the audio is too short /
    is a known hallucination.
    """
    duration = len(pcm_bytes) / (2 * sample_rate)
    if duration < MIN_AUDIO_DURATION:
        print(f"[STT] Too short ({duration:.2f}s) — skipping")
        return ""

    wav_bytes = _pcm16_to_wav(pcm_bytes, sample_rate)

    auth = get_auth()

    # -- attempt with current JWT; retry once on 401 (expired token) --
    for attempt in range(2):
        headers = auth.auth_header()
        try:
            t0 = time.time()
            resp = requests.post(
                DARTMOUTH_STT_URL,
                headers=headers,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                timeout=15,
            )
            elapsed = time.time() - t0

            if resp.status_code == 401 and attempt == 0:
                print("[STT] JWT expired — refreshing...")
                auth._refresh_jwt()
                continue

            resp.raise_for_status()
            data = resp.json()

            # The endpoint may return {"text": "..."} or {"transcript": "..."}
            text = (
                data.get("text", "")
                or data.get("transcript", "")
                or data.get("result", "")
            ).strip()

            rtf = duration / elapsed if elapsed > 0 else 0
            print(f"[STT] {duration:.1f}s audio -> {elapsed:.2f}s request ({rtf:.1f}x RT)")

            # -- post-processing filters --
            if text and text.strip(".!?, ").lower() in HALLUCINATION_PHRASES:
                print(f'[STT] Hallucination "{text}" — discarding')
                return ""
            if text and _is_repetitive(text):
                print(f'[STT] Repetitive "{text[:60]}..." — discarding')
                return ""

            return text

        except requests.RequestException as exc:
            print(f"[STT] Request error: {exc}")
            return ""

    return ""
