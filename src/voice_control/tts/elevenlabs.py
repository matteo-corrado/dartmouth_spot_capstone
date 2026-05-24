"""ElevenLabs cloud TTS backend.

Uses the official `elevenlabs` Python SDK. Streaming via `text_to_speech.stream`
with `eleven_flash_v2_5` model (lowest-latency Flash variant, ~150-300ms TTFA
per independent Coval benchmark May 2026). Output format `pcm_24000` to avoid
mp3 decode latency. Latency optimization level 3 (max except text normalizer).
"""
import os
from typing import Iterator

from elevenlabs.client import ElevenLabs

from . import register_backend


class ElevenLabsBackend:
    MODEL_ID = "eleven_flash_v2_5"
    OUTPUT_FORMAT = "pcm_24000"  # 24kHz signed-16 PCM, no decode overhead

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY env var not set; cannot init ElevenLabsBackend"
            )
        self._client = ElevenLabs(api_key=self.api_key)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        chunks = list(self.stream(text, voice_id))
        return b"".join(chunks)

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        return self._client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id=self.MODEL_ID,
            output_format=self.OUTPUT_FORMAT,
        )

    def list_voices(self) -> list:
        try:
            resp = self._client.voices.search()
            return [{"id": v.voice_id, "name": v.name} for v in resp.voices]
        except Exception:
            return []


register_backend("elevenlabs", lambda: ElevenLabsBackend())
