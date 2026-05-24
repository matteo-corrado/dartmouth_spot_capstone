import os
from unittest.mock import MagicMock, patch
import pytest


def test_elevenlabs_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    from src.voice_control.tts.elevenlabs import ElevenLabsBackend
    with pytest.raises(RuntimeError) as exc:
        ElevenLabsBackend()
    assert "ELEVENLABS_API_KEY" in str(exc.value)


def test_synthesize_calls_sdk_with_flash_model(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    with patch("src.voice_control.tts.elevenlabs.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.text_to_speech.stream.return_value = iter([b"chunk1", b"chunk2"])

        from src.voice_control.tts.elevenlabs import ElevenLabsBackend
        b = ElevenLabsBackend()
        pcm = b.synthesize("hello", "voice123")
        assert pcm == b"chunk1chunk2"
        call_kwargs = instance.text_to_speech.stream.call_args.kwargs
        assert call_kwargs["model_id"] == "eleven_flash_v2_5"
        assert call_kwargs["voice_id"] == "voice123"
        assert call_kwargs["output_format"] == "pcm_24000"


def test_stream_yields_chunks(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    with patch("src.voice_control.tts.elevenlabs.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.text_to_speech.stream.return_value = iter([b"a", b"b", b"c"])

        from src.voice_control.tts.elevenlabs import ElevenLabsBackend
        b = ElevenLabsBackend()
        chunks = list(b.stream("hi", "voice123"))
        assert chunks == [b"a", b"b", b"c"]
