import os
from unittest.mock import MagicMock, patch
import pytest


def test_find_and_claim_returns_voice_id_on_match(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    # Reset module-level client singleton between tests
    import src.voice_control.chatbot.voice_discovery as vd
    vd._client = None
    with patch("src.voice_control.chatbot.voice_discovery.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        mock_voice = MagicMock(voice_id="library_voice_abc",
                                public_owner_id="owner_xyz",
                                name="Cowboy Joe")
        instance.voices.get_shared.return_value = MagicMock(voices=[mock_voice])
        instance.voices.share.return_value = MagicMock(voice_id="claimed_voice_123")

        result = vd.find_and_claim("deep gravelly cowboy", "cowboy")
        assert result == "claimed_voice_123"
        instance.voices.get_shared.assert_called_once()
        instance.voices.share.assert_called_once_with(
            public_user_id="owner_xyz",
            voice_id="library_voice_abc",
            new_name="spot_cowboy",
        )


def test_find_and_claim_returns_none_on_no_match(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    import src.voice_control.chatbot.voice_discovery as vd
    vd._client = None
    with patch("src.voice_control.chatbot.voice_discovery.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.voices.get_shared.return_value = MagicMock(voices=[])
        result = vd.find_and_claim("wholly nonsensical persona", "weird")
        assert result is None
        instance.voices.share.assert_not_called()


def test_find_and_claim_returns_none_on_api_failure(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    import src.voice_control.chatbot.voice_discovery as vd
    vd._client = None
    with patch("src.voice_control.chatbot.voice_discovery.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.voices.get_shared.side_effect = Exception("network error")
        result = vd.find_and_claim("cowboy", "cowboy")
        assert result is None


def test_lru_eviction_retries_on_quota_fail(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    import src.voice_control.chatbot.voice_discovery as vd
    vd._client = None
    with patch("src.voice_control.chatbot.voice_discovery.ElevenLabs") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        mock_voice = MagicMock(voice_id="lib_v", public_owner_id="own", name="V")
        instance.voices.get_shared.return_value = MagicMock(voices=[mock_voice])
        # First share raises quota error; eviction succeeds; retry succeeds
        instance.voices.share.side_effect = [
            Exception("quota_exceeded"),
            MagicMock(voice_id="post_evict_id"),
        ]
        old_voice = MagicMock(voice_id="old_v", name="spot_old")
        instance.voices.search.return_value = MagicMock(voices=[old_voice])

        result = vd.find_and_claim("cowboy", "cowboy")
        assert result == "post_evict_id"
        instance.voices.delete.assert_called_once_with(voice_id="old_v")
