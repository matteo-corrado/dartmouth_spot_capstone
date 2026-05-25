"""Runtime ElevenLabs Library voice discovery + claim.

Library search first (voices.get_shared) — fast (~300ms), library claims may
or may not consume voice slots (research conflict; live-verified at install).
voices.share() to claim into account — returns permanent voice_id.
LRU eviction on quota fail: delete oldest spot_-prefixed custom voice, retry once.
Persistent module-level ElevenLabs client = HTTP/2 reuse (saves ~100ms TLS/call).
Voice Design API NOT used — too slow (1-2s+ gen, two-step) for live demo.
"""
import logging
import os
import threading
from typing import Optional

from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

_CLIENT_LOCK = threading.Lock()
_client: Optional[ElevenLabs] = None


def _get_client() -> ElevenLabs:
    global _client
    with _CLIENT_LOCK:
        if _client is None:
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY not set")
            _client = ElevenLabs(api_key=api_key)
        return _client


def _evict_oldest_spot_voice(client: ElevenLabs) -> bool:
    try:
        existing = client.voices.search()
    except Exception as e:
        logger.warning(f"[VoiceDiscovery] voices.search() for eviction failed: {e}")
        return False
    candidates = [v for v in existing.voices if (v.name or "").startswith("spot_")]
    if not candidates:
        return False
    victim = candidates[0]
    try:
        client.voices.delete(voice_id=victim.voice_id)
        logger.info(f"[VoiceDiscovery] evicted {victim.name} ({victim.voice_id})")
        return True
    except Exception as e:
        logger.warning(f"[VoiceDiscovery] delete failed: {e}")
        return False


def find_and_claim(description: str, persona_name: str) -> Optional[str]:
    """Search ElevenLabs Library, claim top match, return new voice_id.
    Returns None on miss / network error / quota exhaustion (after eviction)."""
    try:
        client = _get_client()
    except RuntimeError as e:
        logger.warning(f"[VoiceDiscovery] {e}")
        return None

    try:
        result = client.voices.get_shared(
            search=description,
            sort="trending",
            page_size=5,
        )
    except Exception as e:
        logger.warning(f"[VoiceDiscovery] get_shared failed: {e}")
        return None
    if not result.voices:
        logger.info(f"[VoiceDiscovery] no library match for '{description}'")
        return None

    top = result.voices[0]
    new_name = f"spot_{persona_name}"
    try:
        added = client.voices.share(
            public_user_id=top.public_owner_id,
            voice_id=top.voice_id,
            new_name=new_name,
        )
        return added.voice_id
    except Exception as e:
        msg = str(e).lower()
        if "quota" not in msg and "limit" not in msg:
            logger.warning(f"[VoiceDiscovery] share failed (non-quota): {e}")
            return None
        if not _evict_oldest_spot_voice(client):
            logger.warning("[VoiceDiscovery] eviction failed; cannot retry share")
            return None
        try:
            added = client.voices.share(
                public_user_id=top.public_owner_id,
                voice_id=top.voice_id,
                new_name=new_name,
            )
            return added.voice_id
        except Exception as e2:
            logger.warning(f"[VoiceDiscovery] share retry failed: {e2}")
            return None
