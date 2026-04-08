"""Boot-time status assessment for the voice pipeline.

Builds a deterministic 2–3 sentence technical assessment of the robot at
startup, optionally polished by the LLM brain. Called from
``client_mic.main()`` between ``stream.start()`` and the LISTENING banner.

Design
------
The template path (``build_template_status``) is the source of truth — it
knows about every field and warning condition. The LLM polish (when
enabled via ``brain.startup_status``) only rewords the template into a
friendlier voice; it cannot soften away a real warning because the
technical facts come from the dict.

When the LLM call fails or is disabled, the template string is spoken
directly. So a degraded boot still produces a useful greeting.
"""
from __future__ import annotations

from typing import Any, Dict, List


# Warning thresholds — tweak here, not at the call sites.
LOW_BATTERY_PCT = 25
LOW_DISK_GB = 3.0


def collect_health(brain, tts, args, audio_player) -> Dict[str, Any]:
    """Snapshot subsystem health from objects already in client_mic.main() scope.

    No new probes — reads existing flags and ``is_available()`` results.

    Args:
        brain: SpotBrain instance or None (when --no-brain or unavailable).
        tts:   SpotTTS instance or None (when --no-tts or unavailable).
        args:  argparse Namespace from client_mic.main().
        audio_player: AudioPlayer instance (always non-None at this point).

    Returns:
        Dict with boolean health flags. Used by ``build_template_status``
        to decide which warnings to surface.
    """
    return {
        "brain_ok": brain is not None and brain.is_available(),
        "tts_ok": tts is not None and tts.is_available(),
        "vlm_warmed": bool(brain and getattr(brain, "_vlm_warmed", False)),
        "audio_player_ok": audio_player is not None and audio_player.is_available(),
        "map_requested": bool(getattr(args, "map", None)),
    }


def _battery_clause(state: Dict[str, Any]) -> str:
    pct = state.get("battery_percent")
    if pct == "unknown" or pct is None:
        return ""
    runtime = state.get("estimated_runtime_minutes")
    if isinstance(runtime, int) and runtime > 0:
        return f"battery at {pct}% (about {runtime} minutes runtime)"
    return f"battery at {pct}%"


def _map_clause(state: Dict[str, Any]) -> str:
    cm = state.get("current_map", "unknown")
    if cm in ("unknown", None, ""):
        return ""
    if state.get("current_map_localized", False):
        return f"loaded on the {cm} map and localized"
    return f"loaded on the {cm} map but not localized"


def _estop_clause(state: Dict[str, Any]) -> str:
    holders = state.get("estop_holders_cut", "")
    if holders:
        return f"the {holders} e-stop is engaged"
    endpoints = state.get("estop_endpoints", "")
    if endpoints and endpoints not in ("unknown", "none"):
        return "all e-stops clear"
    return ""


def _warnings(state: Dict[str, Any], health: Dict[str, Any]) -> List[str]:
    warns: List[str] = []
    pct = state.get("battery_percent")
    if isinstance(pct, (int, float)) and pct < LOW_BATTERY_PCT:
        warns.append(f"battery is low at {int(pct)}%")
    disk = state.get("disk_free_gb")
    if isinstance(disk, (int, float)) and disk < LOW_DISK_GB:
        warns.append(f"only {disk:.1f} GB of disk free")
    if not health.get("brain_ok"):
        warns.append("the LLM brain didn't come up — I'll only respond to safety commands")
    if not health.get("tts_ok"):
        warns.append("text-to-speech is unavailable")
    if health.get("brain_ok") and not health.get("vlm_warmed"):
        warns.append("the vision model isn't warmed up yet")
    if health.get("map_requested") and state.get("current_map") in ("unknown", None, ""):
        warns.append("the requested map didn't load")
    return warns


def build_template_status(state: Dict[str, Any], health: Dict[str, Any]) -> str:
    """Pure-Python deterministic 2–3 sentence boot status.

    Returns a string suitable for both stdout and TTS playback. Drops any
    clause whose data is "unknown" rather than reading the placeholder
    aloud (e.g. won't say "battery at unknown percent").

    Example outputs::

        I'm online, battery at 87%. Loaded on the thayer map and localized,
        all e-stops clear. Standing by — ready when you are.

        Starting up — heads up, battery at 18%. The hardware e-stop is
        engaged. Warnings: battery is low at 18%. I won't move until
        that clears.

        I'm online, battery at 64%. Warnings: the LLM brain didn't come
        up — I'll only respond to safety commands. Standing by.
    """
    warns = _warnings(state, health)

    sentences: List[str] = []

    # Headline
    head_parts: List[str] = []
    head_parts.append("Starting up — heads up" if warns else "I'm online")
    bat = _battery_clause(state)
    if bat:
        head_parts.append(bat)
    sentences.append(", ".join(head_parts) + ".")

    # Map + estop facts
    facts: List[str] = []
    mp = _map_clause(state)
    if mp:
        facts.append(mp)
    es = _estop_clause(state)
    if es:
        facts.append(es)
    if facts:
        # Capitalize first letter of the joined fact sentence.
        joined = ", ".join(facts)
        sentences.append(joined[0].upper() + joined[1:] + ".")

    # Warnings
    if warns:
        sentences.append("Warnings: " + "; ".join(warns) + ".")

    # Closer
    if state.get("estop_holders_cut", ""):
        sentences.append("I won't move until that clears.")
    elif warns:
        sentences.append("Standing by.")
    else:
        sentences.append("Standing by — ready when you are.")

    return " ".join(sentences)
