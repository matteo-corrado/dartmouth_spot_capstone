"""Settings feature handlers: persona, volume and TTS backend. Moved verbatim
from src/voice_control/spot_dispatch.py. None of these need a Spot session.

_current_tts_volume_percent stays in src/spot/robot/state.py: the status
snapshot reads it, so it is not only a settings helper.
"""
import os
import re
import threading


def _handle_set_volume(params):
    """Handle set_volume action — does not require a Spot session.

    Accepts {"level": 1-100} (percentage) and applies it to both the TTS
    output gain and the audio-feedback beep gain. The percentage maps
    linearly to internal gain 0.01..1.0 (1.0 = full volume).
    """
    try:
        raw = params.get("level")
        if raw is None:
            print("[Spot] set_volume: missing 'level' param")
            return False
        pct = float(raw)
        # Clamp 1-100 (LLM-facing range)
        pct = max(1.0, min(100.0, pct))
        gain = pct / 100.0

        from src.voice_control.spot_tts import get_tts
        from src.voice_control.audio_feedback import beep
        tts = get_tts()
        if tts is None:
            # The voice pipeline hasn't registered the TTS singleton yet.
            # In production this never happens — set_volume only fires from
            # an LLM action inside process_utterance, which only runs after
            # client_mic.main() has assigned _tts_instance. Handle the
            # degenerate case anyway so we never crash on missing state.
            print("[Spot] set_volume: TTS not initialized — only updating beep volume")
            applied_beep = beep.set_volume(gain)
            print(f"[Spot] ✓ Volume set to {int(pct)}% (tts=skipped, beep={applied_beep:.2f})")
            return True
        applied_tts = tts.set_volume(gain)
        applied_beep = beep.set_volume(gain)
        print(f"[Spot] ✓ Volume set to {int(pct)}% (tts={applied_tts:.2f}, beep={applied_beep:.2f})")
        return True
    except Exception as e:
        print(f"[Spot] ✗ set_volume failed: {e}")
        return False


def _handle_set_backend(params):
    """Switch active TTS backend at runtime. Accepts {"name": "kokoro"|"elevenlabs"}.

    Sets SPOT_TTS_BACKEND so subsequent get_backend() calls route to the new
    backend, then warms the instance so the first post-switch synth doesn't
    pay cold-init latency.
    """
    name = (params or {}).get("name", "").strip().lower()
    if name not in {"kokoro", "elevenlabs"}:
        print(f"[Spot] set_backend: invalid name '{name}' (expected 'kokoro' or 'elevenlabs')")
        return False
    os.environ["SPOT_TTS_BACKEND"] = name
    try:
        from src.voice_control.tts import get_backend, get_active_backend_name
        get_backend()  # lazy init / cache warm
        active = get_active_backend_name()
        print(f"[Spot] ✓ TTS backend switched to '{active}'")
        return True
    except Exception as e:
        print(f"[Spot] ✗ set_backend warm failed: {e}")
        return False


def do_set_persona(params: dict, brain) -> dict:
    """Switch active persona via runtime action.

    Params:
        name: persona registry key (e.g. "pirate", "butler", "tour_guide").

    Unknown name → auto-upgrade to add_persona with an LLM-generated voice
    description, instead of the prior silent fallback to tour_guide. This
    salvages the common case where the LLM emits set_persona for a name
    that isn't registered yet (e.g. "cowboy", "wizard").
    """
    name = (params or {}).get("name", "").strip().lower()
    if not name:
        return {"ok": False, "error": "set_persona requires 'name' param"}
    if brain is None:
        return {"ok": False, "error": "set_persona requires a brain instance"}
    if name not in brain._persona_registry:
        print(f"[Dispatch] Unknown persona '{name}' — auto-upgrading to add_persona")
        description = brain.describe_persona(name) if hasattr(brain, "describe_persona") else f"{name} character voice"
        return do_add_persona({"name": name, "description": description}, brain)
    from src.voice_control.chatbot.persona import get_persona
    persona = get_persona(name, brain._persona_registry)
    brain.session_state.current_persona = persona.name
    # Stage 2E.2: scale gripper-mouth swing to this persona.
    try:
        from src.voice_control.spot_tts import get_tts
        _t = get_tts()
        if _t is not None and _t.mouth is not None:
            _t.mouth.intensity = getattr(persona, "mouth_intensity", 1.0)
    except Exception:
        pass
    brain.warm_up_async()  # dedup'd bg warm-up; never stacks threads / collides with a live turn
    print(f"[Dispatch] Persona switched to '{persona.name}' (warming new prefix in bg)")
    return {"ok": True, "persona": persona.name}


KOKORO_GENDER_FALLBACK = {"male": "am_fenrir", "female": "af_sarah"}

PROMPT_PREFIX_TEMPLATE = (
    "You are Spot, but in {name} mode. Speak like {description}. "
    "Keep responses short — one or two sentences."
)


def do_add_persona(params: dict, brain) -> dict:
    """Discover + claim ElevenLabs voice matching description, register as persona.
    ACK plays in CURRENT voice; claim runs in parallel worker thread."""
    name = (params.get("name") or "").strip().lower()
    description = (params.get("description") or "").strip()
    if not name or not description:
        return {"ok": False, "error": "add_persona needs name + description"}
    if brain is None:
        return {"ok": False, "error": "add_persona requires a brain instance"}

    # Cache hit — flip without API call
    if name in brain._persona_registry:
        brain.session_state.current_persona = name
        return {"ok": True, "persona": name, "cached": True}

    from src.voice_control.chatbot.persona import get_persona, Persona
    from src.voice_control.chatbot.voice_discovery import find_and_claim
    from src.voice_control.spot_tts import get_tts

    outgoing = get_persona(brain.session_state.current_persona, brain._persona_registry)
    spoken = name.replace("_", " ")
    ack_text = (outgoing.ack_template or "Give me a second.").format(name=spoken)

    tts = get_tts()

    print(f"[Dispatch] add_persona name='{name}' description='{description}'")
    result_box = {"voice_id": None, "error": None}
    def worker():
        try:
            result_box["voice_id"] = find_and_claim(description, name)
        except Exception as e:
            result_box["error"] = f"{type(e).__name__}: {e}"
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    if tts is not None:
        tts.speak(ack_text)  # ACK plays through speakers; claim happens in parallel
    t.join(timeout=15.0)  # ElevenLabs Library search + share can take 8-12s; was 8s (too tight)

    new_voice_id = result_box["voice_id"]
    if not new_voice_id:
        reason = ("worker raised: " + result_box["error"]) if result_box["error"] else (
            "thread still running after 15s budget" if t.is_alive() else "voice library returned no match"
        )
        print(f"[Dispatch] add_persona FAILED for '{name}': {reason}")
        if tts is not None:
            current = brain.session_state.current_persona.replace("_", " ")
            tts.speak(f"I couldn't find a {spoken} voice, so I'll stay as {current}.")
        return {"ok": False, "error": reason}

    # Word-boundary match — substring would misfire ("female" contains "male").
    desc_words = set(re.findall(r"\b\w+\b", description.lower()))
    gender = "male" if desc_words & {"male", "man", "guy", "boy", "men"} else "female"
    kokoro_slug = KOKORO_GENDER_FALLBACK.get(gender, "af_sarah")

    persona = Persona(
        name=name,
        prompt_prefix=PROMPT_PREFIX_TEMPLATE.format(name=name, description=description),
        voices={"elevenlabs": new_voice_id, "kokoro_v1": kokoro_slug},
        sampling_overrides={},
        ack_template=f"One moment while I become a {spoken}.",
    )
    brain._persona_registry[name] = persona
    brain.session_state.current_persona = name
    # describe_persona's side-call evicts the llama-server single-slot cache.
    warm = getattr(brain, "warm_up_async", None)
    if warm is not None:
        warm()
    print(f"[Dispatch] Persona '{name}' added (voice={new_voice_id}, kokoro={kokoro_slug})")
    return {"ok": True, "persona": name, "voice_id": new_voice_id}


# ---------------------------------------------------------------------------
# Registered actions (the legacy _dispatch_core branches; no session)
# ---------------------------------------------------------------------------
def set_persona_action(params, ctx):
    return do_set_persona(params, ctx.brain).get("ok", False)


def add_persona_action(params, ctx):
    return do_add_persona(params, ctx.brain).get("ok", False)


def set_volume_action(params, ctx):
    return _handle_set_volume(params)


def set_backend_action(params, ctx):
    return _handle_set_backend(params)
