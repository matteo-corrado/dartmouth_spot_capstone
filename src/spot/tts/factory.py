"""TextToSpeech factory. The backend (kokoro / elevenlabs) is still chosen
inside SpotTTS from SPOT_TTS_BACKEND (src/spot/tts/__init__.get_backend),
exactly as today; this only builds the speaker on the shared player."""


def make_tts(player, volume: float, output_device=None, cls=None):
    """SpotTTS on `player`, or None when it is not available. Registers the
    live instance as the module singleton (the set_volume action finds it
    there). `cls` lets main pass its SpotTTS name (the golden harness fakes it)."""
    if cls is None:
        from src.spot.tts.speaker import SpotTTS as cls
    print("\nInitializing TTS...")
    tts = cls(player=player, volume=volume)
    if tts.is_available():
        print(f"[TTS] Ready (output device: {output_device}, "
              f"volume={volume}, mic will mute during playback)")
    else:
        print("[TTS] Not available (kokoro-onnx model missing). Continuing without speech output.")
        tts = None
    # Register the live TTS instance as the module singleton so the LLM
    # set_volume action (in spot_dispatch._handle_set_volume) can find it.
    if tts is not None:
        import src.voice_control.spot_tts as _spot_tts_mod
        _spot_tts_mod._tts_instance = tts
    return tts
