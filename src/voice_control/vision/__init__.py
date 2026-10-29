"""Vision package public API (spec 9).

The names below re-export the action functions, so ``vision.describe`` etc. are
functions, not the submodules: import a submodule with
``importlib.import_module("src.voice_control.vision.<name>")`` or
``from src.voice_control.vision.<name> import ...``.
"""
from .describe import describe
from .find import find
from .follow import follow
from .goto import goto
from .people import select_person
from .speaker import refine_speaker_bearing, set_speaker_bearing_getter

__all__ = ["describe", "find", "follow", "goto", "refine_speaker_bearing",
           "select_person", "set_speaker_bearing_getter"]
