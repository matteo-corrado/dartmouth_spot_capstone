"""Moved to src/spot/brain/ and src/spot/conversation/. Compatibility aliases; removed in Task 19."""
import importlib
import sys

for _sub, _new in (
    ("persona", "src.spot.brain.persona"),
    ("voice_discovery", "src.spot.brain.voice_discovery"),
    ("conversation_log", "src.spot.conversation.conversation_log"),
    ("session_state", "src.spot.conversation.session_state"),
):
    _mod = importlib.import_module(_new)
    sys.modules[f"{__name__}.{_sub}"] = _mod
    globals()[_sub] = _mod
