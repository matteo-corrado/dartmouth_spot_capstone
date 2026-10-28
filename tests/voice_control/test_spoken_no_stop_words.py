"""Spot's own spoken lines must never contain stop/halt/freeze/e-stop: the
always-on safety KWS hears Spot's voice and would halt the robot."""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "voice_control"
BAD = re.compile(r"\b(stop|halt|freez\w*|e-?stop|estop)\b", re.IGNORECASE)
_SPEAK_FUNCS = {"_say", "speak"}


def _strings(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.value


def _spoken_literals(path):
    tree = ast.parse(path.read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name in _SPEAK_FUNCS:
                for a in list(n.args) + [k.value for k in n.keywords]:
                    out.extend(_strings(a))
        elif isinstance(n, ast.Assign):
            targets = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if any(t.endswith("_LINE") or t == "_ESTOP_SPOKEN" for t in targets):
                out.extend(_strings(n.value) if not isinstance(n.value, ast.Dict)
                           else [s for v in n.value.values for s in _strings(v)])
    return out


def test_scanner_finds_lines():
    lines = _spoken_literals(ROOT / "spot_dispatch.py")
    assert any("battery" in s for s in lines) and any("released" in s for s in lines)


def test_no_stop_words_in_dispatcher_or_client_spoken_lines():
    for fname in ("spot_dispatch.py", "client_mic.py"):
        bad = [s for s in _spoken_literals(ROOT / fname) if BAD.search(s)]
        assert bad == [], f"{fname}: {bad}"


def test_vlm_prompt_forbids_stop_words():
    src = (ROOT / "llm_brain.py").read_text()
    i = src.index("def query_vlm")
    assert "'stop', 'halt', 'freeze'" in src[i:i + 2000]
