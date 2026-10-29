"""scripts/setup_vision_models.py: exit status reflects the TensorRT warm-build (D1)."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
RT = importlib.import_module("src.voice_control.vision.runtime")


def _script():
    spec = importlib.util.spec_from_file_location("setup_vision_models",
                                                  ROOT / "scripts" / "setup_vision_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def out(tmp_path):
    for m in ("yolo26l-seg", "yolo26l-pose", "yoloe-26l-seg"):
        (tmp_path / f"{m}.onnx").write_bytes(b"")
    return tmp_path


def test_all_trt_exits_zero(monkeypatch, out):
    monkeypatch.setattr(RT, "load_model", lambda m, model_dir=None: SimpleNamespace(provider="trt"))
    assert _script().main(["--out", str(out)]) == 0


def test_fallback_off_trt_warns_and_exits_nonzero(monkeypatch, out, capsys):
    monkeypatch.setattr(RT, "load_model", lambda m, model_dir=None: SimpleNamespace(
        provider="cuda" if m == "yolo26l-pose" else "trt"))
    assert _script().main(["--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "WARNING" in err and "yolo26l-pose=cuda" in err


def test_no_warm_skips_check(monkeypatch, out):
    monkeypatch.setattr(RT, "load_model", lambda *a, **k: pytest.fail("no warm"))
    assert _script().main(["--out", str(out), "--no-warm"]) == 0
