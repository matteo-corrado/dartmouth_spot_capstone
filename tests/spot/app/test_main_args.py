"""app.main keeps client_mic's CLI byte-identical (golden captured on the
base commit ebed8d0 with COLUMNS=80: client_mic.py --help)."""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
GOLDEN = pathlib.Path(__file__).resolve().parent / "golden_help.txt"


def test_argparse_identical(monkeypatch, capsys):
    from src.spot.app.main import main
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(sys, "argv", ["client_mic.py"])  # argparse prog, as when run as a script
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == GOLDEN.read_text()


def test_shim_script_help_identical():
    """The old entry point still runs: python src/voice_control/client_mic.py."""
    env = dict(os.environ, COLUMNS="80")
    out = subprocess.run([sys.executable, str(ROOT / "src" / "voice_control" / "client_mic.py"), "--help"],
                         cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout == GOLDEN.read_text()
