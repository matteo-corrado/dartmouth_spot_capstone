import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_MODS = [
    ("asr", "backends"),
    ("asr.parakeet", "backends.parakeet"),
    ("asr.nemotron", "backends.nemotron"),
    ("asr_pb2", "asr_pb2"),
    ("asr_pb2_grpc", "asr_pb2_grpc"),
]


@pytest.mark.parametrize("old_name,new_name", _MODS)
def test_alias_identity(old_name, new_name):
    old = importlib.import_module(f"src.voice_control.{old_name}")
    new = importlib.import_module(f"src.spot.stt.{new_name}")
    assert old is new


def test_bare_pb2_identity():
    import asr_pb2
    import asr_pb2_grpc

    new = importlib.import_module("src.spot.stt.asr_pb2")
    new_grpc = importlib.import_module("src.spot.stt.asr_pb2_grpc")
    assert asr_pb2 is new
    assert asr_pb2_grpc is new_grpc
    assert new_grpc.asr__pb2 is new


def test_server_script_help():
    # server.py has no argparse, so --help would start the gRPC server on 50055.
    # Check the script shim parses and the moved server imports instead.
    shim = ROOT / "src" / "voice_control" / "server.py"
    compile(shim.read_text(), str(shim), "exec")
    server = importlib.import_module("src.spot.stt.server")
    assert callable(server.serve)
    r = subprocess.run(
        [sys.executable, "-c",
         "import runpy,sys; sys.argv=['server.py']; "
         "import src.spot.stt.server as s; s.serve=lambda: print('SERVE_CALLED'); "
         f"runpy.run_path({str(shim)!r}, run_name='__main__')"],
        cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    assert "SERVE_CALLED" in r.stdout
