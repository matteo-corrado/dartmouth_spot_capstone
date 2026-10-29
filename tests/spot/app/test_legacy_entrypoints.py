"""Task 15: the legacy entry points scripts/run_voice_control.py launches
still start: src/voice_control/client_mic.py and src/voice_control/server.py."""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
GOLDEN = pathlib.Path(__file__).resolve().parent / "golden_help.txt"


def test_legacy_entrypoints_start():
    env = dict(os.environ, COLUMNS="80")
    out = subprocess.run([sys.executable, "src/voice_control/client_mic.py", "--help"],
                         cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout == GOLDEN.read_text()

    # server.py has no CLI (--help would load the ASR model and serve on
    # :50055), so run the shim as __main__ with serve() replaced: exit 0 and
    # the shim reached src.spot.stt.server.serve.
    code = ("import runpy, sys; sys.path.insert(0, '.'); "
            "import src.spot.stt.server as s; "
            "s.serve = lambda: print('SERVE-CALLED'); "
            "runpy.run_path('src/voice_control/server.py', run_name='__main__')")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("SERVE-CALLED")


def test_launcher_still_runs_the_shim_paths():
    """run_voice_control.py launches the same two files, so its
    `pgrep -f voice_control/server.py` check still matches."""
    src = (ROOT / "scripts" / "run_voice_control.py").read_text()
    assert 'VOICE_DIR = PROJECT_ROOT / "src" / "voice_control"' in src
    assert 'str(VOICE_DIR / "server.py")' in src
    assert 'str(VOICE_DIR / "client_mic.py")' in src
    assert '"voice_control/server.py"' in src
