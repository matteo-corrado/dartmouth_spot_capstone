"""Every operator script must still import cleanly after the src.spot port.

Scripts are imported (not run) in a subprocess with run_name='not_main', so
their ``if __name__ == "__main__"`` blocks never fire. Nothing here opens the
mic, the XVF3800, the robot or a port.
"""
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    p for p in (REPO / "scripts").glob("*.py") if p.name != "__init__.py"
)

# Scripts that touch hardware or the network at import time: name -> reason.
# Empty on purpose: every script guards its work under __main__ and only
# defines functions / imports modules at import time (verified in Task 17).
# A script that starts needing an entry here gets a --help check instead.
IMPORT_SKIP: dict[str, str] = {}

_RUNNER = "import runpy,sys; runpy.run_path(sys.argv[1], run_name='not_main')"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_script_imports(path):
    if path.name in IMPORT_SKIP:
        pytest.skip(IMPORT_SKIP[path.name])
    r = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(path)],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr[-2000:]


def test_scripts_found():
    assert len(SCRIPTS) >= 20


def test_deleted_scripts_stay_deleted():
    for name in ("preflight_2a", "audition_personas",
                 "audition_kokoro_voices", "save_self_intro"):
        assert not (REPO / "scripts" / f"{name}.py").exists()


def test_tests_record_demo_clips_still_pass():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:warnings", "-o", "addopts=",
         "-q", "tests/voice_control/test_record_demo_clips.py"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stdout[-2000:]


def test_script_src_imports_resolve():
    """Function-local ``from src... import`` lines are not run by the import test."""
    import ast
    import importlib.util

    sys.path.insert(0, str(REPO))
    bad = []
    for p in SCRIPTS:
        for n in ast.walk(ast.parse(p.read_text())):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("src"):
                mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names if a.name.startswith("src")]
            if not mods:
                continue
            try:
                if importlib.util.find_spec(mods[0]) is None:
                    bad.append((p.name, mods[0]))
            except ImportError:
                bad.append((p.name, mods[0]))
    assert not bad, bad
