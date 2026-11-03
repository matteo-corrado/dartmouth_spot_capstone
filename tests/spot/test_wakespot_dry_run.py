"""wakespot --dry-run: no E-Stop claim, no map upload, --dry-run forwarded."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ws(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("wakespot_under_test", ROOT / "scripts" / "wakespot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "LOCKFILE", tmp_path / "wakespot.pid", raising=False)
    monkeypatch.setattr(mod, "_acquire_lockfile", lambda: True)
    monkeypatch.setattr(mod, "_release_lockfile", lambda: None)
    monkeypatch.setattr(mod.signal, "signal", lambda *a: None)
    return mod


def _args(ws, argv):
    return ws._build_parser().parse_args(argv)


def test_new_flags_forwarded(ws):
    cmd = ws._build_voice_control_cmd(
        _args(ws, ["--no-brain", "--no-server", "--latency-out", "x.jsonl"]),
        None)
    assert "--no-brain" in cmd and "--no-server" in cmd
    assert cmd[cmd.index("--latency-out") + 1] == "x.jsonl"
    assert "--dry-run" not in cmd


def test_dry_run_forwards_flag_and_never_a_map(ws):
    cmd = ws._build_voice_control_cmd(_args(ws, ["--dry-run", "jackson"]), Path("/maps/jackson"))
    assert "--dry-run" in cmd
    assert "--map" not in cmd


class _FakeProc:
    pid = 12345
    returncode = 0

    def poll(self):
        return 0


def test_dry_run_main_never_claims_estop_or_resolves_map(ws, monkeypatch):
    def no_robot(*a, **k):
        raise AssertionError("dry run touched the robot")
    monkeypatch.setattr(ws, "estop_session", no_robot)
    monkeypatch.setattr(ws.map_loader, "resolve_map_path", no_robot)
    monkeypatch.setattr(ws.map_loader, "write_last_used", no_robot)
    seen = {}

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        return _FakeProc()
    monkeypatch.setattr(ws.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", ["wakespot", "--dry-run", "jackson"])
    assert ws.main() == 0
    assert "--dry-run" in seen["cmd"] and "--map" not in seen["cmd"]


def test_no_estop_stop_never_raises(ws):
    ws._NoEstop().stop()


def test_without_dry_run_still_claims_estop(ws, monkeypatch):
    called = {}

    class _Ctx:
        def __enter__(self):
            called["estop"] = True
            return {"keepalive": ws._NoEstop()}

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(ws, "estop_session", lambda **k: _Ctx())
    monkeypatch.setattr(ws.subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    monkeypatch.setattr(sys, "argv", ["wakespot", "--no-map"])
    assert ws.main() == 0
    assert called.get("estop") is True


def test_allow_no_safety_kws_not_offered(ws):
    with pytest.raises(SystemExit):
        _args(ws, ["--allow-no-safety-kws"])


def test_dry_run_emergency_never_claims_a_cut(ws, monkeypatch, capsys):
    monkeypatch.setattr(ws, "_DRY_RUN", True)
    ws._emergency_cut(_FakeProc(), ws._NoEstop())
    out = capsys.readouterr().out
    assert "motors de-energized" not in out and "nothing was cut" in out


def test_real_emergency_message_unchanged(ws, monkeypatch, capsys):
    monkeypatch.setattr(ws, "_DRY_RUN", False)
    ws._emergency_cut(_FakeProc(), ws._NoEstop())
    assert "E-Stop CUT issued" in capsys.readouterr().out
