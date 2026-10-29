"""Task 19: src/voice_control/ is reduced to two launch stubs (client_mic.py,
server.py; scripts/run_voice_control.py launches them by path). They hold no
state: top level is only the docstring, imports, the sys.path bootstrap and
the ``if __name__ == "__main__"`` call into src/spot (AST check)."""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUBS = {
    "client_mic.py": ("src.spot.app.main", "main"),
    "server.py": ("src.spot.stt.server", "serve"),
}


def _is_sys_path(node):
    """`sys.path` or a call/compare on it."""
    return any(isinstance(n, ast.Attribute) and n.attr == "path"
               and isinstance(n.value, ast.Name) and n.value.id == "sys"
               for n in ast.walk(node))


def _is_main_guard(node):
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == "__name__" and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


@pytest.mark.parametrize("fname", sorted(STUBS))
def test_split_shims_hold_no_state(fname):
    tree = ast.parse((ROOT / "src" / "voice_control" / fname).read_text())
    body = tree.body
    assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)  # docstring
    mains = []
    for node in body[1:]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr):  # sys.path.append(...) only
            assert _is_sys_path(node), ast.dump(node)
        elif isinstance(node, ast.Assign):  # the one project-root name for the bootstrap
            assert [t.id for t in node.targets] == ["_root"], ast.dump(node)
        elif isinstance(node, ast.If) and _is_main_guard(node):
            mains.append(node)
        elif isinstance(node, ast.If):  # `if _root not in sys.path: sys.path.insert(...)`
            assert _is_sys_path(node.test) and not node.orelse
            assert all(isinstance(n, ast.Expr) and _is_sys_path(n) for n in node.body)
        else:
            pytest.fail(f"{fname}: unexpected top-level {type(node).__name__}")
    assert len(mains) == 1
    module, func = STUBS[fname]
    imp, call = mains[0].body
    assert isinstance(imp, ast.ImportFrom) and imp.module == module
    assert [a.name for a in imp.names] == [func]
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    assert call.value.func.id == func and not call.value.args


def test_voice_control_holds_only_the_two_stubs():
    vc = ROOT / "src" / "voice_control"
    py = sorted(p.name for p in vc.rglob("*.py"))
    assert py == sorted(STUBS)
