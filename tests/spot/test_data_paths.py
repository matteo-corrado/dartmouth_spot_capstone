"""Data-path constants of the moved modules still resolve to the repo's
config / runs / logs / models / grammar files. Restored from the T19-deleted
tests/spot/test_aliases_{robot,wake,brain}.py (base 23522e9) on the new module
paths only; assertions unchanged (the two old-symlink lines dropped)."""
import importlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO


# ---- from test_aliases_robot.py ----

def test_tuning_default_path_resolves_to_repo_config():
    from src.spot.core import tuning
    assert tuning.DEFAULT_PATH.resolve() == (REPO / "config" / "tuning.yaml").resolve()


def test_latency_run_dir_resolves_to_repo_runs():
    from src.spot.core import latency
    assert latency.DEFAULT_RUN_DIR.resolve() == (REPO / "runs").resolve()


def test_location_manager_project_root():
    from src.spot.nav import location_manager
    assert Path(location_manager.PROJECT_ROOT).resolve() == REPO.resolve()


def test_visual_nav_project_root():
    from src.spot.features.visual_nav import navigator
    assert Path(navigator._PROJECT_ROOT).resolve() == REPO.resolve()


# ---- from test_aliases_wake.py ----

def test_path_constants_unchanged():
    sherpa = importlib.import_module("src.spot.wake.sherpa_onnx")
    kws = importlib.import_module("src.spot.safety.kws")
    phrases = importlib.import_module("src.spot.wake.phrases")
    assert sherpa._PROJECT_ROOT == ROOT
    assert kws._PROJECT_ROOT == ROOT
    assert phrases._PHRASES_FILE == ROOT / "config" / "wake_phrases.txt"


# ---- from test_aliases_brain.py ----

def test_data_path_constants_exist():
    from src.spot.brain import llm_brain
    from src.spot.brain.backends import llamacpp_backend
    from src.spot.brain import persona
    from src.spot.conversation import conversation_log, voiceprint
    assert llm_brain._GRAMMAR_PATH.is_file()
    assert llm_brain._KNOWLEDGE_DIR.is_dir()
    for name in llm_brain._KNOWLEDGE_FILES:
        assert (llm_brain._KNOWLEDGE_DIR / name).is_file()
    assert Path(llamacpp_backend.GRAMMAR_PATH).is_file()
    assert (persona._REPO_ROOT / "config" / "personas.yaml").is_file()
    assert Path(conversation_log.DEFAULT_LOG_DIR).parent == ROOT / "logs"
    assert Path(voiceprint.DEFAULT_MODEL_PATH).parent.parent == ROOT / "models"
