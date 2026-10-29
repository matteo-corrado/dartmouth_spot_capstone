import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

PAIRS = [
    ("src.session", "src.spot.robot.session"),
    ("src.estop", "src.spot.safety.estop"),
    ("src.graph_nav_utils", "src.spot.nav.graph_nav_utils"),
    ("src.map_loader", "src.spot.nav.map_loader"),
    ("src.location_manager", "src.spot.nav.location_manager"),
    ("src.voice_control.perception", "src.spot.perception"),
    ("src.voice_control.perception.camera_intrinsics", "src.spot.perception.camera_intrinsics"),
    ("src.voice_control.perception.local_grid_helpers", "src.spot.perception.local_grid_helpers"),
    ("src.voice_control.perception.obstacle_query", "src.spot.perception.obstacle_query"),
    ("src.voice_control.perception.raycast_helpers", "src.spot.perception.raycast_helpers"),
    ("src.voice_control.world_objects", "src.spot.perception.world_objects"),
    ("src.voice_control.visual_nav", "src.spot.features.visual_nav.navigator"),
    ("src.voice_control.emotes", "src.spot.features.dance.emotes"),
    ("src.voice_control.gaze", "src.spot.features.gaze.math"),
    ("src.voice_control.tuning", "src.spot.core.tuning"),
    ("src.voice_control.latency", "src.spot.core.latency"),
    ("src.voice_control.startup_status", "src.spot.core.startup_status"),
    ("src.voice_control.voice_state", "src.spot.core.voice_state"),
    ("src.voice_control.control_server", "src.spot.operator.control_server"),
]


@pytest.mark.parametrize("old_name,new_name", PAIRS)
def test_alias_identity(old_name, new_name):
    old = importlib.import_module(old_name)
    new = importlib.import_module(new_name)
    assert old is new


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
