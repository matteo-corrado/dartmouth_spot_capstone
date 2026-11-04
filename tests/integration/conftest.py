"""Fixtures for the golden conversation harness (tests/integration/harness.py)."""
import json

import pytest

from tests.integration.harness import GOLDEN_PATH, resolve_clip


@pytest.fixture
def golden():
    """golden/conversation.json plus its clips resolved to paths. Skips when
    the operator recordings (gitignored, SPOT_DEMO_AUDIO) are not present."""
    data = json.loads(GOLDEN_PATH.read_text())
    paths = [resolve_clip(c) for c in data["clips"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"demo recordings not present ({len(missing)} missing, e.g. {missing[0]})")
    data["paths"] = paths
    return data
