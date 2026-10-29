"""Golden conversation gate: today's client_mic.main() decisions on recorded
clips must not change across the modular restructure.

Feeds audio at real time (several minutes). Regenerate only on purpose, at a
commit whose behaviour is the new reference:
    spot-env/bin/python tests/integration/harness.py generate tests/integration/golden/conversation.json
"""
import pytest

from tests.integration.harness import normalize, run_pipeline


@pytest.mark.slow
def test_conversation_matches_golden(golden):
    records = normalize(run_pipeline(golden["paths"]))
    assert records == golden["records"]
