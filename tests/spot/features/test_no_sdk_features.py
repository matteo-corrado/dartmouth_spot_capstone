"""CR #3: every enabled feature registers without the bosdyn SDK (handlers
import it at call time, walk/handlers.py's rule), so the regex intents and
the grammar never depend on it. Fresh interpreter with bosdyn blocked."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

CODE = """
import sys
sys.modules['bosdyn'] = None
from src.spot.features import REGISTRY
from src.spot.features.enabled import ENABLED
failed = REGISTRY.load(ENABLED)
print('FAILED', [n for n, _, _ in failed], [n for n, _, _ in REGISTRY.failures])
print('LOADED', sorted(f.name for f in REGISTRY.features))
from src.spot.brain.intent import parse_intent
for t in ('go to the kitchen', 'tour the map', 'find the chair', 'come here'):
    r = parse_intent(t)
    print('INTENT', t, '->', r and r['intent'])
from src.spot.brain import llm_brain
for a in ('go_to', 'tour', 'go_to_object', 'load_map'):
    print('GRAMMAR', a, '"\\\\"%s\\\\""' % a in llm_brain._GRAMMAR_TEXT)
print('ROBOT_IMPORTED', 'src.spot.robot' in sys.modules)
"""


def test_all_features_register_and_parse_without_bosdyn():
    out = subprocess.run([sys.executable, "-c", CODE], cwd=str(ROOT), capture_output=True,
                         text=True, timeout=120, env=dict(os.environ, SPOT_DRY_RUN="1"))
    assert out.returncode == 0, out.stderr[-2000:]
    o = out.stdout
    assert "FAILED [] []" in o
    assert ("LOADED ['dance', 'follow', 'gaze', 'mouth', 'navigation', 'settings', "
            "'visual_nav', 'walk']") in o
    assert "INTENT go to the kitchen -> go_to" in o
    assert "INTENT tour the map -> tour" in o
    assert "INTENT find the chair -> go_to_object" in o
    assert "INTENT come here -> walk_to" in o
    for a in ("go_to", "tour", "go_to_object", "load_map"):
        assert f"GRAMMAR {a} True" in o
    assert "ROBOT_IMPORTED False" in o
