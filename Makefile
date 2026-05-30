# Stage 2F voice-eval suite. Run: make eval-voice
PY := spot-env/bin/python
PYTEST := spot-env/bin/pytest

.PHONY: eval-voice
eval-voice:
	$(PYTEST) tests/voice_control/test_barge_in.py tests/voice_control/wake/test_safety_kws.py tests/voice_control/tts/test_chunker.py tests/audio/eval_bargein.py -v
	$(PY) -m tests.audio.eval_wake
	$(PY) -m tests.audio.eval_safety_recall
