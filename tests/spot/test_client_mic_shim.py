"""client_mic is a stateless shim after Task 10: re-exports are the moved
objects themselves and old state names forward live to their owners."""
import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
import src.spot.mic.ring_control as _ring
import src.spot.safety.listener as _sl
import src.voice_control.client_mic as cm


def test_reexports_are_the_moved_objects():
    assert cm.process_utterance.__self__ is _loop.CONVERSATION
    assert cm._on_safety_kw.__self__ is _sl.LISTENER
    assert cm._update_ring.__self__ is _ring.RING
    assert cm.execute_on_spot is _loop.execute_on_spot
    assert cm._dispatch_safety is _sl._dispatch_safety
    assert cm.safety_kws_loop is _sl.safety_kws_loop
    # The golden harness tells the main stream apart by callback identity.
    assert cm.audio_callback is _cap.CAPTURE._callback


def test_state_forwards_read_the_owner_live(monkeypatch):
    for name, (mod, owner, attr) in cm._FORWARDS.items():
        assert name not in vars(cm), f"stale module-level copy of {name}"
        obj = mod if owner is None else getattr(mod, owner)
        sentinel = object()
        monkeypatch.setattr(obj, attr, sentinel)
        assert getattr(cm, name) is sentinel, name
