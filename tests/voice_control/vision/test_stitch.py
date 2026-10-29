import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.voice_control.vision import stitch as st
from src.voice_control.vision.cameras import frame_from_response

PRIV = Path(__file__).parents[2] / "fixtures" / "vision" / "private"
BIN, REF = PRIV / "front_pair.bin", PRIV / "stitch_ref.png"
pytestmark = pytest.mark.skipif(not (BIN.exists() and REF.exists()),
                                reason="private stitch fixtures missing")


@pytest.fixture(scope="module")
def pair():
    from bosdyn.api import image_pb2
    r = image_pb2.GetImageResponse()
    r.ParseFromString(BIN.read_bytes())
    by = {i.source.name: i for i in r.image_responses}
    return (frame_from_response(by["frontright_fisheye_image"]),
            frame_from_response(by["frontleft_fisheye_image"]))


def test_stitch_matches_bd_reference(pair):
    st._MAPS.clear()
    out = st.stitch_front(*pair)
    ref = cv2.imread(str(REF), cv2.IMREAD_GRAYSCALE)
    assert out.shape == ref.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - ref.astype(int)).mean() < 1.0
    assert (out == 0).sum() == (ref == 0).sum()


def test_stitch_maps_cached(pair, monkeypatch):
    st._MAPS.clear()
    calls = []
    real = st._build_maps
    monkeypatch.setattr(st, "_build_maps", lambda *a: calls.append(1) or real(*a))
    st.stitch_front(*pair)
    st.stitch_front(*pair)
    assert len(calls) == 1


def test_stitch_time(pair):
    st.stitch_front(*pair)
    t = time.perf_counter()
    st.stitch_front(*pair)
    assert time.perf_counter() - t < 0.060
