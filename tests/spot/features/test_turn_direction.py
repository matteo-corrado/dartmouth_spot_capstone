import math

from src.spot.features.walk.handlers import turn_rad


def test_left_is_counter_clockwise_positive():
    assert math.isclose(turn_rad(90, "left"), math.pi / 2)
    assert math.isclose(turn_rad(90, "right"), -math.pi / 2)


def test_string_and_negative_degrees():
    assert math.isclose(turn_rad("45", "left"), math.pi / 4)  # fast-path "$deg" capture
    assert math.isclose(turn_rad(-30, "right"), -math.pi / 6)
