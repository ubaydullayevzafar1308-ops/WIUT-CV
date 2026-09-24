"""accident / near_miss helpers: road-plane kinematics and tracker glitches."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_params
from src.features import FEATURE_DTYPE
from src.rules.collisions import ground_kinematics, leaves_frame, track_jumps

PARAMS = load_params()
ASPECT = 2160 / 3840


def track(xs: list[float], y: float = 0.6, width: float = 0.08, dt: float = 0.1, track_id: int = 1) -> np.ndarray:
    """A feature table of one box moving along a row."""
    f = np.zeros(len(xs), dtype=FEATURE_DTYPE)
    f["track_id"], f["t"] = track_id, np.arange(len(xs)) * dt
    f["x"], f["y"] = xs, y
    f["x1"], f["x2"] = np.array(xs) - width / 2, np.array(xs) + width / 2
    f["y1"], f["y2"] = y - width, y
    return f


def context(features: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(features=features, params=PARAMS, aspect=ASPECT)


def test_constant_motion_along_a_row_has_no_acceleration():
    g = PARAMS["risk"]["ground"]
    width = g["width_slope"] * (0.6 - g["horizon_y"])            # one car box width at this row per second
    kin = ground_kinematics(context(track(list(0.3 + width * np.arange(20) * 0.1))))
    inner = slice(5, 15)
    assert kin["speed"][inner] == pytest.approx(1.0, rel=1e-3)
    assert np.abs(kin["along"][inner]).max() < 1e-3 and np.abs(kin["lateral"][inner]).max() < 1e-3


def test_track_jumps_flag_leaps_and_size_changes():
    f = track([0.30, 0.31, 0.32, 0.60, 0.61])                      # leaps ~3.5 box widths in 0.1 s
    assert track_jumps(context(f), 8.0, 1.5).tolist() == [False, False, False, True, False]
    g = track([0.30, 0.31, 0.32, 0.33])
    g["x2"][2] += 0.1                                               # the box suddenly doubles in width
    assert track_jumps(context(g), 8.0, 1.5)[2]


def test_a_track_ending_at_the_border_left_the_frame():
    f = track([0.90, 0.95, 0.99])
    assert leaves_frame(context(f), np.arange(3), 0.03)
    assert not leaves_frame(context(track([0.40, 0.45, 0.50])), np.arange(3), 0.03)
