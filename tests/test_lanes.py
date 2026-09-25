"""Lane geometry of illegal_turn / solid_line_crossing."""
from __future__ import annotations

import numpy as np
import pytest

from src.rules.lanes import boundary_along, signed_distance
from src.rules.signal_rules import StopLine


def test_signed_distance_tells_the_side_and_the_position_along_the_line():
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    dist, pos = signed_distance(line, np.array([0.25, 0.75]), np.array([0.1, -0.2]), aspect=1.0)
    assert np.sign(dist[0]) == -np.sign(dist[1])
    assert np.abs(dist) == pytest.approx([0.1, 0.2])
    assert pos == pytest.approx([0.25, 0.75])


def test_lane_boundary_meets_the_stop_line_where_extended():
    stop = StopLine(p0=np.array([0.0, 0.5]), unit=np.array([1.0, 0.0]), length=1.0, upstream=1.0)
    boundary = np.array([[0.1, 0.1], [0.3, 0.3]])            # a diagonal that would reach y = 0.5 at x = 0.5
    assert boundary_along(stop, boundary, aspect=1.0) == pytest.approx(0.5)
