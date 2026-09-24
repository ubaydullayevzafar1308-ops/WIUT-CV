"""Flags -> segments: runs, gap merging, short-segment removal, same-class union, clipping."""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import OFFICIAL_CLASSES
from run_submission import clean_events
from src.postprocess import class_params, flags_to_runs, flags_to_segments, merge_segments

STEP = 0.1
PARAMS = {
    "default": {"merge_gap": 1.0, "min_len": 0.5},
    "classes": {"stopped_vehicle": {"merge_gap": 2.0, "min_len": 10.0}},
}


def series(duration: float, *on: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Samples every STEP seconds, flagged inside the given [start, end) intervals."""
    t = np.round(np.arange(0, duration, STEP), 6)
    flags = np.zeros(len(t), dtype=bool)
    for start, end in on:
        flags |= (t >= start - 1e-9) & (t < end - 1e-9)
    return t, flags


def test_runs_end_where_the_next_sample_begins():
    t, flags = series(10, (2.0, 3.0))
    assert flags_to_runs(t, flags) == [(2.0, pytest.approx(3.0))]


def test_run_at_the_end_is_extended_by_one_sample():
    t, flags = series(10, (9.0, 10.0))
    assert flags_to_runs(t, flags) == [(9.0, pytest.approx(10.0))]


def test_no_flags_no_segments():
    t, flags = series(10)
    assert flags_to_segments(t, flags, "jaywalking", PARAMS, 10.0) == []
    assert flags_to_segments(np.array([]), np.array([], dtype=bool), "jaywalking", PARAMS, 10.0) == []


def test_short_gaps_are_merged_long_gaps_are_not():
    t, flags = series(20, (1, 3), (3.5, 5), (7, 9))           # gaps of 0.5 s and 2 s
    assert flags_to_segments(t, flags, "jaywalking", PARAMS, 20.0) == [[1.0, 5.0, "jaywalking"],
                                                                        [7.0, 9.0, "jaywalking"]]


def test_blips_are_dropped():
    t, flags = series(20, (2, 2.3), (5, 8))
    assert flags_to_segments(t, flags, "jaywalking", PARAMS, 20.0) == [[5.0, 8.0, "jaywalking"]]


def test_per_class_parameters():
    assert class_params("stopped_vehicle", PARAMS) == {"merge_gap": 2.0, "min_len": 10.0}
    assert class_params("wrong_way", PARAMS) == PARAMS["default"]
    t, flags = series(60, (0, 8), (9.5, 15), (30, 38))         # 1.5 s gap merged; 8 s alone is too short
    assert flags_to_segments(t, flags, "stopped_vehicle", PARAMS, 60.0) == [[0.0, 15.0, "stopped_vehicle"]]


def test_simultaneous_events_of_one_class_become_one_segment():
    segments = [[2.0, 6.0, "wrong_way"], [4.0, 9.0, "wrong_way"], [5.0, 5.5, "wrong_way"]]
    assert merge_segments(segments, PARAMS, 20.0) == [[2.0, 9.0, "wrong_way"]]


def test_different_classes_may_overlap():
    segments = [[2.0, 6.0, "wrong_way"], [4.0, 9.0, "accident"]]
    assert merge_segments(segments, PARAMS, 20.0) == [[4.0, 9.0, "accident"], [2.0, 6.0, "wrong_way"]]


def test_clipped_to_video_duration():
    segments = [[-1.0, 3.0, "congestion"], [18.0, 25.0, "congestion"], [21.0, 22.0, "congestion"]]
    assert merge_segments(segments, PARAMS, 20.0) == [[0.0, 3.0, "congestion"], [18.0, 20.0, "congestion"]]


def test_irregular_sampling():
    t = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5])
    flags = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    assert flags_to_runs(t, flags) == [(0.1, 1.2), (2.0, pytest.approx(2.5 + 0.4))]


def test_output_passes_the_harness_unchanged():
    rng = np.random.default_rng(0)
    segments = []
    for label in ("wrong_way", "jaywalking", "stopped_vehicle", "congestion"):
        for _ in range(40):
            start = rng.uniform(-5, 300)
            segments.append([start, start + rng.uniform(0.1, 30), label])
    events = merge_segments(segments, PARAMS, 300.0)
    kept, problems = clean_events(events, list(OFFICIAL_CLASSES), 300.0)
    assert problems == []
    assert sorted(kept) == sorted(events)
