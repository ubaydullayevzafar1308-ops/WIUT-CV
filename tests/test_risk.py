"""Part B: risk components, causality and determinism of the RiskEstimator."""
from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from src import risk
from src.config import ROOT, runtime_params
from src.risk import Components, RiskEstimator, ground_plane, pedestrian_alert, risk_score, smooth, time_to_contact

P = runtime_params()["risk"]
SAMPLE = ROOT / "samples" / "C3905.MP4"
CLIP_FRAMES = 240          # 8 s of video
CUT_FRAMES = 150           # the "truncated video" ends here


def pair(rel, vel, pedestrian=0.0) -> np.ndarray:
    return np.array([[*rel, *vel, pedestrian]], dtype=np.float64)


FAST = 1.5 * P["min_closing_speed"]   # a closing speed that counts (car box widths / s)


def test_ground_plane_unit_is_the_car_box_width():
    g, aspect = P["ground"], 2160 / 3840
    for y in (0.2, 0.5, 0.9):
        width = g["width_slope"] * (y - g["horizon_y"])        # car box width at this row
        a, b = ground_plane(np.array([0.4, 0.4 + width]), np.array([y, y]), g, aspect)
        assert np.hypot(*(b - a)) == pytest.approx(1.0)
    near, far = ground_plane(0.5, 0.9, g, aspect), ground_plane(0.5, 0.2, g, aspect)
    assert far[1] > near[1]                                     # higher in the image = further away


def test_head_on_pair_time_to_contact():
    # 5 car box widths apart, contact at the contact radius
    ttc = time_to_contact(pair((5.0, 0.0), (-FAST, 0.0)), P)
    assert ttc == pytest.approx((5.0 - P["contact_radius"]) / FAST)


def test_pairs_that_do_not_count():
    assert time_to_contact(pair((5.0, 0.0), (FAST, 0.0)), P) == np.inf              # moving apart
    assert time_to_contact(pair((5.0, 0.0), (-0.5 * P["min_closing_speed"], 0.0)), P) == np.inf  # too slow
    touching = P["contact_radius"] + P["min_gap"] / 2
    assert time_to_contact(pair((touching, 0.0), (-FAST, 0.0)), P) == np.inf        # already side by side (queue)
    assert time_to_contact(pair((5.0, 3.0), (-FAST, 0.0)), P) == np.inf             # passes 3 box widths aside
    assert time_to_contact(np.zeros((0, 5)), P) == np.inf


def test_risk_grows_as_contact_nears():
    def score(distance):
        return risk_score(Components(0.0, pair((distance, 0.0), (-FAST, 0.0)), 0.0, False, np.zeros((0, 3))), P)
    assert score(0.5 * FAST) > score(2.0 * FAST) > score(4.0 * FAST) >= 0.0
    assert 0.0 <= score(0.3 * FAST) <= 1.0


def test_pedestrian_alert_needs_an_approaching_vehicle_away_from_crossings():
    near = P["pedestrian_radius"] / 2
    far_from_crossing = P["pedestrian_crossing_margin"] + 1
    assert pedestrian_alert(np.array([[near, 1.0, far_from_crossing]]), P)
    assert not pedestrian_alert(np.array([[near, -1.0, far_from_crossing]]), P)     # vehicle moving away
    assert not pedestrian_alert(np.array([[near, 1.0, 0.0]]), P)                    # at a crossing
    assert not pedestrian_alert(np.zeros((0, 3)), P)


def test_smoothing_stays_in_range():
    assert smooth(0.0, 1.0, P) == pytest.approx(P["ema_alpha"])
    assert 0.0 <= smooth(0.9, 1.0, P) <= 1.0


def frames(n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(SAMPLE))
    out = []
    while len(out) < n:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def run(clip: list[np.ndarray], n_frames: int) -> np.ndarray:
    fps = 30000 / 1001
    estimator = RiskEstimator()
    estimator.reset({"video_id": SAMPLE.name, "fps": fps, "width": 3840, "height": 2160, "n_frames": n_frames})
    return np.array([estimator.step(frame, i / fps) for i, frame in enumerate(clip)])


@pytest.fixture(scope="module")
def clip(request):
    if not SAMPLE.exists():
        pytest.skip("C3905.MP4 not available (videos are not in git)")
    return frames(CLIP_FRAMES)


@pytest.fixture
def fixed_pace(monkeypatch):
    """Switch off the wall-clock pacing, which by design reacts to machine speed."""
    params = copy.deepcopy(runtime_params())
    params["risk"]["realtime_factor"] = 1e9
    monkeypatch.setattr(risk, "runtime_params", lambda: params)


def test_causal_and_deterministic(clip, fixed_pace):
    full = run(clip, n_frames=3825)
    truncated = run(clip[:CUT_FRAMES], n_frames=CUT_FRAMES)
    again = run(clip, n_frames=3825)
    assert np.array_equal(full[:CUT_FRAMES], truncated)     # the score on [0, t] ignores everything after t
    assert np.array_equal(full, again)                       # same frames, same scores
    assert np.all((full >= 0) & (full <= 1))
