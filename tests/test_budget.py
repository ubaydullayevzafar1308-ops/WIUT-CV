"""Time budget: Part A allowance and the sampling-stride controller."""
from __future__ import annotations

import pytest

from src import budget
from src.budget import StrideController, plan_budget
from src.config import load_params
from src.video import VideoInfo

BP = load_params()["budget"]
INFO = VideoInfo(fps=30.0, n_frames=3000, width=3840, height=2160)   # 100 s video


def run(controller: StrideController, seconds_per_sample: float, batch: int = 8) -> list[int]:
    """Feed the controller batches processed at a fixed speed; return the stride after every batch."""
    now, index, strides = 0.0, 0, []
    while index < INFO.n_frames:
        index += batch * controller.stride
        now += batch * seconds_per_sample
        strides.append(controller.update(min(index, INFO.n_frames - 1), now))
    return strides


def test_stride_stays_when_on_schedule():
    controller = StrideController(3, deadline=100.0, n_frames=INFO.n_frames, params=BP)
    assert set(run(controller, seconds_per_sample=0.05)) == {3}      # 1000 samples x 0.05 s = 50 s < 100 s
    assert not controller.adapted


def test_stride_grows_until_the_deadline_is_met():
    controller = StrideController(3, deadline=60.0, n_frames=INFO.n_frames, params=BP)
    strides = run(controller, seconds_per_sample=0.2)                 # stride 3 would need 200 s
    assert controller.adapted and strides == sorted(strides)          # only ever increases
    assert strides[-1] <= BP["max_sample_stride"]
    assert controller.stride >= 9                                     # needs >= 3.3x fewer samples


def test_stride_is_capped():
    controller = StrideController(3, deadline=1.0, n_frames=INFO.n_frames, params=BP)
    run(controller, seconds_per_sample=1.0)
    assert controller.stride == BP["max_sample_stride"]


def test_plan_gives_part_a_the_rest_of_the_target(monkeypatch):
    monkeypatch.setattr(budget, "harness_seconds_per_frame", lambda path, params: 0.01)   # Part B: 30 s
    plan = plan_budget("video.mp4", INFO, start=budget.time.perf_counter(), params={"budget": BP})
    reserve = (0.01 * INFO.n_frames + BP["part_b_model_factor"] * INFO.duration) * BP["part_b_safety"]
    assert plan.part_b_reserve == pytest.approx(reserve)
    assert plan.part_a_allowance == pytest.approx(BP["target_factor"] * INFO.duration - reserve - BP["part_a_tail_sec"])


def test_plan_falls_back_to_the_hard_limit(monkeypatch):
    monkeypatch.setattr(budget, "harness_seconds_per_frame", lambda path, params: 0.05)   # Part B > 2.0x alone
    plan = plan_budget("video.mp4", INFO, start=budget.time.perf_counter(), params={"budget": BP})
    hard = BP["hard_margin"] * BP["time_factor"] * INFO.duration
    assert plan.part_a_allowance == pytest.approx(hard - plan.part_b_reserve - BP["part_a_tail_sec"])
    assert plan.part_a_allowance > 0
