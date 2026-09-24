"""Time budget: Part A allowance and the sampling-stride controller."""
from __future__ import annotations

import pytest

from src import budget
from src.budget import StrideController, plan_budget, sampling_stride
from src.config import deep_merge, load_params
from src.video import VideoInfo

BP = load_params()["budget"]
INFO = VideoInfo(fps=30.0, n_frames=3000, width=3840, height=2160)   # 100 s video
INFO_4K_2997 = VideoInfo(fps=30000 / 1001, n_frames=3825, width=3840, height=2160)   # the sample videos


def runtime_profile(kind: str) -> dict:
    params = load_params()
    return deep_merge(params, params["device_profiles"][kind])


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


def test_plan_puts_the_fuse_at_the_fuse_factor(monkeypatch):
    monkeypatch.setattr(budget, "harness_seconds_per_frame", lambda path, params: 0.01)   # decoding: 30 s
    plan = plan_budget("video.mp4", INFO, start=budget.time.perf_counter(), params={"budget": BP})
    reserve = (0.01 * INFO.n_frames + BP["part_b_model_factor"] * INFO.duration) * BP["part_b_safety"]
    assert plan.part_b_reserve == pytest.approx(reserve)
    assert plan.part_a_allowance == pytest.approx(BP["fuse_factor"] * INFO.duration - reserve - BP["part_a_tail_sec"])


def test_sampling_stride_depends_only_on_device_profile_and_metadata():
    gpu, cpu = load_params(), runtime_profile("cpu")
    assert sampling_stride(INFO_4K_2997, gpu) == 3
    assert sampling_stride(INFO_4K_2997, cpu) == 6
    assert sampling_stride(VideoInfo(fps=59.94, n_frames=6000, width=3840, height=2160), gpu) == 6
    assert sampling_stride(VideoInfo(fps=29.97, n_frames=3000, width=7680, height=4320), gpu) == 12
    assert sampling_stride(VideoInfo(fps=25.0, n_frames=2500, width=1920, height=1080), gpu) == 3
