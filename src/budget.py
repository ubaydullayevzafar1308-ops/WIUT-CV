"""Wall-clock budget: Part A adapts its frame sampling so that Part A + Part B fit the harness limit.

The harness gives each video ``time_factor x duration`` for Part A and Part B
together and then streams every full-resolution frame through Part B with
OpenCV. Before tracking starts, that Part B cost is estimated on this machine by
reading a few 4K frames the same way; Part A gets the rest of the target budget
and raises its sampling stride whenever its projected finish is past the deadline.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2

from src.video import VideoInfo

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Plan:
    """Budget of one video. Times are ``time.perf_counter()`` values or seconds."""

    start: float
    duration: float
    part_b_reserve: float
    part_a_deadline: float

    @property
    def part_a_allowance(self) -> float:
        return self.part_a_deadline - self.start


def harness_seconds_per_frame(video_path: str, params: dict[str, Any]) -> float:
    """Time to read one full-resolution frame with OpenCV, as the harness does in Part B."""
    cap = cv2.VideoCapture(video_path)
    try:
        for _ in range(params["part_b_warmup_frames"]):
            cap.read()
        t0 = time.perf_counter()
        n = 0
        for _ in range(params["part_b_probe_frames"]):
            if not cap.read()[0]:
                break
            n += 1
        return (time.perf_counter() - t0) / max(n, 1)
    finally:
        cap.release()


def plan_budget(video_path: str, info: VideoInfo, start: float, params: dict[str, Any]) -> Plan:
    """Part A deadline for this video; ``start`` is when detect_events began.

    Part A gets ``target_factor x duration`` minus the Part B reserve and a tail
    for rules. If that leaves nothing, it falls back to ``hard_margin`` of the
    harness limit and logs a warning.
    """
    bp = params["budget"]
    per_frame = harness_seconds_per_frame(video_path, bp)
    reserve = (per_frame * info.n_frames + bp["part_b_model_factor"] * info.duration) * bp["part_b_safety"]
    target_end = start + bp["target_factor"] * info.duration - reserve - bp["part_a_tail_sec"]
    now = time.perf_counter()
    if target_end <= now:
        target_end = start + bp["hard_margin"] * bp["time_factor"] * info.duration - reserve - bp["part_a_tail_sec"]
        log.warning("target %.1fx is out of reach (Part B needs ~%.0f s); Part A aims at %.0f%% of the %.0fx limit",
                    bp["target_factor"], reserve, 100 * bp["hard_margin"], bp["time_factor"])
    plan = Plan(start=start, duration=info.duration, part_b_reserve=reserve, part_a_deadline=target_end)
    log.info("budget: video %.1f s, Part B reserve %.1f s (%.1f ms/frame), Part A allowance %.1f s",
             info.duration, reserve, 1000 * per_frame, plan.part_a_allowance)
    return plan


class StrideController:
    """Raises the sampling stride when tracking would finish after the Part A deadline.

    The speed is measured per video frame since the last change (after
    ``warmup_batches`` batches); the stride only ever grows, up to ``max_stride``.
    """

    def __init__(self, stride: int, deadline: float, n_frames: int, params: dict[str, Any]):
        self.stride = stride
        self.initial_stride = stride
        self.deadline = deadline
        self.n_frames = n_frames
        self.max_stride = params["max_sample_stride"]
        self.warmup = params["warmup_batches"]
        self.batches = 0
        self.ref: tuple[float, int] | None = None
        self.changes: list[tuple[int, int]] = []

    @property
    def adapted(self) -> bool:
        return self.stride != self.initial_stride

    def update(self, frame_index: int, now: float) -> int:
        """Call after each detector batch with the last processed frame index; returns the stride to use."""
        self.batches += 1
        if self.batches <= self.warmup or self.ref is None:
            self.ref = (now, frame_index)
            return self.stride
        t_ref, i_ref = self.ref
        if frame_index <= i_ref:
            return self.stride
        per_frame = (now - t_ref) / (frame_index - i_ref)
        projected = now + (self.n_frames - frame_index) * per_frame
        if projected <= self.deadline or self.stride >= self.max_stride:
            return self.stride
        time_left = self.deadline - now
        if time_left <= 0:
            new_stride = self.max_stride
        else:
            factor = (projected - now) / time_left
            new_stride = min(self.max_stride, max(self.stride + 1, math.ceil(self.stride * factor)))
        log.warning("behind schedule at frame %d (projected %.0f s over the deadline): stride %d -> %d",
                    frame_index, projected - self.deadline, self.stride, new_stride)
        self.changes.append((frame_index, new_stride))
        self.stride = new_stride
        self.ref = (now, frame_index)
        return self.stride
