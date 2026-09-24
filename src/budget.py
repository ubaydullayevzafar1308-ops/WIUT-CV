"""Sampling stride of Part A and the time-limit safety fuse.

The stride is chosen deterministically (``sampling_stride``: device profile and
video metadata, no timing), so two runs give the same predictions. Timing is
only a fuse: the harness gives each video ``time_factor x duration`` for Part A
and Part B together and then streams every full-resolution frame through Part B
with OpenCV. That Part B cost is estimated by reading a few 4K frames the same
way; if Part A + Part B are projected over ``fuse_factor x duration``, Part A
raises its stride and logs a warning. On a normal machine the fuse never blows.
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


def sampling_stride(info: VideoInfo, params: dict[str, Any]) -> int:
    """Detector sampling stride from the device profile and the video's metadata only (deterministic).

    Aims at ``sample_fps`` detector frames per second of video (device profile:
    10 on cuda / mps, 5 on cpu), scaled up for frames larger than
    ``reference_pixels`` (decoding cost grows with the pixel count), as a multiple
    of the decode stride. Duration does not enter: the time limit grows with it.
    """
    vp = params["video"]
    stride = info.fps / vp["sample_fps"] * max(1.0, info.width * info.height / vp["reference_pixels"])
    decode = vp["stride"]
    return int(decode * max(1, round(stride / decode)))


def plan_budget(video_path: str, info: VideoInfo, start: float, params: dict[str, Any]) -> Plan:
    """The fuse for this video; ``start`` is when detect_events began.

    Part A may run until ``fuse_factor x duration`` minus the estimated Part B
    time and a tail for rules; only then does its stride grow.
    """
    bp = params["budget"]
    per_frame = harness_seconds_per_frame(video_path, bp)
    reserve = (per_frame * info.n_frames + bp["part_b_model_factor"] * info.duration) * bp["part_b_safety"]
    deadline = start + bp["fuse_factor"] * info.duration - reserve - bp["part_a_tail_sec"]
    plan = Plan(start=start, duration=info.duration, part_b_reserve=reserve, part_a_deadline=deadline)
    log.info("budget: video %.1f s, Part B estimate %.1f s (%.1f ms/frame); fuse at %.1fx: Part A may take %.1f s",
             info.duration, reserve, 1000 * per_frame, bp["fuse_factor"], plan.part_a_allowance)
    return plan


class StrideController:
    """The fuse: raises the sampling stride when tracking would finish after the Part A deadline.

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
        log.warning("time fuse blown at frame %d (Part A + B projected %.0f s over the fuse): stride %d -> %d",
                    frame_index, projected - self.deadline, self.stride, new_stride)
        self.changes.append((frame_index, new_stride))
        self.stride = new_stride
        self.ref = (now, frame_index)
        return self.stride
