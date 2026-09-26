"""Light analysis mode of the live demo (demo/api.py): events and the risk curve of one uploaded video.

Made for a CPU server: the small detector, fewer detector frames and one decoding
pass for both parts. Part A tracks as in ``pipeline.detect_events``; every frame it
decodes (every ``video.stride``-th frame, downscaled) is also handed to the risk
estimator, which processes every ``risk.sample_fps``-th second of them. The
parameters are params.yaml with the device profile and ``demo.overrides`` applied.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.budget import Plan
from src.config import deep_merge, runtime_params
from src.pipeline import context_from_tracks, scene_matched
from src.postprocess import merge_segments
from src.risk import RiskEstimator
from src.rules import apply_rules
from src.scene import align_scene
from src.tracking import track_video
from src.video import probe

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


def demo_params() -> dict[str, Any]:
    """params.yaml with the device profile and the demo's light mode (``demo.overrides``) applied."""
    params = runtime_params()
    return deep_merge(params, params["demo"]["overrides"])


def thin_curve(points: list[tuple[float, float]], bin_sec: float) -> list[list[float]]:
    """One point per ``bin_sec`` window that has scores: ``[window start, largest score in it]``."""
    best: dict[int, float] = {}
    for t, score in points:
        k = math.floor(t / bin_sec + 1e-9)
        best[k] = max(best.get(k, 0.0), score)
    return [[round(k * bin_sec, 3), round(score, 4)] for k, score in sorted(best.items())]


def visible_events(events: list[list], matched: bool, unmatched_classes: list[str]) -> list[list]:
    """All events when the scene matches the camera; otherwise only those not tied to our intersection."""
    return events if matched else [e for e in events if e[2] in unmatched_classes]


def analyze(video_path: str, progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Events, risk curve and scene match of one video in the demo's light mode.

    Args:
        video_path: the video file.
        progress: called with the share of the work done, in [0, 1] (scene
            alignment, then decoded frames, then the rules).

    Returns:
        ``{"duration", "fps", "events": [[start, end, label]], "risk": [[t, score]],
        "scene_matched"}``. The scene matches when it could be aligned to the
        reference edge map and the traffic follows its lanes
        (``pipeline.scene_matched``); otherwise only ``demo.unmatched_classes``
        are returned, the events not tied to this intersection.
    """
    report = progress or (lambda share: None)
    start = time.perf_counter()
    params = demo_params()
    dp = params["demo"]
    info = probe(video_path)
    scene, registration = align_scene(video_path, params)
    report(dp["progress"]["aligned"])

    decode_stride = params["video"]["stride"]
    estimator = RiskEstimator(params=params)
    estimator.reset({"video_id": Path(video_path).stem, "fps": info.fps / decode_stride, "width": info.width,
                     "height": info.height, "n_frames": math.ceil(info.n_frames / decode_stride)})
    scores: list[tuple[float, float]] = []
    span = dp["progress"]["tracked"] - dp["progress"]["aligned"]

    def on_frame(index: int, t_sec: float, bgr: np.ndarray) -> None:
        scores.append((t_sec, estimator.step(bgr, t_sec)))
        report(dp["progress"]["aligned"] + span * min(1.0, (index + 1) / max(info.n_frames, 1)))

    plan = Plan(start=start, duration=info.duration, part_b_reserve=0.0,
                part_a_deadline=start + dp["time_factor"] * info.duration + dp["time_extra_sec"])
    tracks = track_video(video_path, params, use_cache=False, scene=scene, plan=plan, on_frame=on_frame)
    report(dp["progress"]["tracked"])
    ctx = context_from_tracks(video_path, scene, tracks, params)
    events = merge_segments(apply_rules(ctx), params["postprocess"], info.duration)
    follows_lanes, _ = scene_matched(ctx.features, ctx.scene, params)
    matched = registration.ok and follows_lanes
    events = visible_events(events, matched, dp["unmatched_classes"])

    elapsed = time.perf_counter() - start
    log.info("%s: demo analysis of %.1f s of video in %.1f s (%.2fx); %d events, %d risk updates, scene %s",
             Path(video_path).name, info.duration, elapsed, elapsed / max(info.duration, 1e-9), len(events),
             estimator.processed, "matched" if matched else "not matched")
    return {
        "duration": round(info.duration, 3),
        "fps": info.fps,
        "events": [[round(s, 3), round(e, 3), label] for s, e, label in events],
        "risk": thin_curve(scores, dp["risk_bin_sec"]),
        "scene_matched": bool(matched),
    }
