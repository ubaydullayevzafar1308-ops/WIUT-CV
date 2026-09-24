"""Part A: video -> tracks + scene + signal phases -> events."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.budget import Plan, plan_budget
from src.config import runtime_params
from src.features import compute_features
from src.postprocess import merge_segments
from src.rules import VideoContext, apply_rules
from src.scene import scene_for_video
from src.tracking import track_video
from src.video import probe

log = logging.getLogger(__name__)


def video_context(video_path: str, params: dict, plan: Plan | None = None) -> VideoContext:
    """Align the scene, track (one decoding pass, within ``plan``) and build the rules' view of the video."""
    info = probe(video_path)
    scene = scene_for_video(video_path, params)
    tracks = track_video(video_path, params, scene=scene, plan=plan)
    return VideoContext(
        features=compute_features(tracks, scene, params),
        scene=scene,
        signal=tracks.signal_timeline(params),
        sample_t=tracks.frames / info.fps,
        duration=info.duration,
        aspect=info.height / info.width,
        params=params,
        final_stride=int(tracks.timing.get("final_stride", params["video"]["sample_stride"])),
    )


def detect_events(video_path: str) -> list[list]:
    """Return ``[[start_sec, end_sec, label], ...]`` for one video.

    The scene is aligned to the video, then one decoding pass tracks road users
    and reads the traffic light while keeping Part A inside its time budget.
    Track features, the scene and the signal timeline go through the per-class
    rules (src/rules/), whose segments are merged and filtered per class.
    """
    start = time.perf_counter()
    params = runtime_params()
    info = probe(video_path)
    plan = plan_budget(video_path, info, start, params)
    ctx = video_context(video_path, params, plan)
    events = merge_segments(apply_rules(ctx), params["postprocess"], info.duration)
    elapsed = time.perf_counter() - start
    log.info("%s: %d events; Part A %.1f s = %.2fx duration (allowance %.1f s, final stride %d)",
             Path(video_path).name, len(events), elapsed, elapsed / info.duration, plan.part_a_allowance,
             ctx.final_stride)
    return events
