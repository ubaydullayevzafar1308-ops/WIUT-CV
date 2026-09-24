"""Part A: video -> tracks + scene + signal phases -> events."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.budget import plan_budget
from src.config import runtime_params
from src.scene import scene_for_video
from src.tracking import track_video
from src.video import probe

log = logging.getLogger(__name__)


def detect_events(video_path: str) -> list[list]:
    """Return ``[[start_sec, end_sec, label], ...]`` for one video.

    The scene is aligned to the video, then one decoding pass tracks road users
    and reads the traffic light while keeping Part A inside its time budget.
    Per-class rules and segment post-processing plug in on top of the tracks,
    scene and signal timeline, so no events are emitted yet.
    """
    start = time.perf_counter()
    params = runtime_params()
    info = probe(video_path)
    plan = plan_budget(video_path, info, start, params)
    scene = scene_for_video(video_path, params)
    tracks = track_video(video_path, params, scene=scene, plan=plan)
    tracks.signal_timeline(params)
    elapsed = time.perf_counter() - start
    log.info("%s: Part A %.1f s = %.2fx duration (allowance %.1f s, final stride %d)",
             Path(video_path).name, elapsed, elapsed / info.duration, plan.part_a_allowance,
             tracks.timing.get("final_stride", params["video"]["sample_stride"]))
    return []
