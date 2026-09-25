"""Part A: video -> tracks + scene + signal phases -> events."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.budget import Plan, plan_budget, sampling_stride
from src.config import runtime_params
from src.features import compute_features, fit_car_width
from src.postprocess import merge_segments
from src.rules import VideoContext, apply_rules
from src.scene import Scene, scene_for_video
from src.tracking import track_video
from src.video import probe

log = logging.getLogger(__name__)


def scene_matched(features: np.ndarray, scene: Scene, params: dict) -> tuple[bool, float]:
    """Whether the scene fits this camera, judged by the traffic itself (no reference frame needed).

    The share of moving vehicles inside a lane (outside crossings and the
    intersection) heading within ``max_angle_deg`` of one of the lane's allowed
    directions; the scene matches when it is at least ``min_share``.
    """
    p = params["scene_match"]
    f = features
    moving = (np.isin(f["cls"], params["rules"]["vehicle_classes"]) & (f["speed"] >= params["features"]["heading_min_speed"])
              & (f["lane"] >= 0) & ~f["in_crossing"] & ~f["in_intersection"] & ~f["edge"])
    if not moving.any():
        return False, 0.0
    unit = np.stack([f["vx"], f["vy"]], axis=1)[moving].astype(np.float64)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-9)
    best = np.array([np.max(scene.lanes[lane].directions @ u) for lane, u in zip(f["lane"][moving], unit)])
    share = float(np.mean(best >= np.cos(np.radians(p["max_angle_deg"]))))
    return share >= p["min_share"], share


def video_context(video_path: str, params: dict, plan: Plan | None = None) -> VideoContext:
    """Align the scene, track (one decoding pass, within ``plan``) and build the rules' view of the video."""
    info = probe(video_path)
    scene = scene_for_video(video_path, params)
    tracks = track_video(video_path, params, scene=scene, plan=plan)
    features = compute_features(tracks, scene, params)
    matched, share = scene_matched(features, scene, params)
    log.log(logging.INFO if matched else logging.WARNING, "%s: scene %s the camera (%.1f%% of moving vehicles follow their "
            "lane directions)", Path(video_path).name, "matches" if matched else "does NOT match", 100 * share)
    return VideoContext(
        features=features,
        scene=scene,
        signal=tracks.signal_timeline(params),
        sample_t=tracks.frames / info.fps,
        duration=info.duration,
        aspect=info.height / info.width,
        params=params,
        car_width_fit=fit_car_width(features),
        final_stride=int(tracks.timing.get("final_stride", sampling_stride(info, params))),
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
