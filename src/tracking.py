"""YOLO detection + ByteTrack tracking over a video, with an on-disk cache of the tracks table."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config import load_params, resolve, select_device, set_seeds
from src.video import VideoInfo, prefetch, probe, read_frames

log = logging.getLogger(__name__)

# One row per tracked box. Box coordinates are normalised to [0, 1] of the frame.
TRACK_DTYPE = np.dtype([
    ("frame", np.int32),
    ("track_id", np.int32),
    ("cls", np.int16),
    ("conf", np.float32),
    ("x1", np.float32),
    ("y1", np.float32),
    ("x2", np.float32),
    ("y2", np.float32),
])

_MODELS: dict[str, Any] = {}


@dataclass
class Tracks:
    """Tracker output for one video.

    ``frames`` lists every sampled frame index (including frames with no boxes);
    ``rows`` holds the boxes in ``TRACK_DTYPE``; ``timing`` has stage durations in seconds.
    """

    info: VideoInfo
    frames: np.ndarray
    rows: np.ndarray
    timing: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            info=np.array([self.info.fps, self.info.n_frames, self.info.width, self.info.height], dtype=np.float64),
            frames=self.frames,
            rows=self.rows,
        )

    @classmethod
    def load(cls, path: Path) -> Tracks:
        with np.load(path) as data:
            fps, n_frames, width, height = data["info"]
            info = VideoInfo(fps=float(fps), n_frames=int(n_frames), width=int(width), height=int(height))
            return cls(info=info, frames=data["frames"], rows=data["rows"])


def get_model(weights: str):
    """Load a YOLO model once per process and reuse it (Part A and Part B share it)."""
    if weights not in _MODELS:
        from ultralytics import YOLO

        _MODELS[weights] = YOLO(str(resolve(weights)), task="detect")
    return _MODELS[weights]


def make_tracker(tracker_params: dict[str, Any], sampled_fps: float):
    """Create a fresh ByteTrack instance; the lost-track buffer is given in seconds."""
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(
        track_high_thresh=tracker_params["track_high_thresh"],
        track_low_thresh=tracker_params["track_low_thresh"],
        new_track_thresh=tracker_params["new_track_thresh"],
        track_buffer=max(1, round(tracker_params["track_buffer_sec"] * sampled_fps)),
        match_thresh=tracker_params["match_thresh"],
        fuse_score=tracker_params["fuse_score"],
    )
    return BYTETracker(args)


def params_hash(params: dict[str, Any]) -> str:
    """Short hash of everything that changes the tracks table."""
    relevant = {k: params[k] for k in ("video", "detector", "tracker")}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(video_path: str, params: dict[str, Any]) -> Path:
    return resolve(params["cache"]["dir"]) / f"{Path(video_path).stem}_{params_hash(params)}.npz"


def track_video(video_path: str, params: dict[str, Any] | None = None, use_cache: bool | None = None) -> Tracks:
    """Detect and track road users in a video, reading or writing the cache.

    Args:
        video_path: path to the video file.
        params: full parameter dict; defaults to ``configs/params.yaml``.
        use_cache: overrides ``params["cache"]["enabled"]``.
    """
    params = params or load_params()
    use_cache = params["cache"]["enabled"] if use_cache is None else use_cache
    path = cache_path(video_path, params)
    if use_cache and path.exists():
        log.info("tracks cache hit: %s", path.name)
        return Tracks.load(path)

    tracks = _run_tracking(video_path, params)
    if use_cache:
        try:
            tracks.save(path)
        except OSError as err:
            log.warning("could not write tracks cache %s: %s", path, err)
    return tracks


def _run_tracking(video_path: str, params: dict[str, Any]) -> Tracks:
    set_seeds(params["seed"])
    vp, dp = params["video"], params["detector"]
    info = probe(video_path)
    device = select_device()
    model = get_model(dp["weights"])
    tracker = make_tracker(params["tracker"], info.fps / vp["stride"])
    predict_kwargs = dict(
        imgsz=dp["imgsz"],
        conf=dp["conf"],
        iou=dp["iou"],
        classes=dp["classes"],
        device=device,
        quantize=16 if dp["fp16"] and device.startswith("cuda") else None,
        verbose=False,
    )
    frames = prefetch(read_frames(
        video_path,
        stride=vp["stride"],
        target_width=vp["target_width"],
        threads=vp["decoder_threads"],
        skip_nonref=vp["skip_nonref"],
        interpolation=vp["interpolation"],
        skip_check_frames=vp["skip_check_frames"],
    ), vp["prefetch"])

    sampled: list[int] = []
    chunks: list[np.ndarray] = []
    timing = {"decode_wait": 0.0, "detect": 0.0, "track": 0.0}
    t_start = time.perf_counter()
    batch_idx: list[int] = []
    batch_img: list[np.ndarray] = []
    t0 = time.perf_counter()
    for index, _, bgr in frames:
        batch_idx.append(index)
        batch_img.append(bgr)
        if len(batch_img) < dp["batch"]:
            continue
        timing["decode_wait"] += time.perf_counter() - t0
        chunks += _detect_and_track(model, tracker, batch_idx, batch_img, predict_kwargs, timing)
        sampled += batch_idx
        batch_idx, batch_img = [], []
        t0 = time.perf_counter()
    timing["decode_wait"] += time.perf_counter() - t0
    if batch_img:
        chunks += _detect_and_track(model, tracker, batch_idx, batch_img, predict_kwargs, timing)
        sampled += batch_idx
    timing["total"] = time.perf_counter() - t_start

    rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=TRACK_DTYPE)
    log.info(
        "%s: %d sampled frames, %d boxes, %d tracks on %s in %.1fs (waiting for decoder %.1fs, detect %.1fs, track %.1fs)",
        Path(video_path).name, len(sampled), len(rows), len(np.unique(rows["track_id"])), device,
        timing["total"], timing["decode_wait"], timing["detect"], timing["track"],
    )
    return Tracks(info=info, frames=np.asarray(sampled, dtype=np.int32), rows=rows, timing=timing)


def _detect_and_track(
    model,
    tracker,
    indices: list[int],
    images: list[np.ndarray],
    predict_kwargs: dict[str, Any],
    timing: dict[str, float],
) -> list[np.ndarray]:
    """Run one detector batch, then feed each frame's boxes to the tracker in order."""
    t0 = time.perf_counter()
    results = model.predict(images, **predict_kwargs)
    t1 = time.perf_counter()
    out = []
    for index, result in zip(indices, results):
        height, width = result.orig_shape
        tracked = tracker.update(result.boxes.cpu().numpy())
        if len(tracked) == 0:
            continue
        rows = np.empty(len(tracked), dtype=TRACK_DTYPE)
        rows["frame"] = index
        rows["x1"], rows["x2"] = tracked[:, 0] / width, tracked[:, 2] / width
        rows["y1"], rows["y2"] = tracked[:, 1] / height, tracked[:, 3] / height
        rows["track_id"] = tracked[:, 4]
        rows["conf"] = tracked[:, 5]
        rows["cls"] = tracked[:, 6]
        out.append(rows)
    timing["detect"] += t1 - t0
    timing["track"] += time.perf_counter() - t1
    return out
