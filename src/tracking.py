"""YOLO detection + ByteTrack tracking over a video, with an on-disk cache of the tracks table.

The same decoding pass also reads the traffic-light heads (src/signal.py), so a
video is decoded once in Part A.
"""
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

from src.budget import Plan, StrideController, sampling_stride
from src.config import resolve, runtime_params, set_seeds
from src.scene import Scene, scene_for_video
from src.signal import PHASES, SignalTimeline, read_heads, timeline_from_samples
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
# One row per signal sample: time and the reading of each head (index into signal.PHASES).
SIGNAL_DTYPE = np.dtype([("t", np.float64), ("ped", np.int8), ("veh", np.int8)])

_MODELS: dict[str, Any] = {}


@dataclass
class Tracks:
    """Tracker output for one video.

    ``frames`` lists every sampled frame index (including frames with no boxes);
    ``rows`` holds the boxes in ``TRACK_DTYPE``; ``signal`` the traffic-light
    samples in ``SIGNAL_DTYPE``; ``timing`` has stage durations in seconds and
    the final sampling stride.
    """

    info: VideoInfo
    frames: np.ndarray
    rows: np.ndarray
    signal: np.ndarray
    timing: dict[str, float] = field(default_factory=dict)

    def signal_timeline(self, params: dict[str, Any]) -> SignalTimeline:
        """Phase timeline from the signal samples; ``params`` is the full parameter dict."""
        names = np.array(PHASES, dtype=object)
        return timeline_from_samples(self.signal["t"], names[self.signal["ped"]], names[self.signal["veh"]],
                                     params["signal"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            info=np.array([self.info.fps, self.info.n_frames, self.info.width, self.info.height], dtype=np.float64),
            frames=self.frames,
            rows=self.rows,
            signal=self.signal,
        )

    @classmethod
    def load(cls, path: Path) -> Tracks:
        with np.load(path) as data:
            fps, n_frames, width, height = data["info"]
            info = VideoInfo(fps=float(fps), n_frames=int(n_frames), width=int(width), height=int(height))
            return cls(info=info, frames=data["frames"], rows=data["rows"], signal=data["signal"])


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


def light_rois(scene: Scene) -> dict[str, list[float]]:
    return {light["id"]: light["roi"] for light in scene.traffic_lights}


def params_hash(params: dict[str, Any], scene: Scene) -> str:
    """Short hash of everything that changes the cached output (device profile and head positions included)."""
    relevant = {k: params[k] for k in ("video", "detector", "tracker", "signal")}
    relevant["rois"] = {k: np.round(v, 4).tolist() for k, v in light_rois(scene).items()}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(video_path: str, params: dict[str, Any], scene: Scene) -> Path:
    return resolve(params["cache"]["dir"]) / f"{Path(video_path).stem}_{params_hash(params, scene)}.npz"


def track_video(
    video_path: str,
    params: dict[str, Any] | None = None,
    use_cache: bool | None = None,
    scene: Scene | None = None,
    plan: Plan | None = None,
) -> Tracks:
    """Detect and track road users and read the traffic light, using the cache when possible.

    Args:
        video_path: path to the video file.
        params: full parameter dict for this device; defaults to ``runtime_params()``.
        use_cache: overrides ``params["cache"]["enabled"]``.
        scene: scene aligned to this video; computed if not given.
        plan: the time fuse; without it the sampling stride never changes. Results
            of a run whose stride was raised are not cached.
    """
    params = params or runtime_params()
    scene = scene or scene_for_video(video_path, params)
    use_cache = params["cache"]["enabled"] if use_cache is None else use_cache
    path = cache_path(video_path, params, scene)
    if use_cache and path.exists():
        log.info("tracks cache hit: %s", path.name)
        return Tracks.load(path)

    tracks, adapted = _run_tracking(video_path, params, scene, plan)
    if use_cache and not adapted:
        try:
            tracks.save(path)
        except OSError as err:
            log.warning("could not write tracks cache %s: %s", path, err)
    return tracks


def _run_tracking(video_path: str, params: dict[str, Any], scene: Scene, plan: Plan | None) -> tuple[Tracks, bool]:
    set_seeds(params["seed"])
    vp, dp, sp = params["video"], params["detector"], params["signal"]
    info = probe(video_path)
    device = params["device"]
    model = get_model(dp["weights"])
    stride = sampling_stride(info, params)
    tracker = make_tracker(params["tracker"], info.fps / stride)
    controller = StrideController(stride, plan.part_a_deadline if plan else float("inf"),
                                  info.n_frames, params["budget"])
    rois = light_rois(scene)
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
    signal: list[tuple[float, int, int]] = []
    timing = {"decode_wait": 0.0, "detect": 0.0, "track": 0.0}
    t_start = time.perf_counter()
    batch_idx: list[int] = []
    batch_img: list[np.ndarray] = []
    next_sample, next_signal_t = 0, 0.0
    t0 = time.perf_counter()
    for index, t_sec, bgr in frames:
        if t_sec >= next_signal_t:
            next_signal_t = t_sec + 1.0 / sp["sample_fps"]
            ped, veh = read_heads(bgr, rois, sp)
            signal.append((t_sec, PHASES.index(ped), PHASES.index(veh)))
        if index < next_sample:
            continue
        next_sample = index + controller.stride
        batch_idx.append(index)
        batch_img.append(bgr)
        if len(batch_img) < dp["batch"]:
            continue
        timing["decode_wait"] += time.perf_counter() - t0
        chunks += _detect_and_track(model, tracker, batch_idx, batch_img, predict_kwargs, timing)
        sampled += batch_idx
        controller.update(batch_idx[-1], time.perf_counter())
        batch_idx, batch_img = [], []
        t0 = time.perf_counter()
    timing["decode_wait"] += time.perf_counter() - t0
    if batch_img:
        chunks += _detect_and_track(model, tracker, batch_idx, batch_img, predict_kwargs, timing)
        sampled += batch_idx
    timing["total"] = time.perf_counter() - t_start
    timing["final_stride"] = controller.stride

    rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=TRACK_DTYPE)
    log.info(
        "%s: %d sampled frames (stride %d%s), %d boxes, %d tracks, %d signal samples on %s in %.1fs "
        "(waiting for decoder %.1fs, detect %.1fs, track %.1fs)",
        Path(video_path).name, len(sampled), controller.stride,
        f", raised from {controller.initial_stride}" if controller.adapted else "",
        len(rows), len(np.unique(rows["track_id"])), len(signal), device,
        timing["total"], timing["decode_wait"], timing["detect"], timing["track"],
    )
    tracks = Tracks(info=info, frames=np.asarray(sampled, dtype=np.int32), rows=rows,
                    signal=np.array(signal, dtype=SIGNAL_DTYPE), timing=timing)
    return tracks, controller.adapted


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
