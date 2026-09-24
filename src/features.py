"""Per-sample kinematics and scene zones of tracked objects.

Units: positions are normalised [0, 1] image coordinates of the anchor (bottom
centre of the box, where the object touches the road). Velocities and
accelerations are in frame widths per second (per second squared), with the
vertical component scaled by height / width so that speeds are the same in every
direction on the image. Heading is in degrees in the image plane: 0 = right,
90 = down. Times are seconds (``frame / fps``, as the harness counts them).
All derivatives use the real sample times, so they hold when the sampling
stride changes during a video.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from shapely import contains_xy

from src.scene import Scene
from src.tracking import Tracks

NO_LANE = -1

FEATURE_DTYPE = np.dtype([
    ("frame", np.int32),
    ("t", np.float64),
    ("track_id", np.int32),
    ("cls", np.int16),
    ("x1", np.float32),           # raw box, normalised
    ("y1", np.float32),
    ("x2", np.float32),
    ("y2", np.float32),
    ("x", np.float32),            # smoothed anchor, normalised
    ("y", np.float32),
    ("size", np.float32),         # smoothed box width, frame widths: the local scale of the scene (perspective)
    ("edge", np.bool_),           # the box touches the frame border (object cut off)
    ("vx", np.float32),           # frame widths / s
    ("vy", np.float32),
    ("speed", np.float32),
    ("accel", np.float32),        # along the direction of motion (braking < 0), frame widths / s^2
    ("heading", np.float32),      # degrees, NaN when too slow to have a direction
    ("lane", np.int8),            # index into scene.lanes, NO_LANE outside every lane
    ("on_carriageway", np.bool_),  # for pedestrians: at least pedestrian_inset inside the edge
    ("in_crossing", np.bool_),
    ("in_intersection", np.bool_),
    ("dwell", np.float32),        # seconds the object has been standing, up to this sample
])


def track_slices(ids: np.ndarray) -> Iterator[slice]:
    """Slices of consecutive equal track ids (rows sorted by track, then frame)."""
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(ids)]):
        yield slice(int(start), int(end))


def windowed_mean(t: np.ndarray, values: np.ndarray, window: float) -> np.ndarray:
    """Centred moving average over +-window/2 seconds (irregular sampling allowed)."""
    lo = np.searchsorted(t, t - window / 2, side="left")
    hi = np.searchsorted(t, t + window / 2, side="right")
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def central_difference(t: np.ndarray, values: np.ndarray, span: float) -> np.ndarray:
    """d(values)/dt from the samples about span/2 before and after each sample (one-sided at the ends)."""
    if len(t) < 2:
        return np.zeros_like(values)
    lo = np.clip(np.searchsorted(t, t - span / 2, side="left"), 0, len(t) - 1)
    hi = np.clip(np.searchsorted(t, t + span / 2, side="right") - 1, 0, len(t) - 1)
    same = hi == lo
    lo = np.where(same, np.maximum(lo - 1, 0), lo)
    hi = np.where(same & (hi == lo), np.minimum(hi + 1, len(t) - 1), hi)
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    return (values[hi] - values[lo]) / dt[:, None]


def dwell_times(t: np.ndarray, standing: np.ndarray) -> np.ndarray:
    """Seconds each sample has been part of an uninterrupted standing run."""
    dwell = np.zeros(len(t))
    for i in range(1, len(t)):
        if standing[i] and standing[i - 1]:
            dwell[i] = dwell[i - 1] + t[i] - t[i - 1]
    return dwell


def compute_features(tracks: Tracks, scene: Scene, params: dict[str, Any]) -> np.ndarray:
    """Feature table (``FEATURE_DTYPE``), one row per tracked box, sorted by track id then frame.

    Args:
        tracks: tracker output of one video.
        scene: scene aligned to the same video.
        params: full parameter dict.
    """
    fp = params["features"]
    rows = np.sort(tracks.rows, order=["track_id", "frame"])
    aspect = tracks.info.height / tracks.info.width
    out = np.zeros(len(rows), dtype=FEATURE_DTYPE)
    out["frame"], out["track_id"], out["cls"] = rows["frame"], rows["track_id"], rows["cls"]
    for key in ("x1", "y1", "x2", "y2"):
        out[key] = rows[key]
    out["t"] = rows["frame"] / tracks.info.fps
    anchor = np.stack([(rows["x1"] + rows["x2"]) / 2, rows["y2"]], axis=1).astype(np.float64)
    width = (rows["x2"] - rows["x1"]).astype(np.float64)[:, None]
    margin = fp["edge_margin"]
    out["edge"] = ((rows["x1"] < margin) | (rows["y1"] < margin) | (rows["x2"] > 1 - margin)
                   | (rows["y2"] > 1 - margin))

    for sl in track_slices(rows["track_id"]):
        t = out["t"][sl]
        xy = windowed_mean(t, anchor[sl], fp["smooth_window_sec"])
        vel = central_difference(t, xy * [1.0, aspect], fp["velocity_window_sec"])
        acc = central_difference(t, vel, fp["velocity_window_sec"])
        speed = np.hypot(vel[:, 0], vel[:, 1])
        direction = vel / np.maximum(speed, 1e-9)[:, None]
        out["x"][sl], out["y"][sl] = xy[:, 0], xy[:, 1]
        out["size"][sl] = windowed_mean(t, width[sl], fp["smooth_window_sec"])[:, 0]
        out["vx"][sl], out["vy"][sl], out["speed"][sl] = vel[:, 0], vel[:, 1], speed
        out["accel"][sl] = np.sum(acc * direction, axis=1)
        heading = np.degrees(np.arctan2(vel[:, 1], vel[:, 0])) % 360
        out["heading"][sl] = np.where(speed >= fp["heading_min_speed"], heading, np.nan)
        out["dwell"][sl] = dwell_times(t, speed < fp["stationary_speed"])

    x, y = out["x"].astype(np.float64), out["y"].astype(np.float64)
    out["lane"] = NO_LANE
    for i, lane in enumerate(scene.lanes):
        inside = contains_xy(lane.polygon, x, y) & (out["lane"] == NO_LANE)
        out["lane"][inside] = i
    xy = np.stack([x, y], axis=1)
    person = np.isin(out["cls"], fp["pedestrian_classes"])
    out["on_carriageway"] = scene.on_carriageway(xy)
    out["on_carriageway"][person] = scene.on_carriageway(xy[person], fp["pedestrian_inset"], aspect)
    out["in_crossing"] = scene.in_crossing(xy)
    out["in_intersection"] = contains_xy(scene.intersection, x, y)
    return out


def fit_car_width(features: np.ndarray, car_class: int = 2, min_samples: int = 200) -> np.ndarray:
    """Linear fit of car box width (frame widths) against image y: the scene's perspective scale.

    Returns ``[slope, intercept]`` for ``np.polyval``; measured ~[0.15, 0.01] on the samples.
    """
    car = (features["cls"] == car_class) & ~features["edge"]
    edges = np.linspace(0.0, 1.0, 21)
    ys, widths = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = car & (features["y"] >= lo) & (features["y"] < hi)
        if m.sum() >= min_samples:
            ys.append((lo + hi) / 2)
            widths.append(float(np.median(features["size"][m])))
    return np.polyfit(ys, widths, 1)
