"""Which person detections are real pedestrians (shared by jaywalking and failure_to_yield)."""
from __future__ import annotations

import numpy as np
import shapely

from src.features import track_slices
from src.rules.base import VideoContext


def riding(ctx: VideoContext, person: np.ndarray, rider_classes: list[int]) -> np.ndarray:
    """Person samples whose anchor lies inside a vehicle or bicycle box of the same frame (rider, passenger)."""
    f = ctx.features
    out = np.zeros(len(f), dtype=bool)
    carriers = np.flatnonzero(np.isin(f["cls"], rider_classes))
    order = carriers[np.argsort(f["frame"][carriers], kind="stable")]
    frames = f["frame"][order]
    for row in np.flatnonzero(person):
        lo, hi = np.searchsorted(frames, f["frame"][row], side="left"), np.searchsorted(frames, f["frame"][row], side="right")
        boxes = order[lo:hi]
        if len(boxes):
            px, py = f["x"][row], f["y"][row]
            out[row] = bool(np.any((f["x1"][boxes] <= px) & (px <= f["x2"][boxes])
                                   & (f["y1"][boxes] <= py) & (py <= f["y2"][boxes])))
    return out


def on_structures(ctx: VideoContext, rows: np.ndarray) -> np.ndarray:
    """Which rows' boxes overlap a structure of the scene (signal gantry, poles): heads detected as people."""
    if not ctx.scene.structures:
        return np.zeros(len(rows), dtype=bool)
    f = ctx.features
    union = shapely.union_all(list(ctx.scene.structures.values()))
    boxes = shapely.box(f["x1"][rows], f["y1"][rows], f["x2"][rows], f["y2"][rows])
    return shapely.intersects(union, boxes)


def static_tracks(ctx: VideoContext, person: np.ndarray, min_travel_boxes: float) -> np.ndarray:
    """Person rows of tracks whose whole life spans fewer than ``min_travel_boxes`` of their own box widths."""
    f = ctx.features
    out = np.zeros(len(f), dtype=bool)
    for sl in track_slices(f["track_id"]):
        if not person[sl.start]:
            continue
        x, y = f["x"][sl].astype(np.float64), f["y"][sl].astype(np.float64) * ctx.aspect
        span = np.hypot(x.max() - x.min(), y.max() - y.min()) / max(float(np.median(f["size"][sl])), 1e-6)
        out[sl] = span < min_travel_boxes
    return out


def reliable_pedestrians(ctx: VideoContext) -> dict[str, np.ndarray]:
    """Person rows and the reason masks that disqualify them.

    Returns ``{"person": ..., "static detection": ..., "on a gantry or pole": ...,
    "riding": ..., "too small": ..., "cut off by the frame edge": ...}``; a row is a
    reliable pedestrian if it is a person and in none of the other masks.
    """
    p = ctx.params["rules"]["pedestrians"]
    f = ctx.features
    person = np.isin(f["cls"], ctx.params["features"]["pedestrian_classes"])
    rows = np.flatnonzero(person)
    structure = np.zeros(len(f), dtype=bool)
    structure[rows] = on_structures(ctx, rows)
    return {
        "person": person,
        "static detection": static_tracks(ctx, person, p["min_track_travel_boxes"]),
        "on a gantry or pole": structure,
        "riding": riding(ctx, person, p["rider_classes"]),
        "too small": person & ((f["y2"] - f["y1"]) < p["min_height"]),
        "cut off by the frame edge": person & f["edge"],
    }


def reliable_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    out = masks["person"].copy()
    for name, mask in masks.items():
        if name != "person":
            out &= ~mask
    return out
