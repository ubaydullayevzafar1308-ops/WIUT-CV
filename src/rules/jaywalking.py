"""jaywalking: a pedestrian on the carriageway outside a crossing."""
from __future__ import annotations

import numpy as np
from shapely import affinity, contains_xy

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs


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


def near_crossing(ctx: VideoContext, rows: np.ndarray, margin: float) -> np.ndarray:
    """Which rows lie on a crossing widened by ``margin`` frame widths (people walking at the zebra's edge)."""
    f = ctx.features
    x, y = f["x"][rows].astype(np.float64), f["y"][rows].astype(np.float64) * ctx.aspect
    out = np.zeros(len(rows), dtype=bool)
    for polygon in ctx.scene.crossings.values():
        widened = affinity.scale(polygon, xfact=1.0, yfact=ctx.aspect, origin=(0, 0)).buffer(margin)
        out |= contains_xy(widened, x, y)
    return out


class Jaywalking:
    """A pedestrian on the carriageway outside a crossing for at least ``min_duration_sec``.

    On the carriageway uses the zone shrunk by ``pedestrian_inset`` (a person
    waiting on the kerb is not on the road). Excluded: islands, crossings widened
    by ``crossing_margin`` (people crossing at the edge of the zebra), stopping
    zones (boarding at the bus stop, walking to parked cars), and people riding
    (anchor inside a bicycle / motorcycle / vehicle box: riders, passengers).
    Tracks that never move (a signal head detected as a person) and people cut
    off by the frame edge are rejected. The segment runs from the step onto the
    road to leaving it.
    """

    label = "jaywalking"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        """Every run on the road of >= review_min_sec, with the reason it is kept or rejected."""
        p = ctx.params["rules"]["jaywalking"]
        f = ctx.features
        person = np.isin(f["cls"], ctx.params["features"]["pedestrian_classes"])
        xy = np.stack([f["x"], f["y"]], axis=1).astype(np.float64)
        on_road = person & f["on_carriageway"] & ~f["in_crossing"] & ~ctx.scene.in_stopping_zone(xy)
        idx = np.flatnonzero(on_road)
        on_road[idx[near_crossing(ctx, idx, p["crossing_margin"])]] = False
        rider = riding(ctx, on_road, p["rider_classes"])
        found = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            if not on_road[sl].any():
                continue
            x, y = f["x"][sl].astype(np.float64), f["y"][sl].astype(np.float64) * ctx.aspect
            travel = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
            for start, end in merged_runs(t, on_road[sl], p["run_merge_sec"]):
                if end - start < p["review_min_sec"]:
                    continue
                rows = np.flatnonzero((t >= start) & (t < end) & on_road[sl]) + sl.start
                c = {"track_id": int(f["track_id"][rows[0]]), "start": start, "end": end,
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                if travel < p["min_track_travel"]:
                    c["reason"], c["kept"] = f"static detection (track spans {travel:.3f} frame widths)", False
                elif f["edge"][rows].mean() > 0.5:
                    c["reason"], c["kept"] = "cut off by the frame edge", False
                elif rider[rows].mean() > 0.5:
                    c["reason"], c["kept"] = "riding (inside a bicycle / vehicle box)", False
                elif end - start < p["min_duration_sec"]:
                    c["reason"], c["kept"] = f"on the road only {end - start:.1f} s", False
                else:
                    c["reason"], c["kept"] = f"{end - start:.1f} s on the carriageway outside crossings", True
                c["close"] = not c["kept"] and (c["reason"].startswith("on the road only")
                                                or c["reason"] == "cut off by the frame edge")
                found.append(c)
        return found
