"""jaywalking: a pedestrian on the carriageway away from any crossing."""
from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity, contains_xy

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs
from src.rules.stopping import sample_durations


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


def distance_to_refuges(ctx: VideoContext, rows: np.ndarray) -> np.ndarray:
    """Distance (frame widths, aspect-corrected) from each row to the nearest crossing or island."""
    f = ctx.features
    zones = [*ctx.scene.crossings.values(), *ctx.scene.islands.values()]
    union = shapely.union_all([affinity.scale(z, xfact=1.0, yfact=ctx.aspect, origin=(0, 0)) for z in zones])
    points = shapely.points(f["x"][rows].astype(np.float64), f["y"][rows].astype(np.float64) * ctx.aspect)
    return shapely.distance(union, points)


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
    """A pedestrian on the carriageway, away from every crossing and island.

    On the carriageway uses the zone shrunk by ``pedestrian_inset`` (a person
    waiting on the kerb is not on the road). Excluded: crossings widened by
    ``crossing_margin`` (people at the zebra's edge), islands, stopping zones
    (boarding at the bus stop, walking to parked cars) and riders / passengers
    (anchor inside a bicycle / motorcycle / vehicle box). A run on the road is
    kept if it lasts ``min_duration_sec`` and the person is more than
    ``far_lanes`` lane widths from every crossing and island for at least
    ``far_min_sec`` of it: a short walk from one zebra to the next (cutting the
    corner) is not an event. Tracks that never move (a signal head detected as a
    person) and people cut off by the frame edge are rejected. The segment runs
    from the step onto the road to leaving it.
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
        idx = np.flatnonzero(on_road)
        far = np.zeros(len(f), dtype=bool)
        far[idx] = distance_to_refuges(ctx, idx) > p["far_lanes"] * ctx.lane_width(f["y"][idx])
        rider = riding(ctx, on_road, p["rider_classes"])
        found = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            if not on_road[sl].any():
                continue
            x, y = f["x"][sl].astype(np.float64), f["y"][sl].astype(np.float64) * ctx.aspect
            travel = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
            dt = sample_durations(t)
            for start, end in merged_runs(t, on_road[sl], p["run_merge_sec"]):
                if end - start < p["review_min_sec"]:
                    continue
                inside = (t >= start) & (t < end)
                rows = np.flatnonzero(inside & on_road[sl]) + sl.start
                far_sec = float((dt * far[sl])[inside].sum())
                c = {"track_id": int(f["track_id"][rows[0]]), "start": start, "end": end, "far_sec": round(far_sec, 1),
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                if travel < p["min_track_travel"]:
                    c["reason"], c["kept"] = f"static detection (track spans {travel:.3f} frame widths)", False
                elif f["edge"][rows].mean() > 0.5:
                    c["reason"], c["kept"] = "cut off by the frame edge", False
                elif rider[rows].mean() > 0.5:
                    c["reason"], c["kept"] = "riding (inside a bicycle / vehicle box)", False
                elif end - start < p["min_duration_sec"]:
                    c["reason"], c["kept"] = f"on the road only {end - start:.1f} s", False
                elif far_sec < p["far_min_sec"]:
                    c["reason"], c["kept"] = (f"stays near a crossing or island (far from them {far_sec:.1f} s): "
                                              "walking from one zebra to another"), False
                else:
                    c["reason"], c["kept"] = (f"{end - start:.1f} s on the road, {far_sec:.1f} s of it more than "
                                              f"{p['far_lanes']} lanes from any crossing"), True
                c["close"] = not c["kept"] and (c["reason"].startswith(("on the road only", "stays near"))
                                                or c["reason"] == "cut off by the frame edge")
                found.append(c)
        return found
