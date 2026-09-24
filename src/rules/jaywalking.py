"""jaywalking: a pedestrian on the carriageway outside a crossing."""
from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity

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


def zone_distances(ctx: VideoContext, rows: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Names of all crossings and islands and each row's distance to each (frame widths, aspect-corrected)."""
    f = ctx.features
    zones = {**{f"crossing:{k}": v for k, v in ctx.scene.crossings.items()},
             **{f"island:{k}": v for k, v in ctx.scene.islands.items()}}
    points = shapely.points(f["x"][rows].astype(np.float64), f["y"][rows].astype(np.float64) * ctx.aspect)
    dist = np.stack([shapely.distance(affinity.scale(z, xfact=1.0, yfact=ctx.aspect, origin=(0, 0)), points)
                     for z in zones.values()], axis=1)
    return list(zones), dist


class Jaywalking:
    """A pedestrian on the carriageway outside a crossing, not just cutting the corner between two.

    On the carriageway uses the zone shrunk by ``pedestrian_inset`` (a person
    waiting on the kerb is not on the road). Excluded: points within
    ``crossing_margin_lanes`` lane widths of a crossing (people at the zebra's
    edge), islands, stopping zones (boarding at the bus stop, walking to parked
    cars) and riders / passengers (anchor inside a bicycle / motorcycle / vehicle
    box). A run on the road is kept if it lasts ``min_duration_sec``, unless it
    goes from one crossing or island to a different one within
    ``corner_cut_sec`` (cutting the corner). People walking along the kerb on the
    road count. Tracks that never move (a signal head detected as a person) and
    people cut off by the frame edge are rejected. The segment runs from the step
    onto the road to leaving it.
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
        names, dist = zone_distances(ctx, idx)
        lane = ctx.lane_width(f["y"][idx])
        crossing_cols = [i for i, n in enumerate(names) if n.startswith("crossing:")]
        on_road[idx[dist[:, crossing_cols].min(axis=1) < p["crossing_margin_lanes"] * lane]] = False
        nearest = np.full(len(f), -1)
        nearest_lanes = np.full(len(f), np.inf)
        nearest[idx] = dist.argmin(axis=1)
        nearest_lanes[idx] = dist.min(axis=1) / lane
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
                first, last = rows[0], rows[-1]
                c = {"track_id": int(f["track_id"][first]), "start": start, "end": end,
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                from_zone = names[nearest[first]] if nearest_lanes[first] <= p["corner_near_lanes"] else None
                to_zone = names[nearest[last]] if nearest_lanes[last] <= p["corner_near_lanes"] else None
                if travel < p["min_track_travel"]:
                    c["reason"], c["kept"] = f"static detection (track spans {travel:.3f} frame widths)", False
                elif f["edge"][rows].mean() > 0.5:
                    c["reason"], c["kept"] = "cut off by the frame edge", False
                elif rider[rows].mean() > 0.5:
                    c["reason"], c["kept"] = "riding (inside a bicycle / vehicle box)", False
                elif end - start < p["min_duration_sec"]:
                    c["reason"], c["kept"] = f"on the road only {end - start:.1f} s", False
                elif from_zone and to_zone and from_zone != to_zone and end - start <= p["corner_cut_sec"]:
                    c["reason"], c["kept"] = f"cutting the corner: {from_zone} -> {to_zone} in {end - start:.1f} s", False
                else:
                    route = f"{from_zone or 'road'} -> {to_zone or 'road'}"
                    c["reason"], c["kept"] = f"{end - start:.1f} s on the road outside crossings ({route})", True
                c["close"] = not c["kept"] and (c["reason"].startswith(("on the road only", "cutting the corner"))
                                                or c["reason"] == "cut off by the frame edge")
                found.append(c)
        return found
