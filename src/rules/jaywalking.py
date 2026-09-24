"""jaywalking: a pedestrian on the carriageway outside a crossing."""
from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs
from src.rules.pedestrians import reliable_pedestrians


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
    """A pedestrian walking on the carriageway outside a crossing, not just cutting a corner.

    On the carriageway uses the zone shrunk by ``pedestrian_inset`` (a person
    waiting on the kerb is not on the road). Excluded: points within
    ``crossing_margin_lanes`` lane widths of a crossing (people at the zebra's
    edge), islands and stopping zones (boarding at the bus stop, walking to
    parked cars). A run on the road is kept if it lasts ``min_duration_sec`` and
    the person walks ``min_walk_boxes`` of their own box widths (not standing at
    the kerb or median), unless it starts and ends at crossings or islands within
    ``corner_cut_sec`` (cutting the corner, or a loop off an island and back).
    Unreliable detections are rejected (see ``pedestrians.reliable_pedestrians``:
    static objects such as signal heads, gantry and poles, riders, tiny far-away
    boxes, frame edge). The segment runs from the step onto the road to leaving it.
    """

    label = "jaywalking"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        """Every run on the road of >= review_min_sec, with the reason it is kept or rejected."""
        p = ctx.params["rules"]["jaywalking"]
        f = ctx.features
        masks = reliable_pedestrians(ctx)
        xy = np.stack([f["x"], f["y"]], axis=1).astype(np.float64)
        on_road = masks["person"] & f["on_carriageway"] & ~f["in_crossing"] & ~ctx.scene.in_stopping_zone(xy)
        idx = np.flatnonzero(on_road)
        names, dist = zone_distances(ctx, idx)
        lane = ctx.lane_width(f["y"][idx])
        crossing_cols = [i for i, n in enumerate(names) if n.startswith("crossing:")]
        on_road[idx[dist[:, crossing_cols].min(axis=1) < p["crossing_margin_lanes"] * lane]] = False
        nearest = np.full(len(f), -1)
        nearest_lanes = np.full(len(f), np.inf)
        nearest[idx] = dist.argmin(axis=1)
        nearest_lanes[idx] = dist.min(axis=1) / lane
        found = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            if not on_road[sl].any():
                continue
            for start, end in merged_runs(t, on_road[sl], p["run_merge_sec"]):
                if end - start < p["review_min_sec"]:
                    continue
                rows = np.flatnonzero((t >= start) & (t < end) & on_road[sl]) + sl.start
                first, last = rows[0], rows[-1]
                walked = float(np.hypot(f["x"][last] - f["x"][first], (f["y"][last] - f["y"][first]) * ctx.aspect)
                               / max(float(np.median(f["size"][rows])), 1e-6))
                c = {"track_id": int(f["track_id"][first]), "start": start, "end": end, "walked_boxes": round(walked, 1),
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                from_zone = names[nearest[first]] if nearest_lanes[first] <= p["corner_near_lanes"] else None
                to_zone = names[nearest[last]] if nearest_lanes[last] <= p["corner_near_lanes"] else None
                unreliable = {name: float(m[rows].mean()) for name, m in masks.items() if name != "person"}
                worst = max(unreliable, key=unreliable.get)
                if unreliable[worst] > 0.5:
                    c["reason"], c["kept"] = worst, False
                elif end - start < p["min_duration_sec"]:
                    c["reason"], c["kept"] = f"on the road only {end - start:.1f} s", False
                elif from_zone and to_zone and end - start <= p["corner_cut_sec"]:
                    kind = "loop off" if from_zone == to_zone else "cutting the corner:"
                    c["reason"], c["kept"] = f"{kind} {from_zone} -> {to_zone} in {end - start:.1f} s", False
                elif walked < p["min_walk_boxes"]:
                    c["reason"], c["kept"] = f"standing on the road, walked only {walked:.1f} box widths", False
                else:
                    route = f"{from_zone or 'road'} -> {to_zone or 'road'}"
                    c["reason"], c["kept"] = (f"walked {walked:.1f} box widths in {end - start:.1f} s on the road "
                                              f"outside crossings ({route})"), True
                c["close"] = not c["kept"] and c["reason"].startswith(
                    ("on the road only", "cutting the corner", "loop off", "standing on the road", "cut off"))
                found.append(c)
        return found
