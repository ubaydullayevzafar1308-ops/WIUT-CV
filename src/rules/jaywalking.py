"""jaywalking: a pedestrian on the carriageway outside a crossing."""
from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs
from src.risk import ground_plane
from src.rules.ground import crossing_distance_m, road_depth_m
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


def near_standing_vehicles(ctx: VideoContext, rows: np.ndarray, radius: float) -> dict[str, np.ndarray]:
    """Which person ``rows`` are within ``radius`` (road plane, car box widths) of a vehicle standing at the kerb
    (in a stopping zone or within ``kerb_lanes`` of the carriageway edge) or on a crossing, in the same frame.

    People there are getting in or out, loading, or walking round a car that blocks the zebra.
    """
    p = ctx.params["rules"]["jaywalking"]
    f = ctx.features
    g = ctx.params["risk"]["ground"]
    out = {"next to a car standing at the kerb": np.zeros(len(rows), dtype=bool),
           "walking round a car standing on the zebra": np.zeros(len(rows), dtype=bool)}
    vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
    standing = vehicle & (f["dwell"] >= p["standing_car_sec"])
    cars = np.flatnonzero(standing)
    if not len(cars) or not len(rows):
        return out
    xy = np.stack([f["x"][cars], f["y"][cars]], axis=1).astype(np.float64)
    edge = shapely.distance(affinity.scale(ctx.scene.carriageway.exterior, xfact=1.0, yfact=ctx.aspect, origin=(0, 0)),
                            shapely.points(xy[:, 0], xy[:, 1] * ctx.aspect))
    kerb = ctx.scene.in_stopping_zone(xy) | (edge <= p["kerb_lanes"] * ctx.lane_width(xy[:, 1]))
    zebra = ctx.scene.in_crossing(xy)
    car_pos = ground_plane(f["x"][cars], f["y"][cars], g, ctx.aspect)
    person_pos = ground_plane(f["x"][rows], f["y"][rows], g, ctx.aspect)
    order = np.argsort(f["frame"][cars], kind="stable")
    frames = f["frame"][cars][order]
    for k, row in enumerate(rows):
        lo, hi = np.searchsorted(frames, f["frame"][row], "left"), np.searchsorted(frames, f["frame"][row], "right")
        same = order[lo:hi]
        close = same[np.hypot(*(car_pos[same] - person_pos[k]).T) <= radius]
        out["next to a car standing at the kerb"][k] = bool(kerb[close].any())
        out["walking round a car standing on the zebra"][k] = bool(zebra[close].any())
    return out


class Jaywalking:
    """A pedestrian walking over the asphalt of the carriageway outside the crossings.

    On the road = on the carriageway at least ``min_road_depth_m`` inside its
    edge and away from the islands (a person on the kerb or the far pavement is
    not on it), more than ``crossing_margin_m`` from every crossing (people walk
    along the edge of a zebra) and outside stopping zones (the bus stop, parking);
    distances on the road plane (``rules.ground``). A run on the road is kept if it
    lasts ``min_duration_sec`` and the person covers ``min_walk_m`` - including
    cutting the corner between zebras and islands - at a median speed of at most
    ``max_speed_mps`` (faster: a scooter or a bicycle, not a pedestrian).
    Rejected: runs mostly within ``car_radius_m`` of a vehicle
    standing at the kerb (getting in or out, loading) or on a crossing (walking
    round it), and unreliable detections (see ``pedestrians.reliable_pedestrians``:
    static objects such as signal heads, gantry and poles, riders, tiny far-away
    boxes, frame edge). The segment runs from the step onto the road to leaving
    it; simultaneous walkers merge into one event in post-processing.
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
        on_road[idx[(crossing_distance_m(ctx, idx) <= p["crossing_margin_m"])
                    | (road_depth_m(ctx, idx) < p["min_road_depth_m"])]] = False
        idx = np.flatnonzero(on_road)
        names, dist = zone_distances(ctx, idx)
        lane = ctx.lane_width(f["y"][idx])
        nearest = np.full(len(f), -1)
        nearest_lanes = np.full(len(f), np.inf)
        nearest[idx] = dist.argmin(axis=1)
        nearest_lanes[idx] = dist.min(axis=1) / lane
        road_rows = np.flatnonzero(on_road)
        near_cars = {name: np.zeros(len(f), dtype=bool) for name in ("next to a car standing at the kerb",
                                                                      "walking round a car standing on the zebra")}
        metres = ctx.params["risk"]["ground"]["box_width_m"]
        for name, mask in near_standing_vehicles(ctx, road_rows, p["car_radius_m"] / metres).items():
            near_cars[name][road_rows] = mask
        ground = ground_plane(f["x"], f["y"], ctx.params["risk"]["ground"], ctx.aspect) * metres
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
                walked = float(np.hypot(*(ground[last] - ground[first])))
                steps = np.hypot(*np.diff(ground[rows], axis=0).T) / np.maximum(np.diff(f["t"][rows]), 1e-6)
                speed = float(np.median(steps)) if len(steps) else 0.0
                c = {"track_id": int(f["track_id"][first]), "start": start, "end": end, "walked_m": round(walked, 1),
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                from_zone = names[nearest[first]] if nearest_lanes[first] <= p["corner_near_lanes"] else "road"
                to_zone = names[nearest[last]] if nearest_lanes[last] <= p["corner_near_lanes"] else "road"
                unreliable = {name: float(m[rows].mean()) for name, m in masks.items() if name != "person"}
                unreliable |= {name: float(m[rows].mean()) for name, m in near_cars.items()}
                worst = max(unreliable, key=unreliable.get)
                if unreliable[worst] > 0.5:
                    c["reason"], c["kept"] = worst, False
                elif end - start < p["min_duration_sec"]:
                    c["reason"], c["kept"] = f"on the road only {end - start:.1f} s", False
                elif speed > p["max_speed_mps"]:
                    c["reason"], c["kept"] = f"moving at {speed:.1f} m/s: a scooter or a bicycle, not a pedestrian", False
                elif walked < p["min_walk_m"]:
                    c["reason"], c["kept"] = f"on the road {end - start:.1f} s, but walked only {walked:.1f} m", False
                else:
                    c["reason"], c["kept"] = (f"walked {walked:.1f} m in {end - start:.1f} s on the road outside "
                                              f"crossings ({from_zone} -> {to_zone})"), True
                c["close"] = not c["kept"] and c["reason"].startswith(("on the road", "next to a car", "walking round",
                                                                          "cut off"))
                found.append(c)
        return found
