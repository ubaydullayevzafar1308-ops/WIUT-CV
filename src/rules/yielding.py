"""failure_to_yield: a vehicle drives onto a crossing while a pedestrian is on it in the vehicle's path."""
from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity, contains_xy

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs
from src.rules.pedestrians import reliable_mask, reliable_pedestrians
from src.rules.stopping import travel_direction

SIGNAL_CROSSING = "avenue_near"   # the crossing on the approach whose signal is read
STEP_LOOKAHEAD_SEC = 0.5          # "walking towards the crossing" = closer to it this much later
CROSSING_TOUCH = 0.02             # lanes within this distance (normalised) of a crossing are the ones that cross it


def crossing_axis(polygon, aspect: float) -> np.ndarray:
    """Unit vector of a crossing's long axis (the direction pedestrians walk), aspect-corrected."""
    rect = affinity.scale(polygon, xfact=1.0, yfact=aspect, origin=(0, 0)).minimum_rotated_rectangle
    corners = np.asarray(rect.exterior.coords)[:3]
    sides = np.diff(corners, axis=0)
    longest = sides[np.argmax(np.hypot(sides[:, 0], sides[:, 1]))]
    return longest / np.hypot(*longest)


def along_is_measurable(ctx: VideoContext, polygon, axis: np.ndarray, min_angle: float) -> bool:
    """Whether image headings can tell 'along the crossing' from 'across it': its axis must be at least
    ``min_angle`` degrees from the directions of the lanes that cross it (perspective can make them look parallel)."""
    near = polygon.buffer(CROSSING_TOUCH)
    angles = [np.degrees(np.arccos(min(1.0, abs(float(axis @ d)))))
              for lane in ctx.scene.lanes if lane.polygon.intersects(near) for d in lane.directions]
    return bool(angles) and min(angles) >= min_angle


class FailureToYield:
    """A vehicle drives through a crossing while a pedestrian is on it (or stepping onto it) in its path.

    The vehicle drives onto the crossing (its anchor, the bottom centre of the box,
    is on it at some point; vehicles in stopping zones, e.g. parking at the
    corner, are not checked); a reliable pedestrian (see
    ``pedestrians.reliable_pedestrians``) walking on the same crossing or within
    ``ped_margin_lanes`` of it, at most ``path_lanes`` lane widths to the side of
    the vehicle's line of travel (its lane or a step beside it) and not more than
    ``behind_lanes`` behind it, for at least ``min_conflict_samples`` samples,
    while the vehicle keeps moving across the crossing (it drives through rather
    than waits, and does not ride along it with the pedestrians, which is only
    measurable where perspective does not make the crossing look parallel to its
    traffic). The segment is the whole pass: from the first sample where the
    vehicle's box touches the crossing to the first where it is fully off it (or
    the vehicle leaves the frame); passes of one vehicle closer than
    ``merge_track_gap_sec`` are one event. On the near crossing, whose signal is
    read, it only counts while the vehicles do not have green: then the
    pedestrians cross legally; on green they cross against the signal.
    Crossings in ``excluded_crossings`` (signal not visible) are not checked.
    """

    label = "failure_to_yield"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        """Every vehicle pass over a crossing with a pedestrian on it, with the reason it is kept or rejected."""
        p = ctx.params["rules"]["failure_to_yield"]
        f = ctx.features
        vehicle = (np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
                   & ~ctx.scene.in_stopping_zone(np.stack([f["x"], f["y"]], axis=1).astype(np.float64)))
        vehicle_rows = np.flatnonzero(vehicle)
        boxes = shapely.box(f["x1"][vehicle_rows], f["y1"][vehicle_rows], f["x2"][vehicle_rows], f["y2"][vehicle_rows])
        pedestrian = reliable_mask(reliable_pedestrians(ctx))
        direction = travel_direction(ctx)
        unit = np.stack([f["vx"], f["vy"]], axis=1).astype(np.float64)
        speed = np.hypot(unit[:, 0], unit[:, 1])
        moving = speed >= ctx.params["features"]["heading_min_speed"]
        direction[moving] = unit[moving] / speed[moving][:, None]
        xy = np.stack([f["x"], f["y"] * ctx.aspect], axis=1).astype(np.float64)
        found = []
        for name, polygon in ctx.scene.crossings.items():
            if name in p["excluded_crossings"]:
                continue
            on_crossing = np.zeros(len(f), dtype=bool)
            on_crossing[vehicle_rows] = shapely.intersects(polygon, boxes)
            wheels_on = vehicle & contains_xy(polygon, f["x"].astype(np.float64), f["y"].astype(np.float64))
            peds = self._pedestrians_at(ctx, polygon, pedestrian, p)
            axis = crossing_axis(polygon, ctx.aspect)
            check_along = along_is_measurable(ctx, polygon, axis, p["along_check_min_angle"])
            for sl in track_slices(f["track_id"]):
                if not on_crossing[sl].any():
                    continue
                t = f["t"][sl]
                for start, end in merged_runs(t, on_crossing[sl], ctx.params["rules"]["wrong_way"]["run_merge_sec"]):
                    rows = np.flatnonzero((t >= start) & (t < end) & on_crossing[sl]) + sl.start
                    if not wheels_on[rows].any():
                        continue   # only the box overlaps the crossing (tall vehicle passing beside it)
                    present, conflict = 0, 0
                    for row in rows[~f["edge"][rows]]:
                        others = peds.get(int(f["frame"][row]))
                        if others is None:
                            continue
                        present += 1
                        d = direction[row]
                        if np.isnan(d).any():
                            continue
                        rel = xy[others] - xy[row]
                        lane = float(ctx.lane_width(f["y"][row]))
                        lateral, ahead = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]), rel @ d
                        conflict += bool(np.any((lateral <= p["path_lanes"] * lane) & (ahead >= -p["behind_lanes"] * lane)))
                    if not present:
                        continue
                    c = {"track_id": int(f["track_id"][sl.start]), "start": start, "end": end, "crossing": name,
                         "conflict_samples": conflict, "x": float(np.median(f["x"][rows])),
                         "y": float(np.median(f["y"][rows]))}
                    vehicle_phase = str(ctx.signal.phase_at(start))
                    phase = f", vehicle phase {vehicle_phase}" if name == SIGNAL_CROSSING else ""
                    known = rows[~np.isnan(direction[rows, 0])]
                    heading = np.median(direction[known], axis=0) if len(known) else np.zeros(2)
                    along = abs(float(heading @ axis)) / max(float(np.hypot(*heading)), 1e-9)
                    if f["edge"][rows].mean() > 0.5:
                        c["reason"], c["kept"] = "cut off by the frame edge", False
                    elif np.median(f["speed"][rows]) < p["min_speed"]:
                        c["reason"], c["kept"] = f"waited on crossing {name}{phase}", False
                    elif check_along and along > np.cos(np.radians(p["along_angle"])):
                        c["reason"], c["kept"] = f"rode along crossing {name} (not across it)", False
                    elif name == SIGNAL_CROSSING and vehicle_phase not in p["legal_phases"]:
                        c["reason"], c["kept"] = (f"pedestrian on crossing {name} against the signal "
                                                  f"(vehicles have {vehicle_phase})"), False
                    elif conflict < p["min_conflict_samples"]:
                        c["reason"], c["kept"] = (f"pedestrian on crossing {name} but in the path only "
                                                  f"{conflict} sample(s){phase}"), False
                    else:
                        c["reason"], c["kept"] = (f"drove over crossing {name} with a pedestrian in its path "
                                                  f"({conflict} samples){phase}"), True
                    c["close"] = not c["kept"] and conflict > 0
                    found.append(c)
        return self._one_per_vehicle(found, p["merge_track_gap_sec"])

    @staticmethod
    def _one_per_vehicle(found: list[dict], gap: float) -> list[dict]:
        """Join the kept passes of one vehicle that overlap or are closer than ``gap`` seconds."""
        kept = sorted((c for c in found if c["kept"]), key=lambda c: (c["track_id"], c["start"]))
        joined: list[dict] = []
        for c in kept:
            last = joined[-1] if joined else None
            if last and last["track_id"] == c["track_id"] and c["start"] - last["end"] < gap:
                last["end"] = max(last["end"], c["end"])
                last["conflict_samples"] += c["conflict_samples"]
                if c["crossing"] not in last["crossing"]:
                    last["crossing"] += f"+{c['crossing']}"
                    last["reason"] += f"; then {c['reason']}"
            else:
                joined.append(dict(c))
        return joined + [c for c in found if not c["kept"]]

    @staticmethod
    def _pedestrians_at(ctx: VideoContext, polygon, pedestrian: np.ndarray, p: dict) -> dict[int, np.ndarray]:
        """Rows of reliable pedestrians walking on the crossing, or stepping onto it, grouped by frame.

        Stepping onto it = within ``ped_margin_lanes`` of the crossing and walking towards it;
        people standing (waiting at the kerb or on its edge) do not count.
        """
        f = ctx.features
        rows = np.flatnonzero(pedestrian)
        x, y = f["x"][rows].astype(np.float64), f["y"][rows].astype(np.float64)
        scaled = affinity.scale(polygon, xfact=1.0, yfact=ctx.aspect, origin=(0, 0))
        dist = shapely.distance(scaled, shapely.points(x, y * ctx.aspect))
        ahead = shapely.points(x + f["vx"][rows] * STEP_LOOKAHEAD_SEC, y * ctx.aspect + f["vy"][rows] * STEP_LOOKAHEAD_SEC)
        towards = shapely.distance(scaled, ahead) < dist
        walking = f["speed"][rows] >= p["walk_speed"]
        stepping = (dist <= p["ped_margin_lanes"] * ctx.lane_width(f["y"][rows])) & towards
        rows = rows[walking & ((dist == 0) | stepping)]
        by_frame: dict[int, list[int]] = {}
        for row in rows:
            by_frame.setdefault(int(f["frame"][row]), []).append(int(row))
        return {frame: np.array(r) for frame, r in by_frame.items()}
