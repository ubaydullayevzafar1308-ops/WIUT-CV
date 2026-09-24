"""Standing traffic: stopped_vehicle and congestion."""
from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity, contains_xy

from src.features import track_slices
from src.postprocess import Segment, flags_to_runs
from src.rules.base import VideoContext

SIGNAL_LANE = "avenue_near"   # the approach controlled by the visible signal


def sample_durations(t: np.ndarray) -> np.ndarray:
    """Time each sample stands for: up to the next sample (the last one gets the median spacing)."""
    if len(t) < 2:
        return np.zeros(len(t))
    dt = np.diff(t)
    return np.r_[dt, np.median(dt)]


def travel_direction(ctx: VideoContext) -> np.ndarray:
    """Unit direction each vehicle was last seen moving in (pixel-aspect space); NaN before it first moved."""
    f = ctx.features
    moving = f["speed"] >= ctx.params["features"]["heading_min_speed"]
    direction = np.full((len(f), 2), np.nan)
    for sl in track_slices(f["track_id"]):
        v = np.stack([f["vx"][sl], f["vy"][sl]], axis=1).astype(np.float64)
        idx = np.where(moving[sl], np.arange(len(v)), -1)
        last = np.maximum.accumulate(idx)
        seen = last >= 0
        unit = v / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)[:, None]
        direction[np.flatnonzero(seen) + sl.start] = unit[last[seen]]
    return direction


def in_queue(ctx: VideoContext, standing: np.ndarray) -> np.ndarray:
    """Standing vehicles that are part of a queue: another standing vehicle just ahead, or several around.

    "Ahead" follows the direction the vehicle last moved in (the lane direction if
    it never moved), so it works inside the intersection, where no lane is set.
    Distances are measured in the vehicle's own box widths, which follows the
    perspective (a car near the camera is ~3x wider than one at the far kerb).
    """
    p = ctx.params["rules"]["stopped_vehicle"]
    f = ctx.features
    direction = travel_direction(ctx)
    out = np.zeros(len(f), dtype=bool)
    idx = np.flatnonzero(standing)
    order = idx[np.argsort(f["frame"][idx], kind="stable")]
    frames = f["frame"][order]
    bounds = np.flatnonzero(np.r_[True, frames[1:] != frames[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        if len(rows) < 2:
            continue
        xy = np.stack([f["x"][rows], f["y"][rows] * ctx.aspect], axis=1).astype(np.float64)
        for k, row in enumerate(rows):
            rel = (np.delete(xy, k, axis=0) - xy[k]) / max(float(f["size"][row]), 1e-6)
            dist = np.hypot(rel[:, 0], rel[:, 1])
            if np.sum(dist < p["jam_radius"]) >= p["jam_neighbours"]:
                out[row] = True
                continue
            heads = [direction[row]] if not np.isnan(direction[row]).any() else (
                list(ctx.scene.lanes[f["lane"][row]].directions) if f["lane"][row] >= 0 else [])
            for d in heads:
                ahead, lateral = rel @ d, np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
                if np.any((ahead > 0) & (ahead < p["queue_gap"]) & (lateral < p["queue_lateral"])):
                    out[row] = True
                    break
    return out


def yielding_at_crossing(ctx: VideoContext, standing: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Standing vehicles waiting for a pedestrian on the crossing right in front of them.

    A sample is yielding while the vehicle stands at a crossing (anchor within
    ``yield_distance`` box widths of it) and a pedestrian is on that crossing
    ahead of the vehicle, in its lane. Only the first ``max_yield_sec`` of each
    uninterrupted yielding run are excused; waiting longer counts as standing.
    """
    p = ctx.params["rules"]["stopped_vehicle"]
    f = ctx.features
    person = np.isin(f["cls"], ctx.params["features"]["pedestrian_classes"])
    xy = np.stack([f["x"], f["y"] * ctx.aspect], axis=1).astype(np.float64)
    out = np.zeros(len(f), dtype=bool)
    candidates = np.flatnonzero(standing & ~np.isnan(direction[:, 0]))
    for polygon in ctx.scene.crossings.values():
        scaled = affinity.scale(polygon, xfact=1.0, yfact=ctx.aspect, origin=(0, 0))
        at_crossing = candidates[shapely.distance(scaled, shapely.points(xy[candidates, 0], xy[candidates, 1]))
                                 < p["yield_distance"] * f["size"][candidates]]
        on_crossing = np.flatnonzero(person & contains_xy(polygon, f["x"].astype(np.float64), f["y"].astype(np.float64)))
        by_frame: dict[int, list[int]] = {}
        for row in on_crossing:
            by_frame.setdefault(int(f["frame"][row]), []).append(row)
        for row in at_crossing:
            peds = by_frame.get(int(f["frame"][row]))
            if peds is None:
                continue
            rel = (xy[peds] - xy[row]) / f["size"][row]
            d = direction[row]
            ahead, lateral = rel @ d, np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
            out[row] |= bool(np.any((ahead > 0) & (ahead < p["yield_reach"]) & (lateral < p["yield_lateral"])))
    for sl in track_slices(f["track_id"]):
        t, flags = f["t"][sl], out[sl]
        for start, end in flags_to_runs(t, flags):
            flags[(t >= start + p["max_yield_sec"]) & (t < end)] = False
        out[sl] = flags
    return out


class StoppedVehicle:
    """A vehicle stationary on the carriageway >= 10 s that traffic goes round.

    The segment runs from the moment the vehicle stops until it moves again.
    Standing time counts towards the 10 s only while the box is inside the
    frame, the vehicle is not in a stopping zone (bus stop, kerbside parking),
    not held by the visible signal (avenue_near until the queue has discharged),
    not part of a queue (``in_queue``) and not yielding to a pedestrian on the
    crossing in front of it (``yielding_at_crossing``). A stop is kept only if
    the vehicle was seen driving up to it, at least ``min_overtakers`` other vehicles
    drive past it in its direction while it stands (when the whole lane stands,
    it is a queue), no standing neighbour ahead moves off together with it, and, on
    an approach held by a signal that is not visible, no other vehicle stands
    next to it.
    """

    label = "stopped_vehicle"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        """Every stop of >= min_stop_sec standing, with the reason it is kept or rejected."""
        p = ctx.params["rules"]["stopped_vehicle"]
        f = ctx.features
        xy = np.stack([f["x"], f["y"]], axis=1).astype(np.float64)
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        standing = vehicle & f["on_carriageway"] & (f["speed"] < ctx.params["features"]["stationary_speed"])
        since_green = ctx.seconds_since_green(f["t"])
        direction = travel_direction(ctx)
        reasons = {
            "cut off by the frame edge": f["edge"],
            "stopping zone (bus stop / parking)": ctx.scene.in_stopping_zone(xy),
            "held by the visible signal": (f["lane"] == ctx.lane_index(SIGNAL_LANE))
            & ~(since_green >= p["signal_discharge_sec"]),
            "queue or jam": in_queue(ctx, standing),
            "yielding to pedestrians": yielding_at_crossing(ctx, standing, direction),
        }
        excluded = np.zeros(len(f), dtype=bool)
        for mask in reasons.values():
            excluded |= mask
        counts = standing & ~excluded

        found = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            if not standing[sl].any():
                continue
            dt = sample_durations(t)
            for start, stop_end in self._stops(t, standing[sl], p["run_merge_sec"]):
                inside = (t >= start) & (t < stop_end)
                if (dt * standing[sl])[inside].sum() < p["min_stop_sec"]:
                    continue
                rows = np.flatnonzero(inside) + sl.start
                counted = float((dt * counts[sl])[inside].sum())
                candidate = {"track_id": int(f["track_id"][rows[0]]), "start": start, "end": stop_end,
                             "counted_sec": round(counted, 1), "x": float(np.median(f["x"][rows])),
                             "y": float(np.median(f["y"][rows])), "size": float(np.median(f["size"][rows]))}
                candidate["overtakers"] = self._overtakers(ctx, rows, direction, p)
                candidate["arrived"] = bool(np.any(f["speed"][sl][t < start] > p["arrive_speed"]))
                candidate["reason"], candidate["kept"] = self._verdict(ctx, rows, candidate, standing, direction, reasons, p)
                found.append(candidate)
        return found

    def _verdict(self, ctx: VideoContext, rows: np.ndarray, c: dict, standing: np.ndarray, direction: np.ndarray,
                 reasons: dict[str, np.ndarray], p: dict) -> tuple[str, bool]:
        if not c["arrived"]:
            return "standing since it first appeared (not seen driving up)", False
        if c["counted_sec"] < p["min_stop_sec"]:
            share = {name: float(mask[rows].mean()) for name, mask in reasons.items()}
            return max(share, key=share.get), False
        if self._queued_at_hidden_signal(ctx, rows, standing, p):
            return "queue at the signal that is not visible", False
        if self._departs_together(ctx, rows, c, standing, direction, p):
            return "moved off together with standing neighbours (queue)", False
        if c["overtakers"] < p["min_overtakers"]:
            return f"not overtaken ({c['overtakers']} vehicles passed): the lane stands", False
        return f"stands {c['counted_sec']} s while {c['overtakers']} vehicles drive round it", True

    @staticmethod
    def _neighbours(ctx: VideoContext, row: int, others: np.ndarray, radius: float) -> np.ndarray:
        f = ctx.features
        d = np.hypot(f["x"][others] - f["x"][row], (f["y"][others] - f["y"][row]) * ctx.aspect)
        return others[d < radius * f["size"][row]]

    def _overtakers(self, ctx: VideoContext, rows: np.ndarray, direction: np.ndarray, p: dict) -> int:
        """Distinct other vehicles moving past the stopped one, in its direction, while it stands."""
        f = ctx.features
        heads = direction[rows[0]]
        lane = f["lane"][rows[0]]
        dirs = [heads] if not np.isnan(heads).any() else (list(ctx.scene.lanes[lane].directions) if lane >= 0 else [])
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        moving = np.flatnonzero(vehicle & (f["speed"] > p["overtake_speed"]) & (f["track_id"] != f["track_id"][rows[0]])
                                & (f["t"] >= f["t"][rows[0]]) & (f["t"] <= f["t"][rows[-1]]))
        near = self._neighbours(ctx, rows[len(rows) // 2], moving, p["overtake_radius"])
        if dirs and len(near):
            v = np.stack([f["vx"][near], f["vy"][near]], axis=1).astype(np.float64)
            v /= np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)[:, None]
            same = np.max(v @ np.stack(dirs).T, axis=1) >= np.cos(np.radians(p["overtake_max_angle"]))
            near = near[same]
        return len(np.unique(f["track_id"][near]))

    def _queued_at_hidden_signal(self, ctx: VideoContext, rows: np.ndarray, standing: np.ndarray, p: dict) -> bool:
        f = ctx.features
        mid = rows[len(rows) // 2]
        if not ctx.scene.in_signal_queue_zone(np.array([[f["x"][mid], f["y"][mid]]], dtype=np.float64))[0]:
            return False
        others = np.flatnonzero(standing & (f["track_id"] != f["track_id"][mid])
                                & (f["t"] >= f["t"][rows[0]]) & (f["t"] <= f["t"][rows[-1]]))
        return len(self._neighbours(ctx, mid, others, p["hidden_signal_radius"])) > 0

    def _departs_together(self, ctx: VideoContext, rows: np.ndarray, c: dict, standing: np.ndarray,
                          direction: np.ndarray, p: dict) -> bool:
        """A neighbour ahead, standing just before this vehicle moved off, also moves off within the window.

        Only neighbours ahead count: vehicles queued behind a stopped vehicle
        naturally move off when it does.
        """
        f = ctx.features
        last = rows[-1]
        window = p["depart_window_sec"]
        before = np.flatnonzero(standing & (f["track_id"] != f["track_id"][last])
                                & (f["t"] >= c["end"] - window) & (f["t"] < c["end"]))
        near = self._neighbours(ctx, last, before, p["depart_radius"])
        d = direction[last]
        if not np.isnan(d).any():
            rel = np.stack([f["x"][near] - f["x"][last], (f["y"][near] - f["y"][last]) * ctx.aspect], axis=1)
            near = near[rel @ d > 0]
        for track in np.unique(f["track_id"][near]):
            after = (f["track_id"] == track) & (f["t"] >= c["end"] - window) & (f["t"] <= c["end"] + window)
            if np.any(f["speed"][after] > p["overtake_speed"]):
                return True
        return False

    @staticmethod
    def _stops(t: np.ndarray, standing: np.ndarray, merge_sec: float) -> list[tuple[float, float]]:
        """Standing runs of one track, joined across short movements."""
        stops: list[list[float]] = []
        for start, end in flags_to_runs(t, standing):
            if stops and start - stops[-1][1] < merge_sec:
                stops[-1][1] = end
            else:
                stops.append([start, end])
        return [(s, e) for s, e in stops]


class Congestion:
    """Standstill or crawling traffic in a lane for longer than a signal cycle.

    Per sampled frame and lane: number of vehicles and their median speed,
    smoothed with a rolling median. The lane is congested while it holds at
    least ``min_vehicles`` with a median speed below ``crawl_speed``; only
    stretches longer than ``min_duration_sec`` count, so ordinary red-light
    queues (cleared every cycle) do not.
    """

    label = "congestion"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        p = ctx.params["rules"]["congestion"]
        f = ctx.features
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        t = ctx.sample_t
        frame_of = np.searchsorted(t, f["t"])
        step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        half = max(1, round(p["smooth_sec"] / step / 2))
        segments: list[Segment] = []
        for lane_idx in range(len(ctx.scene.lanes)):
            m = np.flatnonzero(vehicle & (f["lane"] == lane_idx))
            count = np.bincount(frame_of[m], minlength=len(t))
            speed = np.full(len(t), np.inf)
            order = m[np.argsort(frame_of[m], kind="stable")]
            frames, starts = np.unique(frame_of[order], return_index=True)
            for i, chunk in zip(frames, np.split(f["speed"][order], starts[1:])):
                speed[i] = np.median(chunk)
            count_s = np.array([np.median(count[max(0, i - half):i + half + 1]) for i in range(len(t))])
            speed_s = np.array([np.median(speed[max(0, i - half):i + half + 1]) for i in range(len(t))])
            jammed = (count_s >= p["min_vehicles"]) & (speed_s < p["crawl_speed"])
            segments += [[s, e, self.label] for s, e in flags_to_runs(t, jammed) if e - s >= p["min_duration_sec"]]
        return segments
