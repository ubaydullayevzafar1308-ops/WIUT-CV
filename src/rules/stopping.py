"""Standing traffic: stopped_vehicle and congestion."""
from __future__ import annotations

import numpy as np

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


class StoppedVehicle:
    """A vehicle stationary on the carriageway >= 10 s, not in a queue at the signal or behind another vehicle.

    The segment runs from the moment the vehicle stops until it moves again.
    Standing time only counts towards the 10 s while the vehicle is not held by
    the signal (avenue_near outside green, or in the first seconds of green while
    the queue discharges), not part of a queue (see ``in_queue``) and not in a
    stopping zone of the scene (bus stop, kerbside parking).
    """

    label = "stopped_vehicle"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[stop["start"], stop["end"], self.label] for stop in self.stops(ctx)]

    def stops(self, ctx: VideoContext) -> list[dict]:
        """Qualifying stops with the vehicle: track_id, start, end, counted seconds, anchor x/y and box size."""
        p = ctx.params["rules"]["stopped_vehicle"]
        f = ctx.features
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        standing = vehicle & f["on_carriageway"] & (f["speed"] < ctx.params["features"]["stationary_speed"])
        parked = ctx.scene.in_stopping_zone(np.stack([f["x"], f["y"]], axis=1).astype(np.float64))
        since_green = ctx.seconds_since_green(f["t"])
        held_by_signal = (f["lane"] == ctx.lane_index(SIGNAL_LANE)) & ~(since_green >= p["signal_discharge_sec"])
        counts = standing & ~parked & ~held_by_signal & ~in_queue(ctx, standing)

        found: list[dict] = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            if not standing[sl].any():
                continue
            weight = sample_durations(t) * counts[sl]
            for start, end in self._stops(t, standing[sl], p["run_merge_sec"]):
                inside = (t >= start) & (t < end)
                counted = float(weight[inside].sum())
                if counted >= p["min_stop_sec"]:
                    found.append({"track_id": int(f["track_id"][sl][0]), "start": start, "end": end,
                                  "counted_sec": round(counted, 1),
                                  "x": float(np.median(f["x"][sl][inside])), "y": float(np.median(f["y"][sl][inside])),
                                  "size": float(np.median(f["size"][sl][inside]))})
        return found

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
