"""accident and near_miss: pairs of road users on the road plane.

Positions and velocities use the image -> road-plane mapping of Part B
(``risk.ground_plane``, car box widths) and its time to contact
(``risk.pair_ttc``). Unlike Part B, the rules see the whole track, so they
can check what happens after a contact or a close approach.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import shapely

from src.features import central_difference, track_slices
from src.risk import ground_plane, pair_ttc
from src.rules.base import VideoContext, merged_runs
from src.rules.pedestrians import reliable_mask, reliable_pedestrians


def ground_kinematics(ctx: VideoContext) -> dict[str, np.ndarray]:
    """Per feature row: road-plane position, velocity, speed, and acceleration along / across the motion."""
    f = ctx.features
    fp = ctx.params["features"]
    pos = ground_plane(f["x"], f["y"], ctx.params["risk"]["ground"], ctx.aspect)
    vel, acc = np.zeros_like(pos), np.zeros_like(pos)
    for sl in track_slices(f["track_id"]):
        vel[sl] = central_difference(f["t"][sl], pos[sl], fp["velocity_window_sec"])
        acc[sl] = central_difference(f["t"][sl], vel[sl], fp["velocity_window_sec"])
    speed = np.hypot(vel[:, 0], vel[:, 1])
    unit = vel / np.maximum(speed, 1e-9)[:, None]
    return {"pos": pos, "vel": vel, "speed": speed, "along": np.sum(acc * unit, axis=1),
            "lateral": unit[:, 0] * acc[:, 1] - unit[:, 1] * acc[:, 0]}


def footprints(ctx: VideoContext, depth: float) -> np.ndarray:
    """Road-plane polygon under each box: its bottom strip, ``depth`` of the box height tall."""
    f = ctx.features
    g = ctx.params["risk"]["ground"]
    top = f["y2"] - depth * (f["y2"] - f["y1"])
    xs = np.stack([f["x1"], f["x2"], f["x2"], f["x1"]], axis=1)
    ys = np.stack([f["y2"], f["y2"], top, top], axis=1)
    return shapely.polygons(ground_plane(xs, ys, g, ctx.aspect))


def participants(ctx: VideoContext) -> tuple[np.ndarray, np.ndarray]:
    """Rows that can be involved, and which are vehicles.

    Vehicles at least ``min_vehicle_width`` wide (far away the road-plane positions are too
    coarse) and reliable pedestrians on the carriageway; boxes cut off by the frame edge are left out.
    """
    f = ctx.features
    vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
    walker = reliable_mask(reliable_pedestrians(ctx)) & f["on_carriageway"]
    near = vehicle & (f["size"] >= ctx.params["rules"]["near_miss"]["min_vehicle_width"])
    return (near | walker) & ~f["edge"], vehicle


def track_jumps(ctx: VideoContext, max_box_speed: float, max_size_ratio: float) -> np.ndarray:
    """Rows where the box leaps from the previous sample of its track: faster than ``max_box_speed``
    of its own widths per second, or its width changes by more than ``max_size_ratio`` (the tracker
    switched to another object or merged two)."""
    f = ctx.features
    out = np.zeros(len(f), dtype=bool)
    x, y = (f["x1"] + f["x2"]) / 2, f["y2"] * ctx.aspect
    width = np.maximum(f["x2"] - f["x1"], 1e-6)
    for sl in track_slices(f["track_id"]):
        if sl.stop - sl.start < 2:
            continue
        dt = np.maximum(np.diff(f["t"][sl]), 1e-6)
        step = np.hypot(np.diff(x[sl]), np.diff(y[sl])) / width[sl][1:] / dt
        ratio = np.exp(np.abs(np.diff(np.log(width[sl]))))
        out[sl.start + 1:sl.stop] = (step > max_box_speed) | (ratio > max_size_ratio)
    return out


def pair_samples(ctx: VideoContext, kin: dict[str, np.ndarray], radius: float) -> dict[str, np.ndarray]:
    """Every pair of participants seen in the same frame within ``radius``, at least one of them a vehicle.

    Returns rows ``a``, ``b`` (``track_id[a] < track_id[b]``) sorted by pair, then time.
    """
    f = ctx.features
    member, vehicle = participants(ctx)
    rows = np.flatnonzero(member)
    rows = rows[np.argsort(f["frame"][rows], kind="stable")]
    bounds = np.flatnonzero(np.r_[True, np.diff(f["frame"][rows]) != 0, True])
    a_all, b_all = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = rows[lo:hi]
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        rel = kin["pos"][b] - kin["pos"][a]
        keep = (vehicle[a] | vehicle[b]) & (np.hypot(rel[:, 0], rel[:, 1]) < radius)
        a_all.append(a[keep])
        b_all.append(b[keep])
    a = np.concatenate(a_all or [np.zeros(0, int)])
    b = np.concatenate(b_all or [np.zeros(0, int)])
    swap = f["track_id"][a] > f["track_id"][b]
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    order = np.lexsort((f["t"][a], f["track_id"][b], f["track_id"][a]))
    return {"a": a[order], "b": b[order]}


def leaves_frame(ctx: VideoContext, rows: np.ndarray, margin: float) -> bool:
    """Whether a track's last box is within ``margin`` of the frame border (it drove out, not got occluded)."""
    f = ctx.features
    last = rows[-1]
    return bool(f["edge"][last] or f["x1"][last] < margin or f["x2"][last] > 1 - margin or f["y2"][last] > 1 - margin)


def standing_after(ctx: VideoContext, track_id: int, t0: float, within: float, stand: float,
                   exit_margin: float) -> float | None:
    """When the track comes to stand for ``stand`` s, or leaves the frame, within ``within`` s of ``t0``.

    A track that ends inside the frame (occlusion, lost) neither stands nor leaves: None.
    """
    f = ctx.features
    rows = np.flatnonzero((f["track_id"] == track_id) & (f["t"] >= t0))
    if not len(rows):
        return None
    t = f["t"][rows]
    still = f["speed"][rows] < ctx.params["features"]["stationary_speed"]
    for s, e in merged_runs(t, still, 0.0):
        if s > t0 + within:
            break
        if e - s >= stand:
            return float(s)
    return float(t[-1]) if t[-1] <= t0 + within and leaves_frame(ctx, rows, exit_margin) else None


class Collisions:
    """Shared pair analysis of one video for ``Accident`` and ``NearMiss`` (cached per context)."""

    def __init__(self, ctx: VideoContext) -> None:
        self.ctx = ctx
        rp = ctx.params["rules"]
        self.acc_p, self.nm_p = rp["accident"], rp["near_miss"]
        f = ctx.features
        self.kin = ground_kinematics(ctx)
        self.jump = track_jumps(ctx, self.acc_p["max_box_speed"], self.acc_p["max_size_ratio"])
        self.pairs = pair_samples(ctx, self.kin, self.nm_p["pair_radius"])
        a, b = self.pairs["a"], self.pairs["b"]
        rel = self.kin["pos"][b] - self.kin["pos"][a]
        vel = self.kin["vel"][b] - self.kin["vel"][a]
        person = np.isin(f["cls"], ctx.params["features"]["pedestrian_classes"])
        rows = np.column_stack([rel, vel, (person[a] | person[b]).astype(np.float64)])
        self.dist = np.hypot(rel[:, 0], rel[:, 1])
        self.closing = -np.sum(rel * vel, axis=1) / np.maximum(self.dist, 1e-9)
        self.ttc = pair_ttc(rows, self.nm_p) if len(rows) else np.zeros(0)
        self.gap = np.full(len(a), np.inf)
        near = np.flatnonzero(self.dist < self.acc_p["check_radius"])
        if len(near):
            shapes = footprints(ctx, self.acc_p["footprint_depth"])
            self.gap[near] = shapely.distance(shapes[a[near]], shapes[b[near]])
        key = f["track_id"][a].astype(np.int64) * 1_000_000 + f["track_id"][b]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, int)
        self.groups = [slice(int(s), int(e)) for s, e in zip(starts, np.r_[starts[1:], len(key)])]

    def pair_info(self, sl: slice, k: int) -> dict[str, Any]:
        f = self.ctx.features
        a, b = self.pairs["a"][k], self.pairs["b"][k]
        return {"track_id": int(f["track_id"][a]), "other_id": int(f["track_id"][b]),
                "x": float((f["x"][a] + f["x"][b]) / 2), "y": float((f["y"][a] + f["y"][b]) / 2)}

    def speed_drop(self, row_track: int, t0: float) -> float:
        """Largest road-plane speed before ``t0`` minus the smallest after it (car box widths / s)."""
        f = self.ctx.features
        p = self.acc_p
        rows = np.flatnonzero(f["track_id"] == row_track)
        t, speed = f["t"][rows], self.kin["speed"][rows]
        before = speed[(t >= t0 - p["before_sec"]) & (t <= t0)]
        after = speed[(t >= t0) & (t <= t0 + p["after_sec"])]
        return float(before.max() - after.min()) if len(before) and len(after) else 0.0

    def accidents(self) -> list[dict]:
        """Contacts of two road users that approached each other, with the reason they are kept or rejected."""
        p = self.acc_p
        f = self.ctx.features
        found = []
        for sl in self.groups:
            contact = np.flatnonzero(self.gap[sl] <= p["contact_gap"])
            if not len(contact):
                continue
            k0 = sl.start + contact[0]
            t = f["t"][self.pairs["a"][sl]]
            t0 = float(t[contact[0]])
            before = (t >= t0 - p["before_sec"]) & (t <= t0)
            closing = float(self.closing[sl][before].max())
            if closing < p["min_closing"]:
                continue   # touching without driving into each other: queues, side by side, occlusion
            c = self.pair_info(sl, k0) | {"start": t0, "closing": closing}
            ids = (c["track_id"], c["other_id"])
            drops = [self.speed_drop(i, t0) for i in ids]
            settled = [standing_after(self.ctx, i, t0, p["settle_sec"], p["stand_sec"], p["exit_margin"]) for i in ids]
            c["end"] = max([s for s in settled if s is not None] + [t0 + p["min_len_sec"]])
            window = np.isin(f["track_id"], ids) & (f["t"] >= t0 - p["before_sec"]) & (f["t"] <= c["end"])
            if min(drops) < p["min_speed_drop"]:
                c["reason"], c["kept"] = (f"contact at closing {closing:.1f}, but speed drops {drops[0]:.1f} / "
                                          f"{drops[1]:.1f} (need {p['min_speed_drop']})"), False
            elif np.any(self.jump & window):
                c["reason"], c["kept"] = "contact and hard stop, but a box leaps (tracker switched objects)", False
            elif any(s is None for s in settled):
                c["reason"], c["kept"] = "contact and hard stop, but a participant drives on or its track is lost", False
            elif not self._stay_together(sl, settled, p):
                c["reason"], c["kept"] = "contact and both stopped, but apart (stopped next to each other)", False
            else:
                c["reason"], c["kept"] = (f"contact at closing {closing:.1f}, speed drops {drops[0]:.1f} / "
                                          f"{drops[1]:.1f}, both stopped or left"), True
            c["close"] = min(drops) >= p["min_speed_drop"]
            found.append(c)
        return found

    def _stay_together(self, sl: slice, settled: list[float], p: dict) -> bool:
        """Whether the two, once both stopped, keep touching for ``stand_sec`` (or one of them left the frame)."""
        t = self.ctx.features["t"][self.pairs["a"][sl]]
        start = max(settled)
        rows = (t >= start) & (t <= start + p["stand_sec"])
        if not rows.any():
            return True   # a participant left the frame
        return bool(np.median(self.gap[sl][rows]) <= p["contact_gap"])

    def near_misses(self) -> list[dict]:
        """Approaches with a very small time to contact, with the reason they are kept or rejected."""
        p = self.nm_p
        f = self.ctx.features
        found = []
        for sl in self.groups:
            a, b = self.pairs["a"][sl], self.pairs["b"][sl]
            t = f["t"][a]
            risky = self.ttc[sl] <= p["max_ttc"]
            for rs, re in merged_runs(t, risky, p["run_merge_sec"]):
                in_run = (t >= rs) & (t <= re)
                k_close = int(np.flatnonzero(in_run)[np.argmin(self.dist[sl][in_run])])
                t_close = float(t[k_close])
                after = t > t_close
                apart = np.flatnonzero(after & (self.dist[sl] >= p["separation"]))
                end = float(t[apart[0]]) if len(apart) else float(t[-1])
                c = self.pair_info(sl, sl.start + k_close) | {"ttc": float(self.ttc[sl][in_run].min())}
                evasive = self._evasive_start(c, rs - p["react_before_sec"], re + p["react_after_sec"])
                c["start"] = evasive if evasive is not None else rs
                c["end"] = max(end, c["start"] + p["min_len_sec"])
                touched = bool(np.any(self.gap[sl][(t >= rs) & (t <= end)] <= self.acc_p["contact_gap"]))
                stopped = [i for i in (c["track_id"], c["other_id"]) if self._stops(i, t_close)]
                if evasive is None:
                    c["reason"], c["kept"] = f"time to contact {c['ttc']:.2f} s, no hard braking or swerve", False
                elif np.any(self.jump & np.isin(f["track_id"], (c["track_id"], c["other_id"]))
                            & (f["t"] >= c["start"] - p["react_before_sec"]) & (f["t"] <= c["end"])):
                    c["reason"], c["kept"] = (f"time to contact {c['ttc']:.2f} s, but a box leaps "
                                              f"(tracker switched objects)"), False
                elif touched:
                    c["reason"], c["kept"] = f"time to contact {c['ttc']:.2f} s, but they touched", False
                elif stopped:
                    c["reason"], c["kept"] = (f"time to contact {c['ttc']:.2f} s and evasive action, "
                                              f"but a participant stopped after (queue)"), False
                else:
                    c["reason"], c["kept"] = (f"time to contact {c['ttc']:.2f} s, {c['action']}, "
                                              f"no contact, both drove on"), True
                c["close"] = True
                found.append(c)
        return found

    def _stops(self, track_id: int, t0: float) -> bool:
        """Whether the track stands still for ``stop_sec`` within ``after_sec`` of ``t0``."""
        p = self.nm_p
        f = self.ctx.features
        rows = np.flatnonzero((f["track_id"] == track_id) & (f["t"] >= t0) & (f["t"] <= t0 + p["after_sec"]))
        still = f["speed"][rows] < self.ctx.params["features"]["stationary_speed"]
        return any(e - s >= p["stop_sec"] for s, e in merged_runs(f["t"][rows], still, 0.0))

    def _evasive_start(self, c: dict, t_from: float, t_to: float) -> float | None:
        """First hard braking or swerve of either participant in [t_from, t_to]; records which in ``c``."""
        p = self.nm_p
        f = self.ctx.features
        best = None
        for track in (c["track_id"], c["other_id"]):
            rows = np.flatnonzero((f["track_id"] == track) & (f["t"] >= t_from) & (f["t"] <= t_to))
            moving = self.kin["speed"][rows] >= p["min_speed"]
            brake = moving & (self.kin["along"][rows] <= -p["brake_decel"])
            swerve = moving & (np.abs(self.kin["lateral"][rows]) >= p["swerve_accel"])
            for mask, action in ((brake, "hard braking"), (swerve, "swerve")):
                if mask.any():
                    t = float(f["t"][rows[mask][0]])
                    if best is None or t < best[0]:
                        best = (t, f"{action} of track {track}")
        if best is None:
            return None
        c["action"] = best[1]
        return best[0]


def analysis(ctx: VideoContext) -> Collisions:
    """The pair analysis of this context, computed once for both rules."""
    cached = getattr(ctx, "_collisions", None)
    if cached is None:
        cached = Collisions(ctx)
        object.__setattr__(ctx, "_collisions", cached)
    return cached


class Accident:
    """Two road users collide: their road-plane footprints touch while they drive into each other
    (closing speed at least ``min_closing``), both lose at least ``min_speed_drop`` of speed, and
    both then stand still (or leave the frame) within ``settle_sec``. The segment runs from the
    first contact until every participant has stopped or left."""

    label = "accident"

    def apply(self, ctx: VideoContext) -> list[list]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        return analysis(ctx).accidents()


class NearMiss:
    """A very small time to contact (``max_ttc``) with hard braking or a swerve of at least one
    participant, no contact, and neither stops afterwards. The segment runs from the start of the
    evasive action until the two are ``separation`` car box widths apart."""

    label = "near_miss"

    def apply(self, ctx: VideoContext) -> list[list]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        return analysis(ctx).near_misses()
