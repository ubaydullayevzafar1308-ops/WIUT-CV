"""illegal_turn and solid_line_crossing: lane discipline on approach A (the near carriageway of the avenue).

Lanes of approach A are numbered from the kerb (1, by the signal pole) towards
the median; a vehicle's lane is where its anchor crosses the stop line, between
the lane boundaries drawn in scene.json (``approach_lanes``). From lane 1 the
only way is right into the side street; from lanes 2 and beyond the side street
is forbidden. Vehicles from the right edge of the frame (approach B) are free to
go anywhere and are not checked.
"""
from __future__ import annotations

import numpy as np
from shapely import contains_xy

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs
from src.rules.collisions import track_jumps
from src.rules.signal_rules import ALONG_SLACK, StopLine, stop_line

SIDE_STREET = "side_street"


def boundary_along(line: StopLine, coords: np.ndarray, aspect: float) -> float:
    """Where a lane boundary (extended as a straight line) meets the stop line, as a position along it."""
    p, q = coords[0] * [1.0, aspect], coords[-1] * [1.0, aspect]
    d = q - p
    denom = d[0] * line.unit[1] - d[1] * line.unit[0]
    s = ((line.p0[0] - p[0]) * line.unit[1] - (line.p0[1] - p[1]) * line.unit[0]) / denom
    point = p + s * d
    return float((point - line.p0) @ line.unit / line.length)


def signed_distance(coords: np.ndarray, x: np.ndarray, y: np.ndarray, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of points to a polyline (aspect-corrected frame widths; the sign tells the side) and
    the position of their nearest point along it (0 at its start, 1 at its end)."""
    pts = np.stack([x, y * aspect], axis=1).astype(np.float64)
    c = coords * [1.0, aspect]
    seg = np.diff(c, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.r_[0.0, np.cumsum(seg_len)]
    rel = pts[:, None, :] - c[None, :-1, :]
    t = np.clip(np.sum(rel * seg[None], axis=2) / seg_len ** 2, 0.0, 1.0)
    nearest = c[None, :-1, :] + t[..., None] * seg[None]
    dist = np.hypot(*(pts[:, None, :] - nearest).transpose(2, 0, 1))
    k = dist.argmin(axis=1)
    rows = np.arange(len(pts))
    cross = seg[k, 0] * rel[rows, k, 1] - seg[k, 1] * rel[rows, k, 0]
    return np.sign(cross) * dist[rows, k], (cum[k] + t[rows, k] * seg_len[k]) / cum[-1]


class IllegalTurn:
    """A vehicle from approach A that leaves its lane the wrong way.

    Only turns count (task definition: a turn from the wrong lane). From lane 1
    the only turn allowed is right into the side street: turning left (reaching
    one of ``left_lanes``) is an illegal turn, going straight on (``straight_lanes``)
    is not a turn. From lanes 2 and beyond it must not enter the side street (its anchor reaches the side street
    exit, beyond the zebra), also not by driving through the intersection and
    round the triangle island. Vehicles crossing the stop line within
    ``lane_margin`` (along the line) of a lane boundary have an uncertain lane
    and are not judged. The segment runs from the start of the manoeuvre (the
    vehicle moves off after the stop line and its heading leaves the approach
    direction by ``turn_start_deg``) until it reaches its
    destination.
    """

    label = "illegal_turn"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        """Every vehicle from approach A that reaches a destination, with its lane and verdict."""
        p = ctx.params["rules"]["illegal_turn"]
        f = ctx.features
        scene = ctx.scene
        if not scene.lane_boundaries or SIDE_STREET not in scene.exits:
            return []
        line = stop_line(scene, ctx.aspect, scene.approach_stop_line)
        bounds = np.array([boundary_along(line, np.asarray(scene.solid_lines[b].coords), ctx.aspect)
                           for b in scene.lane_boundaries])
        dist, along = line.distance(f["x"], f["y"], ctx.aspect), line.along(f["x"], f["y"], ctx.aspect)
        x, y = f["x"].astype(np.float64), f["y"].astype(np.float64)
        side = contains_xy(scene.exits[SIDE_STREET], x, y)
        straight = np.isin(f["lane"], [ctx.lane_index(name) for name in p["straight_lanes"]])
        left = np.isin(f["lane"], [ctx.lane_index(name) for name in p["left_lanes"]])
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        approach_dir = next(ln for ln in scene.lanes if ln.id == scene.approach_stop_line).directions[0]
        found = []
        for sl in track_slices(f["track_id"]):
            if not vehicle[sl.start]:
                continue
            d, a, t = dist[sl], along[sl], f["t"][sl]
            cross = np.flatnonzero((d[:-1] >= 0) & (d[1:] < 0) & (a[1:] > -ALONG_SLACK) & (a[1:] < 1 + ALONG_SLACK)) + 1
            if not len(cross):
                continue
            k = int(cross[0])
            after = np.arange(len(t)) >= k
            to_side = np.flatnonzero(after & side[sl])
            to_left, to_straight = np.flatnonzero(after & left[sl]), np.flatnonzero(after & straight[sl])
            if len(to_side):   # the side street wins: round the triangle island it passes the exit leg first
                destination, arrive = SIDE_STREET, int(to_side[0])
            elif len(to_left):
                destination, arrive = "left", int(to_left[0])
            elif len(to_straight):
                destination, arrive = "straight", int(to_straight[0])
            else:
                continue   # lost in the intersection: destination unknown
            into_side = destination == SIDE_STREET
            lane = int(np.sum(a[k] > bounds)) + 1
            margin = float(np.min(np.abs(a[k] - bounds)))
            start = self._manoeuvre_start(ctx, sl, k, arrive, None if destination == "straight" else approach_dir, p)
            c = {"track_id": int(f["track_id"][sl.start]), "start": start, "end": float(t[arrive]),
                 "lane": lane, "along": round(float(a[k]), 3), "destination": destination,
                 "x": float(f["x"][sl.start + k]), "y": float(f["y"][sl.start + k])}
            wrong = (lane == 1 and destination == "left") or (lane > 1 and into_side)
            where = {SIDE_STREET: "into the side street", "left": "left", "straight": "straight on"}[destination]
            if margin < p["lane_margin"]:
                c["reason"], c["kept"] = (f"lane uncertain (along {c['along']}, {margin:.3f} from a lane boundary), "
                                          f"went {where}"), False
            elif wrong:
                c["reason"], c["kept"] = f"from lane {lane} {where}", True
            else:
                c["reason"], c["kept"] = f"from lane {lane} {where}: allowed", False
            c["close"] = c["kept"] or margin < p["lane_margin"]
            found.append(c)
        return found

    @staticmethod
    def _manoeuvre_start(ctx: VideoContext, sl: slice, k: int, arrive: int, approach_dir: np.ndarray | None,
                         p: dict) -> float:
        """First sample from the stop-line crossing on where the vehicle moves (and, for a turn, heads away from
        the approach direction by ``turn_start_deg``)."""
        f = ctx.features
        rows = np.arange(sl.start + k, sl.start + arrive + 1)
        ok = f["speed"][rows] >= p["min_speed"]
        if approach_dir is not None:
            heading = np.stack([f["vx"][rows], f["vy"][rows]], axis=1) / np.maximum(f["speed"][rows], 1e-9)[:, None]
            ok &= heading @ approach_dir < np.cos(np.radians(p["turn_start_deg"]))
        hits = np.flatnonzero(ok)
        return float(f["t"][rows[hits[0]] if len(hits) else rows[-1]])


class SolidLineCrossing:
    """A vehicle crosses a solid line (``scene.solid_lines``) within its drawn extent.

    Its anchor changes side, having been on the old side for ``stable_sec``
    before and staying on the new side for ``stable_sec`` after, at least
    ``clear_boxes`` of its box width away from the line on both sides (not
    jitter along it), while moving and without box leaps (tracker switching
    objects). Crossings beyond the ends of a line (``extent_margin``) do not
    count. The segment runs from the first sample where the bottom edge of its
    box straddles the line (a wheel on the line) until it is fully on the other
    side (in the new lane).
    """

    label = "solid_line_crossing"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        p = ctx.params["rules"]["solid_line_crossing"]
        f = ctx.features
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        jump = track_jumps(ctx, ctx.params["rules"]["accident"]["max_box_speed"],
                           ctx.params["rules"]["accident"]["max_size_ratio"])
        found = []
        for name, geometry in ctx.scene.solid_lines.items():
            coords = np.asarray(geometry.coords)
            dist, pos = signed_distance(coords, f["x"], f["y"], ctx.aspect)
            left, _ = signed_distance(coords, f["x1"], f["y2"], ctx.aspect)
            right, _ = signed_distance(coords, f["x2"], f["y2"], ctx.aspect)
            straddle = np.sign(left) != np.sign(right)
            clear = p["clear_boxes"] * f["size"]
            for sl in track_slices(f["track_id"]):
                if not vehicle[sl.start]:
                    continue
                s, t = np.sign(dist[sl]), f["t"][sl]
                changes = np.flatnonzero((s[:-1] * s[1:] < 0) & ~f["edge"][sl][1:]) + 1
                for k in changes:
                    inside = p["extent_margin"] < pos[sl][k] < 1 - p["extent_margin"]
                    before = (t >= t[k] - p["stable_sec"]) & (t < t[k])
                    after = (t >= t[k]) & (t <= t[k] + p["stable_sec"])
                    stable = (t[k] - t[0] >= p["stable_sec"] and t[-1] - t[k] >= p["stable_sec"]
                              and np.all(s[before] == s[k - 1]) and np.all(s[after] == s[k])
                              and np.abs(dist[sl][before]).max(initial=0) >= clear[sl][k]
                              and np.abs(dist[sl][after]).max(initial=0) >= clear[sl][k])
                    if not inside or not stable:
                        continue
                    start_k = k - 1
                    while start_k > 0 and straddle[sl][start_k - 1]:
                        start_k -= 1
                    start_k = start_k if straddle[sl][start_k] else k
                    end_k = k
                    while end_k < len(t) - 1 and straddle[sl][end_k]:
                        end_k += 1
                    c = {"track_id": int(f["track_id"][sl.start]), "line": name, "start": float(t[start_k]),
                         "end": float(max(t[end_k], t[start_k] + p["min_len_sec"])),
                         "x": float(f["x"][sl.start + k]), "y": float(f["y"][sl.start + k])}
                    window = (t >= c["start"] - p["stable_sec"]) & (t <= c["end"] + p["stable_sec"])
                    if f["speed"][sl][k] < p["min_speed"]:
                        c["reason"], c["kept"] = f"on {name}, but barely moving (box jitter)", False
                    elif jump[sl][window].any():
                        c["reason"], c["kept"] = f"on {name}, but the box leaps (tracker switched objects)", False
                    else:
                        c["reason"], c["kept"] = f"crossed solid line {name}", True
                    c["close"] = True
                    found.append(c)
        return found
