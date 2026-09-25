"""Signal violations on the avenue_near approach: red_light and stop_line.

The vehicle phase comes from the signal timeline (src/signal.py), which already
contains the 3 s of flashing green and 3 s of amber that follow the green man
going red. The stop line is ``scene.stop_lines["avenue_near"]``; distances to it
are signed (positive before the line, negative past it) in frame widths with
the vertical axis aspect-corrected. The anchor (bottom centre of the box) is the
front of a vehicle driving down the image, so "past the line" is measured on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs
from src.scene import Scene
from src.signal import GREEN

APPROACH = "avenue_near"
ALONG_SLACK = 0.05      # a crossing counts slightly beyond the drawn ends of the stop line


@dataclass(frozen=True)
class StopLine:
    """The stop line in aspect-corrected coordinates: start, unit direction, length, upstream sign."""

    p0: np.ndarray
    unit: np.ndarray
    length: float
    upstream: float

    def distance(self, x: np.ndarray, y: np.ndarray, aspect: float) -> np.ndarray:
        """Signed distance of points to the line (frame widths): > 0 before it, < 0 past it."""
        rel = np.stack([x, y * aspect], axis=1) - self.p0
        return self.upstream * (self.unit[0] * rel[:, 1] - self.unit[1] * rel[:, 0])

    def along(self, x: np.ndarray, y: np.ndarray, aspect: float) -> np.ndarray:
        """Position along the line: 0 at its start, 1 at its end."""
        rel = np.stack([x, y * aspect], axis=1) - self.p0
        return rel @ self.unit / self.length


def stop_line(scene: Scene, aspect: float, approach: str = APPROACH) -> StopLine:
    """The approach's stop line, oriented so that its lane lies on the positive side."""
    coords = np.asarray(scene.stop_lines[approach].coords) * [1.0, aspect]
    d = coords[-1] - coords[0]
    length = float(np.hypot(*d))
    line = StopLine(p0=coords[0], unit=d / length, length=length, upstream=1.0)
    centre = np.asarray(next(lane for lane in scene.lanes if lane.id == approach).polygon.centroid.coords)
    return StopLine(p0=line.p0, unit=line.unit, length=length,
                    upstream=float(np.sign(line.distance(centre[:, 0], centre[:, 1], aspect)[0])))


def next_green_start(green_starts: np.ndarray, t: float) -> float:
    """First green start after ``t`` (infinity if the video ends first)."""
    later = green_starts[green_starts > t]
    return float(later[0]) if len(later) else float("inf")


def phase_changes(ctx: VideoContext, phases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Times when the vehicle phase enters one of ``phases`` and when it turns green."""
    phase = ctx.signal.phase
    inside = np.isin(phase, phases)
    enter = ctx.signal.t[np.flatnonzero(inside & ~np.r_[False, inside[:-1]])]
    green = ctx.signal.t[np.flatnonzero((phase == GREEN) & np.r_[True, phase[:-1] != GREEN])]
    return enter, green


class RedLight:
    """A vehicle crosses the avenue_near stop line while its phase is red and drives on into the intersection.

    It must reach the intersection (or get beyond the near crossing) before the
    next green without standing still past the line; a vehicle that stops past
    the line is a stop_line case, never both.

    Crossings in the first or last ``phase_grace_sec`` of red are within the
    uncertainty of the phase reading (2 samples/s, smoothed) and are only kept
    for review, as are crossings on amber. The segment runs from crossing the line to
    leaving the intersection (or the frame).
    """

    label = "red_light"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        p = ctx.params["rules"]["red_light"]
        f = ctx.features
        line = stop_line(ctx.scene, ctx.aspect)
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        dist = line.distance(f["x"], f["y"], ctx.aspect)
        along = line.along(f["x"], f["y"], ctx.aspect)
        red_starts, green_starts = phase_changes(ctx, p["red_phases"])
        found = []
        for sl in track_slices(f["track_id"]):
            if not vehicle[sl.start]:
                continue
            d, t = dist[sl], f["t"][sl]
            hits = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)
                                  & (along[sl][1:] > -ALONG_SLACK) & (along[sl][1:] < 1 + ALONG_SLACK)) + 1
            for i in hits:
                tc = float(t[i])
                phase = str(ctx.signal.phase_at(tc))
                next_red = red_starts[red_starts > tc]
                red_soon = bool(len(next_red)) and next_red[0] - tc < p["review_window_sec"]
                if phase == GREEN and not red_soon:
                    continue
                c = {"track_id": int(f["track_id"][sl.start]), "start": tc, "end": self._leaves(f, sl, i),
                     "phase": phase, "x": float(f["x"][sl][i]), "y": float(f["y"][sl][i])}
                entered, stopped = self._goes_through(ctx, sl, i, dist, next_green_start(green_starts, tc))
                since_red = tc - red_starts[red_starts <= tc].max() if np.any(red_starts <= tc) else np.inf
                next_green = green_starts[green_starts > tc]
                to_green = next_green[0] - tc if len(next_green) else np.inf
                if f["edge"][sl][i]:
                    c["reason"], c["kept"] = "cut off by the frame edge", False
                elif phase not in p["red_phases"]:
                    c["reason"], c["kept"] = (f"crossed on {phase} (red {next_red[0] - tc:.1f} s later)"
                                              if red_soon else f"crossed on {phase}"), False
                elif since_red < p["phase_grace_sec"]:
                    c["reason"], c["kept"] = f"crossed {since_red:.1f} s after red began (phase boundary)", False
                elif to_green < p["phase_grace_sec"]:
                    c["reason"], c["kept"] = f"crossed {to_green:.1f} s before green (phase boundary)", False
                elif stopped:
                    c["reason"], c["kept"] = "crossed on red and stopped past the line (stop_line)", False
                elif not entered:
                    c["reason"], c["kept"] = "crossed on red but did not reach the intersection before green", False
                else:
                    c["reason"], c["kept"] = f"crossed {since_red:.1f} s into {phase} and drove into the intersection", True
                c["close"] = not c["kept"]
                found.append(c)
        return found

    @staticmethod
    def _goes_through(ctx: VideoContext, sl: slice, i: int, dist: np.ndarray, green: float) -> tuple[bool, bool]:
        """(entered, stopped): whether after crossing at sample ``i`` the vehicle drove into the intersection
        (or beyond the near crossing) before ``green``, and whether it stood still past the line on the way."""
        f = ctx.features
        rows = np.arange(sl.start + i, sl.stop)
        rows = rows[f["t"][rows] < green]
        beyond = f["in_intersection"][rows] | (dist[rows] < -ctx.params["rules"]["stop_line"]["max_past"] * f["size"][rows])
        until = rows[: int(np.argmax(beyond)) + 1] if beyond.any() else rows
        standing = f["speed"][until] < ctx.params["features"]["stationary_speed"]
        t = f["t"][until]
        stood = float(np.sum(np.r_[np.diff(t), 0.0] * standing)) if len(t) else 0.0
        return bool(beyond.any()), stood >= ctx.params["rules"]["stop_line"]["min_stop_sec"]

    @staticmethod
    def _leaves(f: np.ndarray, sl: slice, i: int) -> float:
        """Time the vehicle leaves the intersection after crossing at sample ``i`` (or its last sample)."""
        inside = f["in_intersection"][sl][i:]
        t = f["t"][sl][i:]
        if inside.any():
            first = int(np.argmax(inside))
            out = np.flatnonzero(~inside[first:])
            if len(out):
                return float(t[first + out[0]])
        return float(max(t[-1], t[0] + 1e-3))


class StopLineViolation:
    """On red, a vehicle stops with its front past the stop line without entering the intersection.

    "Past" means more than ``front_past`` and at most ``max_past`` box widths
    beyond the line (between the line and the intersection: on the near crossing).
    The vehicle must have driven past the line (its anchor ``front_past`` beyond
    it) on red or amber (``arrive_phases``) or at the end of green, at most
    ``arrive_before_red_sec`` before red: a vehicle that crossed it earlier on
    green and got stuck in a jam is not a stop_line case.

    The segment runs from the moment it stops to the next green (or the end of the video).
    """

    label = "stop_line"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        p = ctx.params["rules"]["stop_line"]
        f = ctx.features
        line = stop_line(ctx.scene, ctx.aspect)
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        dist = line.distance(f["x"], f["y"], ctx.aspect)
        along = line.along(f["x"], f["y"], ctx.aspect)
        standing = vehicle & (f["speed"] < ctx.params["features"]["stationary_speed"])
        red = np.isin(ctx.signal.phase_at(f["t"]), p["red_phases"])
        at_line = (standing & red & ~f["in_intersection"] & ~f["edge"] & (dist < 0)
                   & (dist > -p["max_past"] * f["size"])
                   & (along > -ALONG_SLACK) & (along < 1 + ALONG_SLACK))
        past_line = dist < -p["front_past"] * f["size"]
        past = at_line & past_line
        red_starts, green_starts = phase_changes(ctx, p["red_phases"])
        found = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            for start, end in merged_runs(t, at_line[sl], ctx.params["rules"]["stopped_vehicle"]["run_merge_sec"]):
                inside = (t >= start) & (t < end)
                rows = np.flatnonzero(inside) + sl.start
                past_sec = float(np.sum(np.r_[np.diff(t), 0.0][inside] * past[sl][inside]))
                green = green_starts[green_starts > start]
                c = {"track_id": int(f["track_id"][sl.start]), "start": start,
                     "end": float(green[0]) if len(green) else ctx.duration,
                     "past_by": round(float(-np.median(dist[rows] / f["size"][rows])), 2),
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                crossed = np.flatnonzero(past_line[sl])
                t_cross = float(t[crossed[0]]) if len(crossed) else start
                arrived = str(ctx.signal.phase_at(t_cross))
                red = red_starts[red_starts >= t_cross]
                before_red = float(red[0]) - t_cross if len(red) else np.inf
                if (past_sec >= p["min_stop_sec"] and arrived not in p["arrive_phases"]
                        and before_red > p["arrive_before_red_sec"]):
                    c["reason"], c["kept"] = (f"stood past the stop line on red, but crossed it on {arrived} "
                                              f"{before_red:.0f} s before red (stuck in a jam)"), False
                elif past_sec >= p["min_stop_sec"]:
                    c["reason"], c["kept"] = (f"stood {past_sec:.1f} s on red with its front "
                                              f"{c['past_by']} box widths past the stop line"), True
                else:
                    c["reason"], c["kept"] = (f"at the line: past by {c['past_by']} box widths, "
                                              f"{past_sec:.1f} s beyond the margin"), False
                c["close"] = not c["kept"]
                found.append(c)
        return found
