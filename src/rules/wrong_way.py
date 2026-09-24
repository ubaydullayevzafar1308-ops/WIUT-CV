"""wrong_way: a vehicle moving against every allowed direction of its lane."""
from __future__ import annotations

import numpy as np

from src.features import track_slices
from src.postprocess import Segment
from src.rules.base import VideoContext, merged_runs


class WrongWay:
    """A vehicle driving against the traffic direction of its lane.

    A sample is against the lane when the vehicle is inside a lane, moving
    faster than ``min_speed``, its box does not touch the frame edge, and its
    heading is more than ``against_angle`` from every allowed direction of that
    lane (two-way sections list both). Areas without a lane (the intersection,
    the U-turn area below the far crossing) and crossings (vehicles cross them;
    couriers on mopeds ride along them with the pedestrians) are not checked. A run is kept if it
    lasts ``min_duration_sec`` and the vehicle covered ``min_travel`` of its own
    box widths meanwhile (box jitter on the spot does not count). The segment
    runs from entering the opposing direction to leaving it (or the frame).
    """

    label = "wrong_way"

    def apply(self, ctx: VideoContext) -> list[Segment]:
        return [[c["start"], c["end"], self.label] for c in self.candidates(ctx) if c["kept"]]

    def candidates(self, ctx: VideoContext) -> list[dict]:
        """Every run against the lane of >= review_min_sec, with the reason it is kept or rejected."""
        p = ctx.params["rules"]["wrong_way"]
        f = ctx.features
        vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
        moving = vehicle & (f["lane"] >= 0) & ~f["in_crossing"] & (f["speed"] > p["min_speed"])
        unit = np.stack([f["vx"], f["vy"]], axis=1).astype(np.float64)
        unit /= np.maximum(np.hypot(unit[:, 0], unit[:, 1]), 1e-9)[:, None]
        against = np.zeros(len(f), dtype=bool)
        cos_limit = np.cos(np.radians(p["against_angle"]))
        for i, lane in enumerate(ctx.scene.lanes):
            m = moving & (f["lane"] == i)
            against[m] = np.max(unit[m] @ lane.directions.T, axis=1) < cos_limit
        lanes = [lane.id for lane in ctx.scene.lanes]

        found = []
        for sl in track_slices(f["track_id"]):
            t = f["t"][sl]
            if not against[sl].any():
                continue
            for start, end in merged_runs(t, against[sl], p["run_merge_sec"]):
                if end - start < p["review_min_sec"]:
                    continue
                rows = np.flatnonzero((t >= start) & (t < end) & against[sl]) + sl.start
                path = np.hypot(np.diff(f["x"][rows]), np.diff(f["y"][rows]) * ctx.aspect).sum()
                travel = float(path / max(float(np.median(f["size"][rows])), 1e-6))
                c = {"track_id": int(f["track_id"][rows[0]]), "start": start, "end": end,
                     "travel_boxes": round(travel, 1), "lane": lanes[int(np.bincount(f["lane"][rows]).argmax())],
                     "x": float(np.median(f["x"][rows])), "y": float(np.median(f["y"][rows]))}
                if f["edge"][rows].mean() > 0.5:
                    c["reason"], c["kept"] = "cut off by the frame edge", False
                elif end - start < p["min_duration_sec"]:
                    c["reason"], c["kept"] = f"against the lane only {end - start:.1f} s", False
                elif travel < p["min_travel"]:
                    c["reason"], c["kept"] = f"moved only {travel:.1f} box widths (jitter)", False
                else:
                    c["reason"], c["kept"] = (f"{end - start:.1f} s against {c['lane']}, "
                                              f"{travel:.1f} box widths travelled"), True
                c["close"] = not c["kept"]
                found.append(c)
        return found
