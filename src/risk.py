"""Part B: causal accident-risk estimator.

``step`` only ever sees the frames the harness has passed so far: it never opens
the video and never uses Part A output. Every ``stride``-th frame is resized to
960 px, run through the shared YOLO model and the estimator's own ByteTrack; the
scene is aligned on the median of the frames seen in the first seconds.

At each processed frame the estimator measures (``Components``):
- every pair of a moving vehicle with another object nearby (vehicle, or a
  pedestrian on the carriageway): relative position and velocity on the ground
  plane (``ground_plane``), in car box widths,
- the hardest braking of a moving vehicle,
- whether a vehicle has been driving against its lane,
- pedestrians on the road near a vehicle: distance, closing speed, distance to
  the nearest crossing.
``risk_score`` turns them into a score (time to contact through a sigmoid, plus
bonuses); it is a pure function of the measurements, so the calibration can be
replayed from a log. The score is EMA-smoothed and clipped to [0, 1].
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import shapely
from shapely import affinity, contains_xy

from src.config import runtime_params, set_seeds
from src.registration import register_frame
from src.scene import Scene, load_scene
from src.tracking import get_model, make_tracker

PERSON = 0
NO_PAIRS = np.zeros((0, 5))
NO_WALKERS = np.zeros((0, 3))


@dataclass
class Components:
    """What the estimator measured at one processed frame.

    ``pairs``: rows [rel_x, rel_y, vel_x, vel_y, pedestrian] of a moving vehicle and
    another object on the ground plane (car box widths, car box widths / s;
    pedestrian = 1 for a person).
    ``walkers``: rows [distance, closing speed, crossing distance] of a pedestrian on
    the road and a moving vehicle (car box widths, car box widths / s, lane widths).
    """

    t: float
    pairs: np.ndarray
    brake: float
    wrong_way: bool
    walkers: np.ndarray


def ground_plane(x: np.ndarray | float, y: np.ndarray | float, g: dict[str, Any], aspect: float) -> np.ndarray:
    """Image anchor (normalised x, y) -> position on the road plane, in car box widths.

    A pinhole camera over flat ground: lateral and depth positions are both
    proportional to 1 / (row - horizon row), with the horizon where the car box
    width model ``width = width_slope * (y - horizon_y)`` reaches zero and the
    focal length (frame widths) from the avenue's and the crossings' vanishing
    points. The unit is the box width of a car at that spot, so image distances
    along the rows keep their scale and distances in depth are stretched.
    """
    depth = (np.asarray(y, dtype=np.float64) - g["horizon_y"]) * aspect
    scale = aspect / g["width_slope"]
    return np.stack([scale * (np.asarray(x, dtype=np.float64) - 0.5) / depth, scale * g["focal"] / depth], axis=-1)


def time_to_contact(pairs: np.ndarray, p: dict[str, Any]) -> float:
    """Smallest time until a pair comes within contact distance, over pairs approaching fast enough."""
    if not len(pairs):
        return np.inf
    rel, vel, pedestrian = pairs[:, :2], pairs[:, 2:4], pairs[:, 4] > 0
    radius = np.where(pedestrian, p["pedestrian_contact_radius"], p["contact_radius"])
    dist = np.hypot(rel[:, 0], rel[:, 1])
    closing = -np.sum(rel * vel, axis=1) / np.maximum(dist, 1e-9)
    ok = (dist > radius + p["min_gap"]) & (closing >= p["min_closing_speed"])
    a = np.sum(vel * vel, axis=1)
    b = 2.0 * np.sum(rel * vel, axis=1)
    c = dist ** 2 - radius ** 2
    disc = b * b - 4 * a * c
    ok &= (a > 0) & (disc >= 0)
    t = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2 * np.where(a > 0, a, 1.0))
    t = t[ok & (t > 0) & (t <= p["horizon_sec"])]
    return float(t.min()) if len(t) else np.inf


def pedestrian_alert(walkers: np.ndarray, p: dict[str, Any]) -> bool:
    """A pedestrian away from crossings with a vehicle close and approaching."""
    if not len(walkers):
        return False
    dist, closing, crossing = walkers[:, 0], walkers[:, 1], walkers[:, 2]
    return bool(np.any((dist < p["pedestrian_radius"]) & (closing >= p["pedestrian_min_closing"])
                       & (crossing > p["pedestrian_crossing_margin"])))


def risk_score(c: Components, p: dict[str, Any]) -> float:
    """Unsmoothed risk of one processed frame (before EMA), in [0, 1]."""
    ttc = time_to_contact(c.pairs, p)
    score = 0.0 if not np.isfinite(ttc) else 1.0 / (1.0 + np.exp(-p["slope"] * (p["tau0"] - ttc)))
    if c.brake > p["brake_decel"]:
        score += p["brake_bonus"] * min(c.brake / p["brake_decel"] - 1.0, 1.0)
    score += p["wrong_way_bonus"] * c.wrong_way + p["pedestrian_bonus"] * pedestrian_alert(c.walkers, p)
    return float(np.clip(score, 0.0, 1.0))


def smooth(previous: float, raw: float, p: dict[str, Any]) -> float:
    return float(np.clip(p["ema_alpha"] * raw + (1.0 - p["ema_alpha"]) * previous, 0.0, 1.0))


def velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares velocity of a short track segment."""
    return np.array([np.polyfit(t, x, 1)[0], np.polyfit(t, y, 1)[0]])


class RiskEstimator:
    """Per-frame P(an accident starts within the next 5 s), using only frames seen so far."""

    def __init__(self, record: bool = False) -> None:
        """``record`` keeps every processed frame's ``Components`` in ``self.log`` (calibration)."""
        self.record = record

    def reset(self, meta: dict) -> None:
        """Start a new video. ``meta`` has video_id, fps, width, height, n_frames."""
        self.params = runtime_params()
        self.p = self.params["risk"]
        set_seeds(self.params["seed"])
        self.meta = meta
        self.aspect = meta["height"] / meta["width"]
        self.model = get_model(self.params["detector"]["weights"])
        self.stride = self.p["stride"]
        self.tracker = make_tracker(self.params["tracker"], meta["fps"] / self.stride)
        self.scene: Scene = load_scene()
        self.aligned = False
        self.registration_frames: list[np.ndarray] = []
        self.history: dict[int, deque] = {}
        self.boxes: dict[int, tuple[float, float, float, float]] = {}
        self.against_since: dict[int, float] = {}
        self.index = 0
        self.next_index = 0
        self.processed = 0
        self.pace_origin = (time.perf_counter(), 0.0)
        self.last_score = 0.0
        self.log: list[Components] = []

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next 5 s) for the frame at ``t_sec``."""
        index = self.index
        self.index += 1
        if index < self.next_index:
            return self.last_score
        self._pace(t_sec)
        self.next_index = index + self.stride
        height = round(frame.shape[0] * self.p["frame_width"] / frame.shape[1])
        small = cv2.resize(frame, (self.p["frame_width"], height), interpolation=cv2.INTER_AREA)
        self._align(small, t_sec)
        c = self._measure(small, t_sec)
        if self.record:
            self.log.append(c)
        self.last_score = smooth(self.last_score, risk_score(c, self.p), self.p)
        return self.last_score

    # --- timing and scene -------------------------------------------------

    def _pace(self, t_sec: float) -> None:
        """Process fewer frames when Part B runs slower than ``realtime_factor`` (the stride only grows).

        The pace is measured from the end of the warm-up (model loading, first
        inferences), as seconds spent per second of video since then.
        """
        self.processed += 1
        if self.processed <= self.p["pace_warmup_steps"]:
            self.pace_origin = (time.perf_counter(), t_sec)
            return
        started, t0 = self.pace_origin
        if time.perf_counter() - started > self.p["realtime_factor"] * (t_sec - t0) and self.stride < self.p["max_stride"]:
            self.stride += 1
            self.pace_origin = (time.perf_counter(), t_sec)

    def _align(self, small: np.ndarray, t_sec: float) -> None:
        """Collect frames of the first seconds, then align the scene once on their median."""
        if self.aligned:
            return
        if t_sec < self.p["registration_sec"]:
            self.registration_frames.append(small)
            return
        frames = self.registration_frames
        if frames:
            picks = np.linspace(0, len(frames) - 1, min(len(frames), self.p["registration_frames"])).astype(int)
            reg = register_frame(np.median(np.stack([frames[i] for i in picks]), axis=0).astype(np.uint8),
                                 self.params["registration"])
            if reg.ok:
                self.scene = load_scene().warped(reg.warp)
        self.registration_frames = []
        self.aligned = True

    # --- measurement ------------------------------------------------------

    def _measure(self, small: np.ndarray, t_sec: float) -> Components:
        dp = self.params["detector"]
        device = self.params["device"]
        result = self.model.predict(small, imgsz=self.p["imgsz"], conf=dp["conf"], iou=dp["iou"], classes=dp["classes"],
                                    device=device, verbose=False,
                                    quantize=16 if dp["fp16"] and device.startswith("cuda") else None)[0]
        tracked = self.tracker.update(result.boxes.cpu().numpy())
        h, w = small.shape[:2]
        alive = set()
        self.boxes = {}
        for x1, y1, x2, y2, track_id, _, cls, _ in tracked:
            tid = int(track_id)
            alive.add(tid)
            self.boxes[tid] = (x1 / w, y1 / h, x2 / w, y2 / h)
            hist = self.history.setdefault(tid, deque())
            gx, gy = ground_plane((x1 + x2) / 2 / w, y2 / h, self.p["ground"], self.aspect)
            hist.append((t_sec, (x1 + x2) / 2 / w, y2 / h, gx, gy, int(cls)))
            while hist and t_sec - hist[0][0] > self.p["history_sec"]:
                hist.popleft()
        for tid in list(self.history):
            if tid not in alive and t_sec - self.history[tid][-1][0] > self.p["history_sec"]:
                del self.history[tid]
                self.against_since.pop(tid, None)
        objects = self._kinematics(t_sec, alive)
        movers = [o for o in objects if o["vehicle"] and o["v"] is not None and np.hypot(*o["v"]) > self.p["min_speed"]]
        pedestrians = self._pedestrians_on_road(objects)
        return Components(t=t_sec, pairs=self._pairs(movers, objects, pedestrians),
                          brake=max([-o["accel"] for o in movers] + [0.0]),
                          wrong_way=self._wrong_way(objects, t_sec),
                          walkers=self._walkers(movers, pedestrians))

    def _kinematics(self, t_sec: float, alive: set[int]) -> list[dict]:
        """Image anchor, ground position, ground velocity and acceleration (car box widths) and image heading
        of every object seen in this frame."""
        p = self.p
        objects = []
        vehicles = self.params["rules"]["vehicle_classes"]
        for tid in alive:
            t, x, y, gx, gy, cls = (np.array(v, dtype=np.float64) for v in zip(*self.history[tid]))
            recent = t >= t_sec - p["velocity_window_sec"]
            obj = {"id": tid, "cls": int(cls[-1]), "x": x[-1], "y": y[-1], "g": np.array([gx[-1], gy[-1]]),
                   "v": None, "heading": None, "accel": 0.0, "vehicle": int(cls[-1]) in vehicles}
            tr = t[recent]
            if len(tr) >= p["min_track_samples"] and tr[-1] - tr[0] >= p["min_track_span_sec"]:
                obj["v"] = velocity(tr, gx[recent], gy[recent])
                obj["heading"] = velocity(tr, x[recent], y[recent] * self.aspect)
                half = tr >= tr[0] + (tr[-1] - tr[0]) / 2
                if half.sum() >= 2 and (~half).sum() >= 2:
                    v1 = velocity(tr[~half], gx[recent][~half], gy[recent][~half])
                    v2 = velocity(tr[half], gx[recent][half], gy[recent][half])
                    dt = max(float(np.mean(tr[half]) - np.mean(tr[~half])), 1e-3)
                    speed = np.hypot(*v2)
                    if speed > 0:
                        obj["accel"] = float(((v2 - v1) / dt) @ (v2 / speed))
            objects.append(obj)
        return objects

    def _pairs(self, movers: list[dict], objects: list[dict], pedestrians: set[int]) -> np.ndarray:
        """Relative motion of each moving vehicle and every vehicle or on-road pedestrian within ``pair_radius``."""
        rows = []
        seen = set()
        for a in movers:
            for b in objects:
                if b["id"] == a["id"] or b["v"] is None or (not b["vehicle"] and b["id"] not in pedestrians):
                    continue
                key = tuple(sorted((a["id"], b["id"])))
                if key in seen:
                    continue
                seen.add(key)
                rel = b["g"] - a["g"]
                if np.hypot(*rel) > self.p["pair_radius"]:
                    continue
                vel = b["v"] - a["v"]
                rows.append([rel[0], rel[1], vel[0], vel[1], 0.0 if b["vehicle"] else 1.0])
        return np.array(rows) if rows else NO_PAIRS

    def _pedestrians_on_road(self, objects: list[dict]) -> set[int]:
        """Ids of pedestrians on the carriageway (shrunk by the pedestrian inset), crossings included.

        A person whose anchor lies inside a vehicle or bicycle box of the same frame is a
        rider or passenger, not a pedestrian.
        """
        people = [o for o in objects if o["cls"] == PERSON]
        if not people:
            return set()
        xy = np.array([[o["x"], o["y"]] for o in people])
        inside = self.scene.on_carriageway(xy, self.params["features"]["pedestrian_inset"], self.aspect)
        riders = self.params["rules"]["pedestrians"]["rider_classes"]
        carriers = [b for tid, b in self.boxes.items() if self.history[tid][-1][-1] in riders]
        riding = [any(x1 <= o["x"] <= x2 and y1 <= o["y"] <= y2 for x1, y1, x2, y2 in carriers) for o in people]
        return {o["id"] for o, ok, rides in zip(people, inside, riding) if ok and not rides}

    def _walkers(self, movers: list[dict], pedestrians: set[int]) -> np.ndarray:
        """Each on-road pedestrian with each moving vehicle: distance, closing speed, crossing distance."""
        people = [self.history[tid][-1] for tid in pedestrians]
        if not people or not movers:
            return NO_WALKERS
        crossings = shapely.union_all([affinity.scale(c, xfact=1.0, yfact=self.aspect, origin=(0, 0))
                                       for c in self.scene.crossings.values()])
        g = self.p["ground"]
        rows = []
        for _, px, py, gx, gy, _ in people:
            gap = shapely.distance(crossings, shapely.Point(px, py * self.aspect))
            lane = self.params["rules"]["lane_width_cars"] * g["width_slope"] * (py - g["horizon_y"])
            for car in movers:
                rel = np.array([gx, gy]) - car["g"]
                dist = float(np.hypot(*rel))
                closing = float(rel @ car["v"]) / max(dist, 1e-9)
                rows.append([dist, closing, gap / lane])
        return np.array(rows)

    def _wrong_way(self, objects: list[dict], t_sec: float) -> bool:
        """A vehicle has been moving against every allowed direction of its lane for ``wrong_way_sec``."""
        for o in objects:
            if not o["vehicle"] or o["v"] is None or np.hypot(*o["v"]) < self.p["min_speed"]:
                self.against_since.pop(o["id"], None)
                continue
            heading = o["heading"] / max(float(np.hypot(*o["heading"])), 1e-9)
            on_crossing = any(contains_xy(c, o["x"], o["y"]) for c in self.scene.crossings.values())
            lane = next((ln for ln in self.scene.lanes if contains_xy(ln.polygon, o["x"], o["y"])), None)
            if lane is not None and not on_crossing and np.max(lane.directions @ heading) < -0.5:
                self.against_since.setdefault(o["id"], t_sec)
            else:
                self.against_since.pop(o["id"], None)
        return any(t_sec - since >= self.p["wrong_way_sec"] for since in self.against_since.values())
