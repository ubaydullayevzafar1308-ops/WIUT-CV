"""Signal phase timeline per video, checked against vehicle motion at the avenue_near stop line.

    python tools/signal_timeline.py [--videos samples]

Writes outputs/signal_<video>.png (rows: pedestrian head, vehicle head, fused
phase, stop-line crossings, share of moving vehicles before the stop line) and
outputs/signal_check.json, and prints per phase: time share, crossings per
minute and moving share. A correct phase has almost no crossings on red.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely import contains_xy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, runtime_params  # noqa: E402
from src.scene import Scene, scene_for_video  # noqa: E402
from src.signal import GREEN, PHASES, RED, SignalTimeline  # noqa: E402
from src.tracking import Tracks, track_video  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
VEHICLES = (2, 3, 5, 7)
ASPECT = 9 / 16                   # normalised y -> width units
APPROACH_DEPTH = 0.12             # "before the stop line" = this far upstream (frame widths)
MOVING_SPEED = 0.01               # frame widths per second
MOTION_BIN_SEC = 5.0
COLORS = {"red": (40, 40, 220), "red_amber": (0, 120, 255), "amber": (0, 200, 255), "green": (60, 190, 60),
          "unknown": (150, 150, 150)}
WIDTH, ROW_H, LABEL_W = 1800, 34, 190


def anchors(tracks: Tracks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vehicle rows sorted by track, their bottom-centre anchors and times."""
    rows = np.sort(tracks.rows[np.isin(tracks.rows["cls"], VEHICLES)], order=["track_id", "frame"])
    xy = np.stack([(rows["x1"] + rows["x2"]) / 2, rows["y2"]], axis=1).astype(np.float64)
    return rows, xy, rows["frame"] / tracks.info.fps


def stop_line_frame(scene: Scene) -> tuple[np.ndarray, np.ndarray, float]:
    """Stop line start, direction (width units) and the sign of the upstream side."""
    line = np.asarray(scene.stop_lines["avenue_near"].coords)
    p0, d = line[0], line[-1] - line[0]
    upstream = np.asarray(next(ln for ln in scene.lanes if ln.id == "avenue_near").polygon.centroid.coords)[0]
    return p0, d, float(np.sign(side(upstream[None], p0, d)[0]))


def side(xy: np.ndarray, p0: np.ndarray, d: np.ndarray) -> np.ndarray:
    rel = (xy - p0) * [1, ASPECT]
    return d[0] * rel[:, 1] - d[1] * ASPECT * rel[:, 0]


def stop_line_crossings(tracks: Tracks, scene: Scene) -> np.ndarray:
    """Times at which a vehicle anchor crosses the stop line from upstream to downstream."""
    rows, xy, t = anchors(tracks)
    p0, d, up = stop_line_frame(scene)
    s = side(xy, p0, d) * up
    along = ((xy - p0) * [1, ASPECT]) @ (d * [1, ASPECT]) / np.sum((d * [1, ASPECT]) ** 2)
    same = rows["track_id"][1:] == rows["track_id"][:-1]
    cross = same & (s[:-1] > 0) & (s[1:] <= 0) & (along[1:] > -0.05) & (along[1:] < 1.05)
    return t[1:][cross]


def approach_motion(tracks: Tracks, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Times and moving flags of vehicle samples in avenue_near just before the stop line."""
    rows, xy, t = anchors(tracks)
    p0, d, up = stop_line_frame(scene)
    dist = side(xy, p0, d) * up / np.hypot(d[0], d[1] * ASPECT)
    lane = next(ln for ln in scene.lanes if ln.id == "avenue_near").polygon
    near = contains_xy(lane, xy[:, 0], xy[:, 1]) & (dist > 0) & (dist < APPROACH_DEPTH)
    same = np.r_[False, rows["track_id"][1:] == rows["track_id"][:-1]]
    dt = np.r_[1.0, np.diff(t)]
    speed = np.r_[0.0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]) * ASPECT)] / np.maximum(dt, 1e-6)
    keep = near & same
    return t[keep], speed[keep] > MOVING_SPEED


def phase_stats(timeline: SignalTimeline, crossings: np.ndarray, motion_t: np.ndarray, moving: np.ndarray) -> dict:
    step = np.median(np.diff(timeline.t))
    at_cross, at_motion = timeline.phase_at(crossings), timeline.phase_at(motion_t)
    stats = {}
    for phase in PHASES:
        minutes = np.sum(timeline.phase == phase) * step / 60
        if minutes == 0:
            continue
        m = at_motion == phase
        stats[phase] = {"minutes": round(float(minutes), 2),
                        "crossings": int(np.sum(at_cross == phase)),
                        "crossings_per_min": round(float(np.sum(at_cross == phase) / minutes), 2),
                        "moving_share": round(float(moving[m].mean()), 3) if m.any() else None}
    starts = timeline.t[1:][(timeline.phase[:-1] != GREEN) & (timeline.phase[1:] == GREEN)]
    delays = [float(crossings[crossings >= s][0] - s) for s in starts if np.any(crossings >= s)]
    stats["green_starts"] = len(starts)
    stats["first_crossing_after_green_sec_median"] = round(float(np.median(delays)), 1) if delays else None
    red_ends = timeline.t[1:][(timeline.phase[:-1] == RED) & (timeline.phase[1:] != RED)]
    stats["red_phases"] = len(red_ends)
    return stats


def draw(video: str, duration: float, timeline: SignalTimeline, crossings: np.ndarray,
         motion_t: np.ndarray, moving: np.ndarray, stats: dict) -> np.ndarray:
    rows = ["pedestrian head", "vehicle head", "phase (fused)", "stop-line crossings", "moving before line"]
    img = np.full((ROW_H * (len(rows) + 3), WIDTH, 3), 255, np.uint8)
    x_of = lambda t: (LABEL_W + (np.asarray(t) / duration) * (WIDTH - LABEL_W - 10)).astype(int)  # noqa: E731
    step = np.median(np.diff(timeline.t))
    for r, label in enumerate(rows):
        y = ROW_H * (r + 1)
        cv2.putText(img, label, (8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if r < 3:
            states = (timeline.ped, timeline.veh, timeline.phase)[r]
            for t, s in zip(timeline.t, states):
                cv2.rectangle(img, (int(x_of(t)), y + 4), (int(x_of(t + step)), y + ROW_H - 4), COLORS[s], -1)
        elif r == 3:
            for t in crossings:
                cv2.line(img, (int(x_of(t)), y + 4), (int(x_of(t)), y + ROW_H - 4), (0, 0, 0), 1)
        else:
            edges = np.arange(0, duration + MOTION_BIN_SEC, MOTION_BIN_SEC)
            idx = np.digitize(motion_t, edges) - 1
            for b in range(len(edges) - 1):
                m = idx == b
                if m.any():
                    h = int((ROW_H - 8) * moving[m].mean())
                    cv2.rectangle(img, (int(x_of(edges[b])), y + ROW_H - 4 - h), (int(x_of(edges[b + 1])) - 1, y + ROW_H - 4),
                                  (90, 90, 90), -1)
    y_axis = ROW_H * (len(rows) + 1)
    for t in np.arange(0, duration, 30):
        cv2.line(img, (int(x_of(t)), ROW_H), (int(x_of(t)), y_axis), (215, 215, 215), 1)
        cv2.putText(img, f"{int(t)}s", (int(x_of(t)) + 2, y_axis + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    red, green = stats.get("red", {}), stats.get("green", {})
    title = (f"{video}: crossings/min red {red.get('crossings_per_min')} vs green {green.get('crossings_per_min')}; "
             f"moving before line red {red.get('moving_share')} vs green {green.get('moving_share')}")
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    x = LABEL_W
    for phase, color in COLORS.items():
        cv2.rectangle(img, (x, y_axis + 26), (x + 18, y_axis + 40), color, -1)
        cv2.putText(img, phase, (x + 24, y_axis + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        x += 130
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    args = ap.parse_args()
    params = runtime_params()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    report = {}
    print(f"| video | phase | minutes | crossings | per min | moving before line |")
    print("|---|---|---:|---:|---:|---:|")
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        scene = scene_for_video(str(path), params)
        tracks = track_video(str(path), params, scene=scene)
        timeline = tracks.signal_timeline(params)
        crossings = stop_line_crossings(tracks, scene)
        motion_t, moving = approach_motion(tracks, scene)
        stats = phase_stats(timeline, crossings, motion_t, moving)
        agree = np.isin(timeline.ped, (RED, GREEN)) & np.isin(timeline.veh, (RED, GREEN))
        stats["ped_veh_agreement"] = round(float(np.mean(timeline.ped[agree] == timeline.veh[agree])), 3)
        stats["unknown_share"] = round(float(np.mean(timeline.phase == "unknown")), 3)
        report[path.name] = stats
        cv2.imwrite(str(out / f"signal_{path.stem}.png"),
                    draw(path.name, tracks.info.duration, timeline, crossings, motion_t, moving, stats))
        for phase in PHASES:
            if phase in stats:
                s = stats[phase]
                print(f"| {path.name} | {phase} | {s['minutes']} | {s['crossings']} | {s['crossings_per_min']} | {s['moving_share']} |")
    (out / "signal_check.json").write_text(json.dumps(report, indent=1))
    for video, s in report.items():
        print(f"{video}: ped/veh agreement {s['ped_veh_agreement']}, unknown {s['unknown_share']}, "
              f"green starts {s['green_starts']}, first crossing after green (median) {s['first_crossing_after_green_sec_median']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
