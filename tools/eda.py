"""EDA of the sample videos from the cached tracks: motion heatmap, direction field, trajectories.

    python tools/eda.py [--videos samples] [--reference outputs/frames/C3896_mid.jpg]

Every video is aligned to the reference frame (the camera shifts between
sessions), so plots of all videos together are in one coordinate system.
PNGs go to outputs/eda/: <video>_<plot>.png per video and all_<plot>.png
for all videos together, plus stats.json.
Needs outputs/frames/<video>_mid.jpg (tools/extract_frames.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_params  # noqa: E402
from src.registration import estimate_affine, invert, transform_points  # noqa: E402
from src.tracking import track_video  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLES = (2, 3, 5, 7)
PERSONS = (0,)

CANVAS = (1920, 1080)            # output image size
BACKDROP_DIM = 0.55              # darken the reference frame under overlays
VELOCITY_WINDOW_SEC = 0.5        # central difference half-window for velocity
MOVING_SPEED = 0.01              # frame widths per second; slower anchors count as stationary
HEAT_BINS = (192, 108)
FLOW_CELLS = (40, 22)
FLOW_MIN_SAMPLES = 25            # a cell needs this many moving samples to get an arrow
FLOW_ARROW_SEC = 1.0             # arrow length = displacement over this many seconds
TRAJ_MIN_SEC = 2.0               # trajectories shorter than this are not drawn
TRAJ_MIN_TRAVEL = 0.05           # ... nor ones that travel less (fraction of the frame diagonal)


def anchors_with_velocity(tracks, warp_to_ref: np.ndarray) -> dict[str, np.ndarray]:
    """Per-box anchor (bottom centre) in reference coordinates and its velocity.

    Returns arrays ``track_id, cls, t, x, y, vx, vy`` (positions normalised,
    velocities in normalised units per second; NaN where the window is incomplete).
    """
    rows = np.sort(tracks.rows, order=["track_id", "frame"])
    fps = tracks.info.fps
    xy = np.stack([(rows["x1"] + rows["x2"]) / 2, rows["y2"]], axis=1).astype(np.float64)
    xy = transform_points(warp_to_ref, xy)
    t = rows["frame"] / fps
    vel = np.full_like(xy, np.nan)
    half = max(1, round(VELOCITY_WINDOW_SEC * fps / load_params()["video"]["stride"]))
    ids = rows["track_id"]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    for s, e in zip(starts, ends):
        if e - s <= 2 * half:
            continue
        idx = np.arange(s + half, e - half)
        dt = t[idx + half] - t[idx - half]
        vel[idx] = (xy[idx + half] - xy[idx - half]) / dt[:, None]
    return {"track_id": ids, "cls": rows["cls"], "t": t, "x": xy[:, 0], "y": xy[:, 1], "vx": vel[:, 0], "vy": vel[:, 1]}


def speed_px(a: dict[str, np.ndarray]) -> np.ndarray:
    """Speed in frame widths per second, with the vertical component scaled to width units."""
    w, h = CANVAS
    return np.hypot(a["vx"], a["vy"] * h / w)


def backdrop(reference: np.ndarray) -> np.ndarray:
    return (cv2.resize(reference, CANVAS, interpolation=cv2.INTER_AREA) * BACKDROP_DIM).astype(np.uint8)


def heading_color(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """BGR colours from motion heading (hue = direction), for arrays of pixel displacements."""
    hue = ((np.degrees(np.arctan2(dy, dx)) % 360) / 2).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], axis=-1)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]


def draw_legend(img: np.ndarray, title: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, title, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def draw_compass(img: np.ndarray) -> None:
    """Hue wheel so arrow/trajectory colours can be read as directions."""
    cx, cy, r = img.shape[1] - 80, 110, 50
    for deg in range(0, 360, 3):
        rad = np.radians(deg)
        color = tuple(int(c) for c in heading_color(np.array([np.cos(rad)]), np.array([np.sin(rad)]))[0])
        cv2.line(img, (cx, cy), (int(cx + r * np.cos(rad)), int(cy + r * np.sin(rad))), color, 3)
    cv2.putText(img, "heading", (cx - 42, cy + r + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def plot_heatmap(base: np.ndarray, a: dict[str, np.ndarray], classes: tuple[int, ...], title: str) -> np.ndarray:
    sel = np.isin(a["cls"], classes) & (speed_px(a) > MOVING_SPEED)
    hist, _, _ = np.histogram2d(a["y"][sel], a["x"][sel], bins=(HEAT_BINS[1], HEAT_BINS[0]), range=[[0, 1], [0, 1]])
    heat = np.log1p(hist)
    heat = cv2.resize((255 * heat / max(heat.max(), 1e-6)).astype(np.uint8), CANVAS, interpolation=cv2.INTER_LINEAR)
    heat = cv2.GaussianBlur(heat, (0, 0), 3)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    alpha = (heat.astype(np.float32) / 255)[..., None] * 0.85
    img = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
    draw_legend(img, f"{title}: {int(sel.sum())} moving anchor samples (log density)")
    return img


def plot_flow(base: np.ndarray, a: dict[str, np.ndarray], title: str) -> np.ndarray:
    w, h = CANVAS
    sel = np.isin(a["cls"], VEHICLES) & (speed_px(a) > MOVING_SPEED)
    cx = np.clip((a["x"][sel] * FLOW_CELLS[0]).astype(int), 0, FLOW_CELLS[0] - 1)
    cy = np.clip((a["y"][sel] * FLOW_CELLS[1]).astype(int), 0, FLOW_CELLS[1] - 1)
    cell = cy * FLOW_CELLS[0] + cx
    n = np.bincount(cell, minlength=FLOW_CELLS[0] * FLOW_CELLS[1])
    mvx = np.bincount(cell, a["vx"][sel], minlength=n.size) / np.maximum(n, 1)
    mvy = np.bincount(cell, a["vy"][sel], minlength=n.size) / np.maximum(n, 1)
    img = base.copy()
    for k in np.flatnonzero(n >= FLOW_MIN_SAMPLES):
        x0 = (k % FLOW_CELLS[0] + 0.5) / FLOW_CELLS[0] * w
        y0 = (k // FLOW_CELLS[0] + 0.5) / FLOW_CELLS[1] * h
        dx, dy = mvx[k] * w * FLOW_ARROW_SEC, mvy[k] * h * FLOW_ARROW_SEC
        color = tuple(int(c) for c in heading_color(np.array([dx]), np.array([dy]))[0])
        cv2.arrowedLine(img, (int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)), color, 2, cv2.LINE_AA, tipLength=0.3)
    draw_legend(img, f"{title}: mean vehicle velocity per cell (arrow = {FLOW_ARROW_SEC:g} s of travel)")
    draw_compass(img)
    return img


def plot_trajectories(base: np.ndarray, a: dict[str, np.ndarray], classes: tuple[int, ...], title: str) -> np.ndarray:
    w, h = CANVAS
    layer = np.zeros_like(base)
    ids = a["track_id"]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    drawn = 0
    for s, e in zip(starts, ends):
        if a["cls"][s] not in classes or a["t"][e - 1] - a["t"][s] < TRAJ_MIN_SEC:
            continue
        pts = np.stack([a["x"][s:e] * w, a["y"][s:e] * h], axis=1)
        travel = pts[-1] - pts[0]
        if np.hypot(*travel) < TRAJ_MIN_TRAVEL * np.hypot(w, h):
            continue
        color = tuple(int(c) for c in heading_color(np.array([travel[0]]), np.array([travel[1]]))[0])
        cv2.polylines(layer, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)
        drawn += 1
    img = cv2.addWeighted(base, 1.0, layer, 0.9, 0)
    draw_legend(img, f"{title}: {drawn} trajectories (colour = overall heading)")
    draw_compass(img)
    return img


def video_stats(tracks, a: dict[str, np.ndarray]) -> dict:
    ids, first = np.unique(tracks.rows["track_id"], return_index=True)
    track_cls = tracks.rows["cls"][first]
    per_frame = np.bincount(np.searchsorted(tracks.frames, tracks.rows["frame"]), minlength=len(tracks.frames))
    moving = speed_px(a) > MOVING_SPEED
    return {
        "duration_sec": round(tracks.info.duration, 2),
        "sampled_frames": int(len(tracks.frames)),
        "boxes": int(len(tracks.rows)),
        "tracks_by_class": {COCO_NAMES[int(c)]: int((track_cls == c).sum()) for c in np.unique(track_cls)},
        "boxes_per_frame_mean": round(float(per_frame.mean()), 1),
        "boxes_per_frame_max": int(per_frame.max()),
        "vehicle_moving_share": round(float(moving[np.isin(a["cls"], VEHICLES)].mean()), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--frames", default=str(ROOT / "outputs" / "frames"))
    ap.add_argument("--reference", default=str(ROOT / "outputs" / "frames" / "C3896_mid.jpg"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eda"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = cv2.imread(args.reference)
    if reference is None:   # sample frames are local only: overlays on a dark canvas, no alignment
        print(f"{args.reference} missing: drawing on a blank canvas without alignment")
    base = backdrop(reference if reference is not None else np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8))
    reg_params = load_params()["registration"]

    per_video, stats = [], {}
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        tracks = track_video(str(path))
        frame = cv2.imread(str(Path(args.frames) / f"{path.stem}_mid.jpg"))
        warp, cc = (estimate_affine(reference, frame, reg_params) if reference is not None and frame is not None
                    else (np.eye(2, 3), 0.0))
        a = anchors_with_velocity(tracks, invert(warp))
        per_video.append(a)
        stats[path.name] = {**video_stats(tracks, a), "alignment_cc": round(cc, 3),
                            "warp_to_video": np.round(warp, 5).tolist()}
        plots = {
            "heatmap_vehicles": plot_heatmap(base, a, VEHICLES, f"{path.stem} vehicles"),
            "heatmap_persons": plot_heatmap(base, a, PERSONS, f"{path.stem} persons"),
            "flow": plot_flow(base, a, path.stem),
            "trajectories": plot_trajectories(base, a, VEHICLES, f"{path.stem} vehicles"),
        }
        for name, img in plots.items():
            cv2.imwrite(str(out / f"{path.stem}_{name}.png"), img)
        print(f"{path.name}: aligned (cc={cc:.3f}), {stats[path.name]['tracks_by_class']}")

    merged = {k: np.concatenate([a[k] for a in per_video]) for k in per_video[0]}
    offset = np.cumsum([0] + [a["track_id"].max() + 1 for a in per_video[:-1]])
    merged["track_id"] = np.concatenate([a["track_id"] + o for a, o in zip(per_video, offset)])
    cv2.imwrite(str(out / "all_heatmap_vehicles.png"), plot_heatmap(base, merged, VEHICLES, "all videos, vehicles"))
    cv2.imwrite(str(out / "all_heatmap_persons.png"), plot_heatmap(base, merged, PERSONS, "all videos, persons"))
    cv2.imwrite(str(out / "all_flow.png"), plot_flow(base, merged, "all videos"))
    cv2.imwrite(str(out / "all_trajectories.png"), plot_trajectories(base, merged, VEHICLES, "all videos, vehicles"))
    cv2.imwrite(str(out / "all_trajectories_persons.png"),
                plot_trajectories(base, merged, PERSONS, "all videos, persons"))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
