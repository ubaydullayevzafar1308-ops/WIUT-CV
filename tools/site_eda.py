"""EDA for the website without a single pixel of the sample videos (NDA): figures and numbers from the tracks
cache and configs/scene.json only.

    python tools/site_eda.py [--videos samples] [--out outputs/site/eda]

Writes, 1920 px wide, each with a transparent background and on the site's dark
background (#010909, ``*_dark.png``):
- velocity_field: mean vehicle velocity per grid cell over all videos (arrows
  coloured by heading, colour wheel as legend), as in tools/eda.py;
- scene_map: configs/scene.json (lanes and directions, crossings, stop and
  solid lines, intersection, islands, signal heads), as in tools/scene_overlay.py;
and stats.json: tracks of vehicles and pedestrians per minute of every video, a
density grid of track points over all videos, and one or two example
trajectories per detected event class (polylines in scene.json coordinates).
No frame or edge map is read or drawn: the reference template only aligns each
video's tracks to the scene's coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, runtime_params  # noqa: E402
from src.pipeline import video_context  # noqa: E402
from src.postprocess import merge_segments  # noqa: E402
from src.registration import invert, register_video, transform_points  # noqa: E402
from src.rules import RULES, apply_rules  # noqa: E402
from src.rules.lanes import boundary_along  # noqa: E402
from src.rules.signal_rules import stop_line  # noqa: E402
from src.scene import Scene, load_scene  # noqa: E402
from src.tracking import track_video  # noqa: E402
from tools import eda, scene_overlay  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
W, H = eda.CANVAS
DARK_BG = np.array([9, 9, 1], np.float32) / 255.0   # #010909, BGR
DENSITY_GRID = (48, 27)                             # columns, rows
EXAMPLES_PER_CLASS = 2
EXAMPLE_MARGIN_SEC = 1.0                            # trajectory kept this long before and after the event


class Painter:
    """A premultiplied colour + alpha canvas: every primitive is anti-aliased and composited over it."""

    def __init__(self) -> None:
        self.color = np.zeros((H, W, 3), np.float32)
        self.alpha = np.zeros((H, W), np.float32)

    def _over(self, draw, bgr: tuple[int, int, int], opacity: float = 1.0) -> None:
        mask = np.zeros((H, W), np.uint8)
        draw(mask)
        k = mask.astype(np.float32) / 255.0 * opacity
        self.color = np.array(bgr, np.float32) / 255.0 * k[..., None] + self.color * (1 - k[..., None])
        self.alpha = k + self.alpha * (1 - k)

    def polyline(self, pts, closed: bool, bgr, thickness: int) -> None:
        self._over(lambda m: cv2.polylines(m, [pts], closed, 255, thickness, cv2.LINE_AA), bgr)

    def fill(self, pts, bgr, opacity: float) -> None:
        self._over(lambda m: cv2.fillPoly(m, [pts], 255, cv2.LINE_AA), bgr, opacity)

    def arrow(self, p0, p1, bgr, thickness: int, tip: float) -> None:
        self._over(lambda m: cv2.arrowedLine(m, p0, p1, 255, thickness, cv2.LINE_AA, tipLength=tip), bgr)

    def line(self, p0, p1, bgr, thickness: int) -> None:
        self._over(lambda m: cv2.line(m, p0, p1, 255, thickness, cv2.LINE_AA), bgr)

    def rectangle(self, p0, p1, bgr, thickness: int) -> None:
        self._over(lambda m: cv2.rectangle(m, p0, p1, 255, thickness), bgr)

    def text(self, label: str, org, bgr, scale: float = 0.6) -> None:
        """Text with a dark outline, readable on light and dark backgrounds."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        self._over(lambda m: cv2.putText(m, label, org, font, scale, 255, 4, cv2.LINE_AA), (0, 0, 0))
        self._over(lambda m: cv2.putText(m, label, org, font, scale, 255, 1 if scale < 0.8 else 2, cv2.LINE_AA), bgr)

    def save(self, out: Path, name: str) -> None:
        """``name``.png with a transparent background and ``name``_dark.png on #010909."""
        alpha = np.clip(self.alpha, 0, 1)[..., None]
        straight = np.where(alpha > 1e-6, self.color / np.maximum(alpha, 1e-6), 0.0)
        cv2.imwrite(str(out / f"{name}.png"), np.round(np.dstack([np.clip(straight, 0, 1), alpha]) * 255).astype(np.uint8))
        dark = self.color + DARK_BG * (1 - alpha)
        cv2.imwrite(str(out / f"{name}_dark.png"), np.round(np.clip(dark, 0, 1) * 255).astype(np.uint8))


def px(points) -> np.ndarray:
    return (np.asarray(points) * [W, H]).astype(np.int32)


def heading(dx: float, dy: float) -> tuple[int, int, int]:
    return tuple(int(c) for c in eda.heading_color(np.array([dx]), np.array([dy]))[0])


def compass(p: Painter) -> None:
    """Colour wheel: arrow colours read as directions (as tools/eda.draw_compass)."""
    cx, cy, r = W - 80, 110, 50
    for deg in range(0, 360, 3):
        rad = np.radians(deg)
        p.line((cx, cy), (int(cx + r * np.cos(rad)), int(cy + r * np.sin(rad))), heading(np.cos(rad), np.sin(rad)), 3)
    p.text("heading", (cx - 42, cy + r + 24), (255, 255, 255))


def velocity_field(anchors: list[dict[str, np.ndarray]]) -> Painter:
    """Mean vehicle velocity per cell over all videos (the data and style of tools/eda.plot_flow)."""
    a = {k: np.concatenate([x[k] for x in anchors]) for k in anchors[0]}
    cells = eda.FLOW_CELLS
    sel = np.isin(a["cls"], eda.VEHICLES) & (eda.speed_px(a) > eda.MOVING_SPEED)
    cx = np.clip((a["x"][sel] * cells[0]).astype(int), 0, cells[0] - 1)
    cy = np.clip((a["y"][sel] * cells[1]).astype(int), 0, cells[1] - 1)
    cell = cy * cells[0] + cx
    n = np.bincount(cell, minlength=cells[0] * cells[1])
    mvx = np.bincount(cell, a["vx"][sel], minlength=n.size) / np.maximum(n, 1)
    mvy = np.bincount(cell, a["vy"][sel], minlength=n.size) / np.maximum(n, 1)
    p = Painter()
    for k in np.flatnonzero(n >= eda.FLOW_MIN_SAMPLES):
        x0, y0 = (k % cells[0] + 0.5) / cells[0] * W, (k // cells[0] + 0.5) / cells[1] * H
        dx, dy = mvx[k] * W * eda.FLOW_ARROW_SEC, mvy[k] * H * eda.FLOW_ARROW_SEC
        p.arrow((int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)), heading(dx, dy), 2, 0.3)
    p.text(f"all videos: mean vehicle velocity per cell (arrow = {eda.FLOW_ARROW_SEC:g} s of travel)", (14, 31),
           (255, 255, 255), 0.9)
    compass(p)
    return p


def scene_map(scene: Scene) -> Painter:
    """configs/scene.json with the colours and labels of tools/scene_overlay.render, on no frame."""
    colors, fill_alpha = scene_overlay.COLORS, scene_overlay.FILL_ALPHA
    p = Painter()

    def zone(polygon, bgr, label: str | None = None) -> None:
        pts = px(polygon.exterior.coords)
        p.fill(pts, bgr, fill_alpha)
        p.polyline(pts, True, bgr, 2)
        if label:
            c = px(np.asarray(polygon.centroid.coords))[0]
            p.text(label, (int(c[0]) - 40, int(c[1])), bgr)

    zone(scene.carriageway, colors["carriageway"])
    zone(scene.intersection, colors["intersection"], "intersection")
    for name, polygon in scene.islands.items():
        zone(polygon, colors["island"], name)
    for name, polygon in scene.crossings.items():
        zone(polygon, colors["crossing"], f"crossing:{name}")
    for name, polygon in scene.stopping_zones.items():
        zone(polygon, colors["stopping_zone"], f"stopping:{name}")
    for name, polygon in scene.exits.items():
        zone(polygon, colors["exit"], f"exit:{name}")
    for lane in scene.lanes:
        p.polyline(px(lane.polygon.exterior.coords), True, colors["lane"], 1)
        c = px(np.asarray(lane.polygon.centroid.coords))[0]
        for direction in lane.directions:
            tip = (c + direction * 90).astype(int)
            p.arrow(tuple(int(v) for v in c), tuple(int(v) for v in tip), colors["lane"], 4, 0.35)
        p.text(f"lane:{lane.id}", (int(c[0]) - 50, int(c[1]) - 14), colors["lane"])
    for name, line in scene.stop_lines.items():
        p.polyline(px(line.coords), False, colors["stop_line"], 4)
        p.text(f"stop:{name}", tuple(px(line.coords)[-1] + [6, -6]), colors["stop_line"])
    for name, line in scene.solid_lines.items():
        p.polyline(px(line.coords), False, colors["solid_line"], 2)
        p.text(f"solid:{name}", tuple(px(line.coords)[1] + [8, 0]), colors["solid_line"])
    if scene.lane_boundaries:
        aspect = H / W
        line = stop_line(scene, aspect, scene.approach_stop_line)
        bounds = [0.0] + [boundary_along(line, np.asarray(scene.solid_lines[b].coords), aspect)
                          for b in scene.lane_boundaries] + [1.0]
        for number, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            org = px(((line.p0 + (lo + hi) / 2 * line.length * line.unit) / [1.0, aspect])[None])[0] + [-14, -22]
            p.text(f"A{number}", (int(org[0]), int(org[1])), colors["approach_lane"])
    for light in scene.traffic_lights:
        x1, y1, x2, y2 = px(np.asarray(light["roi"]).reshape(2, 2)).ravel()
        p.rectangle((x1 - 3, y1 - 3), (x2 + 3, y2 + 3), colors["traffic_light"], 2)
        p.text(f"light:{light['id']}", (x2 + 6, y1 + 12), colors["traffic_light"])
    p.text("configs/scene.json (normalised coordinates of the camera view)", (12, 28), (255, 255, 255), 0.8)
    return p


def per_minute(tracks) -> list[dict[str, int]]:
    """Tracks of vehicles and pedestrians seen in every minute (a tracker may split one object into several)."""
    rows = tracks.rows
    minute = (rows["frame"] / tracks.info.fps // 60).astype(int)
    vehicle, person = np.isin(rows["cls"], eda.VEHICLES), np.isin(rows["cls"], eda.PERSONS)
    return [{"minute": m, "vehicles": int(len(np.unique(rows["track_id"][vehicle & (minute == m)]))),
             "pedestrians": int(len(np.unique(rows["track_id"][person & (minute == m)])))}
            for m in range(int(np.ceil(tracks.info.duration / 60)))]


def example_trajectories(ctx, video: str, to_scene: np.ndarray, found: dict[str, list]) -> None:
    """Add up to ``EXAMPLES_PER_CLASS`` trajectories per class of the events this video reports."""
    f = ctx.features
    events = merge_segments(apply_rules(ctx), ctx.params["postprocess"], ctx.duration)
    for rule in RULES:
        examples = found.setdefault(rule.label, [])
        if not hasattr(rule, "candidates") or len(examples) >= EXAMPLES_PER_CLASS:
            continue
        for c in rule.candidates(ctx):
            if len(examples) >= EXAMPLES_PER_CLASS:
                break
            if not c["kept"] or "track_id" not in c or not any(
                    e[2] == rule.label and e[0] < c["end"] and e[1] > c["start"] for e in events):
                continue
            polylines = []
            for key in ("track_id", "other_id"):
                if key in c:
                    rows = np.flatnonzero((f["track_id"] == c[key]) & (f["t"] >= c["start"] - EXAMPLE_MARGIN_SEC)
                                          & (f["t"] <= c["end"] + EXAMPLE_MARGIN_SEC))
                    xy = transform_points(to_scene, np.stack([f["x"][rows], f["y"][rows]], axis=1).astype(np.float64))
                    polylines.append({"t": np.round(f["t"][rows] - c["start"], 2).tolist(), "xy": np.round(xy, 4).tolist()})
            examples.append({"video": video, "start_sec": round(float(c["start"]), 2),
                             "end_sec": round(float(c["end"]), 2), "polylines": polylines})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "site" / "eda"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = runtime_params()
    scene = load_scene()
    anchors, minutes, trajectories = [], {}, {}
    density = {name: np.zeros(DENSITY_GRID[::-1], int) for name in ("vehicles", "pedestrians")}
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        reg = register_video(str(path), params["registration"])   # alignment only: no pixel of it is drawn
        to_scene = invert(reg.warp)
        tracks = track_video(str(path), params, scene=scene.warped(reg.warp))
        a = eda.anchors_with_velocity(tracks, to_scene)
        anchors.append(a)
        minutes[path.name] = per_minute(tracks)
        for name, classes in (("vehicles", eda.VEHICLES), ("pedestrians", eda.PERSONS)):
            mask = np.isin(a["cls"], classes)
            gx = np.clip((a["x"][mask] * DENSITY_GRID[0]).astype(int), 0, DENSITY_GRID[0] - 1)
            gy = np.clip((a["y"][mask] * DENSITY_GRID[1]).astype(int), 0, DENSITY_GRID[1] - 1)
            np.add.at(density[name], (gy, gx), 1)
        example_trajectories(video_context(str(path), params), path.name, to_scene, trajectories)
    velocity_field(anchors).save(out, "velocity_field")
    scene_map(scene).save(out, "scene_map")
    stats = {
        "coordinates": "normalised [0, 1] x / y of the scene.json view; every video is aligned to it",
        "per_minute": minutes,
        "density": {"grid": {"cols": DENSITY_GRID[0], "rows": DENSITY_GRID[1]},
                    "unit": "track points (anchor samples) per cell, all videos",
                    **{name: grid.tolist() for name, grid in density.items()}},
        "trajectories": {label: examples for label, examples in trajectories.items() if examples},
    }
    (out / "stats.json").write_text(json.dumps(stats))
    print(f"wrote {out}: velocity_field(_dark).png, scene_map(_dark).png, stats.json "
          f"(example trajectories: { {k: len(v) for k, v in stats['trajectories'].items()} })")
    return 0


if __name__ == "__main__":
    sys.exit(main())
