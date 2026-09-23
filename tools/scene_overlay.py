"""Draw every zone of configs/scene.json over a frame, for checking and editing the scene.

    python tools/scene_overlay.py [--frame outputs/frames/C3896_mid.jpg] [--out outputs/scene_overlay.png]
    python tools/scene_overlay.py --all-videos     # also warp the scene onto each video's mid frame

With --all-videos, outputs/scene_overlay_<video>.png show the scene after
registration, which is how the pipeline will see it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_params  # noqa: E402
from src.registration import estimate_affine  # noqa: E402
from src.scene import Scene, load_scene  # noqa: E402

CANVAS = (1920, 1080)
FILL_ALPHA = 0.28
COLORS = {  # BGR
    "carriageway": (90, 90, 90),
    "intersection": (0, 200, 255),
    "lane": (255, 160, 0),
    "island": (60, 60, 200),
    "crossing": (255, 255, 255),
    "stop_line": (0, 0, 255),
    "solid_line": (0, 255, 255),
    "traffic_light": (0, 255, 0),
}


def px(points: np.ndarray) -> np.ndarray:
    return (np.asarray(points) * CANVAS).astype(np.int32)


def fill(img: np.ndarray, layer: np.ndarray, polygon, color: tuple[int, int, int], label: str | None = None) -> None:
    pts = px(polygon.exterior.coords)
    cv2.fillPoly(layer, [pts], color)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    if label:
        c = px(np.asarray(polygon.centroid.coords))[0]
        text(img, label, (int(c[0]) - 40, int(c[1])), color)


def text(img: np.ndarray, label: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def render(frame: np.ndarray, scene: Scene, title: str) -> np.ndarray:
    img = cv2.resize(frame, CANVAS, interpolation=cv2.INTER_AREA)
    layer = img.copy()
    fill(img, layer, scene.carriageway, COLORS["carriageway"])
    fill(img, layer, scene.intersection, COLORS["intersection"], "intersection")
    for island_id, polygon in scene.islands.items():
        fill(img, layer, polygon, COLORS["island"], island_id)
    for crossing_id, polygon in scene.crossings.items():
        fill(img, layer, polygon, COLORS["crossing"], f"crossing:{crossing_id}")
    img = cv2.addWeighted(layer, FILL_ALPHA, img, 1 - FILL_ALPHA, 0)

    for lane in scene.lanes:
        pts = px(lane.polygon.exterior.coords)
        cv2.polylines(img, [pts], True, COLORS["lane"], 1, cv2.LINE_AA)
        c = px(np.asarray(lane.polygon.centroid.coords))[0]
        for direction in lane.directions:
            tip = (c + direction * 90).astype(int)
            cv2.arrowedLine(img, tuple(int(v) for v in c), tuple(int(v) for v in tip), COLORS["lane"], 4, cv2.LINE_AA,
                            tipLength=0.35)
        text(img, f"lane:{lane.id}", (int(c[0]) - 50, int(c[1]) - 14), COLORS["lane"])
    for line_id, line in scene.stop_lines.items():
        cv2.polylines(img, [px(line.coords)], False, COLORS["stop_line"], 4, cv2.LINE_AA)
        text(img, f"stop:{line_id}", tuple(px(line.coords)[-1] + [6, -6]), COLORS["stop_line"])
    for line_id, line in scene.solid_lines.items():
        cv2.polylines(img, [px(line.coords)], False, COLORS["solid_line"], 2, cv2.LINE_AA)
        text(img, f"solid:{line_id}", tuple(px(line.coords)[1] + [8, 0]), COLORS["solid_line"])
    for light in scene.traffic_lights:
        x1, y1, x2, y2 = px(np.asarray(light["roi"]).reshape(2, 2)).ravel()
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), COLORS["traffic_light"], 2)
        text(img, f"light:{light['id']}", (x2 + 6, y1 + 12), COLORS["traffic_light"])

    cv2.rectangle(img, (0, 0), (CANVAS[0], 40), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default=str(ROOT / "outputs" / "frames" / "C3896_mid.jpg"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "scene_overlay.png"))
    ap.add_argument("--all-videos", action="store_true")
    ap.add_argument("--frames", default=str(ROOT / "outputs" / "frames"))
    args = ap.parse_args()

    scene = load_scene()
    reference = cv2.imread(args.frame)
    cv2.imwrite(args.out, render(reference, scene, f"configs/scene.json on {Path(args.frame).name} (reference)"))
    print(f"wrote {Path(args.out).relative_to(ROOT)}")
    if args.all_videos:
        for frame_path in sorted(Path(args.frames).glob("*_mid.jpg")):
            frame = cv2.imread(str(frame_path))
            warp, cc = estimate_affine(reference, frame, load_params()["registration"])
            out = Path(args.out).with_name(f"scene_overlay_{frame_path.stem.removesuffix('_mid')}.png")
            cv2.imwrite(str(out), render(frame, scene.warped(warp), f"scene warped onto {frame_path.name} (ECC cc={cc:.2f})"))
            print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
