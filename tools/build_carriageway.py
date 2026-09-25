"""Rebuild the carriageway polygon of configs/scene.json from where vehicles actually drive.

    python tools/build_carriageway.py            # report + outputs/carriageway.png
    python tools/build_carriageway.py --write    # also replace "carriageway" in scene.json

Moving-vehicle anchors of all sample videos (aligned to the reference frame) are
binned; the occupied area is closed, widened by half a vehicle (anchors sit on
the lane centre, the road extends beyond), reduced to its largest component,
and the kerbside parking on the left edge and the islands are cut out.
Prints the share of pedestrian anchors on the carriageway before and after.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, runtime_params  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.registration import invert, load_reference, register_video, transform_points  # noqa: E402
from src.scene import SCENE_PATH, Scene, load_scene  # noqa: E402
from src.tracking import track_video  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
GRID = (480, 270)               # cells over the frame (4 px of a 1920-wide image)
MIN_COUNT = 1                   # anchors a cell needs to count as driven on (the side street is sparse)
MOVING_SPEED = 0.01             # frame widths / s: parked vehicles do not define the road
CLOSE_WIDTH = 0.02              # gaps narrower than this (frame widths) are closed: lane markings, occlusions
HALF_VEHICLE = 0.012            # the road extends this far beyond the outermost anchors
PARKING = (0.0, 0.30, 0.10, 0.55)  # x1, y1, x2, y2: kerbside parking at the left edge, not carriageway
SIMPLIFY = 0.003                # polygon simplification tolerance (frame widths)
CANVAS = (1920, 1080)


def anchors_in_reference(params: dict, videos: Path) -> dict[str, np.ndarray]:
    """Anchors (x, y), speed and class of every tracked box, mapped to the reference frame."""
    parts = []
    for path in sorted(p for p in videos.iterdir() if p.suffix in VIDEO_EXTS):
        reg = register_video(str(path), params["registration"])
        scene = load_scene().warped(reg.warp)
        features = compute_features(track_video(str(path), params, scene=scene), scene, params)
        xy = transform_points(invert(reg.warp), np.stack([features["x"], features["y"]], axis=1).astype(np.float64))
        parts.append((xy, features["speed"], features["cls"]))
    return {"xy": np.concatenate([p[0] for p in parts]), "speed": np.concatenate([p[1] for p in parts]),
            "cls": np.concatenate([p[2] for p in parts])}


def cells(width: float) -> int:
    return max(1, round(width * GRID[0]))


def build_mask(xy: np.ndarray, scene: Scene) -> np.ndarray:
    gx = np.clip((xy[:, 0] * GRID[0]).astype(int), 0, GRID[0] - 1)
    gy = np.clip((xy[:, 1] * GRID[1]).astype(int), 0, GRID[1] - 1)
    counts = np.zeros((GRID[1], GRID[0]), np.int32)
    np.add.at(counts, (gy, gx), 1)
    mask = (counts >= MIN_COUNT).astype(np.uint8)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cells(CLOSE_WIDTH) | 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells(HALF_VEHICLE) + 1,) * 2)
    mask = cv2.dilate(mask, grow)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    x1, y1, x2, y2 = PARKING
    mask[int(y1 * GRID[1]):int(y2 * GRID[1]), int(x1 * GRID[0]):int(x2 * GRID[0])] = 0
    for island in scene.islands.values():
        cv2.fillPoly(mask, [(np.asarray(island.exterior.coords) * GRID).astype(np.int32)], 0)
    return mask


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, SIMPLIFY * GRID[0], True)[:, 0, :]
    return [[round(float(x) / GRID[0], 4), round(float(y) / GRID[1], 4)] for x, y in contour]


def pedestrian_share(scene: Scene, a: dict[str, np.ndarray], params: dict, inset: float) -> float:
    fp = params["features"]
    person = np.isin(a["cls"], fp["pedestrian_classes"])
    return float(scene.on_carriageway(a["xy"][person], inset, CANVAS[1] / CANVAS[0]).mean())


def write_carriageway(polygon: list[list[float]]) -> None:
    text = SCENE_PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    data["carriageway"] = polygon
    out = json.dumps(data, indent=1, ensure_ascii=False)
    out = re.sub(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", r"[\1, \2]", out)
    out = re.sub(r"\[\s*((?:\[[^\[\]]+\],?\s*)+)\]", lambda m: "[" + re.sub(r"\],\s*\[", "], [", m.group(1).strip()) + "]", out)
    out = re.sub(r"\[\s*(-?[\d.]+(?:,\s*-?[\d.]+){2,})\s*\]", lambda m: "[" + re.sub(r",\s*", ", ", m.group(1)) + "]", out)
    SCENE_PATH.write_text(out + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--write", action="store_true", help="replace the carriageway in configs/scene.json")
    args = ap.parse_args()
    params = runtime_params()
    inset = params["features"]["pedestrian_inset"]
    a = anchors_in_reference(params, Path(args.videos))
    vehicle = np.isin(a["cls"], params["rules"]["vehicle_classes"])
    moving = vehicle & (a["speed"] > MOVING_SPEED)

    old = load_scene()
    polygon = mask_to_polygon(build_mask(a["xy"][moving], old))
    new = Scene(**{**old.__dict__, "carriageway": type(old.carriageway)(polygon)})
    print(f"new carriageway: {len(polygon)} vertices, area {new.carriageway.area:.3f} (old {old.carriageway.area:.3f})")
    print(f"moving vehicle anchors on the carriageway: old {old.on_carriageway(a['xy'][moving]).mean():.3f}, "
          f"new {new.on_carriageway(a['xy'][moving]).mean():.3f}")
    print(f"pedestrian anchors on the carriageway: old {pedestrian_share(old, a, params, 0.0):.3f}, "
          f"new {pedestrian_share(new, a, params, 0.0):.3f}, new with {inset} inset {pedestrian_share(new, a, params, inset):.3f}")

    edges = load_reference(params["registration"])   # the reference edge map; None without a template
    frame = (cv2.cvtColor(cv2.resize(np.round(edges * 255).astype(np.uint8), CANVAS), cv2.COLOR_GRAY2BGR)
             if edges is not None else np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8))
    img = (frame * 0.6).astype(np.uint8)
    for scene, color in ((old, (0, 0, 255)), (new, (0, 255, 0))):
        cv2.polylines(img, [(np.asarray(scene.carriageway.exterior.coords) * CANVAS).astype(np.int32)], True, color, 2)
    person = np.isin(a["cls"], params["features"]["pedestrian_classes"])
    for x, y in (a["xy"][person][::25] * CANVAS).astype(int):
        cv2.circle(img, (int(x), int(y)), 1, (255, 200, 0), -1)
    cv2.putText(img, "carriageway: old (red), new (green); dots: pedestrian anchors", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(ROOT / "outputs" / "carriageway.png"), img)
    if args.write:
        write_carriageway(polygon)
        print(f"wrote {SCENE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
