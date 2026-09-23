"""Scene loading, zone queries and registration."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import load_params
from src.registration import estimate_affine, invert, transform_points
from src.scene import load_scene


def test_scene_loads_with_valid_geometry():
    scene = load_scene()
    polygons = [scene.carriageway, scene.intersection, *(ln.polygon for ln in scene.lanes),
                *scene.crossings.values(), *scene.islands.values()]
    assert all(p.is_valid and p.area > 0 for p in polygons)
    assert all(np.isclose(np.linalg.norm(ln.direction), 1.0) for ln in scene.lanes)
    for light in scene.traffic_lights:
        x1, y1, x2, y2 = light["roi"]
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def test_zone_queries():
    scene = load_scene()
    crossing_centre = np.asarray(scene.crossings["side_street"].centroid.coords)
    island_centre = np.asarray(scene.islands["triangle_a"].centroid.coords)
    assert scene.in_crossing(crossing_centre).all()
    assert scene.on_carriageway(crossing_centre).all()
    assert not scene.on_carriageway(island_centre).any()


def test_identity_warp_keeps_scene():
    scene = load_scene()
    warped = scene.warped(np.eye(2, 3))
    assert warped.carriageway.equals(scene.carriageway)


def synthetic_frame(shift_px: tuple[float, float], size: tuple[int, int] = (960, 540)) -> np.ndarray:
    """Textured frame (random rectangles), translated by ``shift_px``."""
    rng = np.random.default_rng(0)
    w, h = size
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, w - 80), rng.integers(0, h - 60)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(20, 80)), int(y + rng.integers(15, 60))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    shift = np.float32([[1, 0, shift_px[0]], [0, 1, shift_px[1]]])
    return cv2.warpAffine(img, shift, size, borderMode=cv2.BORDER_REFLECT)


def test_registration_recovers_translation_under_lighting_change():
    reference = synthetic_frame((0, 0))
    moved = (synthetic_frame((12, -7)) * 0.5).astype(np.uint8)   # shifted and darker
    warp, cc = estimate_affine(reference, moved, load_params()["registration"])
    assert cc > 0.8
    assert warp[0, 2] == pytest.approx(12 / 960, abs=1.5 / 960)
    assert warp[1, 2] == pytest.approx(-7 / 540, abs=1.5 / 540)
    points = np.array([[0.3, 0.4], [0.7, 0.2]])
    assert np.allclose(transform_points(invert(warp), transform_points(warp, points)), points)
