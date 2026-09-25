"""Runtime scene registration: sample videos, rejection of bad alignments, fallback to unshifted zones."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from src import scene as scene_module
from src.config import ROOT, load_params
from src.registration import (IDENTITY, MISSING, REFERENCE_ENV, Registration, edge_map, load_reference, reference_path,
                              register_frame, register_video, transform_points)
from src.scene import Scene, load_scene, scene_for_video
from src.video import probe, read_frame_at

PARAMS = load_params()["registration"]
SAMPLES = ROOT / "samples"

# Centre shift of each session relative to the reference frame (C3896), measured
# with ECC on full-resolution frames when the scene was drawn.
EXPECTED_CENTRE_SHIFT = {
    "C3896.MP4": (0.0, 0.0),
    "C3897.MP4": (0.0003, -0.0003),
    "C3902.MP4": (-0.0221, 0.0268),
    "C3905.MP4": (0.0035, 0.0085),
}
SHIFT_TOLERANCE = 0.004   # ~15 px in 4K
BIG_SHIFT_VIDEO = "C3902.MP4"
MIN_GAIN_ON_BIG_SHIFT = 1.03   # crossing stripes must be clearly better covered after alignment (measured: 1.08)


def centre_shift(warp: np.ndarray) -> np.ndarray:
    return transform_points(warp, np.array([[0.5, 0.5]]))[0] - 0.5


def crossing_edge_strength(edges: np.ndarray, scene: Scene) -> float:
    """Mean gradient magnitude inside the crossing polygons: high when they sit on the zebra stripes."""
    h, w = edges.shape
    mask = np.zeros((h, w), np.uint8)
    for polygon in scene.crossings.values():
        cv2.fillPoly(mask, [(np.asarray(polygon.exterior.coords) * [w, h]).astype(np.int32)], 1)
    return float(edges[mask > 0].mean())


LOCAL_FRAME = ROOT / "configs" / "local" / "reference_frame.jpg"   # sample data: local only, never in git
needs_reference = pytest.mark.skipif(not reference_path(PARAMS).exists(), reason="no reference edge map")
needs_local_frame = pytest.mark.skipif(not LOCAL_FRAME.exists(), reason="reference frame is sample data, local only")


@needs_reference
def test_reference_edge_map_is_small_and_loads():
    assert reference_path(PARAMS).stat().st_size <= 2 * 1024 * 1024
    edges = load_reference(PARAMS)
    assert edges.shape == (540, 960) and edges.dtype == np.float32 and 0.0 <= edges.min() <= edges.max() <= 1.0


@needs_reference
@needs_local_frame
def test_edge_map_is_what_ecc_gets_from_the_frame():
    frame_edges = edge_map(cv2.imread(str(LOCAL_FRAME)), PARAMS["width"])
    assert np.abs(load_reference(PARAMS) - frame_edges).max() < 1e-4   # 16-bit storage


def test_missing_reference_disables_alignment(monkeypatch, caplog):
    monkeypatch.setenv(REFERENCE_ENV, "configs/local/no_such_reference.jpg")
    with caplog.at_level(logging.WARNING, logger="src.registration"):
        frame_reg = register_frame(np.zeros((540, 960, 3), np.uint8), PARAMS)
        video_reg = register_video("no_such_video.mp4", PARAMS)   # returns before opening the video
    for reg in (frame_reg, video_reg):
        assert not reg.ok and reg.reason == MISSING and np.array_equal(reg.warp, IDENTITY)
    assert "reference missing" in caplog.text


@pytest.mark.parametrize("video", sorted(EXPECTED_CENTRE_SHIFT))
@needs_reference
def test_sample_videos_register(video):
    path = SAMPLES / video
    if not path.exists():
        pytest.skip(f"{video} not available (videos are not in git)")
    reg = register_video(str(path), PARAMS)
    assert reg.ok, reg.reason
    assert reg.cc >= PARAMS["min_cc"]
    assert centre_shift(reg.warp) == pytest.approx(EXPECTED_CENTRE_SHIFT[video], abs=SHIFT_TOLERANCE)
    # the aligned zones fit the video: crossing polygons cover the zebra stripes at least as well as unshifted ones
    info = probe(str(path))
    edges = edge_map(read_frame_at(str(path), info.n_frames // 2), PARAMS["width"])
    unshifted = crossing_edge_strength(edges, load_scene())
    aligned = crossing_edge_strength(edges, load_scene().warped(reg.warp))
    assert aligned >= unshifted * 0.995
    if video == BIG_SHIFT_VIDEO:
        assert aligned >= unshifted * MIN_GAIN_ON_BIG_SHIFT


@needs_reference
@needs_local_frame
def test_reference_registers_to_identity():
    reg = register_frame(cv2.imread(str(LOCAL_FRAME)), PARAMS)
    assert reg.ok and reg.cc > 0.99
    assert np.allclose(reg.warp, IDENTITY, atol=1e-3)


@pytest.mark.parametrize("frame", [
    np.random.default_rng(0).integers(0, 255, (540, 960, 3), dtype=np.uint8),   # noise: low correlation
    np.full((540, 960, 3), 90, dtype=np.uint8),                                   # flat: ECC fails
])
@needs_reference
def test_unrelated_frames_are_rejected(frame):
    reg = register_frame(frame, PARAMS)
    assert not reg.ok and reg.reason
    assert np.array_equal(reg.warp, IDENTITY)


def test_rejected_registration_falls_back_to_unshifted_zones(monkeypatch, caplog):
    monkeypatch.setattr(scene_module, "register_video",
                        lambda path, params: Registration(IDENTITY, 0.1, False, "correlation 0.10 < 0.25"))
    with caplog.at_level(logging.WARNING, logger="src.scene"):
        scene = scene_for_video("any.mp4")
    assert scene.carriageway.equals(load_scene().carriageway)
    assert "registration rejected" in caplog.text
