"""Align frames of the same camera across recording sessions.

The camera is re-set between clips (shifts of 1-3 % of the frame width were
measured between sample videos), so scene geometry drawn on one reference frame
has to be warped onto each video. Alignment runs ECC on gradient-magnitude
images, which is robust to the day/dusk lighting differences that break
feature matching.

The reference frame is a frame of the sample videos, which may not be published;
the repository keeps only its edge map (configs/reference_edges.png: exactly
what ECC compares, 16-bit), published with the organisers' permission. The
template is ``registration.reference`` or the ``WIUT_REFERENCE`` environment
variable: an edge map (16-bit, one channel) or a frame (its edge map is
computed). Without it, alignment is disabled and the scene is used as drawn.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import resolve
from src.video import probe, read_frame_at

IDENTITY = np.eye(2, 3)
REFERENCE_ENV = "WIUT_REFERENCE"   # overrides registration.reference
MISSING = "reference missing, alignment disabled"
EDGE_SCALE = 65535                 # an edge map in [0, 1] is stored as 16-bit PNG

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Registration:
    """Result of aligning a video to the reference frame.

    ``warp`` maps reference coordinates to video coordinates (normalised); it is
    the identity when the alignment was rejected (``ok`` is False, ``reason`` says why).
    """

    warp: np.ndarray
    cc: float
    ok: bool
    reason: str


def edge_map(bgr: np.ndarray, width: int) -> np.ndarray:
    """Lighting-normalised gradient magnitude of a frame, resized to ``width``, in [0, 1]."""
    height = round(bgr.shape[0] * width / bgr.shape[1])
    gray = cv2.cvtColor(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray).astype(np.float32)
    magnitude = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), 2)
    return magnitude / (magnitude.max() + 1e-6)


def estimate_affine(reference: np.ndarray, frame: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Affine warp from reference to frame, in normalised [0, 1] coordinates.

    Args:
        reference: BGR reference frame (the one the scene was drawn on), or its edge map
            (``edge_map`` at ``params["width"]``: 2-D, [0, 1]).
        frame: BGR frame of the video to align.
        params: the ``registration`` section of params.yaml.

    Returns:
        ``(warp, cc)``: a 2x3 matrix with ``frame_xy = warp @ [ref_x, ref_y, 1]``
        in normalised coordinates, and the ECC correlation (higher is better).
    """
    ref_edges = reference.astype(np.float32) if reference.ndim == 2 else edge_map(reference, params["width"])
    frame_edges = edge_map(frame, params["width"])
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params["iterations"], params["eps"])
    cc, warp = cv2.findTransformECC(ref_edges, frame_edges, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    h, w = ref_edges.shape
    scale = np.diag([w, h]).astype(np.float64)
    inv_scale = np.diag([1.0 / w, 1.0 / h])
    warp = warp.astype(np.float64)
    return np.hstack([inv_scale @ warp[:, :2] @ scale, (inv_scale @ warp[:, 2])[:, None]]), float(cc)


def invert(warp: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine warp."""
    return cv2.invertAffineTransform(warp)


def transform_points(warp: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine warp to an ``(N, 2)`` array of points."""
    return points @ warp[:, :2].T + warp[:, 2]


def reference_path(params: dict[str, Any]) -> Path:
    """Where the template is: ``WIUT_REFERENCE`` if set, else ``registration.reference``."""
    return resolve(os.environ.get(REFERENCE_ENV) or params["reference"])


def save_edges(edges: np.ndarray, path: Path) -> None:
    """Store an edge map in [0, 1] as a 16-bit PNG (no visible loss of precision for ECC)."""
    cv2.imwrite(str(path), np.round(np.clip(edges, 0, 1) * EDGE_SCALE).astype(np.uint16))


@lru_cache(maxsize=None)
def _load(path: Path, width: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        log.warning("%s (%s)", MISSING, path)
        return None
    if image.ndim == 2 and image.dtype == np.uint16:   # a stored edge map
        return image.astype(np.float32) / EDGE_SCALE
    return edge_map(image, width)                       # a frame


def load_reference(params: dict[str, Any]) -> np.ndarray | None:
    """The template's edge map (loaded once), or None if the template is missing (logged once)."""
    return _load(reference_path(params), params["width"])


def rejection_reason(warp: np.ndarray, cc: float, params: dict[str, Any]) -> str:
    """Why an alignment is implausible, or an empty string if it is accepted."""
    if cc < params["min_cc"]:
        return f"correlation {cc:.2f} < {params['min_cc']}"
    shift = float(np.hypot(*(transform_points(warp, np.array([[0.5, 0.5]]))[0] - 0.5)))
    if shift > params["max_shift"]:
        return f"centre shift {shift:.3f} > {params['max_shift']}"
    scale = float(np.sqrt(abs(np.linalg.det(warp[:, :2]))))
    if abs(scale - 1) > params["max_scale_change"]:
        return f"scale {scale:.3f} outside 1 +- {params['max_scale_change']}"
    rotation = float(np.degrees(np.arctan2(warp[1, 0] - warp[0, 1], warp[0, 0] + warp[1, 1])))
    if abs(rotation) > params["max_rotation_deg"]:
        return f"rotation {rotation:.1f} deg > {params['max_rotation_deg']}"
    return ""


def register_frame(frame: np.ndarray, params: dict[str, Any]) -> Registration:
    """Align one BGR frame (any size) to the reference frame, rejecting implausible results."""
    reference = load_reference(params)
    if reference is None:
        return Registration(IDENTITY, 0.0, False, MISSING)
    try:
        warp, cc = estimate_affine(reference, frame, params)
    except cv2.error as err:
        return Registration(IDENTITY, 0.0, False, f"ECC failed: {str(err).strip().splitlines()[-1]}")
    reason = rejection_reason(warp, cc, params)
    return Registration(IDENTITY if reason else warp, cc, not reason, reason)


def register_video(path: str, params: dict[str, Any]) -> Registration:
    """Align a video to the reference frame using the median of a few frames (moving traffic drops out)."""
    if load_reference(params) is None:
        return Registration(IDENTITY, 0.0, False, MISSING)
    info = probe(path)
    width = params["width"]
    height = round(info.height * width / info.width)
    frames = [
        cv2.resize(read_frame_at(path, int(info.n_frames * f)), (width, height), interpolation=cv2.INTER_AREA)
        for f in params["sample_fractions"]
    ]
    return register_frame(np.median(np.stack(frames), axis=0).astype(np.uint8), params)
