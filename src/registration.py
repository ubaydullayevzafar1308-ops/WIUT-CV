"""Align frames of the same camera across recording sessions.

The camera is re-set between clips (shifts of 1-3 % of the frame width were
measured between sample videos), so scene geometry drawn on one reference frame
has to be warped onto each video. Alignment runs ECC on gradient-magnitude
images, which is robust to the day/dusk lighting differences that break
feature matching.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


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
        reference: BGR reference frame (the one the scene was drawn on).
        frame: BGR frame of the video to align.
        params: the ``registration`` section of params.yaml.

    Returns:
        ``(warp, cc)``: a 2x3 matrix with ``frame_xy = warp @ [ref_x, ref_y, 1]``
        in normalised coordinates, and the ECC correlation (higher is better).
    """
    ref_edges = edge_map(reference, params["width"])
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
