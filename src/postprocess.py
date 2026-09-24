"""Per-frame flags -> event segments ``[start_sec, end_sec, label]``.

Steps, with per-class ``merge_gap`` / ``min_len`` from params.yaml (``postprocess``):
runs of flagged samples become segments; segments of one class that overlap
(simultaneous events, several objects) or are separated by less than
``merge_gap`` are joined; segments shorter than ``min_len`` are dropped; the
rest is clipped to the video duration. The output never has overlapping
segments of the same class, as the harness requires.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Segment = list  # [start_sec, end_sec, label]
DECIMALS = 3    # the harness rounds times to milliseconds


def class_params(label: str, params: dict[str, Any]) -> dict[str, float]:
    """``merge_gap`` and ``min_len`` for a class; ``params`` is the ``postprocess`` section."""
    return {**params["default"], **params["classes"].get(label, {})}


def flags_to_runs(t: np.ndarray, flags: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) of every run of True samples.

    A run starts at its first flagged sample and ends where the next sample
    begins (the last flagged sample covers the interval up to the next one);
    a run ending at the last sample is extended by the median sample spacing.
    """
    t, flags = np.asarray(t, dtype=np.float64), np.asarray(flags, dtype=bool)
    if not flags.any():
        return []
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    edges = np.diff(np.r_[0, flags.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(float(t[s]), float(t[e + 1]) if e + 1 < len(t) else float(t[e]) + step) for s, e in zip(starts, ends)]


def merge_segments(segments: list[Segment], params: dict[str, Any], duration: float) -> list[Segment]:
    """Join, filter and clip segments per class; returns them sorted by (label, start)."""
    out: list[Segment] = []
    for label in sorted({s[2] for s in segments}):
        cp = class_params(label, params)
        spans = sorted((max(0.0, float(s)), min(duration, float(e))) for s, e, lab in segments if lab == label)
        joined: list[list[float]] = []
        for start, end in spans:
            if end <= start:
                continue
            if joined and start - joined[-1][1] < cp["merge_gap"]:
                joined[-1][1] = max(joined[-1][1], end)
            else:
                joined.append([start, end])
        out += [[round(s, DECIMALS), round(e, DECIMALS), label] for s, e in joined
                if e - s >= cp["min_len"] and round(e, DECIMALS) > round(s, DECIMALS)]
    return out


def flags_to_segments(t: np.ndarray, flags: np.ndarray, label: str, params: dict[str, Any],
                      duration: float) -> list[Segment]:
    """Segments of one class from one flag series (``params`` is the ``postprocess`` section)."""
    return merge_segments([[s, e, label] for s, e in flags_to_runs(t, flags)], params, duration)
