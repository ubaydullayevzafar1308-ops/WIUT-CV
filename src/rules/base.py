"""Common interface of the per-class rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.postprocess import Segment
from src.scene import Scene
from src.signal import GREEN, SignalTimeline


@dataclass
class VideoContext:
    """Everything a rule sees about one video.

    ``features`` is the table from src/features.py (sorted by track, then frame);
    ``sample_t`` are the times of all sampled frames, including frames without boxes;
    ``aspect`` is height / width, the factor that makes normalised y comparable with x;
    ``final_stride`` is the sampling stride tracking ended with.
    """

    features: np.ndarray
    scene: Scene
    signal: SignalTimeline
    sample_t: np.ndarray
    duration: float
    aspect: float
    params: dict[str, Any]
    final_stride: int = 0

    def lane_index(self, lane_id: str) -> int:
        return next(i for i, lane in enumerate(self.scene.lanes) if lane.id == lane_id)

    def seconds_since_green(self, t: np.ndarray) -> np.ndarray:
        """Seconds since the current green phase began; NaN while the phase is not green."""
        phase = self.signal.phase
        starts = self.signal.t[np.flatnonzero((phase == GREEN) & np.r_[True, phase[:-1] != GREEN])]
        idx = np.searchsorted(starts, t, side="right") - 1
        since = np.where(idx >= 0, t - starts[np.clip(idx, 0, None)], np.nan)
        return np.where(self.signal.phase_at(t) == GREEN, since, np.nan)


class Rule(Protocol):
    """A rule turns a video context into raw segments of its class (post-processing is applied later)."""

    label: str

    def apply(self, ctx: VideoContext) -> list[Segment]: ...
