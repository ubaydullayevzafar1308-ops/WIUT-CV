"""Common interface of the per-class rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.postprocess import Segment, flags_to_runs
from src.scene import Scene
from src.signal import GREEN, SignalTimeline


@dataclass
class VideoContext:
    """Everything a rule sees about one video.

    ``features`` is the table from src/features.py (sorted by track, then frame);
    ``sample_t`` are the times of all sampled frames, including frames without boxes;
    ``aspect`` is height / width, the factor that makes normalised y comparable with x;
    ``car_width_fit`` is the car box width against y (perspective, see features.fit_car_width);
    ``final_stride`` is the sampling stride tracking ended with.
    """

    features: np.ndarray
    scene: Scene
    signal: SignalTimeline
    sample_t: np.ndarray
    duration: float
    aspect: float
    params: dict[str, Any]
    car_width_fit: np.ndarray
    final_stride: int = 0

    def lane_width(self, y: np.ndarray) -> np.ndarray:
        """Width of a traffic lane (frame widths) at image height ``y``, from the perspective fit."""
        return self.params["rules"]["lane_width_cars"] * np.polyval(self.car_width_fit, y)

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


def merged_runs(t: np.ndarray, flags: np.ndarray, merge_sec: float) -> list[tuple[float, float]]:
    """Runs of flagged samples of one track, joined across gaps shorter than ``merge_sec``."""
    runs: list[list[float]] = []
    for start, end in flags_to_runs(t, flags):
        if runs and start - runs[-1][1] < merge_sec:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    return [(s, e) for s, e in runs]
