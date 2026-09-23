"""Part B: causal accident-risk estimator."""
from __future__ import annotations

import numpy as np


class RiskEstimator:
    """Per-frame P(an accident starts within the next 5 s), using only frames seen so far.

    The TTC-based estimator is not implemented yet: the score stays at 0.0 and
    frames are never touched, so Part B costs only the harness's own decoding.
    """

    def reset(self, meta: dict) -> None:
        """Start a new video. ``meta`` has video_id, fps, width, height, n_frames."""
        self.meta = meta
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return the risk score for the frame at ``t_sec``."""
        return self.last_score
