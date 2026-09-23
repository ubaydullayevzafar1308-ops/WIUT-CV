"""Part A: video -> tracks -> events."""
from __future__ import annotations

from src.tracking import track_video


def detect_events(video_path: str) -> list[list]:
    """Return ``[[start_sec, end_sec, label], ...]`` for one video.

    Tracks are computed (and cached) here; per-class rules and segment
    post-processing plug in on top of them, so no events are emitted yet.
    """
    track_video(video_path)
    return []
