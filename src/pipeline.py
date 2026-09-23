"""Part A: video -> tracks + scene -> events."""
from __future__ import annotations

from src.scene import scene_for_video
from src.tracking import track_video


def detect_events(video_path: str) -> list[list]:
    """Return ``[[start_sec, end_sec, label], ...]`` for one video.

    Tracks are computed (and cached) and the scene is aligned to this video;
    per-class rules and segment post-processing plug in on top of both, so no
    events are emitted yet.
    """
    track_video(video_path)
    scene_for_video(video_path)
    return []
