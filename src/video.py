"""Fast frame reading from 4K H.264 with threaded PyAV and in-decoder downscaling."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Stream properties. ``fps`` is the nominal rate the harness uses for timestamps."""

    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def probe(path: str) -> VideoInfo:
    """Read stream properties without decoding."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        return VideoInfo(
            fps=float(stream.average_rate),
            n_frames=stream.frames,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
        )


def scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Output size for a downscale to ``target_width``; height is kept even for swscale."""
    if width <= target_width:
        return width, height
    return target_width, 2 * round(height * target_width / width / 2)


def read_frames(
    path: str,
    stride: int,
    target_width: int,
    threads: int,
    skip_nonref: bool,
    interpolation: str,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t_sec, bgr)`` for roughly every ``stride``-th frame.

    A frame is emitted when its index is at least ``stride`` after the last emitted
    one, so sampling stays even when the decoder drops frames. With ``skip_nonref``
    the decoder skips non-reference (B) frames entirely, which is much cheaper
    than decoding and discarding them.

    Frame indices come from presentation timestamps, and ``t_sec = index / fps``
    matches the harness, so timestamps stay correct whichever frames are skipped.
    Only sampled frames are converted and downscaled (by swscale, in C).
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        stream.thread_count = threads
        if skip_nonref:
            stream.codec_context.skip_frame = "NONREF"
        fps = float(stream.average_rate)
        start_pts = stream.start_time or 0
        out_w, out_h = scaled_size(stream.codec_context.width, stream.codec_context.height, target_width)
        next_index = 0
        for frame in container.decode(stream):
            index = round(float((frame.pts - start_pts) * stream.time_base) * fps)
            if index < next_index:
                continue
            next_index = index + stride
            bgr = frame.to_ndarray(width=out_w, height=out_h, format="bgr24", interpolation=interpolation)
            yield index, index / fps, bgr
