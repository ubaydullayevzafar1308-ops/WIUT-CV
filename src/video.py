"""Fast frame reading from 4K H.264 with threaded PyAV and in-decoder downscaling."""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

import av
import numpy as np

T = TypeVar("T")

log = logging.getLogger(__name__)


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
    """Read stream properties without decoding.

    The frame count comes from the container header; files that do not store it
    (some phone recordings) get it from the stream or container duration.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        n_frames = stream.frames
        if not n_frames:
            if stream.duration is not None:
                seconds = float(stream.duration * stream.time_base)
            else:
                seconds = (container.duration or 0) / av.time_base
            n_frames = round(seconds * fps)
        return VideoInfo(
            fps=fps,
            n_frames=n_frames,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
        )


def scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Output size for a downscale to ``target_width``; height is kept even for swscale."""
    if width <= target_width:
        return width, height
    return target_width, 2 * round(height * target_width / width / 2)


def _open_stream(container: av.container.InputContainer, threads: int, skip_nonref: bool):
    stream = container.streams.video[0]
    stream.thread_type = "FRAME"
    stream.thread_count = threads
    if skip_nonref:
        stream.codec_context.skip_frame = "NONREF"
    return stream


def _frame_index(frame: av.VideoFrame, stream) -> int:
    """Frame number from the presentation timestamp, counted from the first frame."""
    return round(float((frame.pts - (stream.start_time or 0)) * stream.time_base) * float(stream.average_rate))


def nonref_skip_is_regular(path: str, stride: int, check_frames: int, threads: int) -> bool:
    """Whether dropping non-reference frames in the decoder leaves exactly every ``stride``-th frame.

    Checked on the first ``check_frames`` frames: the surviving indices must start
    below ``stride`` and be exactly ``stride`` apart (true for an I/P every 3rd
    frame with two B-frames in between, false for all-P or other GOP layouts).
    """
    indices = []
    with av.open(path) as container:
        stream = _open_stream(container, threads, skip_nonref=True)
        for frame in container.decode(stream):
            index = _frame_index(frame, stream)
            if index >= check_frames:
                break
            indices.append(index)
    return bool(indices) and indices[0] < stride and all(b - a == stride for a, b in zip(indices, indices[1:]))


def read_frames(
    path: str,
    stride: int,
    target_width: int,
    threads: int,
    skip_nonref: bool,
    interpolation: str,
    skip_check_frames: int,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t_sec, bgr)`` for every ``stride``-th frame.

    With ``skip_nonref`` the decoder drops non-reference (B) frames entirely,
    which is much cheaper than decoding and discarding them. It is used only if
    it yields exactly every ``stride``-th frame (see ``nonref_skip_is_regular``);
    otherwise every frame is decoded and every ``stride``-th one is kept.

    Frame indices come from presentation timestamps, and ``t_sec = index / fps``
    matches the harness. Only sampled frames are converted and downscaled
    (by swscale, in C).
    """
    if skip_nonref and not nonref_skip_is_regular(path, stride, skip_check_frames, threads):
        log.info("%s: non-reference skip does not give every %d-th frame, decoding all frames", path, stride)
        skip_nonref = False
    with av.open(path) as container:
        stream = _open_stream(container, threads, skip_nonref)
        fps = float(stream.average_rate)
        out_w, out_h = scaled_size(stream.codec_context.width, stream.codec_context.height, target_width)
        next_index = 0
        warned = False
        for frame in container.decode(stream):
            index = _frame_index(frame, stream)
            if index < next_index:
                continue
            if skip_nonref and next_index and index != next_index and not warned:
                log.warning("%s: GOP changed at frame %d, sampling is no longer regular", path, index)
                warned = True
            next_index = index + stride
            bgr = frame.to_ndarray(width=out_w, height=out_h, format="bgr24", interpolation=interpolation)
            yield index, index / fps, bgr


def read_frame_at(path: str, index: int) -> np.ndarray:
    """Full-resolution BGR frame with the given index (seek to the preceding keyframe, decode forward)."""
    with av.open(path) as container:
        stream = _open_stream(container, threads=0, skip_nonref=False)
        fps = float(stream.average_rate)
        container.seek((stream.start_time or 0) + int(index / fps / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if _frame_index(frame, stream) >= index:
                return frame.to_ndarray(format="bgr24")
    raise ValueError(f"{path}: frame {index} not found")


def prefetch(items: Iterator[T], depth: int) -> Iterator[T]:
    """Run ``items`` in a background thread, keeping up to ``depth`` results ready.

    FFmpeg releases the GIL while decoding, so the next frames are decoded while
    the caller runs the detector. Exceptions from the producer are re-raised here.
    """
    buffer: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def produce() -> None:
        try:
            for item in items:
                buffer.put(item)
            buffer.put(done)
        except BaseException as err:  # noqa: BLE001 - handed over to the consumer thread
            buffer.put(err)

    threading.Thread(target=produce, name="frame-prefetch", daemon=True).start()
    while (item := buffer.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item
