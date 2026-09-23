"""Frame sampling in src/video.py on small synthetic H.264 clips.

Every frame is a flat grey whose level encodes its index, so a decoded frame
can be mapped back to the frame it came from.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from src.video import nonref_skip_is_regular, read_frames

N_FRAMES = 60
WIDTH, HEIGHT = 64, 48
FPS = Fraction(30000, 1001)
THREADS = 4
CHECK_FRAMES = 45


def level(index: int) -> int:
    return 20 + 3 * index


def write_clip(path: Path, bframes: int) -> Path:
    """Near-lossless H.264 with a fixed GOP: IBBP... for bframes=2, all P for bframes=0.

    Lossless mode and adaptive B-frame placement both suppress B-frames in x264,
    hence a small fixed QP and ``b_strategy=0``. An open GOP (as the camera uses)
    keeps I-frames on the same every-3rd-frame grid as the P-frames.
    """
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.codec_context.max_b_frames = bframes
        stream.options = {"b_strategy": "0", "x264-params": "qp=10:b-pyramid=0:keyint=15:open-gop=1:scenecut=0"}
        for i in range(N_FRAMES):
            img = np.full((HEIGHT, WIDTH, 3), level(i), dtype=np.uint8)
            out.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        out.mux(stream.encode())
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("clips")
    return {"ibbp": write_clip(tmp / "ibbp.mp4", 2), "all_p": write_clip(tmp / "all_p.mp4", 0)}


def sample(path: Path, stride: int, skip_nonref: bool) -> list[tuple[int, float, np.ndarray]]:
    return list(read_frames(str(path), stride, WIDTH, THREADS, skip_nonref, "AREA", CHECK_FRAMES))


def assert_frames_match_indices(frames: list[tuple[int, float, np.ndarray]]) -> None:
    levels = np.array([level(i) for i in range(N_FRAMES)])
    for index, t_sec, bgr in frames:
        assert int(np.argmin(np.abs(levels - bgr.mean()))) == index
        assert t_sec == pytest.approx(index / float(FPS))


def test_skip_is_regular_only_when_gop_matches_stride(clips):
    assert nonref_skip_is_regular(str(clips["ibbp"]), 3, CHECK_FRAMES, THREADS)
    assert not nonref_skip_is_regular(str(clips["ibbp"]), 2, CHECK_FRAMES, THREADS)
    assert not nonref_skip_is_regular(str(clips["all_p"]), 3, CHECK_FRAMES, THREADS)


@pytest.mark.parametrize(("clip", "stride", "skip_nonref"), [
    ("ibbp", 3, True),    # skip used: reference frames are exactly every 3rd
    ("ibbp", 3, False),   # skip disabled
    ("ibbp", 2, True),    # skip would give every 3rd frame -> fall back to full decode
    ("all_p", 3, True),   # nothing to skip -> fall back to full decode
])
def test_read_frames_yields_every_stride_th_frame(clips, clip, stride, skip_nonref):
    frames = sample(clips[clip], stride, skip_nonref)
    assert [index for index, _, _ in frames] == list(range(0, N_FRAMES, stride))
    assert_frames_match_indices(frames)
