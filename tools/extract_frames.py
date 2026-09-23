"""Save reference frames (start, middle, end) of every sample video at full resolution.

    python tools/extract_frames.py [--videos samples] [--out outputs/frames]

Frames are named <video>_<start|mid|end>.jpg. They are the backdrop for EDA
plots and for drawing configs/scene.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT  # noqa: E402
from src.video import probe, read_frame_at  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
JPEG_QUALITY = 92
END_MARGIN_SEC = 1.0


def reference_indices(n_frames: int, fps: float) -> dict[str, int]:
    return {"start": 0, "mid": n_frames // 2, "end": n_frames - 1 - round(END_MARGIN_SEC * fps)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "frames"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        info = probe(str(path))
        for tag, index in reference_indices(info.n_frames, info.fps).items():
            target = out / f"{path.stem}_{tag}.jpg"
            cv2.imwrite(str(target), read_frame_at(str(path), index), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            print(f"{target.relative_to(ROOT)}  t={index / info.fps:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
