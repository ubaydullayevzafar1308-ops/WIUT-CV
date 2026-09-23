"""Runtime report for one video: decoding variants and full tracking vs. the 3x-duration budget.

    python tools/benchmark.py samples/C3905.MP4 [--harness-decode]

Writes outputs/benchmark_<video>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_params, select_device  # noqa: E402
from src.tracking import track_video  # noqa: E402
from src.video import probe, read_frames  # noqa: E402

TIME_FACTOR = 3.0


def time_decode(path: str, video_params: dict, **overrides) -> tuple[float, int]:
    """Decode the whole video with ``read_frames``; return (seconds, frames yielded)."""
    vp = {**video_params, **overrides}
    t0 = time.perf_counter()
    n = sum(1 for _ in read_frames(
        path, stride=vp["stride"], target_width=vp["target_width"], threads=vp["decoder_threads"],
        skip_nonref=vp["skip_nonref"], interpolation=vp["interpolation"],
    ))
    return time.perf_counter() - t0, n


def time_harness_decode(path: str) -> tuple[float, int]:
    """Read every full-resolution frame with OpenCV, as run_submission.py does for Part B."""
    import cv2

    cap = cv2.VideoCapture(path)
    t0 = time.perf_counter()
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return time.perf_counter() - t0, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--harness-decode", action="store_true", help="also time the harness's OpenCV 4K read (Part B)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    params = load_params()
    info = probe(args.video)
    budget = TIME_FACTOR * info.duration
    rows: list[tuple[str, float, int]] = []

    vp = params["video"]
    for label, overrides in [
        ("PyAV decode, all frames decoded, every 3rd converted", {"skip_nonref": False}),
        ("PyAV decode, B-frames skipped in decoder (config)", {}),
    ]:
        sec, n = time_decode(args.video, vp, **overrides)
        rows.append((label, sec, n))
    if args.harness_decode:
        sec, n = time_harness_decode(args.video)
        rows.append(("OpenCV 4K read of every frame (harness Part B)", sec, n))

    tracks = track_video(args.video, params, use_cache=False)
    tm = tracks.timing
    rows.append((f"Part A tracking total ({Path(params['detector']['weights']).name}, {select_device()})",
                 tm["total"], len(tracks.frames)))
    for stage in ("decode_wait", "detect", "track"):
        rows.append((f"  of which {stage}", tm[stage], len(tracks.frames)))

    print(f"\n{Path(args.video).name}: {info.width}x{info.height}, {info.n_frames} frames, "
          f"{info.duration:.1f}s, budget {budget:.0f}s ({TIME_FACTOR:g}x)\n")
    print(f"| {'Stage':56s} | {'Time, s':>7s} | {'Frames':>6s} | {'Video fps':>9s} | {'x duration':>10s} |")
    print(f"|{'-' * 58}|{'-' * 9}:|{'-' * 8}:|{'-' * 11}:|{'-' * 12}:|")
    for label, sec, n in rows:
        print(f"| {label:56s} | {sec:7.1f} | {n:6d} | {info.n_frames / sec:9.1f} | {sec / info.duration:10.2f} |")

    report = {
        "video": Path(args.video).name,
        "duration_sec": info.duration,
        "budget_sec": budget,
        "device": select_device(),
        "params": {k: params[k] for k in ("video", "detector", "tracker")},
        "boxes": int(len(tracks.rows)),
        "rows": [{"stage": label, "sec": round(sec, 2), "frames": n} for label, sec, n in rows],
    }
    out = ROOT / "outputs" / f"benchmark_{Path(args.video).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
