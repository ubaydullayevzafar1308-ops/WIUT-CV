"""Estimate the Part B runtime: the harness loop with YOLO + ByteTrack on every stride-th 4K frame.

    python tools/benchmark_part_b.py samples/C3905.MP4 [--imgsz 640 960] [--stride 3]

Replays run_submission.py's Part B loop (OpenCV reads every full-resolution
frame; step() is called for each). On processed frames the frame is resized
with cv2.resize to the detector size, then YOLO (batch 1, causal) and ByteTrack
run; other frames only return the last score. Part A time is read from
outputs/benchmark_<video>.json (tools/benchmark.py) to give the total vs. budget.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_params, select_device  # noqa: E402
from src.tracking import get_model, make_tracker  # noqa: E402
from src.video import probe  # noqa: E402

TIME_FACTOR = 3.0


def run_loop(path: str, imgsz: int | None, stride: int, params: dict) -> dict[str, float]:
    """One pass of the harness loop; ``imgsz=None`` only decodes (the default estimator)."""
    dp = params["detector"]
    info = probe(path)
    device = select_device()
    model = get_model(dp["weights"]) if imgsz else None
    tracker = make_tracker(params["tracker"], info.fps / stride) if imgsz else None
    timing = {"resize": 0.0, "detect": 0.0, "track": 0.0}
    cap = cv2.VideoCapture(path)
    t_start = time.perf_counter()
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if imgsz and index % stride == 0:
            t0 = time.perf_counter()
            height = 2 * round(frame.shape[0] * imgsz / frame.shape[1] / 2)
            small = cv2.resize(frame, (imgsz, height), interpolation=cv2.INTER_AREA)
            t1 = time.perf_counter()
            result = model.predict(small, imgsz=imgsz, conf=dp["conf"], iou=dp["iou"], classes=dp["classes"],
                                   device=device, verbose=False)[0]
            t2 = time.perf_counter()
            tracker.update(result.boxes.cpu().numpy())
            t3 = time.perf_counter()
            timing["resize"] += t1 - t0
            timing["detect"] += t2 - t1
            timing["track"] += t3 - t2
        index += 1
    cap.release()
    timing["total"] = time.perf_counter() - t_start
    return timing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    params = load_params()
    info = probe(args.video)
    budget = TIME_FACTOR * info.duration
    part_a_report = ROOT / "outputs" / f"benchmark_{Path(args.video).stem}.json"
    part_a = None
    if part_a_report.exists():
        rows = json.loads(part_a_report.read_text())["rows"]
        part_a = next(r["sec"] for r in rows if r["stage"].startswith("Part A tracking total"))

    results = {"decode only (current estimator)": run_loop(args.video, None, args.stride, params)}
    for imgsz in args.imgsz:
        results[f"YOLO11s imgsz {imgsz}, every {args.stride}rd frame"] = run_loop(args.video, imgsz, args.stride, params)

    print(f"\n{Path(args.video).name}: {info.duration:.1f}s, budget {budget:.0f}s; device {select_device()}; "
          f"Part A = {part_a if part_a is not None else 'n/a'} s\n")
    print(f"| {'Part B variant':40s} | {'Part B, s':>9s} | {'resize':>6s} | {'detect':>6s} | {'track':>5s} "
          f"| {'A + B, s':>8s} | {'x duration':>10s} |")
    print(f"|{'-' * 42}|{'-' * 10}:|{'-' * 7}:|{'-' * 7}:|{'-' * 6}:|{'-' * 9}:|{'-' * 11}:|")
    report = []
    for label, tm in results.items():
        total = tm["total"] + (part_a or 0.0)
        print(f"| {label:40s} | {tm['total']:9.1f} | {tm['resize']:6.1f} | {tm['detect']:6.1f} | {tm['track']:5.1f} "
              f"| {total:8.1f} | {total / info.duration:10.2f} |")
        report.append({"variant": label, **{k: round(v, 2) for k, v in tm.items()}, "a_plus_b_sec": round(total, 2)})
    out = ROOT / "outputs" / f"benchmark_part_b_{Path(args.video).stem}.json"
    out.write_text(json.dumps({"video": Path(args.video).name, "duration_sec": info.duration, "budget_sec": budget,
                               "device": select_device(), "part_a_sec": part_a, "rows": report}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
