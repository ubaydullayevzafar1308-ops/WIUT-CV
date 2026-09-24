"""Run the rules on the sample videos and cut a review clip per detection.

    python tools/review_clips.py [--videos samples] [--margin 3]

For every rule that reports candidates (stopped_vehicle, wrong_way,
jaywalking, ..., accident, near_miss) one clip per candidate: kept ones go to
outputs/review/<class>/, rejected ones the rule marks as close calls to
outputs/review/<class>_rejected/. The object is boxed in red on every frame,
following its track; the other participant of a pair (accident, near_miss) in
yellow. Congested stretches go to outputs/review/congestion/ without a box.

Clips come from the 720p proxies (samples/proxy/, see tools/make_proxies.sh);
outputs/review/index.csv lists every candidate with the reason it was kept or
rejected, outputs/review/events.json the final (merged) events per video.
Dev-only: uses the system ffmpeg.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, runtime_params  # noqa: E402
from src.pipeline import video_context  # noqa: E402
from src.postprocess import merge_segments  # noqa: E402
from src.rules import RULES, VideoContext, apply_rules  # noqa: E402
from src.rules.stopping import Congestion  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
HOLD_FRAMES = 15      # keep drawing a track's last box this many frames after its last sample
FIELDS = ["class", "video", "kept", "reason", "start_sec", "end_sec", "duration_sec", "track_id", "other_id", "x",
          "y", "clip", "clip_start_sec", "clip_end_sec"]
COLOURS = [(0, 0, 255), (0, 255, 255)]   # BGR: the object, the other participant


def cut(proxy: Path, start: float, end: float, out: Path, tracks: list[dict[int, np.ndarray]]) -> None:
    """Write [start, end] of the proxy as H.264, drawing each track's ``boxes[frame]`` (normalised x1, y1, x2, y2)
    in its colour (``COLOURS``)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(proxy))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first, last = int(round(start * fps)), int(round(end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    encoder = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    for index in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        for boxes, colour in zip(tracks, COLOURS):
            if index in boxes:
                x1, y1, x2, y2 = (boxes[index] * [width, height, width, height]).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    encoder.wait()
    cap.release()


def track_boxes(ctx: VideoContext, track_id: int) -> dict[int, np.ndarray]:
    """Box of a track for every video frame, holding each sample until the next one."""
    f = ctx.features[ctx.features["track_id"] == track_id]
    boxes = {}
    for k, row in enumerate(f):
        until = f["frame"][k + 1] if k + 1 < len(f) else row["frame"] + HOLD_FRAMES
        for index in range(int(row["frame"]), int(until)):
            boxes[index] = np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float64)
    return boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--margin", type=float, default=3.0, help="seconds of context before and after a detection")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    params = runtime_params()
    out_dir = ROOT / "outputs" / "review"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    videos = Path(args.videos)
    rows, events = [], {}
    for path in sorted(p for p in videos.iterdir() if p.suffix in VIDEO_EXTS):
        ctx = video_context(str(path), params)
        events[path.name] = merge_segments(apply_rules(ctx), params["postprocess"], ctx.duration)
        detections = [(rule.label, c) for rule in RULES if hasattr(rule, "candidates") for c in rule.candidates(ctx)]
        detections += [("congestion", {"start": s, "end": e, "kept": True, "close": False, "reason": "lane jammed"})
                       for s, e, _ in Congestion().apply(ctx)]
        proxy = videos / "proxy" / f"{path.stem}_720p.mp4"
        for label, d in detections:
            clip_start, clip_end = max(0.0, d["start"] - args.margin), min(ctx.duration, d["end"] + args.margin)
            folder = label if d["kept"] else f"{label}_rejected"
            name = f"{path.stem}_{d['start']:06.1f}-{d['end']:06.1f}"
            clip = out_dir / folder / (f"{name}_track{d['track_id']}.mp4" if "track_id" in d else f"{name}.mp4")
            review = d["kept"] or d["close"]
            if review:
                cut(proxy, clip_start, clip_end, clip,
                    [track_boxes(ctx, d[key]) for key in ("track_id", "other_id") if key in d])
            rows.append({"class": label, "video": path.name, "kept": d["kept"], "reason": d["reason"],
                         "start_sec": round(d["start"], 3), "end_sec": round(d["end"], 3),
                         "duration_sec": round(d["end"] - d["start"], 3), "track_id": d.get("track_id", ""),
                         "other_id": d.get("other_id", ""),
                         "x": round(d["x"], 3) if "x" in d else "", "y": round(d["y"], 3) if "y" in d else "",
                         "clip": str(clip.relative_to(out_dir)) if review else "",
                         "clip_start_sec": round(clip_start, 3), "clip_end_sec": round(clip_end, 3)})
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "events.json").write_text(json.dumps(events, indent=1))
    clipped = [row for row in rows if row["clip"]]
    print(f"{len(rows)} candidates, {sum(row['kept'] for row in rows)} kept, {len(clipped)} clips in "
          f"{out_dir.relative_to(ROOT)}")
    for row in clipped:
        print(f"  {'KEPT' if row['kept'] else 'rejected':8s} {row['class']:16s} {row['video']} "
              f"{row['start_sec']:7.1f}-{row['end_sec']:7.1f} s track {row['track_id']} at ({row['x']}, {row['y']}): "
              f"{row['reason']}")
    for video, evs in events.items():
        print(f"events {video}: {evs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
