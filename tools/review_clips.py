"""Run the rules on the sample videos and cut a review clip per detection.

    python tools/review_clips.py [--videos samples] [--margin 3]

One clip per vehicle stop (stopped_vehicle, the stopped vehicle boxed in red)
and per congested stretch, before the per-class merging, so every case can be
judged on its own. Clips come from the 720p proxies (samples/proxy/, see
tools/make_proxies.sh) and go to outputs/review/<class>/<video>_<start>-<end>.mp4;
outputs/review/index.csv lists them, outputs/review/events.json holds the final
(merged) events per video. Dev-only: uses the system ffmpeg.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, runtime_params  # noqa: E402
from src.pipeline import video_context  # noqa: E402
from src.postprocess import merge_segments  # noqa: E402
from src.rules import apply_rules  # noqa: E402
from src.rules.stopping import Congestion, StoppedVehicle  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
PROXY_SIZE = (1280, 720)
BOX_ASPECT = 0.8      # drawn box height / width for a stopped vehicle (only a visual marker)
FIELDS = ["class", "video", "start_sec", "end_sec", "duration_sec", "counted_sec", "track_id", "x", "y",
          "clip", "clip_start_sec", "clip_end_sec"]


def cut(proxy: Path, start: float, end: float, out: Path, box: tuple[int, int, int, int] | None) -> None:
    """Re-encode [start, end] of the proxy, optionally with a red box drawn on every frame."""
    out.parent.mkdir(parents=True, exist_ok=True)
    draw = ["-vf", "drawbox=x={}:y={}:w={}:h={}:color=red@0.9:t=3".format(*box)] if box else []
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(proxy),
         "-t", f"{end - start:.3f}", *draw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an",
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def vehicle_box(stop: dict) -> tuple[int, int, int, int]:
    """Box around a stopped vehicle in proxy pixels, from its anchor (bottom centre) and width."""
    w = stop["size"] * PROXY_SIZE[0] * 1.2
    h = w * BOX_ASPECT
    return int(stop["x"] * PROXY_SIZE[0] - w / 2), int(stop["y"] * PROXY_SIZE[1] - h), int(w), int(h)


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
        detections = [("stopped_vehicle", stop) for stop in StoppedVehicle().stops(ctx)]
        detections += [("congestion", {"start": s, "end": e}) for s, e, _ in Congestion().apply(ctx)]
        proxy = videos / "proxy" / f"{path.stem}_720p.mp4"
        for label, d in detections:
            clip_start, clip_end = max(0.0, d["start"] - args.margin), min(ctx.duration, d["end"] + args.margin)
            name = f"{path.stem}_{d['start']:06.1f}-{d['end']:06.1f}"
            clip = out_dir / label / (f"{name}_track{d['track_id']}.mp4" if "track_id" in d else f"{name}.mp4")
            cut(proxy, clip_start, clip_end, clip, vehicle_box(d) if "size" in d else None)
            rows.append({"class": label, "video": path.name, "start_sec": round(d["start"], 3),
                         "end_sec": round(d["end"], 3), "duration_sec": round(d["end"] - d["start"], 3),
                         "counted_sec": d.get("counted_sec", ""), "track_id": d.get("track_id", ""),
                         "x": round(d["x"], 3) if "x" in d else "", "y": round(d["y"], 3) if "y" in d else "",
                         "clip": str(clip.relative_to(out_dir)), "clip_start_sec": round(clip_start, 3),
                         "clip_end_sec": round(clip_end, 3)})
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "events.json").write_text(json.dumps(events, indent=1))
    print(f"{len(rows)} clips in {out_dir.relative_to(ROOT)}")
    for row in rows:
        print(f"  {row['class']:16s} {row['video']}  {row['start_sec']:7.1f}-{row['end_sec']:7.1f} s "
              f"({row['duration_sec']:5.1f} s) track {row['track_id']} at ({row['x']}, {row['y']})")
    for video, evs in events.items():
        print(f"events {video}: {evs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
