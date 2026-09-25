"""List vehicles that turn into the side street not from the rightmost lane of avenue_near, with clips.

    python tools/turn_candidates.py [--videos samples] [--only C3896 C3905]

Candidates for a possible illegal_turn rule (not a class yet): the only lane
allowed to turn into the side street is the rightmost one at the stop line (by
the pole). A vehicle counts when its track crosses the avenue_near stop line and
later enters the side_street lane; its lane is its position along the stop line
(``lanes`` equal lanes, 1 = rightmost). Clips of the others (from the proxies)
go to outputs/review/illegal_turn_candidates/: the vehicle boxed in red, from
``lead`` seconds before it enters the side street. Run after tools/review_clips.py,
which rebuilds outputs/review/. Dev-only: uses the system ffmpeg.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, runtime_params  # noqa: E402
from src.features import track_slices  # noqa: E402
from src.pipeline import video_context  # noqa: E402
from src.rules.base import VideoContext  # noqa: E402
from src.rules.signal_rules import stop_line  # noqa: E402
from tools.review_clips import cut, track_boxes  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}


def turns(ctx: VideoContext, lanes: int) -> list[dict]:
    """Vehicles crossing the avenue_near stop line and then entering the side street, with their lane."""
    f = ctx.features
    line = stop_line(ctx.scene, ctx.aspect)
    dist, along = line.distance(f["x"], f["y"], ctx.aspect), line.along(f["x"], f["y"], ctx.aspect)
    side = ctx.lane_index("side_street")
    vehicle = np.isin(f["cls"], ctx.params["rules"]["vehicle_classes"])
    found = []
    for sl in track_slices(f["track_id"]):
        if not vehicle[sl.start]:
            continue
        d, a, t = dist[sl], along[sl], f["t"][sl]
        cross = np.flatnonzero((d[:-1] >= 0) & (d[1:] < 0) & (a[1:] > -0.1) & (a[1:] < 1.1)) + 1
        if not len(cross):
            continue
        into = np.flatnonzero((f["lane"][sl] == side) & (np.arange(len(t)) > cross[0]))
        if len(into):
            found.append({"track_id": int(f["track_id"][sl.start]), "cross": float(t[cross[0]]),
                          "enter": float(t[into[0]]), "along": float(a[cross[0]]),
                          "lane": int(np.clip(a[cross[0]] * lanes, 0, lanes - 1)) + 1})
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--only", nargs="*", help="video stems to process (default: all)")
    ap.add_argument("--lanes", type=int, default=4, help="lanes of avenue_near at the stop line")
    ap.add_argument("--lead", type=float, default=6.0, help="clip starts this long before entering the side street")
    ap.add_argument("--tail", type=float, default=2.0)
    args = ap.parse_args()
    params = runtime_params()
    out = ROOT / "outputs" / "review" / "illegal_turn_candidates"
    videos = Path(args.videos)
    for path in sorted(p for p in videos.iterdir() if p.suffix in VIDEO_EXTS):
        if args.only and path.stem not in args.only:
            continue
        ctx = video_context(str(path), params)
        found = turns(ctx, args.lanes)
        wrong = [c for c in found if c["lane"] > 1]
        print(f"{path.name}: {len(found)} turns into the side street, {len(wrong)} not from the rightmost lane")
        for c in sorted(found, key=lambda c: c["enter"]):
            mark = "CANDIDATE" if c["lane"] > 1 else "ok       "
            print(f"  {mark} track {c['track_id']}: stop line at {c['cross']:.1f} s (lane {c['lane']}, "
                  f"along {c['along']:.2f}), enters the side street at {c['enter']:.1f} s")
        proxy = videos / "proxy" / f"{path.stem}_720p.mp4"
        for c in wrong:
            start = max(0.0, max(c["cross"], c["enter"] - args.lead) - 1.0)
            end = min(ctx.duration, c["enter"] + args.tail)
            clip = out / f"{path.stem}_{start:06.1f}-{end:06.1f}_track{c['track_id']}_lane{c['lane']}.mp4"
            cut(proxy, start, end, clip, [track_boxes(ctx, c["track_id"])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
