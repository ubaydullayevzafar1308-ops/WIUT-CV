"""Build outputs/review.html: a local page to accept, reject, adjust and add events per sample video.

    python tools/propose_labels.py [--videos samples]

The page embeds the events the rules detect (merged per class, with the reasons
of the detections behind each one) and plays the 720p proxies from
samples/proxy/ (tools/make_proxies.sh). It needs no server: open it with a
double click. Decisions stay in the browser's localStorage; "Export" downloads
my_labels.json in the ground-truth format (accepted and added events; duration
and fps of the original videos) - save it as data/my_labels.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import OFFICIAL_CLASSES  # noqa: E402
from src.config import ROOT, runtime_params  # noqa: E402
from src.pipeline import video_context  # noqa: E402
from src.postprocess import merge_segments  # noqa: E402
from src.rules import RULES, apply_rules  # noqa: E402
from src.video import probe  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
TEMPLATE = Path(__file__).resolve().parent / "templates" / "review.html"
PLACEHOLDER = "/*DATA*/null/*END*/"


def event_notes(ctx, label: str, start: float, end: float) -> str:
    """Reasons of the kept detections that make up a merged event."""
    rule = next((r for r in RULES if r.label == label and hasattr(r, "candidates")), None)
    if rule is None:
        return ""
    reasons = [f"track {c['track_id']}: {c['reason']}" if "track_id" in c else c["reason"]
               for c in rule.candidates(ctx) if c["kept"] and c["start"] < end and c["end"] > start]
    return "\n".join(reasons)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "review.html"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    params = runtime_params()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    videos = []
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        info = probe(str(path))
        ctx = video_context(str(path), params)
        merged = merge_segments(apply_rules(ctx), params["postprocess"], ctx.duration)
        proxy = Path(args.videos) / "proxy" / f"{path.stem}_720p.mp4"
        videos.append({
            "name": path.name,
            "duration": round(info.duration, 3),
            "fps": round(info.fps, 3),
            "proxy": os.path.relpath(proxy.resolve(), out.parent.resolve()),
            "events": [{"id": f"{label}@{start:.1f}", "label": label, "start": start, "end": end,
                        "note": event_notes(ctx, label, start, end)}
                       for start, end, label in merged],
        })
        print(f"{path.name}: {len(merged)} events")
    data = {"classes": list(OFFICIAL_CLASSES), "videos": videos}
    page = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, json.dumps(data, ensure_ascii=False))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
