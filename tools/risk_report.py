"""Part B report: run the RiskEstimator like the harness does, plot the risk and count alarms.

    python tools/risk_report.py [--videos samples]           # full run, logs to outputs/risk_<video>.npz
    python tools/risk_report.py --replay                     # recompute the score from the logs (calibration)
    python tools/risk_report.py --replay --metric-check      # + evaluate.py on a synthetic accident

The full run streams every 4K frame through ``RiskEstimator.step`` exactly like
run_submission.py (OpenCV), records what the estimator measured at each
processed frame, and times it. ``--replay`` re-applies ``risk_score`` + EMA with
the current params.yaml to the recorded measurements, so thresholds can be tuned
in seconds. Writes outputs/risk_<video>.png (risk, alarm threshold, alarms,
time to contact) and prints alarms per video; alarms follow evaluate.py: runs
of score >= 0.5, runs less than 2 s apart merged.

``--metric-check`` exercises evaluate.py's Part B on the samples, which have no
accidents: a synthetic ``accident`` is placed at the contact predicted at the
most dangerous approach (the highest risk over all videos), and the risk curves
are scored against it (outputs/risk_check_{gt,pred}.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_submission import video_meta  # noqa: E402
from src.config import ROOT, runtime_params  # noqa: E402
from src.risk import NO_PAIRS, NO_WALKERS, Components, RiskEstimator, risk_score, smooth, time_to_contact  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}
THRESHOLD = 0.5
MERGE_SEC = 2.0
WIDTH, PLOT_H, TTC_H, MARGIN = 1800, 220, 90, 50


def run(path: Path) -> tuple[np.ndarray, np.ndarray, list[Components], float]:
    """Stream the video through the estimator; returns frame times, scores, measurements, seconds."""
    meta = video_meta(path)
    estimator = RiskEstimator(record=True)
    estimator.reset({k: meta[k] for k in ("video_id", "fps", "width", "height", "n_frames")})
    cap = cv2.VideoCapture(str(path))
    times, scores = [], []
    start = time.perf_counter()
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / meta["fps"]
        scores.append(estimator.step(frame, t))
        times.append(t)
        index += 1
    cap.release()
    return np.array(times), np.array(scores), estimator.log, time.perf_counter() - start


def replay(times: np.ndarray, log: list[Components], params: dict) -> np.ndarray:
    """The score curve the estimator would output with ``params``, from its recorded measurements."""
    processed = np.array([c.t for c in log])
    per_step, last = [], 0.0
    for c in log:
        last = smooth(last, risk_score(c, params), params)
        per_step.append(last)
    idx = np.searchsorted(processed, times, side="right") - 1
    return np.where(idx >= 0, np.array(per_step)[np.clip(idx, 0, None)], 0.0)


def save_log(path: Path, times: np.ndarray, scores: np.ndarray, log: list[Components], seconds: float) -> None:
    """Measurements of every processed frame; pairs and walkers flattened with their step index."""
    pair_step = np.concatenate([np.full(len(c.pairs), i) for i, c in enumerate(log)] or [np.zeros(0)])
    walker_step = np.concatenate([np.full(len(c.walkers), i) for i, c in enumerate(log)] or [np.zeros(0)])
    np.savez_compressed(
        path, times=times, scores=scores, seconds=seconds, t=[c.t for c in log], brake=[c.brake for c in log],
        wrong_way=[c.wrong_way for c in log], pairs=np.concatenate([c.pairs for c in log] or [NO_PAIRS]),
        pair_step=pair_step, walkers=np.concatenate([c.walkers for c in log] or [NO_WALKERS]), walker_step=walker_step)


def load_log(path: Path) -> tuple[np.ndarray, list[Components], float]:
    with np.load(path) as archive:
        data = {k: archive[k] for k in archive.files}   # decompress each array once
    n = len(data["t"])
    pair_bounds = np.searchsorted(data["pair_step"], np.arange(n + 1))
    walker_bounds = np.searchsorted(data["walker_step"], np.arange(n + 1))
    log = [Components(t=float(data["t"][i]), pairs=data["pairs"][pair_bounds[i]:pair_bounds[i + 1]],
                      brake=float(data["brake"][i]), wrong_way=bool(data["wrong_way"][i]),
                      walkers=data["walkers"][walker_bounds[i]:walker_bounds[i + 1]]) for i in range(n)]
    return data["times"], log, float(data["seconds"])


def alarms(times: np.ndarray, scores: np.ndarray) -> list[tuple[float, float]]:
    """Alarm runs as evaluate.py counts them."""
    runs: list[list[float]] = []
    above = scores >= THRESHOLD
    edges = np.diff(np.r_[0, above.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        if runs and times[s] - runs[-1][1] < MERGE_SEC:
            runs[-1][1] = float(times[e])
        else:
            runs.append([float(times[s]), float(times[e])])
    return [(s, e) for s, e in runs]


def metric_check(curves: dict[str, tuple[np.ndarray, np.ndarray, list[Components]]], params: dict,
                 out: Path) -> None:
    """Score the risk curves with evaluate.py against a synthetic accident right after the most dangerous approach.

    The most dangerous approach is the peak of the risk over all videos; the accident starts at the contact
    predicted there (peak time + time to contact).
    """
    peak, t, name = max((float(scores.max()), float(times[np.argmax(scores)]), n) for n, (times, scores, _) in curves.items())
    step = max((c for c in curves[name][2] if c.t <= t), key=lambda c: c.t)
    ttc = time_to_contact(step.pairs, params)
    start = t + (ttc if np.isfinite(ttc) else 0.0)
    gt, pred = {}, {"videos": {}}
    for n, (times, scores, _) in curves.items():
        events = [[round(start, 3), round(start + 2.0, 3), "accident"]] if n == name else []
        gt[n] = {"duration": float(times[-1]), "fps": float(1 / np.median(np.diff(times))), "events": events}
        pred["videos"][n] = {"events": [], "risk": [[round(float(a), 3), round(float(b), 4)] for a, b in zip(times, scores)]}
    (out / "risk_check_gt.json").write_text(json.dumps(gt))
    (out / "risk_check_pred.json").write_text(json.dumps(pred))
    times, scores, _ = curves[name]
    before = [s for s, _ in alarms(times, scores) if start - 10.0 <= s < start]
    lead = f"alarm starts {start - before[0]:.1f} s before it" if before else "no alarm in the 10 s before it"
    print(f"synthetic accident: {name} at {start:.1f} s (risk peak {peak:.2f} at {t:.1f} s, time to contact {ttc:.2f} s); "
          f"{lead}")
    subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", str(out / "risk_check_pred.json"),
                    "--gt", str(out / "risk_check_gt.json")], check=True)


def draw(name: str, times: np.ndarray, scores: np.ndarray, log: list[Components], runs: list,
         params: dict) -> np.ndarray:
    duration = times[-1] if len(times) else 1.0
    img = np.full((MARGIN + PLOT_H + TTC_H + 60, WIDTH, 3), 255, np.uint8)
    x_of = lambda t: int(MARGIN + (t / duration) * (WIDTH - 2 * MARGIN))  # noqa: E731
    top, bottom = MARGIN, MARGIN + PLOT_H
    for s, e in runs:
        cv2.rectangle(img, (x_of(s), top), (max(x_of(e), x_of(s) + 2), bottom), (200, 200, 255), -1)
    y_th = int(bottom - THRESHOLD * PLOT_H)
    cv2.line(img, (MARGIN, y_th), (WIDTH - MARGIN, y_th), (0, 0, 220), 1)
    pts = np.stack([[x_of(t) for t in times], (bottom - scores * PLOT_H).astype(int)], axis=1)
    cv2.polylines(img, [pts.astype(np.int32)], False, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(img, (MARGIN, top), (WIDTH - MARGIN, bottom), (180, 180, 180), 1)
    ttc_top = bottom + 20
    for c in log:
        ttc = time_to_contact(c.pairs, params)
        if np.isfinite(ttc):
            y = int(ttc_top + TTC_H * min(ttc, 5.0) / 5.0)
            cv2.circle(img, (x_of(c.t), y), 1, (200, 120, 0), -1)
    cv2.putText(img, "TTC 0-5 s", (5, ttc_top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 120, 0), 1)
    for t in np.arange(0, duration, 30):
        cv2.putText(img, f"{int(t)}s", (x_of(t) - 10, ttc_top + TTC_H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    title = f"{name}: risk (max {scores.max():.2f}), threshold {THRESHOLD}, {len(runs)} alarm(s)"
    cv2.putText(img, title, (MARGIN, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--replay", action="store_true", help="recompute from outputs/risk_<video>.npz")
    ap.add_argument("--metric-check", action="store_true", help="evaluate.py against a synthetic accident")
    args = ap.parse_args()
    params = runtime_params()["risk"]
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    total = 0
    curves = {}
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        store = out / f"risk_{path.stem}.npz"
        if args.replay:
            times, log, seconds = load_log(store)
            scores = replay(times, log, params)
        else:
            times, scores, log, seconds = run(path)
            save_log(store, times, scores, log, seconds)
        runs = alarms(times, scores)
        total += len(runs)
        curves[path.name] = (times, scores, log)
        cv2.imwrite(str(out / f"risk_{path.stem}.png"), draw(path.name, times, scores, log, runs, params))
        print(f"{path.name}: Part B {seconds:.1f} s = {seconds / times[-1]:.2f}x duration; max risk {scores.max():.2f}; "
              f"{len(runs)} alarm(s) {[(round(s, 1), round(e, 1)) for s, e in runs]}")
    print(f"total alarms: {total}")
    if args.metric_check:
        metric_check(curves, params, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
