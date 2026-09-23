# WIUT Hackathon 2026 — CV Track: traffic event detection

## Task

A fixed CCTV camera over a road. For each `.mp4`:

- **Part A**: `detect_events(video_path) -> [[start_sec, end_sec, label], ...]`
- **Part B**: `RiskEstimator.reset(meta)` / `.step(frame, t_sec) -> float` — probability that an `accident` starts within the next 5 s. Causal: only frames already seen.

Full task statement: `docs/task.pdf`. Exact metric: `evaluate.py`. Scene description: `samples/camera.md`.

## Sample data (measured)

| File | Duration | Frames |
| --- | --- | --- |
| C3896.MP4 | 340.3 s | 10200 |
| C3897.MP4 | 317.8 s | 9525 |
| C3902.MP4 | 317.8 s | 9525 |
| C3905.MP4 | 127.6 s | 3825 |

- All clips: **3840×2160, H.264, 29.97 fps (30000/1001), ~140 Mbit/s**, ~2.3–6.2 GB each. The test set has the same resolution and fps.
- **Decoding 4K is the main runtime cost, not YOLO.** The harness decodes every 4K frame for Part B with OpenCV; Part A decodes the video again. Both must fit in 3 × duration on 8 CPU cores + T4.
  - Part A: decode with threaded PyAV (`av`, `stream.thread_type = "AUTO"`) or a bundled ffmpeg (`imageio-ffmpeg`) that downscales and drops frames in the decoder (e.g. `fps=10,scale=1280:-2`). Never rely on a system `ffmpeg` binary — the eval machine only has what `requirements.txt` installs.
  - Part B: `step()` receives full 4K frames. On skipped frames return the last score without touching the frame; on processed frames `cv2.resize` first, then YOLO.
  - Measure decode fps and end-to-end time per video on the first day; keep a runtime report per run.
- Downscale to ~1280 px wide for detection (`imgsz` 1280 keeps distant pedestrians); scene coordinates stay normalised, so resolution changes do not break `scene.json`.
- Timestamps: `t = frame_index / fps`, exactly as the harness does (fps = 29.97, not 25).
- Extension is upper-case `.MP4` (the harness accepts it).
- Originals never go to git. For labelling and the website use 720p proxies in `samples/proxy/` (`ffmpeg -i X.MP4 -vf scale=1280:-2 -c:v libx264 -crf 23 -an X_720p.mp4`); all timing and scoring is done on the originals.

## Hard rules

- `run_submission.py` and `evaluate.py` come from the organizers — **never modify them**.
- `solution.py` lives in the repo root; keep the names and signatures of `CLASSES`, `detect_events`, `RiskEstimator` exactly. `CLASSES` may only shrink, never gain ids.
- Evaluation runs **offline**: 1× T4 16 GB, 8 CPU cores, 32 GB RAM, no internet. Weights (≤ 5 GB total) live in `weights/` and are loaded by explicit path. Nothing may be downloaded at runtime.
- Time budget per video (Part A + Part B together) ≤ 3 × video duration, otherwise the video scores as empty. Target ≤ 1.5 ×.
- `RiskEstimator` must not open the video file and must not use Part A results. (Part A may use Part B; not the reverse.)
- Determinism: two runs on the same machine must produce the same `predictions.json`. Fix all seeds.
- Open-weights models only. No hosted/paid model APIs at inference.
- No hard-coded answers for specific sample videos. Hard-coding scene geometry from `camera.md` is allowed and expected.
- Official run: `pip install -r requirements.txt` then `python run_submission.py --videos /data/test --out predictions.json`. Must work on a clean machine with no manual steps.

## What matters in the metric

- Part A: per-class F1 at tIoU 0.3 / 0.5 / 0.7, averaged over τ, then over classes. A class we predict that never occurs in the test set scores 0 and still counts in the mean → only predict classes we detect reliably; remove the rest from `CLASSES`.
- Boundaries matter (tIoU 0.7). Follow the start/end conventions from the class table in the task exactly.
- Simultaneous events of the same class = one merged segment. Same-class segments never overlap.
- Part B: chance-normalised AP + alarm F1 (θ = 0.5, runs < 2 s apart merged, match window 10 s before the accident) + mTTA. False alarms kill precision → in normal traffic the risk must stay well below 0.5 and rise smoothly before an impact.

## Architecture

```
video → frames (stride 2–3) → YOLO (COCO: car, bus, truck, motorcycle, bicycle, person)
      → ByteTrack → track features (speed, heading, zone, dwell time) + configs/scene.json
      → per-class rules (per-frame flags) → postprocess (flags → segments) → events
```

- Detector + tracker output is cached to `cache/<video_stem>_<params_hash>.npz` so rules can be re-run in seconds without YOLO. The cache is a dev convenience; the official run starts with it empty.
- The YOLO model is loaded once per process (module-level singleton); Part B reuses the loaded model but runs its own tracker instance, reset in `reset()`.
- Device auto-select: `cuda` → `mps` (local Mac dev) → `cpu`.
- Anchor point of an object = bottom-centre of its bbox (road contact point).
- Coordinates in `scene.json` are normalised to [0, 1] of frame width/height.
- Draw visualisations with our own OpenCV code (`src/viz.py`), not Ultralytics plotting helpers (they may try to fetch fonts online).

## Repository layout

```
solution.py              # interface only: delegates to src/
run_submission.py        # organizers', unchanged
evaluate.py              # organizers', unchanged
requirements.txt         # pinned with ==, includes lap for ByteTrack
README.md
predictions_samples.json
weights/                 # yolo11s.pt (main), yolo11n.pt (CPU demo), loaded by path
configs/
  scene.json             # lanes (+ allowed direction), carriageway, stop_lines, crossings, solid_lines, intersection, traffic_light_roi
  params.yaml            # every threshold; per-class merge_gap / min_len
src/
  video.py               # frame reading with stride
  tracking.py            # YOLO + ByteTrack → tracks table, cache
  scene.py               # load scene.json, point-in-zone queries
  features.py            # smoothed position, speed, acceleration, heading, zone, dwell time
  signal.py              # traffic-light colour from ROI pixels (only if visible)
  rules/                 # one module per class group, common interface in base.py
  postprocess.py         # flags → segments: merge gaps, drop blips, clip to duration
  pipeline.py            # detect_events implementation
  risk.py                # RiskEstimator implementation (TTC-based)
  viz.py                 # boxes, tracks, timelines for the website
tools/
  draw_scene.py          # click on a frame → polygons into scene.json
  eda.py                 # stats, heatmaps, trajectories for the website
  render_samples.py      # annotated sample videos + timelines
data/my_labels.json      # our dev labels (same format as ground truth)
demo/app.py              # Gradio demo for Hugging Face Spaces
tests/                   # small tests for postprocess and rules
docs/task.pdf
```

`.gitignore`: `samples/*.mp4`, `cache/`, `outputs/`, `*.mp4` except small demo clips.

## Rules per class (implementation order)

| Priority | Class | Rule | Start → End |
| --- | --- | --- | --- |
| 1 | stopped_vehicle | speed < thr for ≥ 10 s on carriageway, not queued behind other stopped vehicles / not at a red signal | stops → moves |
| 1 | congestion | ≥ N vehicles in a direction and median speed < thr across all its lanes, longer than a signal cycle | queue stops → clears |
| 1 | wrong_way | angle between motion and lane direction > 120°, speed > min, ≥ 1 s | enters opposing lane → back / leaves |
| 1 | jaywalking | person anchor in carriageway and not in a crossing, ≥ 1 s | steps on road → leaves |
| 2 | red_light, stop_line | only if the signal is visible: crossing / stopping past the stop line on red | per class table |
| 2 | failure_to_yield | vehicle inside a crossing while a person is on it | enters → leaves crossing |
| 2 | solid_line_crossing | trajectory crosses a solid line outside the intersection | wheel on line → fully in new lane |
| 3 | illegal_turn, illegal_u_turn | entry/exit zones → manoeuvre, checked against prohibited manoeuvres from camera.md; U-turn = heading change > 150° | starts turning → completes |
| 3 | accident | bbox overlap of two tracks + sharp speed drop of both + both stationary afterwards | contact → all stopped |
| 3 | near_miss | low TTC + hard braking or swerve, no contact, no stop | evasive action → clear |
| — | road_obstacle, fire_smoke | not implemented by default; remove from CLASSES | — |

All thresholds are tuned on `data/my_labels.json` for mean F1, paying attention to tIoU 0.7.

## Part B

Every 3rd frame: detect + track; per track velocity over the last ~0.5 s; for approaching pairs within R px compute TTC (distance / closing speed, or closest-point-of-approach). risk = sigmoid(a · (τ0 − TTC_min)) + bonuses (hard braking, wrong-way vehicle, person on carriageway near a vehicle), EMA-smoothed, clipped to [0, 1]. Other frames return the last score. Calibrate τ0 and a so that 0.5 ≈ "accident probably within 5 s".

## Code quality

- Code, comments, docstrings, README — in English.
- Small, clear modules; type hints; docstrings on public functions.
- No dead code, no commented-out blocks, no `*_old.py`, no debug prints. Use `logging`.
- No magic numbers in code — thresholds go to `configs/params.yaml`.
- Notebooks are optional and never the only source of logic.
- Seeds fixed in one place (`random`, `numpy`, `torch`; `cudnn.deterministic = True`).
- Third-party code is credited in the README; every dataset listed with its licence.

## Dev loop

```bash
python run_submission.py --videos samples --out predictions_samples.json --team <team>
python evaluate.py --pred predictions_samples.json --gt data/my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

Print runtime per video vs. budget on every run. Check the effect of every change on the per-class table.

`data/my_labels.json` format:

```json
{"sample_01.mp4": {"duration": 300.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}
```

## Git workflow

- `main` is always runnable: `run_submission.py` works and `evaluate.py --validate-only` passes.
- Every task goes on its own branch: `feat/<name>`, `fix/<name>`, `docs/<name>` (e.g. `feat/tracking-cache`, `feat/rule-wrong-way`).
- Small, focused commits with clear messages in English (`feat: add ByteTrack cache`, `fix: merge same-class overlaps`).
- Push the branch and open a PR to `main` with `gh pr create`. PR description: what changed, why, Score A/B before → after on the dev set, runtime vs. budget.
- Before merging: run the dev loop above; no dead code, no debug prints; `run_submission.py` and `evaluate.py` untouched.
- Merge with squash; delete the branch after merge.
- After each working milestone, tag `main` (`v0.1`, `v0.2`, …) and push tags — a known-good fallback.
- Never commit videos, `cache/`, `outputs/` or any file > 50 MB. Check `git status` before every commit.
