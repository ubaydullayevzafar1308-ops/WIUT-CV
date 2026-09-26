# WIUT Hackathon 2026 — Computer Vision track: starter kit

Traffic events from a fixed road camera: **detect** them as time segments
(`[start_sec, end_sec, label]`) and, as a bonus, **anticipate** accidents with a
causal risk score. Three files; read the task description for the rules.

```
solution.py          <- the ONLY file you implement (CLASSES, detect_events, RiskEstimator)
run_submission.py    <- organizers' harness: folder of videos -> predictions.json   (do not modify)
evaluate.py          <- format check + the official metric                          (do not modify)
examples/            <- ground_truth.json and predictions.json in the exact format
requirements.txt     <- numpy + opencv for the harness; add your own deps to YOUR repo
```

## Approach

```
4K video ─► PyAV decode (every 3rd frame, B-frames skipped in the decoder) ─► YOLO11s @1280 ─► ByteTrack
        ─► track features (anchor, speed, heading, lane, zones, dwell) + configs/scene.json + signal phase
        ─► per-class rules (src/rules/) ─► post-processing (merge, drop blips) ─► events
```

- **Models.** Nothing is trained: YOLO11s with COCO weights (`weights/yolo11s.pt`; `yolo11n.pt` at 640 px on
  CPU-only machines) and Ultralytics ByteTrack. All events come from rules on the tracks, with thresholds in
  `configs/params.yaml`, tuned on the four sample videos by reviewing clips.
- **Scene.** Lanes (with allowed directions), carriageway, crossings, stop line, islands, bus stops,
  gantry/poles and the traffic-light heads are drawn once on a reference frame (`configs/scene.json`). Each
  video is aligned to it at runtime (ECC on gradient images; the camera shifts 1–3 % between sessions).
- **Signal.** The phase of the near approach is read from the pedestrian and vehicle heads by relative chroma
  (`src/signal.py`); the rules use it for red_light, stop_line, stopped_vehicle and failure_to_yield.
- **Road plane.** Distances and speeds for collisions use an approximate ground-plane mapping (horizon and
  scale from the car box width across the image, focal length from vanishing points), in car box widths.
- **Determinism and budget.** Sampling strides depend only on the device and the video's metadata
  (cuda / mps: 10 detector frames per second of video = stride 3 at 29.97 fps; cpu: 5 = stride 6), never on
  timing, so two runs give the same predictions. Timing is only a logged fuse: Part A samples more sparsely
  only if Part A + Part B are projected over 2.7× the duration, Part B only if it runs slower than 2× real
  time (the limit is 3×).

| Class | Rule (src/rules/) |
| --- | --- |
| stopped_vehicle | stationary ≥ 10 s on the carriageway while traffic goes round it; not queued, not held by the signal, not yielding, not in a bus stop / parking zone |
| congestion | a lane holds ≥ 5 vehicles with median speed below crawl speed for longer than a signal cycle (90 s) |
| wrong_way | heading > 120° from every allowed direction of the lane for ≥ 2 s and ≥ 2 box widths of travel |
| jaywalking | reliable pedestrian on the carriageway outside crossings / islands for ≥ 2 s, walking, not cutting a corner |
| red_light | crosses the near stop line on red and drives on into the intersection |
| stop_line | on red, stops with its front past the stop line without entering the intersection |
| failure_to_yield | drives across a crossing while a pedestrian walks on it in the vehicle's path (near crossing: only when vehicles do not have green) |
| accident | road-plane footprints touch while closing in, both lose speed sharply, both stop together or drive out of the frame |
| near_miss | time to contact ≤ 1 s with clearly hard braking or a swerve, reliable tracks, not a queue, no contact, both drive on |

**Part B** (`src/risk.py`) is causal: every 3rd frame at 960 px through the shared YOLO11s and its own
ByteTrack; time to contact of approaching pairs on the road plane → `sigmoid(3·(1.5 − TTC))` plus bonuses
(hard braking, wrong way, pedestrian on the road near a vehicle), EMA-smoothed.

## Results

Full run of `run_submission.py` on the four sample videos (Mac M4, MPS, empty cache), events per class in
`predictions_samples.json` (`evaluate.py --validate-only`: VALID, 22 events). Two consecutive runs gave
identical files (events and risk curves); Part A sampled every 3rd frame on every video and no time fuse blew.

| Class | C3896 | C3897 | C3902 | C3905 | Total |
| --- | --- | --- | --- | --- | --- |
| accident | 0 | 0 | 0 | 0 | 0 |
| near_miss | 0 | 0 | 0 | 0 | 0 |
| red_light | 1 | 1 | 0 | 0 | 2 |
| wrong_way | 0 | 0 | 1 | 0 | 1 |
| stopped_vehicle | 0 | 1 | 0 | 0 | 1 |
| jaywalking | 2 | 0 | 1 | 0 | 3 |
| failure_to_yield | 1 | 3 | 2 | 4 | 10 |
| stop_line | 2 | 1 | 1 | 1 | 5 |
| congestion | 0 | 0 | 0 | 0 | 0 |
| **all** | **6** | **6** | **5** | **5** | **22** |

Time per video, Part A + Part B, against the 3× limit:

| Video | Duration | Part A | A + B | × duration | Share of the 3× budget |
| --- | --- | --- | --- | --- | --- |
| C3896 | 340.3 s | 233.1 s | 593.1 s | 1.74× | 58% |
| C3897 | 317.8 s | 220.2 s | 553.3 s | 1.74× | 58% |
| C3902 | 317.8 s | 225.0 s | 570.7 s | 1.80× | 60% |
| C3905 | 127.6 s | 93.4 s | 225.6 s | 1.77× | 59% |

There are no ground-truth labels for the samples, so no scores are given.

## Reference edge map and NDA

`configs/reference_edges.png` is the template the scene alignment (`src/registration.py`) matches every video
against: the gradient-magnitude edge map of the frame `configs/scene.json` was drawn on (C3896, middle frame),
exactly what ECC compares, stored as a 16-bit PNG (960×540, values in [0, 1] × 65535). It was made with

```python
save_edges(edge_map(cv2.imread("configs/local/reference_frame.jpg"), 960), Path("configs/reference_edges.png"))
```

The frame itself is sample data and is not stored in the repository, nor in its history (`configs/local/` is
git-ignored). The edge map was added with the organisers' permission (25.09.2026). Another template can
be given with `WIUT_REFERENCE` (an edge map or a frame); without one, alignment is disabled and the scene is used as
drawn.

## Live demo API

`demo/api.py` serves the website's live demo: upload a clip, poll the job, fetch events and the risk curve.

```bash
pip install -r demo/requirements.txt
python -m demo.api --host 0.0.0.0 --port 8000
```

| Endpoint | Response |
| --- | --- |
| `POST /api/analyze` (multipart, field `video`) | `{"job_id": "..."}` |
| `GET /api/jobs/{job_id}` | `{"status": "queued" \| "running" \| "done" \| "error", "progress": 0..1, "error"?: "..."}` |
| `GET /api/jobs/{job_id}/result` | `{"duration", "fps", "events": [[s, e, label]], "risk": [[t, score]], "scene_matched"}` |
| `GET /api/health` | `{"ok": true}` |

- One video is analysed at a time; later uploads wait in a queue (at most `demo.max_queued`, then 503).
  `progress` counts decoded frames. Refused requests get `{"error": "..."}`: not `.mp4` (415), over 200 MB
  (413, checked from `Content-Length` before the upload is received), longer than 120 s or unreadable (400).
  Uploads are deleted as soon as their analysis ends.
- Light mode for a CPU server (4 vCPU, no GPU), `demo` in `configs/params.yaml` (`src/demo.py`): yolo11n at
  640 px, 5 detector frames per second for Part A and 5 risk updates per second, both fed from one decoding
  pass; a time fuse raises Part A's stride if the analysis is projected over 1.6× the duration + 20 s.
- `risk` is the `RiskEstimator` curve thinned to 10 Hz: the largest score in every 0.1 s window.
- `scene_matched`: the video was aligned to the reference edge map **and** at least 80 % of moving vehicles
  follow the lane directions of `configs/scene.json`. If not, only events not tied to this intersection are
  returned (`accident`, `near_miss`), plus the risk curve.

## System requirements

- Python 3.11, `pip install -r requirements.txt` (nothing else; OpenCV is headless, no `libGL`).
- **Linux + NVIDIA GPU: driver ≥ 525.60.13.** On Linux `requirements.txt` installs the CUDA 12.6
  build of torch 2.14.0 / torchvision 0.29.0 from `https://download.pytorch.org/whl/cu126`
  (the default PyPI build is CUDA 13 and needs driver ≥ 580). Check with `nvidia-smi`.
- macOS installs the regular build (MPS on Apple Silicon, otherwise CPU).

## Quickstart

```bash
pip install -r requirements.txt
# 1. implement solution.py
# 2. label the sample videos yourselves -> my_labels.json (same shape as examples/ground_truth.json)
python run_submission.py --videos samples --out predictions_samples.json --team <your-team>
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only        # format check without labels
```

## The interface (`solution.py`)

```python
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]

def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...]; label in CLASSES; same-class segments don't overlap."""

class RiskEstimator:
    def reset(self, meta: dict) -> None: ...            # meta: video_id, fps, width, height, n_frames
    def step(self, frame: np.ndarray, t_sec: float) -> float: ...   # BGR uint8 frame -> P(accident within 5 s)
```

`step` is called for **every frame in order** by the harness; it must not open the
video itself. Skipping frames internally and returning the last score is fine.
You may remove ids from `CLASSES`; never add.

## What we run (offline, one GPU, no internet)

```bash
pip install -r requirements.txt            # or: docker build -t team .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

Time budget per video: **3 × its duration** for Part A + Part B together; a video
over budget or a crash scores as empty. Events with a bad label, bad times, or a
same-class overlap are dropped by the harness and listed in its log. Weights
≤ 5 GB, shipped in the repo or fetched once by `weights/download.sh` before the
offline run.

## predictions.json

```json
{
  "team": "your-team-name",
  "videos": {
    "test_001.mp4": {
      "events": [[12.4, 18.9, "accident"], [40.0, 43.5, "red_light"]],
      "risk":   [[0.00, 0.01], [0.04, 0.01], [0.08, 0.02]]
    },
    "test_002.mp4": {"events": [], "risk": []}
  }
}
```

`risk` is written by the harness (one `[t_sec, score]` per frame). Keys are file
names. Every test video must be present, even with `"events": []`.
Ground truth: `{"test_001.mp4": {"duration": 600.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}`.

## Metric (exact code in `evaluate.py`)

**Part A.** Per class `c` and per tIoU threshold τ ∈ {0.3, 0.5, 0.7}: greedy
one-to-one matching by descending IoU; TP/FP/FN pooled over all videos; `F1_c(τ)`.
`Score_A = mean_c mean_τ F1_c(τ)`. Classes = those in the ground truth or in your
predictions (a class you predict that never occurs scores 0).

**Part B** (`accident` only; H = 5 s, W = 10 s, θ = 0.5). Frames in `[s−H, s)`
before an accident start `s` are positive; frames inside accidents and around
near-misses are ignored; the rest negative. `AP` = average precision over frames,
chance-normalised (`max(0, (AP_raw − r)/(1 − r))`, `r` = positive rate, so a
constant score gets 0). Alarms = runs of score ≥ θ (runs < 2 s apart merged),
alarm time = run start; an alarm in `[s−W, s)` of an unmatched accident matches it
→ `F1_alarm`; `mTTA` = mean of `s − alarm_time` (0 if unmatched).
`Score_B = 0.4·AP + 0.4·F1_alarm + 0.2·mTTA/W`.

**Model score** `M = 0.7·Score_A + 0.3·Score_B` (M = Score_A if the test set has no
accidents). Elimination score = 0.6·M + 0.25·Website + 0.15·Code.

## Tips

- Label the sample videos yourselves with the conventions from the task
  description and run `evaluate.py` against them. Without a dev set you are guessing.
- Detector + tracker → trajectories; most classes are rules on trajectories plus
  the scene layout. Learned models help most for `accident` / `near_miss`.
- Post-process segments: merge fragments, drop sub-second blips, then check F1@0.7.
- For Part B, time-to-collision from tracks is a strong simple signal; calibrate
  so that 0.5 means "probably within 5 s". A flat 1.0 scores ≈ 0.
- Print your runtime early; sampling every 2nd–5th frame is usually enough.
