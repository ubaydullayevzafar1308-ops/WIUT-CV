# ASILA — WIUT Hackathon 2026, Computer Vision track

Traffic events from a fixed road camera: every event as a time segment `[start_sec, end_sec, label]` (Part A) and,
frame by frame, the risk that an accident starts within 5 s (Part B, causal).

## Quick start

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
```

- **Weights** are in the repository (`weights/yolo11s.pt`, `weights/yolo11n.pt` for CPU) and loaded by path; nothing is
  downloaded at run time, no internet needed.
- **Python 3.10 or newer** (on 3.10 `requirements.txt` picks numpy 2.2.6 and av 17.1.0; on 3.11+ numpy 2.4.6 and
  av 18.1.0). On macOS use Python 3.11+ (some wheels are missing for 3.10 on macOS); Linux 3.10+ is tested.
- **GPU**: Linux + NVIDIA (T4-class). `requirements.txt` installs the CUDA 12.6 build of torch 2.14.0 /
  torchvision 0.29.0, which needs **driver ≥ 525.60.13** (check with `nvidia-smi`). macOS installs the regular build
  (MPS on Apple Silicon).
- **CPU mode** (no GPU, or `WIUT_DEVICE=cpu`): the `cpu` device profile in `configs/params.yaml` uses yolo11n at
  640 px and 5 detector frames per second of video instead of 10.

## Approach

```
4K video ─► decode every 3rd frame at 1280 px (PyAV, B-frames skipped in the decoder)
         ─► YOLO11s (COCO, 1280 px) ─► ByteTrack ─► track features (anchor, speed, heading, lane, zones, dwell)
         ─► scene alignment: ECC of the video against configs/reference_edges.png ─► configs/scene.json warped onto it
         ─► traffic-light phase from the signal heads ─► per-class rules (src/rules/) ─► post-processing ─► events
```

- **Learned vs rules.** The only learned component is the COCO-pretrained YOLO11 detector, used as is (not fine-tuned;
  no training data or training scripts). Tracking (ByteTrack), the scene alignment, the signal reading and every event
  class are rules on the tracks and the hand-drawn scene (`configs/scene.json`), with every threshold in
  `configs/params.yaml`, tuned on our labels of the sample videos (`data/my_labels.json`).
- **Scene.** Lanes with their allowed directions, carriageway, crossings, stop line, islands, bus stops, the solid
  lane lines of the near approach, gantry and poles and the signal heads are drawn once in normalised coordinates;
  each video is aligned to them at run time (the camera shifts 1–3 % between sessions).
- **Road plane.** Collision rules and Part B measure distances on an approximate ground plane (horizon and scale from
  the car box width across the image, focal length from vanishing points), in car box widths (1 ≈ 2.9 m).
- **Frame sampling** depends only on the device and the video's metadata: 10 detector frames per second of video on
  CUDA / MPS (stride 3 at 29.97 fps), 5 on CPU (stride 6); frames above 4K get a proportionally larger stride.
- **Cache.** Tracks are cached in `cache/` (keyed by the video, detector and tracker settings), the signal readings
  separately (they depend on the alignment). A dev convenience: the official run starts with it empty.

## Event classes

`CLASSES` in `solution.py` keeps 11 of the 14 official ids. Start and end follow the conventions of the task.

| Class | Rule (src/rules/) | Start → end |
| --- | --- | --- |
| stopped_vehicle | stationary ≥ 10 s on the carriageway while ≥ 2 vehicles drive round it; not queued, not held by the signal, not yielding, not in a bus stop / parking zone | stops → moves again |
| congestion | a lane holds ≥ 5 vehicles with median speed below crawl speed for longer than a signal cycle (90 s) | queue stops → clears |
| wrong_way | heading > 120° from every allowed direction of the lane for ≥ 2 s and ≥ 2 box widths of travel | enters the opposing direction → back or out of frame |
| jaywalking | reliable pedestrian on the carriageway outside crossings, islands and stopping zones for ≥ 1 s, covering ≥ 3 m on the road plane at ≤ 4 m/s (faster: scooter / bicycle); not next to a car standing at the kerb or on a crossing | steps onto the road → leaves it |
| red_light | crosses the near stop line on red and drives on into the intersection | crosses the line → leaves the intersection or frame |
| stop_line | on red, stops with its front past the stop line without entering the intersection, having got past it on red / amber or at the end of green | stops → next green |
| failure_to_yield | drives across a crossing while a pedestrian walks on it (or steps onto it) in the vehicle's path; on the near crossing only while vehicles do not have green; the far crossing (its signal is not visible) is not checked | enters the crossing → leaves it |
| illegal_turn | near approach: lane = where the vehicle crosses the stop line (not judged within 0.03 of a lane boundary); from lane 1 turning left, from lanes 2+ turning into the side street | starts turning → reaches its destination |
| solid_line_crossing | the anchor changes side of a solid line within its extent, ≥ 1 s on each side; segments shorter than 1 s are dropped | wheel on the line → fully in the new lane |
| accident | road-plane footprints touch while closing in (≥ 2 box widths/s), both lose ≥ 2 box widths/s of speed, both stop together or drive out of the frame; track switches and occlusions rejected | first contact → all stopped or out |
| near_miss | time to contact ≤ 1 s with hard braking (≥ 13 box widths/s²) or a swerve (≥ 8), reliable tracks, not a queue, no contact, both drive on | onset of the evasive action → clear of each other |

Not predicted: **illegal_u_turn**, **road_obstacle** and **fire_smoke**. None of them occurs in the sample videos, so
there is nothing to tune or check a rule on, and COCO has no class for debris or smoke. A class we predict that never
occurs in the test set scores 0 and still enters the mean, so they are removed from `CLASSES`.

## Part B — accident anticipation

`RiskEstimator` (`src/risk.py`) is strictly causal: `step` only sees the frames the harness has passed so far, never
opens the video and never uses Part A output. Every 3rd frame is resized to 960 px and run through the shared YOLO11s
and its own ByteTrack; the scene is aligned on the median of the frames seen in the first 2 s. For approaching pairs
of road users it computes the time to contact on the road plane; `risk = sigmoid(3 · (1.5 s − TTC))` plus bonuses for
hard braking, a vehicle against its lane and a pedestrian on the road near a vehicle, EMA-smoothed and clipped to
[0, 1].

There are no accidents in the sample videos, so Part B is calibrated only to stay quiet in normal traffic: over the
18 minutes of samples the score reaches the alarm threshold 0.5 four times (C3896 at 139.8 s; C3902 at 64.4, 237.2
and 252.1 s), all false alarms.

## Results

`evaluate.py` on `predictions_samples.json` against our labels `data/my_labels.json`. Per class: F1 at temporal IoU
0.3 / 0.5 / 0.7 with TP / FP / FN at each threshold (the same at all three unless shown).

**Dev set, C3896 + C3905 — Score A 0.8811**

| Class | F1 @0.3 | F1 @0.5 | F1 @0.7 | TP / FP / FN |
| --- | --- | --- | --- | --- |
| failure_to_yield | 0.909 | 0.909 | 0.909 | 5 / 0 / 1 |
| illegal_turn | 1.000 | 1.000 | 1.000 | 3 / 0 / 0 |
| jaywalking | 0.800 | 0.667 | 0.667 | 6 / 3 / 0 @0.3; 5 / 4 / 1 @0.5, @0.7 |
| red_light | 1.000 | 1.000 | 1.000 | 1 / 0 / 0 |
| solid_line_crossing | 0.667 | 0.667 | 0.667 | 1 / 0 / 1 |
| stop_line | 1.000 | 1.000 | 1.000 | 2 / 0 / 0 |

**Held-out, C3897 + C3902 — Score A 0.9007**

| Class | F1 @0.3 | F1 @0.5 | F1 @0.7 | TP / FP / FN |
| --- | --- | --- | --- | --- |
| failure_to_yield | 0.889 | 0.889 | 0.889 | 4 / 1 / 0 |
| illegal_turn | 1.000 | 1.000 | 1.000 | 2 / 0 / 0 |
| jaywalking | 0.848 | 0.848 | 0.848 | 14 / 5 / 0 |
| stop_line | 0.667 | 0.667 | 0.667 | 1 / 1 / 0 |
| stopped_vehicle | 1.000 | 1.000 | 1.000 | 1 / 0 / 0 |
| wrong_way | 1.000 | 1.000 | 1.000 | 1 / 0 / 0 |

Limitations, plainly:

- The labels were built by reviewing the rules' candidates (C3905 was watched in full), so events no rule proposed are
  likely missing: recall is an upper bound.
- The "held-out" videos are not truly held out: their errors were used to fix rules (the signal reading behind
  `red_light`, `failure_to_yield`, `illegal_turn`) and to review the labels.
- The samples contain no accident or near miss, so those two rules and Part B are untested on real positives.

## Runtime

The final `predictions_samples.json`: a full run on a clean checkout with an empty cache, on a Mac M4 (MPS), stride 3
on every video, against the 3× limit.

| Video | Duration | Part A | Part B | Total | × duration | Share of 3× |
| --- | --- | --- | --- | --- | --- | --- |
| C3896 | 340.3 s | 173.6 s | 301.1 s | 474.7 s | 1.39× | 46 % |
| C3897 | 317.8 s | 213.0 s | 357.8 s | 570.8 s | 1.80× | 60 % |
| C3902 | 317.8 s | 220.9 s | 385.2 s | 606.1 s | 1.91× | 64 % |
| C3905 | 127.6 s | 97.8 s | 133.2 s | 231.0 s | 1.81× | 60 % |

Part B includes the harness's decoding of every 4K frame.

## Determinism

- Seeds are fixed in one place (`src/config.py`: `random`, `numpy`, `torch`, `PYTHONHASHSEED`;
  `cudnn.deterministic = True`, `cudnn.benchmark = False`).
- The sampling strides of Part A and Part B depend only on the device and the video's metadata, never on timing.
- Timing is only a logged emergency fuse: Part A samples more sparsely only if Part A + Part B are projected over
  2.7× the duration, Part B only if it runs slower than 2× real time over at least 10 s of video. On our runs no fuse
  blew, and two consecutive runs on the four samples gave identical `predictions.json` (events and risk curves).

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

## Live demo

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

## Licences and credits

No dataset was used for training; the detector is used with its COCO-pretrained weights.

| Component | Licence |
| --- | --- |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (`ultralytics-opencv-headless`), including its ByteTrack implementation | AGPL-3.0 |
| YOLO11 weights (`weights/yolo11s.pt`, `weights/yolo11n.pt`, Ultralytics), pretrained on [COCO](https://cocodataset.org) | AGPL-3.0 (weights); COCO annotations CC BY 4.0 |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) (the tracking algorithm) | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) / [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv-python) (`opencv-python-headless`) | Apache-2.0 |
| [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause |
| [lap](https://github.com/gatagat/lap) | BSD-2-Clause |
| [Shapely](https://github.com/shapely/shapely) | BSD-3-Clause |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT |
| Demo: [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [python-multipart](https://github.com/Kludex/python-multipart) | MIT, BSD-3-Clause, Apache-2.0 |

Because the package uses AGPL-3.0 code, this repository is released under the **AGPL-3.0** (`LICENSE`).
`run_submission.py` and `evaluate.py` are the organisers' starter kit, unchanged.

## Team

| Member | Role | Who did what |
| --- | --- | --- |
| Khojiakbar Khakimov | Team captain · Frontend, product & analytics | Analysed the task and the judging criteria and set the product direction; designed and built the website, the dashboard and the demo UI; coordinated the team and the submission |
| Zafar Ubaydullaev | Backend & infrastructure | The offline inference package and repository; video decoding and the runtime budget; deterministic runs; the demo API and deployment |
| Khamid Bustanov | AI engineer · Models & testing | Model experiments; testing and evaluation of the pipeline on our labelled sample videos |

## Repository layout

```
solution.py              interface for the harness (CLASSES, detect_events, RiskEstimator); code lives in src/
run_submission.py        organisers' harness (unchanged)
evaluate.py              organisers' metric (unchanged)
predictions_samples.json our output on the sample videos
configs/                 params.yaml (every threshold), scene.json (scene geometry), reference_edges.png
src/                     decoding, tracking, scene, signal, features, budget, Part B (risk.py), demo mode
src/rules/               one module per group of event classes
demo/                    live demo API (FastAPI)
tools/                   dev tools: EDA, clips and review page, calibration reports, site figures
tests/                   pytest suite
data/my_labels.json      our labels of the sample videos
weights/                 YOLO11 weights (s: main, n: CPU / demo)
```
