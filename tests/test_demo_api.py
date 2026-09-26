"""Demo API (demo/api.py) on small synthetic clips: endpoints, upload limits, job queue, response format.

The clips are drawn here (a grey box sliding over a dark background), never taken
from the sample videos. One test runs the real light-mode analysis (src/demo.py)
on the CPU; the others use a fake analyzer.
"""
from __future__ import annotations

import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from demo.api import create_app  # noqa: E402
from solution import CLASSES  # noqa: E402
from src.config import load_params  # noqa: E402
from src.demo import thin_curve, visible_events  # noqa: E402

FPS = Fraction(30000, 1001)
TIMEOUT_SEC = 120.0
RESULT_KEYS = {"duration", "fps", "events", "risk", "scene_matched"}


def write_clip(path: Path, n_frames: int, rate: Fraction = FPS, size: tuple[int, int] = (320, 180)) -> Path:
    """H.264 MP4 of a grey box moving to the right."""
    width, height = size
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(n_frames):
            img = np.full((height, width, 3), 60, dtype=np.uint8)
            x = (5 + 3 * i) % (width - width // 6)
            img[height // 2:height // 2 + height // 6, x:x + width // 6] = 200
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """3 s at 29.97 fps."""
    return write_clip(tmp_path_factory.mktemp("clips") / "clip.mp4", 90)


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory) -> Path:
    """121 s at 1 fps, tiny frames: longer than the 120 s limit."""
    return write_clip(tmp_path_factory.mktemp("clips") / "long.mp4", 121, rate=Fraction(1), size=(64, 48))


def demo_params(upload_dir: Path, **changes) -> dict:
    return {**load_params()["demo"], "upload_dir": str(upload_dir), **changes}


def upload(client: TestClient, path: Path, name: str | None = None):
    with open(path, "rb") as f:
        return client.post("/api/analyze", files={"video": (name or path.name, f, "video/mp4")})


def wait(client: TestClient, job_id: str, until: tuple[str, ...] = ("done", "error")) -> dict:
    deadline = time.monotonic() + TIMEOUT_SEC
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in until:
            return status
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not reach {until}")


class FakeAnalyzer:
    """Reports half progress, then waits for ``release`` (or fails when ``fail`` is set)."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.release = threading.Event()
        self.running = 0
        self.max_running = 0
        self.lock = threading.Lock()

    def __call__(self, path: str, progress) -> dict:
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            assert Path(path).exists()
            progress(0.5)
            self.release.wait(TIMEOUT_SEC)
            if self.fail:
                raise RuntimeError("decoder exploded")
            return {"duration": 3.0, "fps": 29.97, "events": [], "risk": [[0.0, 0.0]], "scene_matched": False}
        finally:
            with self.lock:
                self.running -= 1


def test_health_and_cors(tmp_path):
    with TestClient(create_app(FakeAnalyzer(), demo_params(tmp_path))) as client:
        response = client.get("/api/health", headers={"Origin": "https://example.org"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["access-control-allow-origin"] == "*"


def test_real_analysis_of_a_synthetic_clip(clip, tmp_path, monkeypatch):
    monkeypatch.setenv("WIUT_DEVICE", "cpu")
    with TestClient(create_app(params=demo_params(tmp_path))) as client:
        response = upload(client, clip)
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        status = wait(client, job_id)
        assert status == {"status": "done", "progress": 1.0}, status
        result = client.get(f"/api/jobs/{job_id}/result").json()

    assert set(result) == RESULT_KEYS
    assert result["duration"] == pytest.approx(90 / float(FPS), abs=1e-3)
    assert result["fps"] == pytest.approx(float(FPS))
    assert isinstance(result["scene_matched"], bool)
    assert not result["scene_matched"]          # a synthetic clip is not our camera
    for start, end, label in result["events"]:
        assert 0 <= start < end <= result["duration"] and label in CLASSES
        assert label in ("accident", "near_miss")
    times = [t for t, _ in result["risk"]]
    assert times and times[0] == 0.0 and times[-1] < result["duration"]
    assert all(b - a >= 0.1 - 1e-6 for a, b in zip(times, times[1:]))
    assert all(0.0 <= score <= 1.0 for _, score in result["risk"])
    assert not list(tmp_path.iterdir())         # the upload is deleted after the analysis


def test_one_job_at_a_time_in_upload_order(clip, tmp_path):
    analyzer = FakeAnalyzer()
    with TestClient(create_app(analyzer, demo_params(tmp_path))) as client:
        first, second = (upload(client, clip).json()["job_id"] for _ in range(2))
        assert wait(client, first, until=("running",))["progress"] == pytest.approx(0.5)
        assert client.get(f"/api/jobs/{second}").json() == {"status": "queued", "progress": 0.0}
        response = client.get(f"/api/jobs/{first}/result")
        assert response.status_code == 409 and "not finished" in response.json()["error"]
        analyzer.release.set()
        assert wait(client, first)["status"] == "done"
        assert wait(client, second)["status"] == "done"
        assert set(client.get(f"/api/jobs/{second}/result").json()) == RESULT_KEYS
    assert analyzer.max_running == 1
    assert not list(tmp_path.iterdir())


def test_failed_analysis_is_reported(clip, tmp_path):
    analyzer = FakeAnalyzer(fail=True)
    analyzer.release.set()
    with TestClient(create_app(analyzer, demo_params(tmp_path))) as client:
        job_id = upload(client, clip).json()["job_id"]
        status = wait(client, job_id)
        result = client.get(f"/api/jobs/{job_id}/result")
    assert status["status"] == "error" and "decoder exploded" in status["error"]
    assert result.status_code == 409 and "decoder exploded" in result.json()["error"]
    assert not list(tmp_path.iterdir())


def test_full_queue_is_refused(clip, tmp_path):
    analyzer = FakeAnalyzer()
    with TestClient(create_app(analyzer, demo_params(tmp_path, max_queued=1))) as client:
        running = upload(client, clip).json()["job_id"]
        wait(client, running, until=("running",))
        assert upload(client, clip).status_code == 200            # waits in the queue
        response = upload(client, clip)
        analyzer.release.set()
    assert response.status_code == 503 and "busy" in response.json()["error"]


def test_unknown_job(tmp_path):
    with TestClient(create_app(FakeAnalyzer(), demo_params(tmp_path))) as client:
        assert client.get("/api/jobs/nope").status_code == 404
        response = client.get("/api/jobs/nope/result")
    assert response.status_code == 404 and response.json() == {"error": "unknown job"}


@pytest.mark.parametrize("name, status, message", [
    ("clip.avi", 415, ".mp4"),
    ("clip", 415, ".mp4"),
    ("CLIP.MP4", 200, None),                     # the extension check ignores case
])
def test_only_mp4_files(clip, tmp_path, name, status, message):
    analyzer = FakeAnalyzer()
    analyzer.release.set()
    with TestClient(create_app(analyzer, demo_params(tmp_path))) as client:
        response = upload(client, clip, name)
        if status == 200:
            wait(client, response.json()["job_id"])
    assert response.status_code == status
    if message:
        assert message in response.json()["error"]
    assert not list(tmp_path.iterdir())


def test_too_long_video(long_clip, tmp_path):
    with TestClient(create_app(FakeAnalyzer(), demo_params(tmp_path))) as client:
        response = upload(client, long_clip)
    assert response.status_code == 400
    assert response.json() == {"error": "video is 121 s long; at most 120 s are allowed"}
    assert not list(tmp_path.iterdir())


def test_too_large_file_by_content_length(tmp_path):
    big = tmp_path / "big.mp4"
    big.write_bytes(np.random.default_rng(0).integers(0, 256, 2 * 1024 * 1024, dtype=np.uint8).tobytes())
    uploads = tmp_path / "uploads"
    with TestClient(create_app(FakeAnalyzer(), demo_params(uploads, max_upload_mb=0.5))) as client:
        response = upload(client, big)
    assert response.status_code == 413 and "0.5 MB" in response.json()["error"]
    assert not list(uploads.iterdir())


def test_too_large_file_while_copying(tmp_path):
    data = tmp_path / "data.mp4"
    data.write_bytes(bytes(50_000))
    uploads = tmp_path / "uploads"
    with TestClient(create_app(FakeAnalyzer(), demo_params(uploads, max_upload_mb=0.01))) as client:
        response = upload(client, data)
    assert response.status_code == 413 and "larger than" in response.json()["error"]
    assert not list(uploads.iterdir())


def test_unreadable_and_missing_files(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all" * 100)
    uploads = tmp_path / "uploads"
    with TestClient(create_app(FakeAnalyzer(), demo_params(uploads))) as client:
        broken = upload(client, junk)
        missing = client.post("/api/analyze", files={"other": ("x.mp4", b"x", "video/mp4")})
    assert broken.status_code == 400 and broken.json() == {"error": "file is not a readable MP4 video"}
    assert missing.status_code == 400 and "'video'" in missing.json()["error"]
    assert not list(uploads.iterdir())


def test_risk_curve_is_thinned_to_the_window_maximum():
    points = [(0.0, 0.1), (0.033, 0.4), (0.067, 0.2), (0.1, 0.3), (0.35, 0.9), (0.36, 0.5)]
    assert thin_curve(points, 0.1) == [[0.0, 0.4], [0.1, 0.3], [0.3, 0.9]]


def test_unmatched_scene_keeps_only_events_not_tied_to_the_intersection():
    events = [[1.0, 2.0, "accident"], [3.0, 4.0, "red_light"], [5.0, 6.0, "near_miss"]]
    assert visible_events(events, True, ["accident", "near_miss"]) == events
    assert visible_events(events, False, ["accident", "near_miss"]) == [events[0], events[2]]
