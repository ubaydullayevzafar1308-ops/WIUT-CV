"""HTTP API of the live demo on the website.

    POST /api/analyze              multipart, field ``video`` -> {"job_id"}
    GET  /api/jobs/{job_id}        -> {"status": queued|running|done|error, "progress": 0..1, "error"?}
    GET  /api/jobs/{job_id}/result -> {"duration", "fps", "events", "risk", "scene_matched"}
    GET  /api/health               -> {"ok": true}

One video is analysed at a time (src/demo.py, light mode for a CPU server); later
uploads wait in a queue. Uploads are checked (``demo`` in params.yaml: extension,
size, duration) and deleted as soon as their analysis ends. Refused requests get
an HTTP error with ``{"error": "..."}``.

Run: ``python -m demo.api --host 0.0.0.0 --port 8000``.
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_params
from src.video import probe

log = logging.getLogger(__name__)

MB = 1024 * 1024
MULTIPART_OVERHEAD = MB   # Content-Length allowance for the multipart envelope around the file

Analyzer = Callable[[str, Callable[[float], None]], dict[str, Any]]


class Refused(Exception):
    """An upload the API does not accept; ``status`` is the HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    """One uploaded video and the state of its analysis."""

    id: str
    path: Path
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "progress": round(self.progress, 4)}
        if self.error is not None:
            out["error"] = self.error
        return out


class JobQueue:
    """Jobs by id and a single worker thread that analyses them one at a time, in upload order."""

    def __init__(self, analyzer: Analyzer, max_queued: int, keep_jobs: int):
        self.analyzer = analyzer
        self.max_queued = max_queued
        self.keep_jobs = keep_jobs
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pending: queue.Queue[Job | None] = queue.Queue()
        self.worker = threading.Thread(target=self._work, name="demo-worker", daemon=True)

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.pending.put(None)
        self.worker.join()

    def is_full(self) -> bool:
        with self.lock:
            return sum(job.status == "queued" for job in self.jobs.values()) >= self.max_queued

    def submit(self, path: Path) -> Job:
        job = Job(id=uuid.uuid4().hex, path=path)
        with self.lock:
            self.jobs[job.id] = job
            self._forget_old()
        self.pending.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def _forget_old(self) -> None:
        finished = sorted((j for j in self.jobs.values() if j.status in ("done", "error")), key=lambda j: j.created)
        for job in finished[:max(0, len(finished) - self.keep_jobs)]:
            del self.jobs[job.id]

    def _work(self) -> None:
        while (job := self.pending.get()) is not None:
            self._run(job)

    def _run(self, job: Job) -> None:
        def progress(share: float) -> None:
            job.progress = max(job.progress, min(1.0, share))

        job.status = "running"
        try:
            job.result = self.analyzer(str(job.path), progress)
            job.progress, job.status = 1.0, "done"
        except Exception as err:  # noqa: BLE001 - any failure is reported to the client as the job's error
            log.exception("job %s failed", job.id)
            job.error, job.status = f"analysis failed: {err}", "error"
        finally:
            job.path.unlink(missing_ok=True)
            with self.lock:
                self._forget_old()


def save_upload(upload: UploadFile, directory: Path, max_bytes: int, chunk: int) -> Path:
    """Copy an upload to a new file in ``directory``, refusing it once it exceeds ``max_bytes``."""
    fd, name = tempfile.mkstemp(suffix=".mp4", dir=directory)
    path = Path(name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while data := upload.file.read(chunk):
                size += len(data)
                if size > max_bytes:
                    raise Refused(413, f"file is larger than {max_bytes // MB} MB")
                out.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def check_video(path: Path, max_duration: float) -> None:
    """Refuse files that are not readable videos or last longer than ``max_duration`` seconds."""
    try:
        info = probe(str(path))
    except Exception as err:  # noqa: BLE001 - PyAV raises many error types for broken files
        raise Refused(400, "file is not a readable MP4 video") from err
    if info.n_frames <= 0 or info.width <= 0:
        raise Refused(400, "file is not a readable MP4 video")
    if info.duration > max_duration:
        raise Refused(400, f"video is {info.duration:.0f} s long; at most {max_duration:.0f} s are allowed")


def refuse(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def default_analyzer(video_path: str, progress: Callable[[float], None]) -> dict[str, Any]:
    from src.demo import analyze   # imports torch and Ultralytics: only when the first video arrives

    return analyze(video_path, progress)


def create_app(analyzer: Analyzer = default_analyzer, params: dict[str, Any] | None = None) -> FastAPI:
    """The demo API; ``analyzer`` does the work (tests pass a fake), ``params`` is the ``demo`` section."""
    dp = params or load_params()["demo"]
    jobs = JobQueue(analyzer, dp["max_queued"], dp["keep_jobs"])
    upload_dir = Path(dp["upload_dir"] or Path(tempfile.gettempdir()) / "wiut-demo")
    max_bytes = int(dp["max_upload_mb"] * MB)
    extensions = tuple(ext.lower() for ext in dp["extensions"])

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        upload_dir.mkdir(parents=True, exist_ok=True)
        jobs.start()
        yield
        jobs.stop()

    app = FastAPI(title="WIUT traffic events demo", lifespan=lifespan)
    app.state.jobs = jobs

    @app.middleware("http")
    async def upload_size_limit(request: Request, call_next):
        """Refuse oversized uploads from their Content-Length, before the body is received."""
        if request.method != "POST" or request.url.path != "/api/analyze":
            return await call_next(request)
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return refuse(411, "the upload needs a Content-Length header")
        if int(length) > max_bytes + MULTIPART_OVERHEAD:
            return refuse(413, f"file is larger than {dp['max_upload_mb']} MB")
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=dp["cors_origins"], allow_methods=["GET", "POST"],
                       allow_headers=["*"])   # added last: outermost, so refusals carry CORS headers too

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/analyze")
    def analyze(video: UploadFile | None = File(None)):
        if video is None or not video.filename:
            return refuse(400, "no file in the form field 'video'")
        if not video.filename.lower().endswith(extensions):
            return refuse(415, f"only {', '.join(extensions)} files are accepted")
        if jobs.is_full():
            return refuse(503, "the demo server is busy; try again in a few minutes")
        try:
            path = save_upload(video, upload_dir, max_bytes, int(dp["upload_chunk_mb"] * MB))
            try:
                check_video(path, dp["max_duration_sec"])
            except Refused:
                path.unlink(missing_ok=True)
                raise
        except Refused as err:
            return refuse(err.status, str(err))
        job = jobs.submit(path)
        log.info("job %s queued: %s", job.id, video.filename)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        return job.public() if job else refuse(404, "unknown job")

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return refuse(404, "unknown job")
        if job.status == "error":
            return refuse(409, job.error or "analysis failed")
        if job.status != "done":
            return refuse(409, f"job is not finished yet (status: {job.status})")
        return job.result

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the demo API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
