from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile

from speechinsight.service.schemas import (
    ExtractedItemOut,
    HealthReport,
    JobCreated,
    JobResult,
    MetricsReport,
    StageTimings,
    UtteranceOut,
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_DURATION_S = 60 * 60 * 2  # 2 hours
ALLOWED_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


class JobStore:
    """Thread-safe in-process job registry. Deliberately simple — the spec calls
    for a drop-in queue in Phase 7; a shared external queue is out of scope."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobResult] = {}
        self._lock = Lock()

    def create(self) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobResult(job_id=job_id, status="queued")
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(update=fields)

    def get(self, job_id: str) -> JobResult | None:
        with self._lock:
            return self._jobs.get(job_id)

    def counts(self) -> MetricsReport:
        with self._lock:
            jobs = list(self._jobs.values())
        return MetricsReport(
            jobs_total=len(jobs),
            jobs_done=sum(1 for j in jobs if j.status == "done"),
            jobs_error=sum(1 for j in jobs if j.status == "error"),
            jobs_running=sum(1 for j in jobs if j.status in {"queued", "running"}),
        )


def _utterances_out(utterances: Any) -> list[UtteranceOut]:
    return [
        UtteranceOut(id=u.id, speaker=u.speaker, start=u.start, end=u.end, text=u.text)
        for u in utterances
    ]


def _items_out(items: Any) -> list[ExtractedItemOut]:
    return [
        ExtractedItemOut(
            text=i.item.text,
            owner=i.item.owner,
            due=i.item.due,
            evidence=list(i.item.evidence),
            supported=i.supported,
            reasons=list(i.reasons),
        )
        for i in items
    ]


def _run_pipeline(
    store: JobStore,
    job_id: str,
    audio_path: str,
    pipeline: Pipeline,
) -> None:
    """Background worker. Updates the job with per-stage timings and the final
    transcript/extraction. If the extractor is unavailable, the ASR+diarization
    result still lands in the response with status 'extractor_unavailable'."""

    store.update(job_id, status="running")
    timings = StageTimings()
    try:
        t0 = time.perf_counter()
        transcript, timings_dict = pipeline.transcribe_and_diarize(audio_path)
        timings = StageTimings(
            asr_s=timings_dict.get("asr_s"),
            diarization_s=timings_dict.get("diarization_s"),
        )
        store.update(job_id, transcript=_utterances_out(transcript.utterances), timings=timings)

        try:
            report = pipeline.extract(transcript)
            timings = StageTimings(
                asr_s=timings.asr_s,
                diarization_s=timings.diarization_s,
                extraction_s=time.perf_counter()
                - t0
                - (timings.asr_s or 0)
                - (timings.diarization_s or 0),
                total_s=time.perf_counter() - t0,
            )
            store.update(
                job_id,
                status="done",
                decisions=_items_out(report.decisions.kept),
                actions=_items_out(report.actions.kept),
                timings=timings,
            )
        except Pipeline.ExtractorUnavailable as exc:
            store.update(
                job_id,
                status="extractor_unavailable",
                error=str(exc),
                timings=StageTimings(
                    asr_s=timings.asr_s,
                    diarization_s=timings.diarization_s,
                    total_s=time.perf_counter() - t0,
                ),
            )
    except Exception as exc:
        store.update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


class Pipeline:
    """Pipeline contract the API depends on. The real implementation lives
    outside the API module so it can be swapped (and so tests can inject a
    fake)."""

    class ExtractorUnavailable(RuntimeError):
        pass

    def transcribe_and_diarize(self, audio_path: str) -> tuple[Any, dict[str, float]]:
        raise NotImplementedError

    def extract(self, transcript: Any) -> Any:
        raise NotImplementedError

    def health(self) -> dict[str, bool]:
        raise NotImplementedError


def create_app(pipeline: Pipeline, store: JobStore | None = None) -> FastAPI:
    """Build the FastAPI app. ``pipeline`` is injected so tests can run the API
    end-to-end without loading real models."""

    app = FastAPI(title="speech-to-insight")
    jobs = store or JobStore()

    @app.post("/process", response_model=JobCreated, status_code=202)
    async def process(file: UploadFile) -> JobCreated:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=422, detail=f"unsupported file type {suffix!r}")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=422, detail="uploaded file exceeds size limit")

        # NamedTemporaryFile is intentionally kept open past this scope via
        # delete=False so the background worker can read it; close explicitly.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_name = tmp.name

        job_id = jobs.create()
        Thread(target=_run_pipeline, args=(jobs, job_id, tmp_name, pipeline), daemon=True).start()
        return JobCreated(job_id=job_id)

    @app.get("/jobs/{job_id}", response_model=JobResult)
    async def get_job(job_id: str) -> JobResult:
        result = jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="job not found")
        return result

    @app.get("/health", response_model=HealthReport)
    async def health() -> HealthReport:
        stages = pipeline.health()
        status = "ok" if all(stages.values()) else "degraded"
        return HealthReport(status=status, stages=stages)

    @app.get("/metrics", response_model=MetricsReport)
    async def metrics() -> MetricsReport:
        return jobs.counts()

    return app
