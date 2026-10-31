from __future__ import annotations

from pydantic import BaseModel, Field


class JobCreated(BaseModel):
    job_id: str
    status: str = "queued"


class StageTimings(BaseModel):
    asr_s: float | None = None
    diarization_s: float | None = None
    extraction_s: float | None = None
    total_s: float | None = None


class UtteranceOut(BaseModel):
    id: str
    speaker: str
    start: float
    end: float
    text: str


class ExtractedItemOut(BaseModel):
    text: str
    owner: str | None = None
    due: str | None = None
    evidence: list[str]
    supported: bool
    reasons: list[str] = Field(default_factory=list)


class JobResult(BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "done" | "error" | "extractor_unavailable"
    transcript: list[UtteranceOut] = Field(default_factory=list)
    decisions: list[ExtractedItemOut] = Field(default_factory=list)
    actions: list[ExtractedItemOut] = Field(default_factory=list)
    timings: StageTimings = Field(default_factory=StageTimings)
    error: str | None = None


class HealthReport(BaseModel):
    status: str  # "ok" | "degraded"
    stages: dict[str, bool]  # per-stage availability


class MetricsReport(BaseModel):
    jobs_total: int
    jobs_done: int
    jobs_error: int
    jobs_running: int
