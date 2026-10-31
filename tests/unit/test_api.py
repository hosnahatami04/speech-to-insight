from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from speechinsight.extract.grounding import GroundedItem, GroundingResult
from speechinsight.extract.schema import ExtractedItem
from speechinsight.service.api import Pipeline, create_app
from speechinsight.types import Transcript, Utterance


@dataclass
class _ExtractReport:
    decisions: GroundingResult
    actions: GroundingResult


class _FakePipeline(Pipeline):
    def __init__(self, extractor_ok: bool = True) -> None:
        self.extractor_ok = extractor_ok

    def transcribe_and_diarize(self, audio_path: str) -> tuple[Any, dict[str, float]]:
        transcript = Transcript(
            utterances=(
                Utterance("u0001", "A", 0.0, 1.0, "hello"),
                Utterance("u0002", "B", 1.0, 2.0, "pick blue"),
            )
        )
        return transcript, {"asr_s": 0.1, "diarization_s": 0.05}

    def extract(self, transcript: Any) -> Any:
        if not self.extractor_ok:
            raise Pipeline.ExtractorUnavailable("Ollama is down")
        item = ExtractedItem(text="Pick blue.", evidence=("u0002",))
        kept = [GroundedItem(item=item, kind="decision", supported=True, reasons=())]
        return _ExtractReport(
            decisions=GroundingResult(
                kept=kept, dropped_no_evidence=0, unsupported_entities=0, unsupported_nli=0
            ),
            actions=GroundingResult(
                kept=[], dropped_no_evidence=0, unsupported_entities=0, unsupported_nli=0
            ),
        )

    def health(self) -> dict[str, bool]:
        return {"asr": True, "diarization": True, "extractor": self.extractor_ok}


def _wav_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(1600, dtype=np.float32), 16_000)
    return path.read_bytes()


def _wait_for(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in {"done", "error", "extractor_unavailable"}:
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_valid_upload_runs_pipeline_end_to_end(tmp_path: Path) -> None:
    app = create_app(_FakePipeline())
    client = TestClient(app)
    resp = client.post("/process", files={"file": ("a.wav", _wav_bytes(tmp_path))})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    body = _wait_for(client, job_id)
    assert body["status"] == "done"
    assert len(body["transcript"]) == 2
    assert len(body["decisions"]) == 1
    assert body["timings"]["asr_s"] is not None


def test_unsupported_file_type_is_rejected() -> None:
    app = create_app(_FakePipeline())
    client = TestClient(app)
    resp = client.post("/process", files={"file": ("notes.txt", b"hello")})
    assert resp.status_code == 422
    assert "unsupported" in resp.json()["detail"]


def test_empty_file_is_rejected() -> None:
    app = create_app(_FakePipeline())
    client = TestClient(app)
    resp = client.post("/process", files={"file": ("a.wav", b"")})
    assert resp.status_code == 422


def test_oversized_file_is_rejected() -> None:
    from speechinsight.service.api import MAX_UPLOAD_BYTES

    app = create_app(_FakePipeline())
    client = TestClient(app)
    big = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    resp = client.post("/process", files={"file": ("a.wav", big)})
    assert resp.status_code == 422


def test_unknown_job_is_404() -> None:
    app = create_app(_FakePipeline())
    client = TestClient(app)
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_extractor_unavailable_does_not_fail_transcription(tmp_path: Path) -> None:
    app = create_app(_FakePipeline(extractor_ok=False))
    client = TestClient(app)
    resp = client.post("/process", files={"file": ("a.wav", _wav_bytes(tmp_path))})
    job_id = resp.json()["job_id"]
    body = _wait_for(client, job_id)
    assert body["status"] == "extractor_unavailable"
    assert len(body["transcript"]) == 2  # ASR+diarization still succeeded
    assert body["error"] is not None


def test_health_reports_per_stage_status() -> None:
    app = create_app(_FakePipeline(extractor_ok=False))
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["stages"]["asr"] is True
    assert body["stages"]["extractor"] is False


def test_metrics_counts_jobs(tmp_path: Path) -> None:
    app = create_app(_FakePipeline())
    client = TestClient(app)
    for _ in range(2):
        client.post("/process", files={"file": ("a.wav", _wav_bytes(tmp_path))})
    # Give workers a moment to finish
    time.sleep(0.3)
    body = client.get("/metrics").json()
    assert body["jobs_total"] == 2


def test_job_response_contains_no_embedding_key(tmp_path: Path) -> None:
    """Spec §4.4: /jobs must never leak speaker embeddings."""

    app = create_app(_FakePipeline())
    client = TestClient(app)
    resp = client.post("/process", files={"file": ("a.wav", _wav_bytes(tmp_path))})
    body = _wait_for(client, resp.json()["job_id"])
    import json

    assert "embedding" not in json.dumps(body).lower()
