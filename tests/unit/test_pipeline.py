from __future__ import annotations

from speechinsight.service.pipeline import DefaultPipeline, build_default_app


def test_build_default_app_returns_fastapi_app() -> None:
    from fastapi import FastAPI

    app = build_default_app()
    assert isinstance(app, FastAPI)


def test_pipeline_subclass_contract() -> None:
    """DefaultPipeline must satisfy the Pipeline abstract contract (methods
    exist and are overridden)."""

    p = DefaultPipeline()
    assert callable(p.transcribe_and_diarize)
    assert callable(p.extract)
    assert callable(p.health)
