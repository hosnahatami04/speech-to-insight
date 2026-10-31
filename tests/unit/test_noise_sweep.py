from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from speechinsight.cache import StageCache
from speechinsight.data.manifests import Manifest, UtteranceEntry
from speechinsight.evaluation.noise_sweep import (
    SweepPoint,
    run_noise_sweep,
    wer_delta_db,
    write_sweep,
)
from speechinsight.types import Word


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        return [Word(self._text, 0.0, 1.0)]


def _wav(tmp_path: Path, uid: str) -> str:
    path = tmp_path / f"{uid}.wav"
    t = np.arange(16_000) / 16_000
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), tone, 16_000)
    return str(path)


def _manifest(tmp_path: Path) -> Manifest:
    entries = tuple(
        UtteranceEntry(f"u{i}", _wav(tmp_path, f"u{i}"), 1.0, "hello", f"spk{i}") for i in range(2)
    )
    return Manifest("librispeech", "dev-clean", entries)


def test_sweep_produces_one_point_per_combination(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    transcriber = _FakeTranscriber("hello")
    points = run_noise_sweep(
        transcriber,
        manifest,
        model_id="base.en",
        noise_kinds=["white", "pink"],
        snrs_db=[20.0, 10.0],
    )
    assert len(points) == 4
    kinds = {(p.noise_kind, p.snr_db) for p in points}
    assert kinds == {("white", 20.0), ("white", 10.0), ("pink", 20.0), ("pink", 10.0)}
    for p in points:
        assert p.model == "base.en"
        assert p.normalizer == "speechinsight-v1"


def test_cache_hits_prevent_retranscribe(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    transcriber = _FakeTranscriber("hello")
    cache = StageCache(tmp_path / "cache")
    run_noise_sweep(transcriber, manifest, "base.en", ["white"], [10.0], cache=cache)
    first = transcriber.calls
    run_noise_sweep(transcriber, manifest, "base.en", ["white"], [10.0], cache=cache)
    assert transcriber.calls == first  # all served from cache the second time


def test_wer_delta_against_clean(tmp_path: Path) -> None:
    points = [
        SweepPoint("base.en", "white", 20.0, 0.1, 0.0, 0.0, 10, "n"),
        SweepPoint("base.en", "white", 0.0, 0.5, 0.0, 0.0, 10, "n"),
    ]
    assert wer_delta_db(points, "base.en", "white", 0.0) == 0.4
    assert wer_delta_db(points, "base.en", "white", 99.0) is None
    assert wer_delta_db(points, "missing", "white", 0.0) is None


def test_write_sweep_produces_json(tmp_path: Path) -> None:
    import json

    point = SweepPoint("m", "white", 10.0, 0.3, 0.2, 0.4, 5, "n")
    out = tmp_path / "sweep.json"
    write_sweep([point], out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["wer"] == 0.3
    assert data[0]["noise_kind"] == "white"
