from __future__ import annotations

from pathlib import Path

import pytest

from speechinsight.asr.whisper import WhisperConfig, WhisperTranscriber
from speechinsight.config import load_settings
from speechinsight.data.manifests import Manifest, read_manifest
from speechinsight.evaluation.noise_sweep import run_noise_sweep


@pytest.mark.ml
def test_sweep_on_real_tiny_manifest(tmp_path: Path) -> None:
    """Load the real tiny LibriSpeech sub-manifest built from dev-clean, run
    the sweep over a minimal grid with the tiny model, and verify the output
    shape. Deselected unless the ``ml`` marker is requested."""

    manifest_path = Path("artifacts/manifests/librispeech_dev-clean.json")
    if not manifest_path.exists():
        pytest.skip("dev-clean manifest not built")

    full = read_manifest(manifest_path)
    tiny = Manifest(corpus=full.corpus, split=full.split, entries=full.entries[:2])

    settings = load_settings()
    pin = settings.raw["models"]["asr_tiny"]
    transcriber = WhisperTranscriber(
        WhisperConfig(repo=pin["repo"], revision=pin["revision"], cpu_threads=2)
    )

    points = run_noise_sweep(
        transcriber, tiny, "asr_tiny", noise_kinds=["white"], snrs_db=[20.0, 10.0]
    )
    assert len(points) == 2
    for p in points:
        assert p.model == "asr_tiny"
        assert p.noise_kind == "white"
        assert 0.0 <= p.wer <= 2.0  # WER can exceed 1 if hypothesis is much longer
        assert p.ci_low <= p.wer <= p.ci_high + 1e-6
