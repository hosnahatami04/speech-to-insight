from __future__ import annotations

from pathlib import Path

import pytest

from speechinsight.audio.io import load_audio
from speechinsight.config import load_settings
from speechinsight.diarize.cluster import cluster_windows, labels_to_segments
from speechinsight.diarize.embed import EcapaEmbedder, iter_windows
from speechinsight.diarize.vad import SileroVad
from speechinsight.evaluation.der import diarization_error_rate
from speechinsight.types import Segment


@pytest.mark.ml
def test_single_speaker_librispeech_file_yields_one_cluster() -> None:
    """End-to-end diarization on a LibriSpeech dev-clean file (one speaker).
    Downloads Silero VAD and ECAPA on first run."""

    manifest_path = Path("artifacts/manifests/librispeech_dev-clean.json")
    if not manifest_path.exists():
        pytest.skip("dev-clean manifest not built")

    from speechinsight.data.manifests import read_manifest

    manifest = read_manifest(manifest_path)
    entry = manifest.entries[0]
    audio = load_audio(entry.path)

    vad = SileroVad()
    regions = vad.detect(audio.samples, audio.sample_rate)
    assert regions, "VAD should find some speech in a LibriSpeech utterance"

    settings = load_settings()
    pin = settings.raw["models"]["speaker_embedding"]
    embedder = EcapaEmbedder(repo=pin["repo"], revision=pin["revision"])

    windows_tr = iter_windows(regions)
    windows = embedder.embed_windows(audio.samples, audio.sample_rate, windows_tr)
    assert windows, "at least one window must have an embedding"

    labels = cluster_windows(windows)
    assert len(set(labels)) == 1, "a one-speaker file should collapse to one cluster"

    hyp_segments = labels_to_segments(windows, labels)
    # Reference: single speaker A over the whole file.
    ref: tuple[Segment, ...] = (Segment("A", 0.0, audio.duration),)
    result = diarization_error_rate(ref, hyp_segments, collar_s=0.25)
    # DER should be dominated by VAD miss on silent tails; a loose bound is fine
    # here — the point is the pipeline runs and the clustering is sensible.
    assert result.der < 0.6
