from __future__ import annotations

import numpy as np
import pytest

from speechinsight.asr.whisper import WhisperConfig, WhisperTranscriber
from speechinsight.config import load_settings


@pytest.mark.ml
def test_real_model_transcribes_short_clip() -> None:
    """Loads the real pinned tiny model and runs it on a short tone. This checks
    the integration path end to end; it is deselected unless the ``ml`` marker is
    requested."""

    settings = load_settings()
    pin = settings.raw["models"]["asr_tiny"]
    transcriber = WhisperTranscriber(
        WhisperConfig(repo=pin["repo"], revision=pin["revision"], cpu_threads=2)
    )
    samples = np.zeros(16_000, dtype=np.float32)
    words = transcriber.transcribe(samples, 16_000)
    # Silence may yield no words; the contract is that it returns a list of Word.
    assert isinstance(words, list)
    for w in words:
        assert w.end >= w.start
