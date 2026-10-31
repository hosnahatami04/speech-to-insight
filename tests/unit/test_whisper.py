from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from speechinsight.asr.factory import whisper_config
from speechinsight.asr.whisper import WhisperConfig, WhisperTranscriber


@dataclass
class _FakeWord:
    word: str
    start: float
    end: float


@dataclass
class _FakeSegment:
    text: str
    start: float
    end: float
    words: list[_FakeWord]


class _FakeModel:
    def transcribe(self, samples, **kwargs):
        segments = [
            _FakeSegment(
                "hi there",
                0.0,
                1.0,
                [_FakeWord(" hi", 0.0, 0.5), _FakeWord(" there", 0.6, 1.0)],
            )
        ]
        return segments, None


def _transcriber() -> WhisperTranscriber:
    cfg = WhisperConfig(repo="r", revision="rev")
    t = WhisperTranscriber(cfg)
    t._model = _FakeModel()  # inject fake so no model is downloaded
    return t


def test_converts_library_output_to_words() -> None:
    t = _transcriber()
    words = t.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)
    assert [w.text for w in words] == ["hi", "there"]
    # timestamps are monotonic
    assert words[0].start <= words[0].end <= words[1].start <= words[1].end


def test_rejects_non_16k_audio() -> None:
    t = _transcriber()
    with pytest.raises(ValueError, match="16 kHz"):
        t.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)


def test_factory_builds_pinned_config() -> None:
    raw = {
        "models": {"asr_base": {"repo": "org/base", "revision": "abc123"}},
        "asr": {"beam_size": 5, "compute_type": "int8"},
    }
    cfg = whisper_config(raw, "asr_base")
    assert cfg.repo == "org/base"
    assert cfg.revision == "abc123"
    assert cfg.beam_size == 5
