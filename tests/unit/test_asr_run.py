from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from speechinsight.asr.run import evaluate_wer, transcribe_text
from speechinsight.cache import StageCache
from speechinsight.data.manifests import Manifest, UtteranceEntry
from speechinsight.types import Word


def _wav(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    sf.write(str(path), np.zeros(1600, dtype=np.float32), 16_000)
    return str(path)


class _OneWord:
    def __init__(self, word: str) -> None:
        self.word = word
        self.calls = 0

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        return [Word(self.word, 0.0, 1.0)]


def test_transcribe_text_uses_cache(tmp_path: Path) -> None:
    path = _wav(tmp_path, "a.wav")
    cache = StageCache(tmp_path / "cache")
    t = _OneWord("hello")
    first = transcribe_text(t, path, "m", {"beam": 5}, cache)
    second = transcribe_text(t, path, "m", {"beam": 5}, cache)
    assert first == second == "hello"
    assert t.calls == 1  # second call served from cache


def test_evaluate_wer_normalizes_both_sides(tmp_path: Path) -> None:
    path = _wav(tmp_path, "a.wav")
    manifest = Manifest(
        corpus="librispeech",
        split="dev-clean",
        entries=(UtteranceEntry("u1", path, 0.1, "Hello, World!", "84"),),
    )
    t = _OneWord("hello world")
    result = evaluate_wer(t, manifest, "m", {"beam": 5})
    assert result.wer == 0.0  # punctuation/casing normalized away
    assert result.normalizer == "speechinsight-v1"
    assert result.utterances == 1
    assert result.reference_words == 2
