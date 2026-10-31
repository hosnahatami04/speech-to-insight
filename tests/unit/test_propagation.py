from __future__ import annotations

import numpy as np
import pytest

from speechinsight.evaluation.propagation import (
    CorruptionConfig,
    corrupt_transcript,
    fit_degradation,
    swap_speaker_labels,
)
from speechinsight.types import Transcript, Utterance


def _long_transcript(n: int = 60, words_per_utt: int = 10) -> Transcript:
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(50)]
    utts = []
    for i in range(n):
        text = " ".join(rng.choice(vocab, size=words_per_utt).tolist())
        utts.append(
            Utterance(
                id=f"u{i + 1:04d}",
                speaker="A" if i % 2 == 0 else "B",
                start=i * 1.0,
                end=i * 1.0 + 0.5,
                text=text,
            )
        )
    return Transcript(utterances=tuple(utts))


def test_zero_corruption_preserves_text() -> None:
    t = _long_transcript()
    corrupted, measured = corrupt_transcript(t, CorruptionConfig())
    assert measured == 0.0
    for a, b in zip(t.utterances, corrupted.utterances, strict=True):
        assert a.text == b.text


def test_measured_wer_tracks_target_rate() -> None:
    t = _long_transcript()
    _, measured = corrupt_transcript(t, CorruptionConfig(substitution_rate=0.2, seed=0))
    # The measured WER should be close to the substitution rate for 20% subs
    # on a long transcript: within a loose tolerance since dp/measurement varies.
    assert 0.10 < measured < 0.30


def test_higher_target_yields_higher_measured_wer() -> None:
    t = _long_transcript()
    _, low = corrupt_transcript(t, CorruptionConfig(substitution_rate=0.05, seed=1))
    _, high = corrupt_transcript(t, CorruptionConfig(substitution_rate=0.30, seed=1))
    assert high > low


def test_corruption_is_deterministic_for_seed() -> None:
    t = _long_transcript()
    a, _ = corrupt_transcript(t, CorruptionConfig(substitution_rate=0.1, seed=42))
    b, _ = corrupt_transcript(t, CorruptionConfig(substitution_rate=0.1, seed=42))
    for ua, ub in zip(a.utterances, b.utterances, strict=True):
        assert ua.text == ub.text


def test_utterance_ids_are_preserved() -> None:
    t = _long_transcript()
    corrupted, _ = corrupt_transcript(t, CorruptionConfig(substitution_rate=0.5, seed=0))
    assert [u.id for u in corrupted.utterances] == [u.id for u in t.utterances]


def test_swap_zero_rate_is_noop() -> None:
    t = _long_transcript()
    out = swap_speaker_labels(t, swap_rate=0.0)
    for a, b in zip(t.utterances, out.utterances, strict=True):
        assert a.speaker == b.speaker


def test_swap_only_assigns_present_speakers() -> None:
    t = _long_transcript()
    speakers = set(t.speakers)
    out = swap_speaker_labels(t, swap_rate=0.5, seed=3)
    for u in out.utterances:
        assert u.speaker in speakers


def test_swap_rate_is_approximate() -> None:
    t = _long_transcript(n=200)
    out = swap_speaker_labels(t, swap_rate=0.4, seed=0)
    changed = sum(
        1 for a, b in zip(t.utterances, out.utterances, strict=True) if a.speaker != b.speaker
    )
    rate = changed / len(t.utterances)
    assert 0.25 < rate < 0.55


def test_fit_degradation_recovers_known_slope() -> None:
    xs = [0.0, 0.05, 0.10, 0.20, 0.30]
    ys = [0.80 - 0.5 * x for x in xs]  # slope -0.5
    fit = fit_degradation(xs, ys)
    assert abs(fit.slope - (-0.5)) < 1e-6
    assert abs(fit.intercept - 0.80) < 1e-6


def test_fit_degradation_requires_two_points() -> None:
    with pytest.raises(ValueError):
        fit_degradation([0.0], [1.0])
    with pytest.raises(ValueError):
        fit_degradation([0.0, 0.1], [1.0])  # mismatched lengths
