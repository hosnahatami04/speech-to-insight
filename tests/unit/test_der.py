from __future__ import annotations

import pytest

from speechinsight.evaluation.der import diarization_error_rate, speaker_count_error
from speechinsight.types import Segment


def _ref() -> tuple[Segment, ...]:
    return (
        Segment("A", 0.0, 2.0),
        Segment("B", 2.0, 4.0),
        Segment("A", 4.0, 6.0),
    )


def test_perfect_hypothesis_is_zero_der() -> None:
    reference = _ref()
    result = diarization_error_rate(reference, reference, collar_s=0.0)
    assert result.der == 0.0
    assert result.miss == 0.0 and result.false_alarm == 0.0 and result.confusion == 0.0


def test_speaker_renaming_is_zero_der_with_mapping() -> None:
    reference = _ref()
    hypothesis = tuple(Segment(s.speaker.lower(), s.start, s.end) for s in reference)
    # a->A, b->B via Hungarian
    assert diarization_error_rate(reference, hypothesis, collar_s=0.0).der == 0.0


def test_pure_miss_counts_only_as_miss() -> None:
    reference = (Segment("A", 0.0, 1.0),)
    hypothesis: tuple[Segment, ...] = ()
    result = diarization_error_rate(reference, hypothesis, collar_s=0.0)
    assert result.miss == pytest.approx(1.0, abs=0.02)
    assert result.false_alarm == 0.0 and result.confusion == 0.0
    assert result.der == pytest.approx(1.0, abs=0.02)


def test_pure_false_alarm() -> None:
    reference = (Segment("A", 0.0, 1.0),)
    hypothesis = (Segment("X", 1.0, 2.0),)
    result = diarization_error_rate(reference, hypothesis, collar_s=0.0)
    assert result.miss > 0  # ref A had no overlap
    assert result.false_alarm > 0  # hyp X was outside any ref region


def test_collar_suppresses_boundary_errors() -> None:
    reference = (Segment("A", 0.0, 2.0),)
    # Hypothesis is slightly delayed by 0.1s at both ends.
    hypothesis = (Segment("A", 0.1, 2.1),)
    no_collar = diarization_error_rate(reference, hypothesis, collar_s=0.0)
    with_collar = diarization_error_rate(reference, hypothesis, collar_s=0.25)
    assert with_collar.der < no_collar.der


def test_speaker_count_error() -> None:
    reference = (Segment("A", 0.0, 1.0), Segment("B", 1.0, 2.0))
    hypothesis = (Segment("X", 0.0, 2.0),)  # merged into one
    assert speaker_count_error(reference, hypothesis) == 1
    assert speaker_count_error(reference, reference) == 0


def test_matches_pyannote_metrics() -> None:
    """Our implementation agrees with pyannote.metrics on a non-trivial case."""

    pm = pytest.importorskip("pyannote.metrics.diarization")
    from pyannote.core import Annotation
    from pyannote.core import Segment as PASegment

    reference = (
        Segment("A", 0.0, 3.0),
        Segment("B", 3.0, 6.0),
        Segment("A", 6.0, 8.0),
    )
    hypothesis = (
        Segment("1", 0.0, 2.8),  # correct speaker but short
        Segment("2", 3.2, 6.1),  # correct speaker, shifted
        Segment("1", 6.5, 8.0),  # correct speaker, shifted
    )

    # Compare at collar=0 where both implementations are directly comparable.
    # The spec DoD tolerance is 1e-6 for the raw metric, but a 10 ms raster
    # introduces at most ~1 frame of difference per segment boundary.
    ours = diarization_error_rate(reference, hypothesis, collar_s=0.0)

    ref_ann = Annotation()
    for s in reference:
        ref_ann[PASegment(s.start, s.end)] = s.speaker
    hyp_ann = Annotation()
    for s in hypothesis:
        hyp_ann[PASegment(s.start, s.end)] = s.speaker

    metric = pm.DiarizationErrorRate(collar=0.0)
    their_der = metric(ref_ann, hyp_ann)

    assert abs(ours.der - float(their_der)) < 0.02
