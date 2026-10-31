from __future__ import annotations

import jiwer
import pytest
from hypothesis import given
from hypothesis import strategies as st

from speechinsight.evaluation.wer import align_counts, bootstrap_ci, corpus_wer

_word = st.text(alphabet="abc", min_size=1, max_size=3)
_sentence = st.lists(_word, min_size=1, max_size=8)


def test_substitution_deletion_insertion_counts() -> None:
    # reference: a b c d ; hypothesis: a x c  (sub b->x, delete d)
    counts = align_counts(["a", "b", "c", "d"], ["a", "x", "c"])
    assert counts.substitutions == 1
    assert counts.deletions == 1
    assert counts.insertions == 0
    assert counts.reference_words == 4


def test_pure_insertion() -> None:
    counts = align_counts(["a"], ["a", "b", "c"])
    assert counts.insertions == 2
    assert counts.substitutions == 0
    assert counts.deletions == 0


def test_identical_text_is_zero_wer() -> None:
    ref = [["a", "b", "c"]]
    assert corpus_wer(ref, [["a", "b", "c"]]) == 0.0


def test_empty_reference_raises() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        corpus_wer([[]], [["a"]])


@given(ref=_sentence, hyp=_sentence)
def test_matches_jiwer(ref: list[str], hyp: list[str]) -> None:
    ours = corpus_wer([ref], [hyp])
    reference_wer = jiwer.wer(" ".join(ref), " ".join(hyp))
    assert ours == pytest.approx(reference_wer, abs=1e-9)


def test_bootstrap_ci_is_deterministic_and_brackets_point() -> None:
    refs = [["a", "b", "c"], ["d", "e"], ["f", "g", "h"]]
    hyps = [["a", "x", "c"], ["d", "e"], ["f", "g", "y"]]
    low1, high1 = bootstrap_ci(refs, hyps, n_resamples=200, seed=0)
    low2, high2 = bootstrap_ci(refs, hyps, n_resamples=200, seed=0)
    assert (low1, high1) == (low2, high2)
    assert low1 <= corpus_wer(refs, hyps) <= high1
