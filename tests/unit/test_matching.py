from __future__ import annotations

import numpy as np

from speechinsight.evaluation.matching import borderline_pairs, one_to_one_match

_VOCAB_SIZE = 64


def _toy_embed(texts: list[str]) -> np.ndarray:
    """Toy embedding with a stable word -> position map (hashed modulo a fixed
    vocab size). Using a stable map lets independent calls produce comparable
    vectors — important because `one_to_one_match` embeds hyp and ref separately.
    """

    out = np.zeros((len(texts), _VOCAB_SIZE), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash(w) % _VOCAB_SIZE] = 1.0
    return out


def test_perfect_overlap_yields_f1_one() -> None:
    result = one_to_one_match(
        ["pick blue casing", "ship by friday"],
        ["pick blue casing", "ship by friday"],
        _toy_embed,
        threshold=0.5,
    )
    assert result.f1 == 1.0
    assert result.precision == 1.0 and result.recall == 1.0


def test_no_matches_yields_zero_f1() -> None:
    result = one_to_one_match(["pick blue"], ["something unrelated"], _toy_embed, threshold=0.5)
    assert result.f1 == 0.0


def test_assignment_is_one_to_one() -> None:
    # Two hypothesis items very similar to a single reference item: at most one
    # can match (one-to-one via Hungarian).
    result = one_to_one_match(
        ["pick blue", "pick blue"],
        ["pick blue"],
        _toy_embed,
        threshold=0.9,
    )
    assert len(result.matches) == 1


def test_borderline_pairs_filter() -> None:
    result = one_to_one_match(
        ["pick blue casing", "nothing matches here"],
        ["pick blue casing", "random reference"],
        _toy_embed,
        threshold=0.3,
    )
    border = borderline_pairs(result, margin=0.15)
    # borderline must be at or near threshold
    for _, _, s in border:
        assert 0.15 <= s <= 0.45


def test_empty_sides() -> None:
    assert one_to_one_match([], [], _toy_embed).f1 == 1.0
    assert one_to_one_match(["x"], [], _toy_embed).f1 == 0.0
    assert one_to_one_match([], ["x"], _toy_embed).f1 == 0.0
