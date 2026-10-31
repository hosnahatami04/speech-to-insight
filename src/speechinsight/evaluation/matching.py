from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class MatchResult:
    precision: float
    recall: float
    f1: float
    matches: tuple[tuple[int, int, float], ...]  # (hyp_idx, ref_idx, similarity)
    threshold: float


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between row-vectors in a and b."""

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return cast("np.ndarray", a_norm @ b_norm.T)


def one_to_one_match(
    hyp_texts: list[str],
    ref_texts: list[str],
    embed: Callable[[list[str]], np.ndarray],
    threshold: float = 0.6,
) -> MatchResult:
    """Match extracted items to AMI reference items by cosine similarity between
    their embeddings. The pairing is one-to-one via Hungarian assignment, and
    only pairs whose similarity is at or above ``threshold`` are counted as
    matches. Precision, recall and F1 are reported in the item-level sense from
    the spec."""

    if not hyp_texts and not ref_texts:
        return MatchResult(1.0, 1.0, 1.0, (), threshold)
    if not hyp_texts or not ref_texts:
        return MatchResult(0.0, 0.0, 0.0, (), threshold)

    hyp_vec = embed(hyp_texts)
    ref_vec = embed(ref_texts)
    sim = _cosine(np.asarray(hyp_vec), np.asarray(ref_vec))

    # Hungarian maximizes assignment; negate for the minimizer.
    row, col = linear_sum_assignment(-sim)
    matches = tuple(
        (int(r), int(c), float(sim[r, c]))
        for r, c in zip(row, col, strict=True)
        if sim[r, c] >= threshold
    )
    tp = len(matches)
    precision = tp / len(hyp_texts) if hyp_texts else 0.0
    recall = tp / len(ref_texts) if ref_texts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return MatchResult(
        precision=precision, recall=recall, f1=f1, matches=matches, threshold=threshold
    )


def borderline_pairs(
    result: MatchResult, margin: float = 0.1
) -> tuple[tuple[int, int, float], ...]:
    """Pairs whose similarity falls within ``margin`` of the threshold. These
    are the ones a human reviewer should check (spec §3.5)."""

    lo = result.threshold - margin
    hi = result.threshold + margin
    return tuple(m for m in result.matches if lo <= m[2] <= hi)
