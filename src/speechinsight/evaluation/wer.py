from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_words + other.reference_words,
        )


def align_counts(reference: list[str], hypothesis: list[str]) -> EditCounts:
    """Levenshtein alignment over word tokens, counting substitutions, deletions
    and insertions. Deletion = a reference word missing from the hypothesis;
    insertion = a hypothesis word not in the reference."""

    n, m = len(reference), len(hypothesis)
    # dp[i][j] = (cost, S, D, I) to align reference[:i] with hypothesis[:j].
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            dele = dp[i - 1][j]
            ins = dp[i][j - 1]
            best = min(sub, dele, ins, key=lambda c: c[0])
            if best is sub:
                dp[i][j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            elif best is dele:
                dp[i][j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            else:
                dp[i][j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
    _, s, d, ins_count = dp[n][m]
    return EditCounts(s, d, ins_count, n)


def _wer(counts: EditCounts) -> float:
    if counts.reference_words == 0:
        raise ValueError("cannot compute WER against an empty reference")
    return counts.errors / counts.reference_words


def corpus_wer(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    """Corpus-level WER: summed errors over summed reference words, not the mean
    of per-sentence rates."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    total = EditCounts(0, 0, 0, 0)
    for ref, hyp in zip(references, hypotheses, strict=True):
        total = total + align_counts(ref, hyp)
    return _wer(total)


def bootstrap_ci(
    references: list[list[str]],
    hypotheses: list[list[str]],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Sentence-level bootstrap confidence interval for corpus WER. Resamples
    sentences with replacement; deterministic for a given seed."""

    per_sentence = [align_counts(r, h) for r, h in zip(references, hypotheses, strict=True)]
    rng = np.random.default_rng(seed)
    n = len(per_sentence)
    rates = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        total = EditCounts(0, 0, 0, 0)
        for i in idx:
            total = total + per_sentence[i]
        if total.reference_words > 0:
            rates.append(total.errors / total.reference_words)
    lower = float(np.quantile(rates, (1 - confidence) / 2))
    upper = float(np.quantile(rates, 1 - (1 - confidence) / 2))
    return lower, upper
