from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from speechinsight.data.manifests import Manifest
from speechinsight.evaluation.wer import bootstrap_ci, corpus_wer


@dataclass(frozen=True, slots=True)
class AccentWer:
    accent: str
    utterances: int
    reference_words: int
    wer: float
    ci_low: float
    ci_high: float
    note: str  # e.g. "report only (<100 samples)"


# A per-accent WER with fewer than this many utterances is reported, not
# concluded from. The threshold is from the spec data card.
MIN_UTTERANCES_FOR_CONCLUSION = 100


def group_by_accent(
    manifest: Manifest, hypotheses_by_id: dict[str, list[str]]
) -> dict[str, tuple[list[list[str]], list[list[str]]]]:
    """Split the (references, hypotheses) parallel arrays into per-accent
    bundles. References are tokenized from the manifest text (callers normalize
    both sides before passing hypotheses in)."""

    grouped: dict[str, tuple[list[list[str]], list[list[str]]]] = defaultdict(lambda: ([], []))
    for entry in manifest.entries:
        if entry.accent is None:
            continue
        hyp = hypotheses_by_id.get(entry.id)
        if hyp is None:
            continue
        refs, hyps = grouped[entry.accent]
        refs.append(entry.text.split())
        hyps.append(hyp)
    return dict(grouped)


def per_accent_wer(
    manifest: Manifest,
    hypotheses_by_id: dict[str, list[str]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[AccentWer]:
    """Corpus WER per accent group with a bootstrap confidence interval. Groups
    with fewer than ``MIN_UTTERANCES_FOR_CONCLUSION`` samples are tagged so the
    reader does not draw conclusions from a thin slice."""

    grouped = group_by_accent(manifest, hypotheses_by_id)
    results: list[AccentWer] = []
    for accent, (refs, hyps) in sorted(grouped.items()):
        if not refs:
            continue
        wer = corpus_wer(refs, hyps)
        low, high = bootstrap_ci(refs, hyps, n_resamples=n_resamples, seed=seed)
        note = (
            f"report only (<{MIN_UTTERANCES_FOR_CONCLUSION} samples)"
            if len(refs) < MIN_UTTERANCES_FOR_CONCLUSION
            else ""
        )
        results.append(
            AccentWer(
                accent=accent,
                utterances=len(refs),
                reference_words=sum(len(r) for r in refs),
                wer=wer,
                ci_low=low,
                ci_high=high,
                note=note,
            )
        )
    return results
