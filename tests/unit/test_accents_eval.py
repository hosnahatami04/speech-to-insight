from __future__ import annotations

from speechinsight.data.manifests import Manifest, UtteranceEntry
from speechinsight.evaluation.accents import MIN_UTTERANCES_FOR_CONCLUSION, per_accent_wer


def _manifest(entries: list[tuple[str, str, str, str]]) -> Manifest:
    return Manifest(
        corpus="accents",
        split="dev",
        entries=tuple(
            UtteranceEntry(uid, f"{uid}.wav", 1.0, text, "spk", accent)
            for uid, text, accent, _ in entries
        ),
    )


def test_groups_by_accent_and_computes_wer() -> None:
    manifest = _manifest(
        [
            ("u1", "the quick fox", "Indian", "hyp"),
            ("u2", "jumps over", "Indian", "hyp"),
            ("u3", "the lazy dog", "Scottish", "hyp"),
        ]
    )
    hyps = {
        "u1": ["the", "quick", "fox"],  # perfect
        "u2": ["jumps", "for"],  # 1 sub out of 2 words
        "u3": ["the", "lazy", "cat"],  # 1 sub out of 3 words
    }
    results = per_accent_wer(manifest, hyps, n_resamples=50)
    by_accent = {r.accent: r for r in results}
    assert set(by_accent) == {"Indian", "Scottish"}
    # Indian: 1 sub / 5 ref words = 0.2
    assert abs(by_accent["Indian"].wer - 0.2) < 1e-9
    assert by_accent["Indian"].utterances == 2
    assert "report only" in by_accent["Indian"].note


def test_small_group_is_tagged_report_only() -> None:
    manifest = _manifest([("u1", "x", "Welsh", "h")])
    results = per_accent_wer(manifest, {"u1": ["x"]}, n_resamples=10)
    assert results[0].note.startswith("report only")
    assert results[0].utterances < MIN_UTTERANCES_FOR_CONCLUSION


def test_entries_without_accent_are_skipped() -> None:
    manifest = Manifest(
        "accents",
        "dev",
        (
            UtteranceEntry("u1", "a.wav", 1.0, "hi", "spk", None),
            UtteranceEntry("u2", "b.wav", 1.0, "hi", "spk", "Irish"),
        ),
    )
    results = per_accent_wer(manifest, {"u1": ["hi"], "u2": ["hi"]}, n_resamples=10)
    assert len(results) == 1
    assert results[0].accent == "Irish"
