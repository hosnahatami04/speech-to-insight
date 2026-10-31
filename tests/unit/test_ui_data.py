from __future__ import annotations

from speechinsight.ui.ui_data import color_for_speaker, utterances_for_evidence


def test_color_is_stable_for_same_speaker() -> None:
    assert color_for_speaker("A") == color_for_speaker("A")
    assert color_for_speaker("B") == color_for_speaker("B")


def test_different_speakers_can_get_different_colors() -> None:
    palette = ("red", "green", "blue")
    seen = {color_for_speaker(s, palette) for s in ("A", "B", "C", "D")}
    # Not strictly required to be fully distinct (hash collisions possible),
    # but at least two speakers must get different colors.
    assert len(seen) >= 2


def test_evidence_lookup_resolves_ids_in_order() -> None:
    transcript = [
        {"id": "u0001", "speaker": "A", "text": "hi"},
        {"id": "u0002", "speaker": "B", "text": "ok"},
        {"id": "u0003", "speaker": "A", "text": "bye"},
    ]
    out = utterances_for_evidence(["u0003", "u0001"], transcript)
    assert [u["id"] for u in out] == ["u0003", "u0001"]


def test_evidence_lookup_skips_unknown_ids() -> None:
    transcript = [{"id": "u0001", "speaker": "A", "text": "hi"}]
    out = utterances_for_evidence(["u0001", "u9999"], transcript)
    assert len(out) == 1 and out[0]["id"] == "u0001"


def test_evidence_lookup_empty_cases() -> None:
    assert utterances_for_evidence([], [{"id": "u0001"}]) == []
    assert utterances_for_evidence(["u0001"], []) == []
