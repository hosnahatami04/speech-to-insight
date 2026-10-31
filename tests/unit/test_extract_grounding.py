from __future__ import annotations

from speechinsight.extract.grounding import ground_items
from speechinsight.extract.schema import ExtractedItem
from speechinsight.types import Transcript, Utterance


def _transcript() -> Transcript:
    return Transcript(
        utterances=(
            Utterance(id="u0001", speaker="A", start=0, end=1, text="We'll pick blue."),
            Utterance(
                id="u0002",
                speaker="B",
                start=1,
                end=2,
                text="Build it by Friday for 25 dollars.",
            ),
        )
    )


def test_item_with_unknown_evidence_id_is_dropped() -> None:
    items = [ExtractedItem(text="x", evidence=("u9999",))]
    result = ground_items(items, "action", _transcript())
    assert result.kept == []
    assert result.dropped_no_evidence == 1


def test_item_with_mix_of_valid_and_invalid_evidence_keeps_valid_only() -> None:
    items = [ExtractedItem(text="Pick blue.", evidence=("u0001", "u9999"))]
    result = ground_items(items, "decision", _transcript())
    assert len(result.kept) == 1
    assert result.kept[0].item.evidence == ("u0001",)


def test_owner_not_in_speakers_is_flagged_unsupported() -> None:
    items = [ExtractedItem(text="Pick blue.", owner="Z", evidence=("u0001",))]
    result = ground_items(items, "decision", _transcript())
    assert result.kept[0].supported is False
    assert any("owner" in r for r in result.kept[0].reasons)


def test_number_in_text_must_appear_in_evidence() -> None:
    items = [
        ExtractedItem(text="Price is 25.", evidence=("u0002",)),  # 25 is in evidence
        ExtractedItem(text="Price is 99.", evidence=("u0002",)),  # 99 is not
    ]
    result = ground_items(items, "decision", _transcript())
    assert result.kept[0].supported is True
    assert result.kept[1].supported is False


def test_nli_rejection_flags_unsupported() -> None:
    def nli(evidence: str, claim: str) -> bool:
        return "pick" in claim.lower() and "pick" in evidence.lower()

    items = [
        ExtractedItem(text="Pick blue.", evidence=("u0001",)),
        ExtractedItem(text="Pick red.", evidence=("u0001",)),  # not supported by evidence
    ]
    result = ground_items(items, "decision", _transcript(), nli=nli)
    assert result.kept[0].supported is True
    assert result.kept[1].supported is True  # 'pick' is still in both

    # Switch to a stricter NLI
    def strict_nli(evidence: str, claim: str) -> bool:
        return claim.lower().strip(".") in evidence.lower()

    result2 = ground_items(items, "decision", _transcript(), nli=strict_nli)
    assert result2.kept[0].supported is True
    assert result2.kept[1].supported is False
