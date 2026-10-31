from __future__ import annotations

import pytest
from pydantic import ValidationError

from speechinsight.extract.schema import ExtractedItem, ExtractionOutput


def test_item_requires_non_empty_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        ExtractedItem(text="hi", evidence=())


def test_evidence_ids_must_look_like_utterance_ids() -> None:
    with pytest.raises(ValidationError, match="u0001"):
        ExtractedItem(text="hi", evidence=("foo",))


def test_owner_and_due_default_to_none() -> None:
    item = ExtractedItem(text="hi", evidence=("u0001",))
    assert item.owner is None
    assert item.due is None


def test_text_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        ExtractedItem(text="", evidence=("u0001",))


def test_extraction_output_defaults_to_empty_lists() -> None:
    out = ExtractionOutput()
    assert out.decisions == [] and out.actions == []


def test_extraction_output_round_trip() -> None:
    raw = {
        "decisions": [{"text": "Pick blue", "evidence": ["u0001"]}],
        "actions": [{"text": "Draft spec", "owner": "A", "due": "Friday", "evidence": ["u0003"]}],
    }
    out = ExtractionOutput.model_validate(raw)
    assert len(out.decisions) == 1
    assert out.actions[0].owner == "A"
    assert out.actions[0].due == "Friday"
