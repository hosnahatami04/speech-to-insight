from __future__ import annotations

from speechinsight.extract.merge import merge_items
from speechinsight.extract.schema import ExtractedItem


def _ratio(a: str, b: str) -> float:
    """Simple Jaccard-like similarity for tests."""

    toks_a = set(a.lower().split())
    toks_b = set(b.lower().split())
    if not toks_a and not toks_b:
        return 1.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def test_near_duplicates_merge_and_union_evidence() -> None:
    items = [
        ExtractedItem(text="Build the prototype", evidence=("u0001",)),
        ExtractedItem(text="Build the prototype", owner="A", evidence=("u0002",)),
    ]
    merged = merge_items(items, _ratio, threshold=0.8)
    assert len(merged) == 1
    assert merged[0].owner == "A"
    assert set(merged[0].evidence) == {"u0001", "u0002"}


def test_different_items_stay_separate() -> None:
    items = [
        ExtractedItem(text="Pick blue casing", evidence=("u0001",)),
        ExtractedItem(text="Draft the user manual", evidence=("u0005",)),
    ]
    merged = merge_items(items, _ratio, threshold=0.85)
    assert len(merged) == 2


def test_order_is_preserved() -> None:
    items = [
        ExtractedItem(text="A", evidence=("u0001",)),
        ExtractedItem(text="B", evidence=("u0002",)),
    ]
    merged = merge_items(items, _ratio, threshold=0.99)
    assert [m.text for m in merged] == ["A", "B"]
