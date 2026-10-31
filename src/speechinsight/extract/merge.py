from __future__ import annotations

from collections.abc import Callable

from speechinsight.extract.schema import ExtractedItem


def merge_items(
    items: list[ExtractedItem],
    similarity: Callable[[str, str], float],
    threshold: float = 0.85,
) -> list[ExtractedItem]:
    """Collapse near-duplicate items whose text similarity is at least
    ``threshold``. The merged item keeps the first text seen, the union of the
    evidence ids, and the first non-null owner/due. Order is preserved."""

    merged: list[ExtractedItem] = []
    for item in items:
        matched = False
        for i, kept in enumerate(merged):
            if similarity(item.text, kept.text) >= threshold:
                merged[i] = ExtractedItem(
                    text=kept.text,
                    owner=kept.owner or item.owner,
                    due=kept.due or item.due,
                    evidence=tuple(dict.fromkeys([*kept.evidence, *item.evidence])),
                )
                matched = True
                break
        if not matched:
            merged.append(item)
    return merged
