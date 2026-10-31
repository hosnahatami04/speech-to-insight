from __future__ import annotations

# Pure, Streamlit-free helpers so the UI logic is unit-testable.

# A short, color-blind-aware palette. Repeats cyclically for meetings with
# more speakers than entries.
_PALETTE = (
    "#cde8f6",
    "#fde9a9",
    "#d6efc6",
    "#f6cdd1",
    "#e4d7f2",
    "#fbd9b4",
)


def color_for_speaker(speaker: str, palette: tuple[str, ...] = _PALETTE) -> str:
    """Stable per-speaker background color. Hashing on the label keeps the
    color consistent across reloads for the same session."""

    return palette[hash(speaker) % len(palette)]


def utterances_for_evidence(
    evidence_ids: list[str], transcript: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Resolve utterance ids against a transcript list, preserving the order of
    the evidence list. Unknown ids are skipped (never an error for the UI)."""

    by_id = {u["id"]: u for u in transcript}
    return [by_id[eid] for eid in evidence_ids if eid in by_id]
