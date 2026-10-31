from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from speechinsight.extract.schema import ExtractedItem
from speechinsight.types import Transcript


@dataclass(frozen=True, slots=True)
class GroundedItem:
    """An extracted item plus the grounding verdict. ``supported`` is True only
    when every deterministic check passes AND the NLI (if provided) agrees."""

    item: ExtractedItem
    kind: str  # "decision" or "action"
    supported: bool
    reasons: tuple[str, ...]  # explanation of any failed check


@dataclass(frozen=True, slots=True)
class GroundingResult:
    kept: list[GroundedItem]
    dropped_no_evidence: int  # items the deterministic step removed entirely
    unsupported_entities: int  # items whose owner/due/number is missing from evidence
    unsupported_nli: int  # items the NLI rejected


_NUMBER_OR_DATE = re.compile(r"\b\d{1,4}([:/\-\.]\d{1,4})*\b|\b\d+\b")


def _evidence_text(evidence: tuple[str, ...], transcript: Transcript) -> str:
    pieces = []
    for uid in evidence:
        utt = transcript.by_id(uid)
        if utt is not None:
            pieces.append(utt.text)
    return " ".join(pieces)


def _check_entities(item: ExtractedItem, evidence_text: str, known_speakers: set[str]) -> list[str]:
    """Return the list of failed entity checks (empty if the item is fine)."""

    reasons: list[str] = []

    if item.owner is not None and item.owner not in known_speakers:
        reasons.append(f"owner {item.owner!r} is not a known speaker label")

    low_ev = evidence_text.lower()
    if item.due is not None:
        for match in _NUMBER_OR_DATE.findall(item.due):
            token = match if isinstance(match, str) else match[0]
            if token and token.lower() not in low_ev:
                reasons.append(f"due contains {token!r} which is not in evidence")

    for match in _NUMBER_OR_DATE.finditer(item.text):
        token = match.group(0)
        if token.lower() not in low_ev:
            reasons.append(f"text contains number {token!r} which is not in evidence")

    return reasons


def ground_items(
    items: list[ExtractedItem],
    kind: str,
    transcript: Transcript,
    nli: Callable[[str, str], bool] | None = None,
) -> GroundingResult:
    """Verify extracted items against the transcript. Items whose ``evidence``
    ids are invalid or missing are *removed* and counted; items that fail the
    entity or NLI checks are kept but flagged ``supported=False`` with reasons.
    """

    known_ids = {u.id for u in transcript.utterances}
    known_speakers = set(transcript.speakers)

    kept: list[GroundedItem] = []
    dropped = 0
    bad_entities = 0
    bad_nli = 0

    for item in items:
        valid_ev = tuple(e for e in item.evidence if e in known_ids)
        if not valid_ev:
            dropped += 1
            continue
        clean = item.model_copy(update={"evidence": valid_ev})

        ev_text = _evidence_text(valid_ev, transcript)
        reasons = _check_entities(clean, ev_text, known_speakers)
        if reasons:
            bad_entities += 1

        if nli is not None and not reasons and not nli(ev_text, clean.text):
            reasons.append("NLI could not entail the item from its evidence")
            bad_nli += 1

        kept.append(
            GroundedItem(item=clean, kind=kind, supported=not reasons, reasons=tuple(reasons))
        )

    return GroundingResult(
        kept=kept,
        dropped_no_evidence=dropped,
        unsupported_entities=bad_entities,
        unsupported_nli=bad_nli,
    )
