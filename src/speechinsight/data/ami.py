from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from speechinsight.types import Segment


@dataclass(frozen=True, slots=True)
class SummaryItem:
    """A reference decision or action, stated as free text in the AMI abstractive
    summaries. ``kind`` is ``"decision"`` or ``"action"``."""

    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class AmiMeeting:
    meeting_id: str
    split: str
    segments: tuple[Segment, ...]
    decisions: tuple[SummaryItem, ...]
    actions: tuple[SummaryItem, ...]

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(sorted({s.speaker for s in self.segments}))

    @property
    def duration(self) -> float:
        return max((s.end for s in self.segments), default=0.0)


def parse_rttm(path: str | Path) -> tuple[Segment, ...]:
    """Parse an RTTM file into speaker segments. Only ``SPEAKER`` lines are read;
    overlapping segments are preserved as written (no merging)."""

    segments: list[Segment] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            start = float(fields[3])
            duration = float(fields[4])
            speaker = fields[7]
            segments.append(Segment(speaker=speaker, start=start, end=start + duration))
    return tuple(segments)


def load_meeting(
    meeting_id: str,
    split: str,
    rttm_path: str | Path,
    summary_path: str | Path | None,
) -> AmiMeeting:
    """Assemble one meeting from its ground-truth RTTM and, when available, its
    abstractive summary. Meetings without a summary carry empty decision and
    action references."""

    segments = parse_rttm(rttm_path)
    if summary_path is not None and Path(summary_path).is_file():
        decisions, actions = parse_summary(summary_path)
    else:
        decisions, actions = (), ()
    return AmiMeeting(
        meeting_id=meeting_id,
        split=split,
        segments=segments,
        decisions=decisions,
        actions=actions,
    )


# The AMI summaries mark an empty section with a placeholder sentence rather than
# omitting it. These must not become reference items.
_PLACEHOLDERS = {"na", "na.", "*na*", "*na*.", "none", "none."}


def _is_placeholder(text: str) -> bool:
    return text.strip().lower() in _PLACEHOLDERS


def _section_sentences(root: ET.Element, tag: str) -> list[str]:
    sentences: list[str] = []
    for section in root.iter():
        if section.tag.rsplit("}", 1)[-1] != tag:
            continue
        for node in section.iter():
            if node.tag.rsplit("}", 1)[-1] == "sentence":
                text = " ".join((node.text or "").split())
                if text and not _is_placeholder(text):
                    sentences.append(text)
    return sentences


def parse_summary(path: str | Path) -> tuple[tuple[SummaryItem, ...], tuple[SummaryItem, ...]]:
    """Parse an AMI abstractive summary file and return its decision and action
    items as free-text sentences. Namespaces are ignored by matching on the
    local tag name."""

    tree = ET.parse(path)
    root = tree.getroot()
    decisions = tuple(SummaryItem("decision", s) for s in _section_sentences(root, "decisions"))
    actions = tuple(SummaryItem("action", s) for s in _section_sentences(root, "actions"))
    return decisions, actions
