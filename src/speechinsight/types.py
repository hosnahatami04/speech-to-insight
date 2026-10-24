from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous speech region attributed to a single speaker."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Utterance:
    """A speaker turn. ``id`` is stable across a transcript so extracted items
    can reference it as evidence."""

    id: str
    speaker: str
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True, slots=True)
class Transcript:
    """The single intermediate representation shared by every stage. Downstream
    stages consume only this structure, never a stage's raw model output."""

    utterances: tuple[Utterance, ...] = field(default_factory=tuple)

    def by_id(self, utterance_id: str) -> Utterance | None:
        for utt in self.utterances:
            if utt.id == utterance_id:
                return utt
        return None

    @property
    def speakers(self) -> tuple[str, ...]:
        seen: list[str] = []
        for utt in self.utterances:
            if utt.speaker not in seen:
                seen.append(utt.speaker)
        return tuple(seen)

    @property
    def duration(self) -> float:
        if not self.utterances:
            return 0.0
        return max(utt.end for utt in self.utterances)


def make_utterance_id(index: int) -> str:
    # One-based so the first turn is u0001, matching evidence ids in reports.
    return f"u{index + 1:04d}"
