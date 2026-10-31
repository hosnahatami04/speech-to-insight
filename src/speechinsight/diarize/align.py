from __future__ import annotations

from dataclasses import dataclass

from speechinsight.types import Segment, Transcript, Utterance, Word, make_utterance_id


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    max_gap_s: float = 1.0  # start a new utterance after a silence longer than this


def _assign_speaker(word: Word, segments: tuple[Segment, ...]) -> str:
    """Return the speaker whose segment overlaps the word most. If no segment
    overlaps the word, fall back to the nearest segment by time."""

    best_overlap = 0.0
    best_speaker: str | None = None
    for s in segments:
        overlap = max(0.0, min(word.end, s.end) - max(word.start, s.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = s.speaker
    if best_speaker is not None:
        return best_speaker

    nearest = min(
        segments,
        key=lambda s: min(abs(word.start - s.end), abs(word.end - s.start)),
        default=None,
    )
    return nearest.speaker if nearest is not None else "unknown"


def build_transcript(
    words: list[Word],
    segments: tuple[Segment, ...],
    config: AlignmentConfig | None = None,
) -> Transcript:
    """Assign a speaker to each word and group words into utterances.

    An utterance ends when either the next word is spoken by a different
    speaker, or the gap between words exceeds ``max_gap_s``. Utterance ids are
    assigned in order so they can be cited later as evidence.
    """

    cfg = config or AlignmentConfig()
    if not words:
        return Transcript()

    labeled = [(w, _assign_speaker(w, segments)) for w in words]

    utterances: list[Utterance] = []
    cur_words: list[Word] = [labeled[0][0]]
    cur_speaker = labeled[0][1]

    for word, speaker in labeled[1:]:
        gap = word.start - cur_words[-1].end
        if speaker != cur_speaker or gap > cfg.max_gap_s:
            utterances.append(_finalize(cur_words, cur_speaker, len(utterances)))
            cur_words = [word]
            cur_speaker = speaker
        else:
            cur_words.append(word)

    utterances.append(_finalize(cur_words, cur_speaker, len(utterances)))
    return Transcript(utterances=tuple(utterances))


def _finalize(words: list[Word], speaker: str, index: int) -> Utterance:
    text = " ".join(w.text for w in words).strip()
    return Utterance(
        id=make_utterance_id(index),
        speaker=speaker,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=tuple(words),
    )
