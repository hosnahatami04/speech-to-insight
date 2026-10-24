from __future__ import annotations

from speechinsight.types import Transcript, Utterance, make_utterance_id


def _utt(uid: str, speaker: str, start: float, end: float) -> Utterance:
    return Utterance(id=uid, speaker=speaker, start=start, end=end, text="x")


def test_make_utterance_id_is_one_based_and_padded() -> None:
    assert make_utterance_id(0) == "u0001"
    assert make_utterance_id(41) == "u0042"


def test_by_id_returns_match_or_none() -> None:
    transcript = Transcript(utterances=(_utt("u0001", "A", 0.0, 1.0),))
    assert transcript.by_id("u0001") is not None
    assert transcript.by_id("u9999") is None


def test_speakers_are_unique_in_first_seen_order() -> None:
    transcript = Transcript(
        utterances=(
            _utt("u0001", "A", 0.0, 1.0),
            _utt("u0002", "B", 1.0, 2.0),
            _utt("u0003", "A", 2.0, 3.0),
        )
    )
    assert transcript.speakers == ("A", "B")


def test_duration_is_last_end_and_zero_when_empty() -> None:
    assert Transcript().duration == 0.0
    transcript = Transcript(utterances=(_utt("u0001", "A", 0.0, 1.0), _utt("u0002", "B", 1.0, 2.5)))
    assert transcript.duration == 2.5
