from __future__ import annotations

from speechinsight.diarize.align import AlignmentConfig, build_transcript
from speechinsight.types import Segment, Word


def test_word_inside_segment_assigned_to_that_speaker() -> None:
    words = [Word("hello", 0.0, 0.5), Word("world", 0.6, 1.0)]
    segments = (Segment("A", 0.0, 2.0),)
    transcript = build_transcript(words, segments)
    assert len(transcript.utterances) == 1
    assert transcript.utterances[0].speaker == "A"
    assert transcript.utterances[0].id == "u0001"
    assert transcript.utterances[0].text == "hello world"


def test_word_on_boundary_goes_to_max_overlap() -> None:
    # Word spans [0.4, 0.9]; A covers [0, 0.5] (overlap 0.1), B covers [0.5, 2] (overlap 0.4).
    words = [Word("x", 0.4, 0.9)]
    segments = (Segment("A", 0.0, 0.5), Segment("B", 0.5, 2.0))
    transcript = build_transcript(words, segments)
    assert transcript.utterances[0].speaker == "B"


def test_speaker_change_splits_utterance() -> None:
    words = [
        Word("a", 0.0, 0.5),
        Word("b", 0.6, 1.0),
        Word("c", 1.1, 1.5),
    ]
    segments = (Segment("A", 0.0, 0.9), Segment("B", 0.9, 2.0))
    transcript = build_transcript(words, segments)
    assert len(transcript.utterances) == 2
    assert transcript.utterances[0].speaker == "A"
    assert transcript.utterances[1].speaker == "B"
    assert transcript.utterances[0].id == "u0001"
    assert transcript.utterances[1].id == "u0002"


def test_long_gap_splits_utterance() -> None:
    words = [Word("hi", 0.0, 0.5), Word("bye", 2.0, 2.5)]
    segments = (Segment("A", 0.0, 3.0),)
    transcript = build_transcript(words, segments, AlignmentConfig(max_gap_s=1.0))
    assert len(transcript.utterances) == 2


def test_empty_words_yields_empty_transcript() -> None:
    assert build_transcript([], ()).utterances == ()


def test_word_outside_all_segments_falls_back_to_nearest() -> None:
    words = [Word("x", 5.0, 5.5)]
    segments = (Segment("A", 0.0, 2.0), Segment("B", 10.0, 12.0))
    # Word at 5.0-5.5: distance to A end (2.0) is 3.0, to B start (10.0) is 4.5. Pick A.
    transcript = build_transcript(words, segments)
    assert transcript.utterances[0].speaker == "A"
