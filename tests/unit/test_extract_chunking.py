from __future__ import annotations

from itertools import pairwise

from speechinsight.extract.chunking import ChunkingConfig, format_chunk, iter_chunks
from speechinsight.types import Transcript, Utterance, make_utterance_id


def _utt(i: int, words: int) -> Utterance:
    return Utterance(
        id=make_utterance_id(i),
        speaker="A" if i % 2 == 0 else "B",
        start=i * 1.0,
        end=i * 1.0 + 0.5,
        text=" ".join(["word"] * words),
    )


def test_small_transcript_fits_one_chunk() -> None:
    transcript = Transcript(utterances=(_utt(0, 5), _utt(1, 5)))
    chunks = iter_chunks(transcript, ChunkingConfig(max_tokens=100))
    assert len(chunks) == 1
    assert chunks[0] == transcript.utterances


def test_overlap_between_chunks_keeps_tail() -> None:
    transcript = Transcript(utterances=tuple(_utt(i, 50) for i in range(10)))
    chunks = iter_chunks(transcript, ChunkingConfig(max_tokens=150, overlap_utterances=2))
    assert len(chunks) > 1
    # Each adjacent chunk pair shares at least one utterance in id.
    for a, b in pairwise(chunks):
        a_ids = {u.id for u in a}
        b_ids = {u.id for u in b}
        assert a_ids & b_ids, "adjacent chunks must overlap"


def test_every_utterance_appears_in_at_least_one_chunk() -> None:
    transcript = Transcript(utterances=tuple(_utt(i, 30) for i in range(8)))
    chunks = iter_chunks(transcript, ChunkingConfig(max_tokens=100, overlap_utterances=1))
    seen: set[str] = set()
    for chunk in chunks:
        for u in chunk:
            seen.add(u.id)
    assert seen == {u.id for u in transcript.utterances}


def test_format_chunk_includes_ids_and_speakers() -> None:
    chunk = (_utt(0, 2), _utt(1, 2))
    text = format_chunk(chunk)
    assert "[u0001]" in text and "[u0002]" in text
    assert "A:" in text and "B:" in text


def test_empty_transcript_yields_no_chunks() -> None:
    assert iter_chunks(Transcript()) == []
