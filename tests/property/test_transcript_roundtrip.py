from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from speechinsight.serialize import transcript_from_dict, transcript_to_dict
from speechinsight.types import Transcript, Utterance, Word, make_utterance_id

_text = st.text(min_size=0, max_size=12)
_time = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def _words(draw: st.DrawFn) -> tuple[Word, ...]:
    count = draw(st.integers(min_value=0, max_value=4))
    return tuple(Word(text=draw(_text), start=draw(_time), end=draw(_time)) for _ in range(count))


@st.composite
def _transcripts(draw: st.DrawFn) -> Transcript:
    count = draw(st.integers(min_value=0, max_value=5))
    utterances = tuple(
        Utterance(
            id=make_utterance_id(i),
            speaker=draw(st.sampled_from(["A", "B", "C"])),
            start=draw(_time),
            end=draw(_time),
            text=draw(_text),
            words=draw(_words()),
        )
        for i in range(count)
    )
    return Transcript(utterances=utterances)


# Timing-based health checks are unreliable under coverage instrumentation.
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(_transcripts())
def test_roundtrip_preserves_transcript(transcript: Transcript) -> None:
    restored = transcript_from_dict(transcript_to_dict(transcript))
    assert restored == transcript


def test_known_transcript_serializes_to_expected_shape() -> None:
    transcript = Transcript(
        utterances=(
            Utterance(
                id="u0001",
                speaker="A",
                start=0.0,
                end=1.5,
                text="hello there",
                words=(Word("hello", 0.0, 0.5), Word("there", 0.6, 1.5)),
            ),
        )
    )
    data = transcript_to_dict(transcript)
    assert data["utterances"][0]["id"] == "u0001"
    assert len(data["utterances"][0]["words"]) == 2
    assert transcript_from_dict(data) == transcript
