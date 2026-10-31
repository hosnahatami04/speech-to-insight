from __future__ import annotations

import json
from dataclasses import fields

from speechinsight.diarize.embed import Window
from speechinsight.serialize import transcript_to_dict
from speechinsight.types import Segment, Transcript, Utterance, Word


def test_transcript_json_has_no_embedding_field() -> None:
    """The Transcript serialization carries words with timestamps and text, but
    never speaker embeddings. Spec §1.2 forbids persisting biometric data."""

    transcript = Transcript(
        utterances=(
            Utterance(
                id="u0001",
                speaker="A",
                start=0.0,
                end=1.0,
                text="hi",
                words=(Word("hi", 0.0, 1.0),),
            ),
        )
    )
    payload = json.dumps(transcript_to_dict(transcript))
    assert "embedding" not in payload


def test_segment_fields_do_not_include_embedding() -> None:
    """Segment carries only (speaker, start, end). Any downstream serialization
    of segments therefore cannot leak embeddings."""

    names = {f.name for f in fields(Segment)}
    assert names == {"speaker", "start", "end"}


def test_window_embedding_lives_on_the_object_not_in_a_shared_cache() -> None:
    """A Window instance carries its embedding by value, so collecting embeddings
    is explicit (iterating windows), never implicit (reading from a global)."""

    import numpy as np

    w = Window(start=0.0, end=1.5, embedding=np.ones(3, dtype=np.float32))
    # The embedding is on the instance only; no module-level cache keeps it.
    import speechinsight.diarize.embed as embed_mod

    for name in dir(embed_mod):
        attr = getattr(embed_mod, name)
        assert not (isinstance(attr, dict) and any(k == w for k in attr))
