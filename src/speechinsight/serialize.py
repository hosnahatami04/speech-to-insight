from __future__ import annotations

from typing import Any

from speechinsight.types import Transcript, Utterance, Word


def transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    return {
        "utterances": [
            {
                "id": utt.id,
                "speaker": utt.speaker,
                "start": utt.start,
                "end": utt.end,
                "text": utt.text,
                "words": [{"text": w.text, "start": w.start, "end": w.end} for w in utt.words],
            }
            for utt in transcript.utterances
        ]
    }


def transcript_from_dict(data: dict[str, Any]) -> Transcript:
    utterances = tuple(
        Utterance(
            id=item["id"],
            speaker=item["speaker"],
            start=item["start"],
            end=item["end"],
            text=item["text"],
            words=tuple(
                Word(text=w["text"], start=w["start"], end=w["end"]) for w in item.get("words", [])
            ),
        )
        for item in data.get("utterances", [])
    )
    return Transcript(utterances=utterances)
