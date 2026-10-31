from __future__ import annotations

from dataclasses import dataclass

from speechinsight.types import Transcript, Utterance


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_tokens: int = 1500  # rough token budget per chunk
    overlap_utterances: int = 2


def _estimate_tokens(text: str) -> int:
    """Rough token count: whitespace-split, inflated by a constant to approximate
    subword expansion for the extractor models in use."""

    return max(1, int(len(text.split()) * 1.3))


def iter_chunks(
    transcript: Transcript, config: ChunkingConfig | None = None
) -> list[tuple[Utterance, ...]]:
    """Split a transcript into token-budgeted chunks of utterances. Adjacent
    chunks share ``overlap_utterances`` utterances so an item spanning a
    boundary is still extractable."""

    cfg = config or ChunkingConfig()
    utterances = transcript.utterances
    if not utterances:
        return []

    chunks: list[tuple[Utterance, ...]] = []
    i = 0
    while i < len(utterances):
        tokens = 0
        j = i
        while j < len(utterances):
            tokens += _estimate_tokens(utterances[j].text)
            if tokens > cfg.max_tokens and j > i:
                break
            j += 1
        chunks.append(tuple(utterances[i:j]))
        if j >= len(utterances):
            break
        # Advance by at least one utterance, backing off by the overlap.
        step = max(1, (j - i) - cfg.overlap_utterances)
        i += step
    return chunks


def format_chunk(chunk: tuple[Utterance, ...]) -> str:
    """Render a chunk as the transcript text handed to the extractor. Each line
    carries the utterance id so the model can cite it in ``evidence``."""

    return "\n".join(f"[{u.id}] {u.speaker}: {u.text}" for u in chunk)
