from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from speechinsight.extract.chunking import ChunkingConfig, format_chunk, iter_chunks
from speechinsight.extract.client import OllamaClient
from speechinsight.extract.grounding import GroundingResult, ground_items
from speechinsight.extract.merge import merge_items
from speechinsight.extract.schema import ExtractedItem, ExtractionOutput
from speechinsight.types import Transcript

PROMPT_VERSION = "v1"


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    decisions: GroundingResult
    actions: GroundingResult
    schema_errors: int  # chunks whose LLM response could not be parsed as ExtractionOutput
    prompt_version: str


def _call_once(
    client: OllamaClient, prompt_template: str, chunk_text: str
) -> ExtractionOutput | None:
    """One LLM call for one chunk. Returns ``None`` if the response does not fit
    the schema (counted as a schema error)."""

    from pydantic import ValidationError

    from speechinsight.extract.client import OllamaError

    try:
        raw = client.generate_json(prompt_template.format(transcript=chunk_text))
        return ExtractionOutput.model_validate(raw)
    except (OllamaError, ValidationError):
        return None


def extract(
    transcript: Transcript,
    client: OllamaClient,
    prompt_template: str,
    similarity: Callable[[str, str], float],
    nli: Callable[[str, str], bool] | None = None,
    chunking: ChunkingConfig | None = None,
    merge_threshold: float = 0.85,
) -> ExtractionReport:
    """Run the extractor over the transcript in token-budgeted chunks, merge
    near-duplicates, then verify every item against the transcript."""

    all_decisions: list[ExtractedItem] = []
    all_actions: list[ExtractedItem] = []
    schema_errors = 0

    for chunk in iter_chunks(transcript, chunking):
        chunk_text = format_chunk(chunk)
        output = _call_once(client, prompt_template, chunk_text)
        if output is None:
            schema_errors += 1
            continue
        all_decisions.extend(output.decisions)
        all_actions.extend(output.actions)

    decisions = merge_items(all_decisions, similarity, merge_threshold)
    actions = merge_items(all_actions, similarity, merge_threshold)

    return ExtractionReport(
        decisions=ground_items(decisions, "decision", transcript, nli),
        actions=ground_items(actions, "action", transcript, nli),
        schema_errors=schema_errors,
        prompt_version=PROMPT_VERSION,
    )
