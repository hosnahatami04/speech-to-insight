from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExtractedItem(BaseModel):
    """A single decision or action extracted from the meeting. Every item cites
    the utterance ids it was taken from so it can be verified. The LLM must not
    guess ``owner`` or ``due`` — ``null`` when they are not stated in the
    cited utterances (see prompt and the grounding step)."""

    text: str = Field(..., min_length=1)
    owner: str | None = None
    due: str | None = None
    evidence: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("evidence")
    @classmethod
    def _non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("evidence must list at least one utterance id")
        for item in v:
            if not item or not item.startswith("u"):
                raise ValueError(f"evidence id {item!r} must look like 'u0001'")
        return tuple(v)


class ExtractionOutput(BaseModel):
    """One LLM call's structured output. Decisions and actions are two parallel
    lists so the matching step can score them separately."""

    decisions: list[ExtractedItem] = Field(default_factory=list)
    actions: list[ExtractedItem] = Field(default_factory=list)
