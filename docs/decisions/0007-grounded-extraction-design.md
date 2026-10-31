# 0007. Grounded extraction with mandatory evidence

## Context

The extractor must turn a transcript into decisions and action items. The
project's core value is traceability: every item has to point back to the
utterances it came from, and items the model invents must be caught and
dropped rather than silently merged into the output. A long meeting
transcript also exceeds the context window of the small local LLMs this
project uses, so chunking is required — but a decision stated once across a
chunk boundary should not end up as two items.

## Decision

- **Prompt (`prompts/extract_v1.txt`)** instructs the model to use only the
  cited utterances, leave `owner`/`due` as `null` when not stated, and return
  JSON with `evidence` as a list of utterance ids. The prompt is versioned on
  disk and the version string travels with every report.
- **Schema (Pydantic)** enforces `evidence` non-empty and ids shaped like
  `u0001`. One malformed response is retried once, then counted as a schema
  error.
- **Chunking** uses a ~1500-token budget with a 2-utterance overlap. The
  transcript handed to the model carries the utterance id on each line so the
  model can cite them.
- **Merging** collapses near-duplicate items (text similarity ≥ 0.85 by
  default) across chunks, keeping the first text and the union of evidence.
- **Grounding** runs two passes:
  1. **Deterministic**: evidence ids must resolve, `owner` must be an existing
     speaker label, and every number or date in `text`/`due` must appear in the
     evidence text.
  2. **NLI** (optional, injected): the item text must be entailed by its
     evidence text.
  Items with no valid evidence are *removed*; items that fail entity or NLI
  checks are *kept but flagged* unsupported with reasons, so a reviewer sees
  what the model wrote and why it was rejected.
- **Matching** against AMI reference items uses sentence-embedding cosine
  similarity with Hungarian assignment at the item level; a borderline-pair
  helper highlights pairs near the threshold for human review.

## Rejected options

- **No evidence requirement**: makes items untraceable and makes the whole
  study pointless.
- **LangChain / LlamaIndex**: adds a large dependency to make one deterministic
  HTTP call; the `urllib`-based client is small enough to audit.
- **Silent LLM re-prompting beyond one retry**: hides failures and inflates
  compute. One retry is the compromise.
- **Dropping unsupported items instead of flagging**: loses the data that
  tells us the model is hallucinating.

## Consequence

Every reported extraction item either cites real utterances and carries a
grounding verdict, or does not appear in the report. Spec §4.3 KPIs — schema
validity, mandatory evidence, dropped-items share, unsupported-entity share —
map directly onto `GroundingResult` fields.
