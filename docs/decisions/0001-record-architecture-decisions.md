# 0001. Record architecture decisions

## Context

The project makes several non-obvious engineering choices whose rationale would
otherwise be lost. Reviewers and future maintainers need the reasoning, not just
the result, and the reasoning should live next to the code rather than in chat
history or commit messages.

## Decision

Every non-trivial decision is recorded as a short ADR in `docs/decisions/` named
`NNNN-title.md`. Each ADR states the context, the decision, the options that were
rejected, and the resulting consequence. ADRs are append-only; a later ADR
supersedes an earlier one rather than editing it.

## Rejected options

- No ADRs, rationale in commit messages only: commit messages are not
  discoverable as a set and are capped at a readable length.
- A single running design document: harder to reference a specific decision and
  to see what was rejected and why.

## Consequence

A small, consistent paper trail of decisions. The overhead is one short file per
real decision, which is acceptable given the project's emphasis on traceability.
