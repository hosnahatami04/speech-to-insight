# 0002. Project scaffold and shared transcript structure

## Context

The pipeline has three independent stages (recognition, diarization,
extraction) that must be evaluated both in isolation and in sequence, so error
can be traced from stage to stage. This requires a stable boundary between
stages and a reproducible environment.

## Decision

- **Dependency management with `uv`** and a committed `uv.lock`, with a
  `pyproject.toml` that also supports `pip install -e ".[dev]"`. Heavy model
  dependencies are split into optional groups (`asr`, `diar`, `nlp`, `service`,
  `report`) so lint, type and non-model tests stay light; `dev` pulls the test
  and reference-metric tools.
- **`src` layout** with a single package `speechinsight`, to prevent accidental
  imports of the working tree and to keep the import surface explicit.
- **A single intermediate structure, `Transcript`**: an ordered list of
  `Utterance(id, speaker, start, end, text, words)`, where `Word` carries
  per-word timestamps and `Segment` is a speaker-attributed interval. Every
  stage after recognition consumes and produces only this structure. Utterance
  ids are stable (`u0001`, ...) so extracted items can cite them as evidence.

## Rejected options

- **Passing each stage's native output downstream** (for example, raw recognizer
  segments): couples stages to one another's model output and makes
  stage-for-reference substitution in the error-propagation study impossible.
- **A flat layout without `src`**: risks importing the checkout instead of the
  installed package and blurs the public surface.
- **Pinning all dependencies with exact `==` in `pyproject.toml`**: duplicates
  the role of the lockfile and makes compatible upgrades noisier; the lockfile
  is the source of truth for exact versions.

## Consequence

Each stage sits behind a typed interface over `Transcript`, which makes
per-stage evaluation and reference substitution straightforward. The optional
dependency groups keep the default developer loop fast.
