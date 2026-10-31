# 0008. Controlled transcript corruption for error propagation

## Context

The error propagation study (spec §3.4) asks: when the recognizer makes
mistakes, how much does the downstream extraction F1 fall? Running the real
recognizer at many quality levels is expensive and does not let us separate
"WER caused this" from "the particular mistakes caused this". So the study
needs a controllable knob — a way to synthesize a transcript at a chosen WER
and feed it into extraction.

## Decision

- **Synthetic corruption** is applied word-by-word to a reference transcript.
  Three independent rates control substitutions (replace a word with a random
  in-domain word), deletions (drop the word) and insertions (add a word after).
  Replacement words are drawn from the transcript's own vocabulary so the
  corrupted text stays in-domain.
- **Measured WER is reported, not the target rate.** The actual WER after
  corruption is computed from the output; the measured value is what travels
  with the point. This avoids claiming a WER the real corruption did not hit.
- **Utterance ids are preserved.** This keeps `evidence` references valid, so
  the extraction stage never fails because an id disappeared.
- **Speaker-label corruption** is a second knob: each utterance is reassigned
  to a different real speaker with a given rate. Only speakers already present
  in the transcript are used, so the hypothesis stays plausible.
- **Degradation fits** are a plain linear regression of the downstream metric
  against measured WER. A linear fit is a baseline, not a claim of the true
  shape — the raw points still travel with the report, and callers add
  bootstrap CIs over per-meeting resamples.
- **Stage attribution** evaluates extraction on the four combinations of
  (ASR reference | ASR system) × (diarization reference | diarization system).
  The deltas between these cells tell you how much each stage contributes to
  the end-to-end F1 drop.

## Limitations

Synthetic corruption at a chosen WER is **not** distributed like real ASR
error: substitutions are uniform in the vocabulary rather than confused by
acoustic similarity, and deletions are independent rather than clustered.
That is a known weakness and is called out next to every reported curve: the
real-ASR points on the same WER axis let the reader see where the two
diverge.

## Rejected options

- **Only real ASR runs**: expensive and cannot isolate "WER" from "the
  specific mistakes made at this WER".
- **Confusion-matrix-aware corruption**: more realistic but requires an error
  model that itself depends on an ASR, which defeats the point of using
  corruption as a cheap, independent knob.
- **Target-WER-only reporting**: hides the gap between what we asked for and
  what we got.
