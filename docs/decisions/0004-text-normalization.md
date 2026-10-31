# 0004. Text normalization for WER

## Context

WER is highly sensitive to text normalization: casing, punctuation, how numbers
and abbreviations are written, and whitespace all change the score. A WER number
is only meaningful next to the normalizer that produced it, and is only
comparable to another number produced by the same normalizer.

## Decision

Use a single documented normalizer, named `speechinsight-v1`, applied to both the
reference and the hypothesis before scoring. Its rules, in order:

1. Lowercase.
2. Unify apostrophe variants (curly, backtick) to a straight quote.
3. Remove punctuation, keeping the apostrophe inside words so contractions like
   `don't` are preserved.
4. Expand a small, fixed set of abbreviations (`mr` → `mister`, `dr` → `doctor`,
   ...) and single-digit numbers (`0`–`10`) to their spoken form. The set is
   deliberately limited so the rule stays predictable; larger numbers are left
   as written.
5. Collapse whitespace.

Every reported WER carries the normalizer name. Comparison to published numbers
is only made while noting the normalizer difference.

## Rejected options

- **No normalization**: inflates WER with casing and punctuation differences
  that are not recognition errors.
- **An external normalizer from another project**: ties results to a dependency
  whose rules may change, and makes the exact behaviour harder to document here.
- **Full number-to-words expansion**: brittle for large or formatted numbers; the
  limited set covers the frequent disagreements without unpredictable edge cases.

## Consequence

WER numbers in this repository are internally comparable and reproducible, and
honestly labelled with their normalizer. The limited number/abbreviation handling
is a known, documented simplification.
