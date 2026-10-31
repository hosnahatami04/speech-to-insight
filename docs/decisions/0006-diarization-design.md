# 0006. Hand-built diarization pipeline

## Context

Speaker diarization decides who spoke when. Standard turnkey pipelines
(pyannote.audio's end-to-end pipeline, WhisperX) exist, but their license and
model-access constraints change — several of pyannote's recent models require
accepting terms, and depending on a packaged pipeline hides the stage behind
one call, which conflicts with the project's goal of measuring error at each
stage and swapping stages for a reference.

## Decision

Build the diarization stage from independent, swappable components:

1. **Silero VAD** (MIT, no auth) for speech regions. Thresholds come from
   `configs/diarization.yaml`.
2. **Windowing**: 1.5 s windows with 0.75 s hop inside each VAD region. Shorter
   regions still get one window so no speech is dropped.
3. **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`, non-gated) for a speaker
   embedding per window. Embeddings live only on the `Window` object and are
   never written to disk, cache or output — they are biometric data (spec §1.2).
4. **Agglomerative clustering** with cosine distance, average linkage, and a
   distance threshold tuned on the AMI dev subset. The main path leaves the
   number of speakers unknown (threshold-based); a fixed-count mode exists only
   as ablation.
5. **Segment reconstruction** merges consecutive windows with the same cluster
   label into speaker segments.
6. **Word-to-speaker alignment** assigns each ASR word to the segment with the
   largest time overlap and groups words into utterances with stable ids
   (`u0001`, `u0002`, ...) that extracted items can later cite.

Scoring uses our own DER with Hungarian speaker mapping, a configurable collar
and an overlap-handling switch. The implementation agrees with
`pyannote.metrics` within the 10 ms rasterization grid.

## Rejected options

- **pyannote.audio pipeline**: gated models and a hidden stage boundary.
- **WhisperX**: couples diarization to recognition and bundles its own choices.
- **Pretrained x-vector** (SpeechBrain): equally good but less common; ECAPA is
  the current baseline.
- **k-means on embeddings**: needs the speaker count up front, which the
  project treats as unknown.

## Consequence

Each diarization step is independently testable and replaceable. The cost is
that threshold tuning becomes the project's responsibility; the clustering
threshold is set from the AMI dev subset and recorded in the config.
