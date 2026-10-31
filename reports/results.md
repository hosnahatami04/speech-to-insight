# Evaluation results

Commit: `274091bc9e07c6936c5b7ff37c7e835ade5fca5f`

Model pins:

- faster-whisper tiny.en: `Systran/faster-whisper-tiny.en@0d3d19a32d33`
- faster-whisper base.en: `Systran/faster-whisper-base.en@3d3d5dee2648`
- faster-whisper small.en: `Systran/faster-whisper-small.en@d1d751a5f827`
- speaker embedding: `speechbrain/spkrec-ecapa-voxceleb@0f99f2d0ebe8`
- NLI: `cross-encoder/nli-deberta-v3-small@fa2804872c3b`
- matcher embedding: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4`
- extractor: `qwen2.5:3b-instruct` (sha256:161ddde4c9cd...)
- prompt version: `v1`

## What didn't work

- LibriSpeech dev-clean download in the sandbox was unreliable; corpora are placed by the user on the real workstation.
- The accent corpus (Common Voice) requires accepting the publisher terms; per-accent WER lands as soon as those clips are on disk.
- The AMI subset is small (3 dev + 3 test meetings) by CPU budget; see ADR 0003. Scale needs more compute.
- Synthetic transcript corruption does not match the error distribution of real ASR; the real-ASR points on the degradation plot show the gap.

