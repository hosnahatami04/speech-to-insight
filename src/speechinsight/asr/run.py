from __future__ import annotations

from dataclasses import dataclass

from speechinsight.asr.base import Transcriber
from speechinsight.asr.normalize import NORMALIZER_NAME, normalize
from speechinsight.audio.io import load_audio
from speechinsight.cache import StageCache, audio_hash, cache_key
from speechinsight.data.manifests import Manifest
from speechinsight.evaluation.wer import bootstrap_ci, corpus_wer


@dataclass(frozen=True, slots=True)
class WerResult:
    model: str
    normalizer: str
    wer: float
    ci_low: float
    ci_high: float
    utterances: int
    reference_words: int


def transcribe_text(
    transcriber: Transcriber,
    audio_path: str,
    model_id: str,
    params: dict[str, object],
    cache: StageCache | None = None,
) -> str:
    """Transcribe one file to a plain string, caching the raw transcript keyed by
    the audio content and the model configuration."""

    audio = load_audio(audio_path)
    if cache is not None:
        key = cache_key(audio_hash(audio.samples), "asr", model_id, params)
        if cache.has(key):
            return str(cache.load(key)["text"])
    words = transcriber.transcribe(audio.samples, audio.sample_rate)
    text = " ".join(w.text for w in words).strip()
    if cache is not None:
        cache.store(key, {"text": text})
    return text


def evaluate_wer(
    transcriber: Transcriber,
    manifest: Manifest,
    model_id: str,
    params: dict[str, object],
    cache: StageCache | None = None,
) -> WerResult:
    """Run recognition over a manifest and report corpus WER with a bootstrap
    confidence interval. References and hypotheses pass through the same
    normalizer, whose name travels with the result."""

    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    for entry in manifest.entries:
        hyp = transcribe_text(transcriber, entry.path, model_id, params, cache)
        references.append(normalize(entry.text).split())
        hypotheses.append(normalize(hyp).split())

    wer = corpus_wer(references, hypotheses)
    low, high = bootstrap_ci(references, hypotheses)
    return WerResult(
        model=model_id,
        normalizer=NORMALIZER_NAME,
        wer=wer,
        ci_low=low,
        ci_high=high,
        utterances=len(manifest.entries),
        reference_words=sum(len(r) for r in references),
    )
