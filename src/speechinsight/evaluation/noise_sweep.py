from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from speechinsight.asr.base import Transcriber
from speechinsight.asr.normalize import NORMALIZER_NAME, normalize
from speechinsight.audio.io import load_audio
from speechinsight.audio.noise import generate_noise, mix_at_snr
from speechinsight.cache import StageCache, audio_hash, cache_key
from speechinsight.data.manifests import Manifest
from speechinsight.evaluation.wer import bootstrap_ci, corpus_wer

NoiseKind = str  # "white" | "pink" | "babble"

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SweepPoint:
    model: str
    noise_kind: NoiseKind
    snr_db: float
    wer: float
    ci_low: float
    ci_high: float
    utterances: int
    normalizer: str


def _babble_sources(manifest: Manifest, exclude_speaker: str, k: int = 6) -> list[FloatArray]:
    """Return up to ``k`` sample arrays from speakers other than
    ``exclude_speaker``, each loaded as 16 kHz mono float32."""

    sources: list[FloatArray] = []
    seen_speakers = set()
    for entry in manifest.entries:
        if entry.speaker == exclude_speaker or entry.speaker in seen_speakers:
            continue
        try:
            audio = load_audio(entry.path)
        except (FileNotFoundError, ValueError):
            continue
        sources.append(audio.samples)
        seen_speakers.add(entry.speaker)
        if len(sources) >= k:
            break
    return sources


def _transcribe_corrupted(
    transcriber: Transcriber,
    audio_path: str,
    model_id: str,
    noise_kind: NoiseKind,
    snr_db: float,
    seed: int,
    babble_sources: list[FloatArray],
    cache: StageCache | None,
) -> list[str]:
    """Load audio, mix noise at the target SNR, transcribe. Caching is keyed by
    the original audio hash plus the corruption parameters so a repeated sweep
    does not re-transcribe."""

    audio = load_audio(audio_path)
    noise = generate_noise(
        noise_kind,
        len(audio.samples),
        audio.sample_rate,
        seed=seed,
        babble_sources=babble_sources if noise_kind == "babble" else None,
    )
    mixed = mix_at_snr(audio.samples, noise, snr_db, audio.sample_rate)

    params = {"noise": noise_kind, "snr_db": snr_db, "seed": seed}
    if cache is not None:
        key = cache_key(audio_hash(audio.samples), f"asr+{noise_kind}", model_id, params)
        if cache.has(key):
            return normalize(str(cache.load(key)["text"])).split()
    words = transcriber.transcribe(mixed, audio.sample_rate)
    text = " ".join(w.text for w in words).strip()
    if cache is not None:
        cache.store(key, {"text": text})
    return normalize(text).split()


def run_noise_sweep(
    transcriber: Transcriber,
    manifest: Manifest,
    model_id: str,
    noise_kinds: list[NoiseKind],
    snrs_db: list[float],
    seed: int = 0,
    cache: StageCache | None = None,
    babble_sources: list[FloatArray] | None = None,
) -> list[SweepPoint]:
    """For each (noise kind, SNR) combination, mix noise into every utterance,
    transcribe and compute corpus WER with a bootstrap CI."""

    # Pre-fetch babble sources once; same for all SNRs since content is scaled, not resampled.
    if babble_sources is None and "babble" in noise_kinds:
        first_speaker = manifest.entries[0].speaker if manifest.entries else ""
        babble_sources = _babble_sources(manifest, first_speaker)

    points: list[SweepPoint] = []
    for kind in noise_kinds:
        for snr in snrs_db:
            refs: list[list[str]] = []
            hyps: list[list[str]] = []
            for entry in manifest.entries:
                hyp_tokens = _transcribe_corrupted(
                    transcriber,
                    entry.path,
                    model_id,
                    kind,
                    snr,
                    seed,
                    babble_sources or [],
                    cache,
                )
                refs.append(normalize(entry.text).split())
                hyps.append(hyp_tokens)
            if not refs:
                continue
            wer = corpus_wer(refs, hyps)
            low, high = bootstrap_ci(refs, hyps, seed=seed)
            points.append(
                SweepPoint(
                    model=model_id,
                    noise_kind=kind,
                    snr_db=snr,
                    wer=wer,
                    ci_low=low,
                    ci_high=high,
                    utterances=len(refs),
                    normalizer=NORMALIZER_NAME,
                )
            )
    return points


def write_sweep(points: list[SweepPoint], path: str | Path) -> None:
    """Write the sweep as JSON for the report step."""

    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in points], f, indent=2, sort_keys=True)


def wer_delta_db(points: list[SweepPoint], model: str, kind: str, snr_db: float) -> float | None:
    """Difference between WER at (model, kind, snr_db) and the clean baseline
    (highest SNR for that model+kind). ``None`` if either point is missing."""

    model_kind = [p for p in points if p.model == model and p.noise_kind == kind]
    if not model_kind:
        return None
    clean = max(model_kind, key=lambda p: p.snr_db)
    target = next((p for p in model_kind if p.snr_db == snr_db), None)
    if target is None:
        return None
    return target.wer - clean.wer
