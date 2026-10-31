from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from speechinsight.evaluation.wer import corpus_wer
from speechinsight.types import Transcript, Utterance


@dataclass(frozen=True, slots=True)
class CorruptionConfig:
    """Rates are fractions of the reference word count. The resulting WER is
    measured on the corrupted output and reported — not assumed to equal the
    target rate exactly, since substitutions and deletions interact."""

    substitution_rate: float = 0.0
    deletion_rate: float = 0.0
    insertion_rate: float = 0.0
    seed: int = 0


def _vocabulary(transcript: Transcript) -> list[str]:
    """All distinct words in the transcript (lowercased, de-duplicated). Used
    as the source for substitutions and insertions so the corrupted text stays
    in-domain, which keeps the simulated recognizer error realistic."""

    words = []
    seen: set[str] = set()
    for utt in transcript.utterances:
        for w in utt.text.split():
            w_lower = w.lower()
            if w_lower and w_lower not in seen:
                words.append(w_lower)
                seen.add(w_lower)
    return words or ["word"]


def corrupt_transcript(
    transcript: Transcript, config: CorruptionConfig
) -> tuple[Transcript, float]:
    """Return a corrupted copy of the transcript plus its measured WER against
    the original. Corruption is applied per-word with the given rates; the
    utterance ids are preserved so evidence links remain meaningful."""

    rng = np.random.default_rng(config.seed)
    vocab = _vocabulary(transcript)

    new_utts: list[Utterance] = []
    for utt in transcript.utterances:
        orig_words = utt.text.split()
        if not orig_words:
            new_utts.append(utt)
            continue
        out: list[str] = []
        for w in orig_words:
            r = rng.random()
            if r < config.deletion_rate:
                continue  # drop this word
            if r < config.deletion_rate + config.substitution_rate:
                out.append(cast("str", rng.choice(vocab)))
                continue
            out.append(w)
            if rng.random() < config.insertion_rate:
                out.append(cast("str", rng.choice(vocab)))
        new_utts.append(
            Utterance(
                id=utt.id,
                speaker=utt.speaker,
                start=utt.start,
                end=utt.end,
                text=" ".join(out) if out else utt.text[:1],
                words=utt.words,
            )
        )

    corrupted = Transcript(utterances=tuple(new_utts))
    references = [u.text.split() for u in transcript.utterances if u.text]
    hypotheses = [u.text.split() for u in corrupted.utterances if u.text]
    measured_wer = corpus_wer(references, hypotheses) if references else 0.0
    return corrupted, measured_wer


def swap_speaker_labels(transcript: Transcript, swap_rate: float, seed: int = 0) -> Transcript:
    """Reassign a fraction of utterances to a different speaker from the set of
    speakers present in the transcript."""

    speakers = list(transcript.speakers)
    if len(speakers) < 2 or swap_rate <= 0:
        return transcript
    rng = np.random.default_rng(seed)
    new_utts: list[Utterance] = []
    for utt in transcript.utterances:
        if rng.random() < swap_rate:
            others = [s for s in speakers if s != utt.speaker]
            new_sp = cast("str", rng.choice(others)) if others else utt.speaker
        else:
            new_sp = utt.speaker
        new_utts.append(
            Utterance(
                id=utt.id,
                speaker=new_sp,
                start=utt.start,
                end=utt.end,
                text=utt.text,
                words=utt.words,
            )
        )
    return Transcript(utterances=tuple(new_utts))


@dataclass(frozen=True, slots=True)
class PropagationFit:
    """Linear fit of a downstream metric against measured WER.

    ``slope`` is the metric change per 1.0 of WER. For readability the sweep
    report multiplies this by 0.1 to get "per 10 WER points".
    """

    slope: float
    intercept: float
    n_points: int


def fit_degradation(wers: list[float], metrics: list[float]) -> PropagationFit:
    """Ordinary least squares fit of ``metrics ~ wers``. ``PropagationFit`` is a
    plain estimate; callers wrap it with bootstrap resampling for a CI."""

    if len(wers) != len(metrics):
        raise ValueError("wers and metrics must have the same length")
    if len(wers) < 2:
        raise ValueError("fit requires at least two points")
    x = np.asarray(wers, dtype=np.float64)
    y = np.asarray(metrics, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return PropagationFit(slope=float(slope), intercept=float(intercept), n_points=len(x))
