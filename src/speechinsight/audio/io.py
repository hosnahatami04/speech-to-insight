from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from speechinsight.audio.resample import TARGET_SR, resample, to_mono


@dataclass(frozen=True, slots=True)
class Audio:
    samples: np.ndarray  # mono, float32, sample_rate Hz
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> Audio:
    """Load a file, downmix to mono, resample to ``target_sr`` and return
    float32 samples. Rejects a missing, empty or unreadable file with an
    explicit error rather than yielding silent or malformed audio downstream."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    try:
        samples, source_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError(f"could not decode audio file {path}: {exc}") from exc

    if samples.size == 0:
        raise ValueError(f"audio file is empty: {path}")

    mono = to_mono(samples)
    resampled = resample(mono, source_sr, target_sr)
    return Audio(samples=resampled.astype(np.float32, copy=False), sample_rate=target_sr)
