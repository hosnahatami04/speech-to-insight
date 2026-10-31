from __future__ import annotations

from math import gcd
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

TARGET_SR = 16_000

FloatArray = npt.NDArray[np.floating]


def to_mono(samples: FloatArray) -> FloatArray:
    """Average channels to a single channel. Accepts (n,) or (n, channels)."""

    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return cast(FloatArray, samples.mean(axis=1))
    raise ValueError(f"expected 1-D or 2-D audio, got shape {samples.shape}")


def resample(
    samples: FloatArray, source_sr: int, target_sr: int = TARGET_SR
) -> npt.NDArray[np.float32]:
    """Polyphase resampling to ``target_sr``. Deterministic and dependency-light
    (no librosa). Returns the input unchanged when the rate already matches."""

    if source_sr <= 0:
        raise ValueError(f"source sample rate must be positive, got {source_sr}")
    if source_sr == target_sr:
        return samples.astype(np.float32, copy=False)
    divisor = gcd(source_sr, target_sr)
    up = target_sr // divisor
    down = source_sr // divisor
    resampled = resample_poly(samples, up, down)
    return cast("npt.NDArray[np.float32]", resampled.astype(np.float32, copy=False))
