from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def real_time_factor(audio_duration: float, processing_time: float) -> float:
    """Processing time divided by audio duration. Below 1.0 means faster than
    real time."""

    if audio_duration <= 0:
        raise ValueError("audio duration must be positive")
    return processing_time / audio_duration


def measure_rtf(
    transcribe: Callable[[npt.NDArray[np.float32], int], object],
    samples: npt.NDArray[np.float32],
    sample_rate: int,
) -> float:
    """Run one transcription and return its real-time factor. The clock is a
    monotonic timer so it is unaffected by wall-clock adjustments."""

    duration = len(samples) / sample_rate
    start = time.perf_counter()
    transcribe(samples, sample_rate)
    elapsed = time.perf_counter() - start
    return real_time_factor(duration, elapsed)
