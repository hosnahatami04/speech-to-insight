from __future__ import annotations

import numpy as np
import pytest

from speechinsight.evaluation.rtf import measure_rtf, real_time_factor


def test_real_time_factor_ratio() -> None:
    assert real_time_factor(10.0, 5.0) == 0.5


def test_zero_duration_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        real_time_factor(0.0, 1.0)


def test_measure_rtf_runs_transcribe_once() -> None:
    calls = []

    def fake(samples, sample_rate):
        calls.append(len(samples))
        return []

    rtf = measure_rtf(fake, np.zeros(16_000, dtype=np.float32), 16_000)
    assert len(calls) == 1
    assert rtf >= 0.0
