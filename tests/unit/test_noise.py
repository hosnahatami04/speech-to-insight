from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from speechinsight.audio.noise import generate_noise, measure_snr, mix_at_snr

SR = 16_000


def _speech(seed: int = 0, duration_s: float = 2.0) -> np.ndarray:
    """Deterministic pseudo-speech: tone with amplitude envelope so the
    speech-active power estimator sees varying frame RMS."""

    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * duration_s)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 200.0 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    noise = 0.02 * rng.standard_normal(len(t))
    return (tone * envelope + noise).astype(np.float32)


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15)
@given(snr=st.sampled_from([20.0, 10.0, 5.0, 0.0]), kind=st.sampled_from(["white", "pink"]))
def test_measured_snr_matches_target_within_tolerance(snr: float, kind: str) -> None:
    speech = _speech()
    noise = generate_noise(kind, len(speech), SR, seed=42)
    mixed = mix_at_snr(speech, noise, snr, SR)
    measured = measure_snr(speech, mixed, SR)
    # 0.3 dB tolerance covers clip-rescaling and the median-based active estimator.
    assert abs(measured - snr) < 0.3, f"{kind} @ {snr}dB: measured {measured:.3f}"


def test_pink_noise_spectral_slope_is_minus_one() -> None:
    noise = generate_noise("pink", 2**15, SR, seed=0)
    spec = np.abs(np.fft.rfft(noise)) ** 2
    freqs = np.fft.rfftfreq(len(noise), d=1 / SR)
    mask = (freqs > 20) & (freqs < 4000)
    log_f = np.log10(freqs[mask])
    log_p = np.log10(spec[mask] + 1e-20)
    # Linear fit of log power vs log frequency; pink noise has slope ~ -1.
    slope = float(np.polyfit(log_f, log_p, 1)[0])
    assert -1.3 < slope < -0.7, f"pink slope {slope:.3f} not near -1"


def test_babble_mixes_distinct_sources() -> None:
    sources = [_speech(seed=i, duration_s=1.0) for i in range(6)]
    a = generate_noise("babble", 16_000, SR, seed=7, babble_sources=sources)
    b = generate_noise("babble", 16_000, SR, seed=7, babble_sources=sources)
    c = generate_noise("babble", 16_000, SR, seed=8, babble_sources=sources)
    assert np.array_equal(a, b)  # deterministic with same seed
    assert not np.array_equal(a, c)  # different seed -> different mix


def test_babble_requires_sources() -> None:
    with pytest.raises(ValueError, match="babble"):
        generate_noise("babble", 100, SR, seed=0)


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        generate_noise("brown", 100, SR, seed=0)


def test_mix_preserves_length_and_rate() -> None:
    speech = _speech()
    noise = generate_noise("white", len(speech), SR, seed=0)
    mixed = mix_at_snr(speech, noise, 10.0, SR)
    assert len(mixed) == len(speech)
    assert mixed.dtype == np.float32


def test_mix_avoids_clipping() -> None:
    speech = _speech()
    # Even at a very low SNR (loud noise), the output must stay within [-1, 1].
    noise = generate_noise("white", len(speech), SR, seed=0)
    mixed = mix_at_snr(speech, noise, -10.0, SR)
    assert float(np.max(np.abs(mixed))) <= 1.0 + 1e-6
