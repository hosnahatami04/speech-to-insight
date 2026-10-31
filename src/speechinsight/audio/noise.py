from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

# Noise is mixed to a target SNR by scaling against the speech power measured on
# the active-speech portion of the signal (not the whole file, which may contain
# silence that would inflate the noise floor and distort the target SNR).


def _speech_power(samples: FloatArray, frame_s: float, sample_rate: int) -> float:
    """Mean square power of frames whose RMS is above the median RMS.

    Using the median as the threshold is deliberately simple and parameter-free:
    half the frames (the louder half) are treated as speech, which is a stable
    proxy without pulling in a VAD dependency here. If every frame is silent the
    overall mean square is returned so a target SNR still has a reference point.
    """

    frame_len = max(1, int(frame_s * sample_rate))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return float(np.mean(samples**2))
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    threshold = float(np.median(rms))
    active = trimmed[rms >= threshold]
    if active.size == 0:
        return float(np.mean(samples**2))
    return float(np.mean(active**2))


def _white_noise(n: int, rng: np.random.Generator) -> FloatArray:
    return rng.standard_normal(n).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> FloatArray:
    """Pink (1/f) noise by shaping white noise in the frequency domain. The
    amplitude spectrum is scaled by 1/sqrt(f) so power falls as 1/f."""

    white = _white_noise(n, rng)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n).astype(np.float32)
    std = float(np.std(shaped))
    if std > 0:
        shaped = shaped / std
    return shaped


def _babble_noise(n: int, sources: list[FloatArray], rng: np.random.Generator) -> FloatArray:
    """Overlap 4-6 utterances from distinct speakers at random offsets. Each
    source is placed starting at a random position; longer target lengths wrap
    around with the source repeated."""

    if not sources:
        raise ValueError("babble noise needs at least one source utterance")
    count = int(rng.integers(4, 7))
    chosen_idx = rng.choice(len(sources), size=min(count, len(sources)), replace=False)
    mix = np.zeros(n, dtype=np.float32)
    for i in chosen_idx:
        src = sources[int(i)].astype(np.float32)
        if len(src) == 0:
            continue
        # Repeat the source to cover the target length, then take a random slice.
        reps = (n // len(src)) + 2
        tiled = np.tile(src, reps)
        offset = int(rng.integers(0, len(src)))
        mix = mix + tiled[offset : offset + n]
    std = float(np.std(mix))
    if std > 0:
        mix = mix / std
    return mix


def generate_noise(
    kind: str,
    n_samples: int,
    sample_rate: int,
    seed: int,
    babble_sources: list[FloatArray] | None = None,
) -> FloatArray:
    """Generate ``n_samples`` of noise with unit standard deviation. The result
    is deterministic for a given seed. ``kind`` is ``"white"``, ``"pink"`` or
    ``"babble"``; babble requires source utterances."""

    rng = np.random.default_rng(seed)
    if kind == "white":
        return _white_noise(n_samples, rng)
    if kind == "pink":
        return _pink_noise(n_samples, rng)
    if kind == "babble":
        if babble_sources is None:
            raise ValueError("babble noise requires babble_sources")
        return _babble_noise(n_samples, babble_sources, rng)
    raise ValueError(f"unknown noise kind: {kind}")


def mix_at_snr(
    speech: FloatArray,
    noise: FloatArray,
    snr_db: float,
    sample_rate: int,
    frame_s: float = 0.025,
) -> FloatArray:
    """Mix noise into speech at ``snr_db`` dB, measured against the speech-active
    power. The output is scaled down to the input peak when necessary to avoid
    clipping; the scaling factor applies to the whole signal, so the SNR is
    preserved."""

    if len(noise) < len(speech):
        reps = (len(speech) // len(noise)) + 2
        noise = np.tile(noise, reps)[: len(speech)]
    else:
        noise = noise[: len(speech)]

    speech_p = _speech_power(speech, frame_s, sample_rate)
    noise_p = float(np.mean(noise**2))
    if speech_p <= 0 or noise_p <= 0:
        return speech.astype(np.float32, copy=False)

    target_noise_p = speech_p / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_p / noise_p)
    mixed = (speech + scale * noise).astype(np.float32)

    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = (mixed / peak).astype(np.float32)
    return cast("FloatArray", mixed)


def measure_snr(
    speech: FloatArray, mixed: FloatArray, sample_rate: int, frame_s: float = 0.025
) -> float:
    """Measured SNR after mixing, for test and reporting. Uses the same speech-
    active power estimator as ``mix_at_snr``."""

    noise = mixed - speech[: len(mixed)]
    speech_p = _speech_power(speech[: len(mixed)], frame_s, sample_rate)
    noise_p = float(np.mean(noise**2))
    if speech_p <= 0 or noise_p <= 0:
        return float("inf")
    return 10.0 * float(np.log10(speech_p / noise_p))
