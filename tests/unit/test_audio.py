from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speechinsight.audio.io import load_audio
from speechinsight.audio.resample import TARGET_SR, resample, to_mono


def test_stereo_averages_to_mono() -> None:
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.tolist() == [2.0, 3.0]


def test_resample_preserves_frequency_and_duration() -> None:
    src_sr = 48_000
    duration = 1.0
    t = np.arange(int(src_sr * duration)) / src_sr
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    out = resample(tone, src_sr, TARGET_SR)

    assert abs(len(out) / TARGET_SR - duration) < 0.01
    freqs = np.fft.rfftfreq(len(out), 1 / TARGET_SR)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(out)))]
    assert abs(peak - 440.0) < 5.0


def test_resample_noop_when_rate_matches() -> None:
    sig = np.ones(10, dtype=np.float32)
    out = resample(sig, TARGET_SR, TARGET_SR)
    assert out.dtype == np.float32
    assert np.array_equal(out, sig)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "absent.wav")


def test_load_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(0, dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="empty"):
        load_audio(empty)


def test_load_resamples_and_downmixes(tmp_path: Path) -> None:
    src = tmp_path / "stereo.wav"
    stereo = np.zeros((48_000, 2), dtype=np.float32)
    sf.write(str(src), stereo, 48_000)
    audio = load_audio(src)
    assert audio.sample_rate == TARGET_SR
    assert audio.samples.ndim == 1
    assert abs(audio.duration - 1.0) < 0.01
