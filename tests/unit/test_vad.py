from __future__ import annotations

import numpy as np
import pytest

from speechinsight.diarize.vad import SileroVad, SpeechRegion, VadConfig


def test_rejects_non_16k_audio() -> None:
    vad = SileroVad()
    with pytest.raises(ValueError, match="16 kHz"):
        vad.detect(np.zeros(8_000, dtype=np.float32), 8_000)


def test_vad_config_defaults_match_spec() -> None:
    cfg = VadConfig()
    assert cfg.threshold == 0.5
    assert cfg.min_speech_s == 0.25
    assert cfg.min_silence_s == 0.1


def test_speech_region_is_frozen() -> None:
    r = SpeechRegion(start=1.0, end=2.0)
    with pytest.raises(AttributeError):
        r.start = 3.0  # type: ignore[misc]
