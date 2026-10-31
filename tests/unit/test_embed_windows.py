from __future__ import annotations

import numpy as np

from speechinsight.diarize.embed import EcapaEmbedder, Window, WindowingConfig, iter_windows
from speechinsight.diarize.vad import SpeechRegion


def test_long_region_produces_multiple_windows() -> None:
    region = SpeechRegion(start=0.0, end=4.0)
    windows = iter_windows([region], WindowingConfig(window_s=1.5, hop_s=0.75))
    starts = [w[0] for w in windows]
    # 0.0, 0.75, 1.5, 2.25 are all valid starts; 3.0 + 1.5 = 4.5 > 4.0 so stops here
    assert starts == [0.0, 0.75, 1.5, 2.25]


def test_short_region_still_yields_one_window() -> None:
    region = SpeechRegion(start=0.0, end=0.5)  # shorter than window_s
    windows = iter_windows([region], WindowingConfig(window_s=1.5, hop_s=0.75))
    assert windows == [(0.0, 0.5)]


def test_empty_regions_yield_nothing() -> None:
    assert iter_windows([]) == []


def test_window_dataclass_holds_embedding() -> None:
    emb = np.ones(192, dtype=np.float32)
    w = Window(start=0.0, end=1.5, embedding=emb)
    assert w.embedding.shape == (192,)
    assert w.embedding.dtype == np.float32


def test_embedder_rejects_non_16k() -> None:
    import pytest

    embedder = EcapaEmbedder(repo="r", revision="v")
    with pytest.raises(ValueError, match="16 kHz"):
        embedder.embed(np.zeros(8_000, dtype=np.float32), 8_000)
