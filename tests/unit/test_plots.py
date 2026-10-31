from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from speechinsight.evaluation.accents import AccentWer  # noqa: E402
from speechinsight.evaluation.noise_sweep import SweepPoint  # noqa: E402
from speechinsight.evaluation.plots import plot_noise_sweep, plot_per_accent_wer  # noqa: E402


def test_plot_noise_sweep_writes_file(tmp_path: Path) -> None:
    points = [
        SweepPoint("base.en", "white", 20.0, 0.08, 0.06, 0.10, 100, "n"),
        SweepPoint("base.en", "white", 10.0, 0.15, 0.12, 0.18, 100, "n"),
        SweepPoint("base.en", "white", 0.0, 0.45, 0.40, 0.50, 100, "n"),
        SweepPoint("base.en", "pink", 20.0, 0.07, 0.05, 0.09, 100, "n"),
        SweepPoint("base.en", "pink", 10.0, 0.14, 0.11, 0.17, 100, "n"),
    ]
    out = tmp_path / "noise.png"
    returned = plot_noise_sweep(points, out)
    assert returned == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_per_accent_writes_file(tmp_path: Path) -> None:
    results = [
        AccentWer("Indian", 200, 2000, 0.11, 0.09, 0.13, ""),
        AccentWer("Scottish", 20, 150, 0.25, 0.18, 0.33, "report only (<100 samples)"),
    ]
    out = tmp_path / "accents.png"
    plot_per_accent_wer(results, out)
    assert out.exists() and out.stat().st_size > 0
