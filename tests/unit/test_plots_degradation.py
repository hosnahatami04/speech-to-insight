from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from speechinsight.evaluation.plots import plot_degradation


def test_plot_degradation_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "degrade.png"
    plot_degradation(
        wers=[0.0, 0.05, 0.1, 0.2, 0.3],
        metrics=[0.80, 0.75, 0.70, 0.60, 0.50],
        out_path=out,
        real_wers=[0.08, 0.15],
        real_metrics=[0.72, 0.63],
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_degradation_without_real_points(tmp_path: Path) -> None:
    out = tmp_path / "degrade2.png"
    plot_degradation([0.0, 0.1], [0.8, 0.7], out)
    assert out.exists()
