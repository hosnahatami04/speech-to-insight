from __future__ import annotations

import pytest

from speechinsight.evaluation.attribution import AttributionCell, build_attribution


def _cells() -> list[AttributionCell]:
    return [
        AttributionCell("reference", "reference", 0.80, None, None, 50),
        AttributionCell("system", "reference", 0.65, 0.15, None, 50),
        AttributionCell("reference", "system", 0.70, None, 0.28, 50),
        AttributionCell("system", "system", 0.55, 0.15, 0.28, 50),
    ]


def test_build_attribution_rejects_duplicate_combo() -> None:
    cells = _cells()
    cells.append(AttributionCell("system", "system", 0.5, 0.15, 0.28, 50))
    with pytest.raises(ValueError, match="duplicate"):
        build_attribution(cells)


def test_asr_delta_isolates_asr_contribution() -> None:
    table = build_attribution(_cells())
    # ref/ref F1 = 0.80, system/ref F1 = 0.65 -> ASR costs 0.15 F1
    assert table.asr_delta() == pytest.approx(0.15)


def test_diarization_delta_isolates_diarization_contribution() -> None:
    table = build_attribution(_cells())
    # ref/ref F1 = 0.80, ref/system F1 = 0.70 -> diarization costs 0.10 F1
    assert table.diarization_delta() == pytest.approx(0.10)


def test_delta_is_none_when_cell_missing() -> None:
    partial = build_attribution(
        [
            AttributionCell("reference", "reference", 0.80, None, None, 50),
        ]
    )
    assert partial.asr_delta() is None
    assert partial.diarization_delta() is None


def test_by_combo_lookup() -> None:
    table = build_attribution(_cells())
    cell = table.by_combo("system", "system")
    assert cell is not None and cell.extraction_f1 == 0.55
    assert table.by_combo("other", "other") is None
