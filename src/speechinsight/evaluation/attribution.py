from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttributionCell:
    """One cell of the stage-attribution table (spec §3.4 propagation task 4).

    Each cell names which stage used the system output and which used the
    reference, and records the resulting extraction F1 and the measured upstream
    WER / DER that explain it.
    """

    asr: str  # "reference" or "system"
    diarization: str  # "reference" or "system"
    extraction_f1: float
    upstream_wer: float | None  # None when ASR is reference
    upstream_der: float | None  # None when diarization is reference
    utterances: int


@dataclass(frozen=True, slots=True)
class AttributionTable:
    cells: tuple[AttributionCell, ...]

    def by_combo(self, asr: str, diarization: str) -> AttributionCell | None:
        for cell in self.cells:
            if cell.asr == asr and cell.diarization == diarization:
                return cell
        return None

    def asr_delta(self) -> float | None:
        """F1 lost when the ASR is switched from reference to system, with
        diarization held at reference. ``None`` if either cell is missing."""

        ref = self.by_combo("reference", "reference")
        sys_asr = self.by_combo("system", "reference")
        if ref is None or sys_asr is None:
            return None
        return ref.extraction_f1 - sys_asr.extraction_f1

    def diarization_delta(self) -> float | None:
        """F1 lost when the diarization is switched from reference to system,
        with ASR held at reference."""

        ref = self.by_combo("reference", "reference")
        sys_diar = self.by_combo("reference", "system")
        if ref is None or sys_diar is None:
            return None
        return ref.extraction_f1 - sys_diar.extraction_f1


def build_attribution(cells: list[AttributionCell]) -> AttributionTable:
    """Validate that at most one cell exists per (asr, diarization) combo and
    freeze the table."""

    seen: set[tuple[str, str]] = set()
    for c in cells:
        key = (c.asr, c.diarization)
        if key in seen:
            raise ValueError(f"duplicate combo {key} in attribution table")
        seen.add(key)
    return AttributionTable(cells=tuple(cells))
