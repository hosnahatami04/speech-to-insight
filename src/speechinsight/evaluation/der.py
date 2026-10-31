from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from speechinsight.types import Segment


@dataclass(frozen=True, slots=True)
class DerComponents:
    miss: float  # reference speech wrongly labeled silence (seconds)
    false_alarm: float  # hypothesis speech outside any reference segment (seconds)
    confusion: float  # reference speech labeled with the wrong speaker (seconds)
    total_reference: float  # seconds of reference speech scored
    der: float  # (miss + false_alarm + confusion) / total_reference


# Resolution used to rasterize segments onto a time axis. 10 ms matches the
# industry convention for diarization scoring and is small enough that the
# measured DER is insensitive to the sampling grid.
FRAME_S = 0.01


def _rasterize(segments: tuple[Segment, ...], end_s: float, speakers: list[str]) -> np.ndarray:
    """Boolean per-speaker, per-frame activity matrix of shape
    ``(n_speakers, n_frames)``. Overlapping segments activate multiple
    speakers at the same frame."""

    n_frames = int(np.ceil(end_s / FRAME_S))
    speaker_index = {s: i for i, s in enumerate(speakers)}
    grid = np.zeros((len(speakers), n_frames), dtype=bool)
    for seg in segments:
        start = max(0, int(seg.start / FRAME_S))
        stop = min(n_frames, int(np.ceil(seg.end / FRAME_S)))
        if seg.speaker in speaker_index and stop > start:
            grid[speaker_index[seg.speaker], start:stop] = True
    return grid


def _apply_collar(active_frames: np.ndarray, collar_s: float) -> np.ndarray:
    """Zero out ``collar_s`` seconds on each side of every reference-speech
    boundary. ``active_frames`` is the per-frame union over all reference
    speakers (True where some speaker is active)."""

    collar = round(collar_s / FRAME_S)
    if collar <= 0:
        return np.ones_like(active_frames, dtype=bool)

    # A boundary is a frame where activity changes.
    padded = np.concatenate([[False], active_frames, [False]])
    change = padded[1:] != padded[:-1]
    boundary_idx = np.where(change)[0]

    scored = np.ones_like(active_frames, dtype=bool)
    for b in boundary_idx:
        lo = max(0, b - collar)
        hi = min(len(scored), b + collar)
        scored[lo:hi] = False
    return scored


def _best_speaker_mapping(
    ref_grid: np.ndarray, hyp_grid: np.ndarray, scored: np.ndarray
) -> dict[int, int]:
    """Hungarian assignment that maximizes overlap in the scored region. Returns
    ``{hyp_index: ref_index}``. Hypothesis speakers with no match stay out of
    the mapping (treated as pure false alarms)."""

    n_ref, _ = ref_grid.shape
    n_hyp, _ = hyp_grid.shape
    if n_hyp == 0 or n_ref == 0:
        return {}

    overlap = np.zeros((n_hyp, n_ref), dtype=np.int64)
    for h in range(n_hyp):
        for r in range(n_ref):
            overlap[h, r] = int(np.sum(hyp_grid[h] & ref_grid[r] & scored))

    # linear_sum_assignment minimizes; we want to maximize, so negate.
    row, col = linear_sum_assignment(-overlap)
    return {int(r): int(c) for r, c in zip(row, col, strict=True) if overlap[r, c] > 0}


def diarization_error_rate(
    reference: tuple[Segment, ...],
    hypothesis: tuple[Segment, ...],
    collar_s: float = 0.25,
    skip_overlap: bool = False,
) -> DerComponents:
    """DER with Hungarian speaker mapping. ``collar_s`` suppresses scoring in a
    window around every reference boundary; ``skip_overlap`` excludes frames
    where more than one reference speaker is active."""

    ref_speakers = sorted({s.speaker for s in reference})
    hyp_speakers = sorted({s.speaker for s in hypothesis})
    end_s = max(
        max((s.end for s in reference), default=0.0),
        max((s.end for s in hypothesis), default=0.0),
    )
    if end_s <= 0:
        return DerComponents(0.0, 0.0, 0.0, 0.0, 0.0)

    ref_grid = _rasterize(reference, end_s, ref_speakers)
    hyp_grid = _rasterize(hypothesis, end_s, hyp_speakers)
    # Pad to equal length (rasterize already does this since both use the same end_s).

    ref_active = np.asarray(ref_grid.any(axis=0), dtype=bool)
    scored = _apply_collar(ref_active, collar_s)
    if skip_overlap:
        scored = scored & (ref_grid.sum(axis=0) <= 1)

    mapping = _best_speaker_mapping(ref_grid, hyp_grid, scored)

    # Build per-frame expected speaker set for the hypothesis under the mapping:
    # for each ref speaker r, the hypothesis speaker(s) mapped to r.
    reverse: dict[int, list[int]] = {}
    for h, r in mapping.items():
        reverse.setdefault(r, []).append(h)

    miss_frames = 0
    fa_frames = 0
    conf_frames = 0
    total_ref_frames = int(np.sum(ref_active & scored))

    scored_idx = np.where(scored)[0]
    for f in scored_idx:
        ref_speaks = set(np.where(ref_grid[:, f])[0].tolist())
        hyp_speaks = set(np.where(hyp_grid[:, f])[0].tolist())
        if not ref_speaks and not hyp_speaks:
            continue
        if not ref_speaks and hyp_speaks:
            fa_frames += 1
            continue
        if ref_speaks and not hyp_speaks:
            miss_frames += len(ref_speaks)
            continue
        # Both active: for each ref speaker, is the mapped hyp speaker present?
        for r in ref_speaks:
            expected = set(reverse.get(r, []))
            if expected & hyp_speaks:
                pass  # correct
            elif hyp_speaks:
                conf_frames += 1
            else:
                miss_frames += 1

    miss_s = miss_frames * FRAME_S
    fa_s = fa_frames * FRAME_S
    conf_s = conf_frames * FRAME_S
    total_s = total_ref_frames * FRAME_S
    der = (miss_s + fa_s + conf_s) / total_s if total_s > 0 else 0.0
    return DerComponents(
        miss=miss_s, false_alarm=fa_s, confusion=conf_s, total_reference=total_s, der=der
    )


def speaker_count_error(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> int:
    """Absolute error in the estimated number of speakers for one meeting."""

    return abs(len({s.speaker for s in hypothesis}) - len({s.speaker for s in reference}))
