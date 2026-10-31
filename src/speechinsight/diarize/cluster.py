from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from speechinsight.diarize.embed import Window
from speechinsight.types import Segment


@dataclass(frozen=True, slots=True)
class ClusteringConfig:
    metric: str = "cosine"
    linkage: str = "average"
    distance_threshold: float = 0.7  # tuned on dev; see spec


def cluster_windows(
    windows: list[Window],
    config: ClusteringConfig | None = None,
    n_clusters: int | None = None,
) -> list[int]:
    """Return a cluster label per window. ``n_clusters=None`` leaves the number
    of speakers unknown and uses the distance threshold; passing an int fixes
    the count (ablation only — the main path leaves it unknown)."""

    cfg = config or ClusteringConfig()
    if not windows:
        return []
    if len(windows) == 1:
        return [0]

    embeddings = np.stack([w.embedding for w in windows]).astype(np.float32)

    if n_clusters is not None:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric=cfg.metric, linkage=cfg.linkage
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cfg.distance_threshold,
            metric=cfg.metric,
            linkage=cfg.linkage,
        )
    labels = model.fit_predict(embeddings)
    return [int(label) for label in labels]


def labels_to_segments(windows: list[Window], labels: list[int]) -> tuple[Segment, ...]:
    """Collapse consecutive same-label windows into speaker segments."""

    if not windows:
        return ()
    segments: list[Segment] = []
    cur_label = labels[0]
    cur_start = windows[0].start
    cur_end = windows[0].end
    for w, lab in zip(windows[1:], labels[1:], strict=True):
        if lab == cur_label and w.start <= cur_end + 1e-6:
            cur_end = max(cur_end, w.end)
        else:
            segments.append(Segment(speaker=f"S{cur_label}", start=cur_start, end=cur_end))
            cur_label, cur_start, cur_end = lab, w.start, w.end
    segments.append(Segment(speaker=f"S{cur_label}", start=cur_start, end=cur_end))
    return tuple(segments)
