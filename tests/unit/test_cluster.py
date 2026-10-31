from __future__ import annotations

import numpy as np

from speechinsight.diarize.cluster import ClusteringConfig, cluster_windows, labels_to_segments
from speechinsight.diarize.embed import Window


def _window(start: float, end: float, direction: tuple[float, ...]) -> Window:
    vec = np.array(direction, dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return Window(start=start, end=end, embedding=vec)


def test_three_well_separated_clusters_recovered() -> None:
    # Three groups of embeddings, each far from the others on the unit sphere.
    windows = [
        _window(0.0, 1.5, (1.0, 0.0, 0.0)),
        _window(1.5, 3.0, (0.9, 0.1, 0.0)),
        _window(3.0, 4.5, (0.0, 1.0, 0.0)),
        _window(4.5, 6.0, (0.1, 0.9, 0.0)),
        _window(6.0, 7.5, (0.0, 0.0, 1.0)),
        _window(7.5, 9.0, (0.0, 0.1, 0.9)),
    ]
    labels = cluster_windows(windows, ClusteringConfig(distance_threshold=0.3))
    # Expect three distinct clusters.
    assert len(set(labels)) == 3


def test_single_window_returns_single_cluster() -> None:
    windows = [_window(0.0, 1.5, (1.0, 0.0))]
    assert cluster_windows(windows) == [0]


def test_empty_input_returns_empty_labels() -> None:
    assert cluster_windows([]) == []


def test_labels_to_segments_merges_consecutive_same_label() -> None:
    windows = [
        _window(0.0, 1.5, (1.0,)),
        _window(0.75, 2.25, (1.0,)),  # overlapping same speaker
        _window(2.5, 4.0, (1.0,)),  # new speaker
    ]
    segs = labels_to_segments(windows, [0, 0, 1])
    assert len(segs) == 2
    assert segs[0].speaker == "S0" and segs[0].start == 0.0 and segs[0].end == 2.25
    assert segs[1].speaker == "S1" and segs[1].start == 2.5
