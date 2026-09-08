"""Statistical outlier removal.

This is commodity. CloudCompare has it, PDAL has it, Open3D has it, every
scanner vendor ships a version of it. It is included because it is a sensible
pre-pass and because the case study should show the whole pipeline, not
because it differentiates anything.

Label it as commodity in the write-up. Claiming credit for SOR in front of a
surveyor is the fastest way to lose the room.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def statistical_outlier_mask(
    points: np.ndarray,
    k: int = 8,
    std_ratio: float = 2.0,
    chunk: int = 200_000,
) -> np.ndarray:
    """True where the point is an outlier.

    Mean distance to k nearest neighbours, thresholded at
    global_mean + std_ratio * global_std.
    """
    n = len(points)
    if n <= k + 1:
        return np.zeros(n, dtype=bool)

    tree = cKDTree(points)
    mean_d = np.empty(n, dtype=np.float64)
    for lo in range(0, n, chunk):
        d, _ = tree.query(points[lo : lo + chunk], k=k + 1, workers=-1)
        mean_d[lo : lo + chunk] = d[:, 1:].mean(axis=1)  # drop self

    thresh = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d > thresh


def radius_outlier_mask(
    points: np.ndarray, radius: float = 0.20, min_neighbours: int = 4
) -> np.ndarray:
    """True where the point has fewer than `min_neighbours` inside `radius`.

    Cheaper and more predictable than the statistical version at survey
    densities, because the threshold is an absolute distance you can defend
    to a client rather than a scene-dependent statistic.
    """
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True, workers=-1)
    return counts < (min_neighbours + 1)  # +1 for self
