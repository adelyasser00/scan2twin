"""Reference implementation: explicit ray marching through the voxel grid.

This is the slow, obvious, hard-to-get-wrong version. It exists so that the
fast range-image path in `visibility.py` has something to be checked against.
If the two disagree by more than a small margin on a scene, the range image's
angular resolution is wrong, and the case study should say so rather than
quietly shipping whichever number looks better.

Marching is done by fixed-step sampling at half a voxel rather than exact
Amanatides-Woo DDA. At a half-voxel step the probability of skipping a voxel
the ray genuinely passes through is small, and we accumulate *evidence counts*
with a threshold, so an occasional skipped cell changes nothing. The honest
cost of this shortcut is that it is an approximation, and it is written down
here rather than buried.
"""

from __future__ import annotations

import numpy as np

from .grid import VoxelGrid, sorted_membership


def march_free_counts(
    grid: VoxelGrid,
    origin: np.ndarray,
    endpoints: np.ndarray,
    target_keys: np.ndarray,
    margin_voxels: float = 1.5,
    step_frac: float = 0.5,
    ray_chunk: int = 20_000,
    max_steps_cap: int = 6000,
) -> np.ndarray:
    """Count how many of this station's rays passed through each target voxel.

    Parameters
    ----------
    target_keys:
        SORTED array of packed voxel keys we care about. Restricting
        accumulation to this set is the memory trick that makes this run in a
        few hundred MB instead of tens of GB: we only ever care about voxels
        some station found occupied, and that set is tiny next to the free
        volume of a site.
    margin_voxels:
        Stop the ray this many voxels short of its own endpoint. Without it,
        every ray carves the surface it terminated on.

    Returns
    -------
    (len(target_keys),) int32 counts.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    target_keys = np.asarray(target_keys, dtype=np.int64)
    counts = np.zeros(len(target_keys), dtype=np.int32)
    if len(target_keys) == 0 or len(endpoints) == 0:
        return counts

    step = grid.size * step_frac
    stop_back = grid.size * margin_voxels

    for lo in range(0, len(endpoints), ray_chunk):
        ep = endpoints[lo : lo + ray_chunk]
        d = ep - origin
        L = np.linalg.norm(d, axis=1)
        good = L > (stop_back + step)
        if not good.any():
            continue
        d = d[good]
        L = L[good]
        u = d / L[:, None]
        stop = L - stop_back

        n_steps = int(min(np.ceil(stop.max() / step), max_steps_cap))
        for s in range(1, n_steps + 1):
            t = s * step
            active = stop > t
            if not active.any():
                break
            pos = origin + u[active] * t
            keys = grid.key_of(pos)
            slot = sorted_membership(keys, target_keys)
            hit = slot >= 0
            if hit.any():
                np.add.at(counts, slot[hit], 1)
    return counts


def march_free_mask(
    grid: VoxelGrid,
    origin: np.ndarray,
    endpoints: np.ndarray,
    target_keys: np.ndarray,
    min_rays: int = 2,
    **kw,
) -> np.ndarray:
    """Boolean per target voxel: was it traversed by at least `min_rays` rays."""
    return march_free_counts(grid, origin, endpoints, target_keys, **kw) >= min_rays
