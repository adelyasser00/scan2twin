"""Sparse voxel grid.

Deliberately *not* a dense 3D array. A 100 x 100 x 30 m plaza at 10 cm voxels
is 3e8 cells; at one byte each that is 300 MB of mostly empty air, and survey
sites get a lot bigger than a plaza. We hash (i, j, k) to a single int64 key
and only ever materialise cells that something actually touched.

The key packing below fits each axis index into 21 bits (0 .. 2,097,151).
At 10 cm voxels that is a 209 km span per axis, which is more than any
terrestrial scan job will ever need. The packing asserts rather than silently
wrapping, because a silent wrap here would produce plausible-looking garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BITS = 21
_MAX = (1 << _BITS) - 1


@dataclass
class VoxelGrid:
    origin: np.ndarray  # (3,) world coords of voxel (0,0,0) corner
    size: float  # edge length, metres

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        if self.size <= 0:
            raise ValueError("voxel size must be positive")

    @classmethod
    def covering(
        cls, lo: np.ndarray, hi: np.ndarray, size: float, pad: float = 2.0
    ) -> "VoxelGrid":
        """Grid whose index space covers [lo, hi] with a pad in metres."""
        lo = np.asarray(lo, dtype=np.float64) - pad
        hi = np.asarray(hi, dtype=np.float64) + pad
        span = hi - lo
        n = np.ceil(span / size).astype(np.int64)
        if (n > _MAX).any():
            raise ValueError(
                f"scene needs {n.max():,} voxels on one axis at size={size} m, "
                f"which exceeds the {_MAX:,} the key packing supports. "
                f"Increase voxel size or tile the scene."
            )
        return cls(origin=lo, size=size)

    def index(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 3) int64 voxel indices. Floor, not round."""
        return np.floor((pts - self.origin) / self.size).astype(np.int64)

    def key(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) int64 indices -> (N,) int64 packed keys."""
        if idx.size and (idx.min() < 0 or idx.max() > _MAX):
            raise ValueError(
                "voxel index out of packing range; grid origin is wrong or "
                "a point lies outside the grid the scene was built for"
            )
        return (idx[:, 0] << (2 * _BITS)) | (idx[:, 1] << _BITS) | idx[:, 2]

    def key_of(self, pts: np.ndarray) -> np.ndarray:
        return self.key(self.index(pts))

    def unkey(self, keys: np.ndarray) -> np.ndarray:
        """(N,) packed keys -> (N, 3) int64 indices."""
        keys = np.asarray(keys, dtype=np.int64)
        i = (keys >> (2 * _BITS)) & _MAX
        j = (keys >> _BITS) & _MAX
        k = keys & _MAX
        return np.stack([i, j, k], axis=1)

    def centre(self, keys: np.ndarray) -> np.ndarray:
        """(N,) packed keys -> (N, 3) world coords of voxel centres."""
        idx = self.unkey(keys)
        return self.origin + (idx.astype(np.float64) + 0.5) * self.size

    def corner(self, keys: np.ndarray) -> np.ndarray:
        idx = self.unkey(keys)
        return self.origin + idx.astype(np.float64) * self.size


def sorted_membership(query: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Positions of `query` keys inside a *sorted* `table`, -1 where absent.

    Vectorised set lookup without a Python dict. This is the hot path: during
    ray traversal we test millions of visited voxels for membership in the
    occupied set, and a dict lookup per voxel would dominate the runtime.
    """
    if table.size == 0:
        return np.full(len(query), -1, dtype=np.int64)
    pos = np.searchsorted(table, query)
    pos_clipped = np.minimum(pos, len(table) - 1)
    hit = table[pos_clipped] == query
    return np.where(hit, pos_clipped, -1)
