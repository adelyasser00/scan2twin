"""Core interchange types for scan2twin.

The single most important design decision in this package lives here.

A `Station` is a *single scanner setup*, and it carries its `origin`: the
position the laser fired from. Most survey software discards the per-station
origin the moment scans are registered and merged into one cloud. That discard
is precisely why ghost removal is still done by hand: the visibility signal
that would automate it has already been thrown away by the time a technician
opens the file.

Every dataset adapter in `scan2twin.io` must produce `Station` objects.
Nothing in `scan2twin.core` is allowed to know what a Semantic3D .txt file or
a Riegl .3d file looks like. That boundary is what makes this swappable across
datasets, and later across a client's own scanner output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass
class Station:
    """One scanner setup.

    Attributes
    ----------
    name:
        Human-readable identifier, e.g. "domfountain1" or "ScanPos003".
    points:
        (N, 3) float64 array of measured points, in the *shared registered
        frame* of the scene. Not scanner-local. If your source data is
        scanner-local, the adapter is responsible for transforming both the
        points and the origin into the shared frame together.
    origin:
        (3,) float64. The scanner position in the same shared frame.
        THIS IS NOT OPTIONAL. If you do not have it, this package cannot help
        you, and you should say so out loud rather than guessing it.
    rgb:
        Optional (N, 3) uint8 colour.
    intensity:
        Optional (N,) float32 return intensity / reflectance.
    labels:
        Optional (N,) int8 ground truth. Convention used throughout:
            0 = static
            1 = dynamic (moving object present during this scan)
           -1 = unlabelled / unknown
        Only present for benchmark datasets. Never fabricated.
    """

    name: str
    points: np.ndarray
    origin: np.ndarray
    rgb: np.ndarray | None = None
    intensity: np.ndarray | None = None
    labels: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"{self.name}: points must be (N, 3), got {self.points.shape}"
            )
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)

        n = len(self.points)
        for attr in ("rgb", "intensity", "labels"):
            arr = getattr(self, attr)
            if arr is not None:
                arr = np.asarray(arr)
                if len(arr) != n:
                    raise ValueError(
                        f"{self.name}: {attr} has {len(arr)} rows, points has {n}"
                    )
                setattr(self, attr, arr)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def has_ground_truth(self) -> bool:
        return self.labels is not None

    def ranges(self) -> np.ndarray:
        """Distance from scanner origin to each point. Metres."""
        return np.linalg.norm(self.points - self.origin, axis=1)

    def subset(self, mask: np.ndarray) -> "Station":
        """Return a new Station keeping only points where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        return Station(
            name=self.name,
            points=self.points[mask],
            origin=self.origin.copy(),
            rgb=None if self.rgb is None else self.rgb[mask],
            intensity=None if self.intensity is None else self.intensity[mask],
            labels=None if self.labels is None else self.labels[mask],
            meta=dict(self.meta),
        )

    def describe(self) -> str:
        r = self.ranges()
        gt = ""
        if self.has_ground_truth:
            n_dyn = int((self.labels == 1).sum())
            gt = f"  gt_dynamic={n_dyn:,} ({100.0 * n_dyn / max(len(self), 1):.2f}%)"
        return (
            f"{self.name:<24s} n={len(self):>10,}  "
            f"range {r.min():6.2f}-{r.max():7.2f} m  "
            f"origin=({self.origin[0]:.1f}, {self.origin[1]:.1f}, {self.origin[2]:.1f})"
            f"{gt}"
        )


@dataclass
class Scene:
    """A set of registered stations covering one site.

    Carving needs at least two stations with meaningful overlap. A Scene with
    one station is legal to construct (you may still want SOR on it) but
    `carve` will refuse it, loudly.
    """

    name: str
    stations: list[Station]
    meta: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator[Station]:
        return iter(self.stations)

    def __len__(self) -> int:
        return len(self.stations)

    @property
    def n_points(self) -> int:
        return sum(len(s) for s in self.stations)

    @property
    def has_ground_truth(self) -> bool:
        return all(s.has_ground_truth for s in self.stations)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds over all points AND all scanner origins.

        Origins are included deliberately: rays start there, and the grid has
        to contain the rays, not just the returns.
        """
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for s in self.stations:
            lo = np.minimum(lo, s.points.min(axis=0))
            hi = np.maximum(hi, s.points.max(axis=0))
            lo = np.minimum(lo, s.origin)
            hi = np.maximum(hi, s.origin)
        return lo, hi

    def describe(self) -> str:
        lo, hi = self.bounds()
        extent = hi - lo
        lines = [
            f"Scene: {self.name}",
            f"  stations : {len(self.stations)}",
            f"  points   : {self.n_points:,}",
            f"  extent   : {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m",
            f"  ground truth: {'yes' if self.has_ground_truth else 'no'}",
            "",
        ]
        lines += ["  " + s.describe() for s in self.stations]
        return "\n".join(lines)


def concat_labels(stations: Sequence[Station]) -> np.ndarray:
    """Concatenate ground truth across stations, for scene-level metrics."""
    out = []
    for s in stations:
        if s.labels is None:
            out.append(np.full(len(s), -1, dtype=np.int8))
        else:
            out.append(s.labels.astype(np.int8))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int8)
