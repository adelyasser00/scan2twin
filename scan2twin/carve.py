"""Cross-station occupancy conflict: the ghost detector.

THE RULE
--------
A point that station A measured, and that station B demonstrably saw straight
through, was there when A scanned and gone when B scanned. Whatever it was,
it moved. That is a ghost.

No temporal data, no tracking, no trained model. Two or more registered
stations and their scanner origins. That is the whole dependency list, and it
is why this is an engineering problem rather than a research problem.

WHY THE QUERY IS AT POINTS, NOT VOXEL CENTRES
---------------------------------------------
The first version of this module tested visibility at the centre of each
occupied voxel. It worked at most voxel sizes and then destroyed 2.4% of the
scene at exactly one of them, with 99% of the damage on the ground plane.

The cause: a voxel centre sits up to half a voxel away from the surface the
points actually lie on. At normal incidence that offset costs you a few
centimetres of apparent range and the margin absorbs it. At GRAZING incidence
it does not. A scanner 1.6 m high looking at the ground 40 m away has a
depression angle of 2.3 degrees, and a 7 cm vertical offset becomes

    0.07 / sin(2.3 deg) = 1.75 m

of apparent range error. The voxel centre appears to float nearly two metres
in front of the ground, gets scored as free space, and the whole far field of
the floor is deleted. Whether this fires at all depends on where the grid
happens to land relative to z=0, which is why it looked like random
non-monotonic behaviour across voxel sizes rather than a bug.

The fix is not a bigger margin. A margin large enough to cover grazing
incidence at 40 m would be large enough to hide a person at 5 m. The fix is
to stop introducing the offset: query at the measured points, which lie on
the surface by construction.

Voxels are still used, for the occupancy bookkeeping behind the
strict-majority rule and for visualising the mechanism. They are no longer
used as query positions.

WHAT THIS CANNOT DO, STATED UP FRONT
------------------------------------
1. A ghost no second station ever had a sightline through stays. That is a
   capture geometry limit, not a code limit, and `metrics.witness_analysis`
   measures exactly how much of the residue is in this category.
2. Vegetation is not removed. A tree is static. It is in the way, but it was
   there. Removing it is a classification problem and is out of scope here
   on purpose.
3. Two people standing in the same spot across two scans defend each other
   from deletion. Correctly, as far as this method can see.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .grid import VoxelGrid, sorted_membership
from .types import Scene
from .visibility import (
    build_range_image,
    free_space_evidence,
    free_space_evidence_dilated,
)


@dataclass
class CarveConfig:
    voxel_size: float = 0.12  # metres, bookkeeping only, not a query position
    margin_m: float = 0.20  # free-space depth margin; absorbs registration error
    min_witnesses: int = 1  # stations that must see through before deleting
    strict_majority: bool = True  # free witnesses must not be outvoted by occupiers
    angular_res_deg: float | None = None  # None -> estimate from the data
    neighbourhood: int = 1  # angular dilation; 0 disables and is DANGEROUS
    max_range: float | None = None

    def describe(self) -> str:
        ang = "auto" if self.angular_res_deg is None else f"{self.angular_res_deg:.3f}"
        return (
            f"voxel={self.voxel_size} m  margin={self.margin_m} m  "
            f"min_witnesses={self.min_witnesses}  strict={self.strict_majority}  "
            f"ang_res={ang} deg  nbhd={self.neighbourhood}"
        )


@dataclass
class CarveResult:
    masks: dict  # station name -> bool array, True = ghost
    witnesses: dict  # station name -> int array, how many stations saw through
    config: CarveConfig
    n_occupied_voxels: int
    timings: dict = field(default_factory=dict)
    per_station: dict = field(default_factory=dict)

    @property
    def n_removed(self) -> int:
        return int(sum(m.sum() for m in self.masks.values()))

    def describe(self, scene: Scene) -> str:
        total = scene.n_points
        L = [
            "Carve result",
            f"  config          : {self.config.describe()}",
            f"  occupied voxels : {self.n_occupied_voxels:,}",
            f"  points removed  : {self.n_removed:,} of {total:,} "
            f"({100.0 * self.n_removed / max(total, 1):.2f}%)",
            "",
        ]
        for name, m in self.masks.items():
            i = self.per_station.get(name, {})
            L.append(
                f"    {name:<20s} removed {int(m.sum()):>8,} / {len(m):>9,}"
                f"  ({100.0 * m.sum() / max(len(m), 1):5.2f}%)"
                f"  img_fill={i.get('fill_ratio', float('nan')):.3f}"
                f"  ang={i.get('angular_res_deg', float('nan')):.3f}deg"
            )
        L.append("")
        for k, v in self.timings.items():
            L.append(f"  t[{k}] = {v:.2f} s")
        return "\n".join(L)


def estimate_angular_resolution(points: np.ndarray, origin: np.ndarray) -> float:
    """Estimate a station's angular sampling in degrees, from the data alone.

    Real scan files often do not record the resolution they were shot at, and
    the range image is worthless if its bins are far from it. Assume a roughly
    regular angular grid and invert the point count.
    """
    d = points - origin
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    el = np.arcsin(np.clip(d[:, 2] / r, -1, 1))
    span = float(el.max() - el.min())
    if len(points) < 100 or span <= 0:
        return 0.1
    return float(np.degrees(np.sqrt(2 * np.pi * span / len(points))))


def carve(scene: Scene, config: CarveConfig | None = None) -> CarveResult:
    cfg = config or CarveConfig()
    if len(scene) < 2:
        raise ValueError(
            f"scene '{scene.name}' has {len(scene)} station(s). Cross-station "
            "carving needs at least 2. There is no way around this: with one "
            "viewpoint there is no evidence that anything moved."
        )
    if cfg.neighbourhood == 0:
        import warnings

        warnings.warn(
            "neighbourhood=0 disables the angular agreement check. Measured on "
            "the synthetic benchmark this raises surface damage from ~0.002% to "
            "between 9% and 50% while ghost recall barely moves. Recall will "
            "look better and the deliverable will be destroyed. Only use this "
            "to reproduce that finding.",
            RuntimeWarning,
            stacklevel=2,
        )

    t = {}
    names = [s.name for s in scene]

    # --- phase 1: range images, one per station ---------------------------
    t0 = time.perf_counter()
    ris, info = {}, {}
    for s in scene:
        ang = cfg.angular_res_deg
        if ang is None:
            ang = estimate_angular_resolution(s.points, s.origin) * 1.05
        ri = build_range_image(s.points, s.origin, angular_res_deg=ang)
        ris[s.name] = ri
        info[s.name] = {
            "fill_ratio": ri.fill_ratio,
            "angular_res_deg": ang,
            "max_recorded_range": float(s.ranges().max()),
        }
    t["range_images"] = time.perf_counter() - t0

    # --- phase 2: occupancy bookkeeping (for the majority rule only) ------
    t0 = time.perf_counter()
    lo, hi = scene.bounds()
    grid = VoxelGrid.covering(lo, hi, cfg.voxel_size)
    point_keys = {s.name: grid.key_of(s.points) for s in scene}
    station_keys = {n: np.unique(k) for n, k in point_keys.items()}
    occupied = np.unique(np.concatenate(list(station_keys.values())))
    n_occ = len(occupied)

    occ = np.zeros((n_occ, len(names)), dtype=bool)
    for i, n in enumerate(names):
        slot = sorted_membership(station_keys[n], occupied)
        occ[slot[slot >= 0], i] = True
    n_occ_total = occ.sum(axis=1)
    t["occupancy"] = time.perf_counter() - t0

    # --- phase 3: point-level visibility, every station against every other
    t0 = time.perf_counter()
    masks, wit_out = {}, {}
    for i, s in enumerate(scene):
        witnesses = np.zeros(len(s), dtype=np.int16)
        for j, other in enumerate(scene):
            if i == j:
                continue
            # cheap cull: a station cannot testify about anything it could
            # never have reached. Matters at 100+ stations, where the naive
            # all-pairs loop is the thing that stops scaling.
            reach = info[other.name]["max_recorded_range"]
            d = np.linalg.norm(s.points - other.origin, axis=1)
            near = d <= reach
            if not near.any():
                continue
            q = s.points[near]
            if cfg.neighbourhood > 0:
                ev = free_space_evidence_dilated(
                    ris[other.name], q, cfg.margin_m,
                    cfg.neighbourhood, cfg.max_range,
                )
            else:
                ev = free_space_evidence(
                    ris[other.name], q, cfg.margin_m, cfg.max_range
                )
            witnesses[np.flatnonzero(near)[ev]] += 1

        is_dyn = witnesses >= cfg.min_witnesses
        if cfg.strict_majority:
            slot = sorted_membership(point_keys[s.name], occupied)
            other_occ = np.zeros(len(s), dtype=np.int16)
            ok = slot >= 0
            other_occ[ok] = (n_occ_total[slot[ok]] - occ[slot[ok], i]).astype(np.int16)
            is_dyn &= witnesses >= other_occ

        masks[s.name] = is_dyn
        wit_out[s.name] = witnesses
        info[s.name]["removed"] = int(is_dyn.sum())
    t["visibility"] = time.perf_counter() - t0

    return CarveResult(
        masks=masks,
        witnesses=wit_out,
        config=cfg,
        n_occupied_voxels=n_occ,
        timings=t,
        per_station=info,
    )


def apply(scene: Scene, result: CarveResult) -> Scene:
    """Return a new Scene with flagged points removed."""
    return Scene(
        name=scene.name + "_carved",
        stations=[s.subset(~result.masks[s.name]) for s in scene],
        meta={**scene.meta, "carve_config": result.config.describe()},
    )
