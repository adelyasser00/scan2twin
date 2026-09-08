#!/usr/bin/env python3
"""Cross-validate the two visibility engines against each other.

There are two independent implementations of the same question in this repo:

  FAST  (scan2twin.visibility)   spherical range image + angular lookup
  SLOW  (scan2twin.traversal)    explicit fixed-step ray marching

They are supposed to answer "did station B see through this location?" the same
way. carve.py only ever runs FAST; traversal.py exists solely as the reference.
This script pits them head to head, at VOXEL granularity, on the synthetic
plaza, for a single ordered station pair.

Nothing here is tuned to make the two agree. The one free choice -- the range
image's angular bin size, which the task does not pin -- is set the way the
production pipeline in carve.py sets it: estimated from the data and widened by
5%. That value is printed below. If the engines disagree, the disagreement is
the result.

Run:  python cross_validate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scan2twin import VoxelGrid, traversal
from scan2twin.io.synthetic import build_plaza_scene, estimate_angular_resolution
from scan2twin.visibility import build_range_image, free_space_evidence_dilated

VOXEL_SIZE = 0.12
MARGIN_M = 0.20
NEIGHBOURHOOD = 1
SLOW_MIN_RAYS = 2

INK = "#14181d"
GREY = "#c2c8d0"
RED = "#d1344a"
BLUE = "#2f6fb5"


def voxel_normal_pca(pts: np.ndarray) -> np.ndarray | None:
    """Smallest-eigenvector normal from a voxel's points. None if too few."""
    if len(pts) < 4:
        return None
    c = pts - pts.mean(axis=0)
    cov = c.T @ c
    w, v = np.linalg.eigh(cov)
    return v[:, 0]


def main() -> None:
    os.makedirs("out", exist_ok=True)

    # --- 1. synthetic scene ------------------------------------------------
    scene = build_plaza_scene(az_steps=500, el_steps=120, n_walkers=10)
    print(scene.describe())
    print()

    # --- 2. one ordered station pair -------------------------------------
    # A is the station whose points we interrogate; B is the witness whose
    # free space we test A's occupied voxels against.
    A = scene.stations[0]
    B = scene.stations[1]
    print(f"ordered pair:  A = {A.name}   B = {B.name}")
    print(f"  A: {len(A):,} points   origin {tuple(round(float(x), 2) for x in A.origin)}")
    print(f"  B: {len(B):,} points   origin {tuple(round(float(x), 2) for x in B.origin)}")
    print()

    # --- 3. shared grid + the target voxel set ---------------------------
    lo, hi = scene.bounds()
    grid = VoxelGrid.covering(lo, hi, VOXEL_SIZE)

    keys_A = grid.key_of(A.points)                 # per A-point voxel key
    target_keys, inv = np.unique(keys_A, return_inverse=True)  # sorted unique
    n_targets = len(target_keys)
    print(f"target voxels (A's occupied set): {n_targets:,}")

    # --- 3a. FAST engine ------------------------------------------------
    ang = estimate_angular_resolution(B) * 1.05
    print(f"  range-image angular bin: {ang:.4f} deg  "
          f"(estimated from B, x1.05, same rule as carve.py)")
    ri_B = build_range_image(B.points, B.origin, angular_res_deg=ang)
    print(f"  range-image fill ratio: {ri_B.fill_ratio:.3f}")

    free_pt = free_space_evidence_dilated(
        ri_B, A.points, margin_m=MARGIN_M, neighbourhood=NEIGHBOURHOOD
    )
    # aggregate per voxel: free if a strict majority of A's points there are free
    free_count = np.zeros(n_targets, dtype=np.int64)
    total_count = np.zeros(n_targets, dtype=np.int64)
    np.add.at(free_count, inv, free_pt.astype(np.int64))
    np.add.at(total_count, inv, 1)
    fast_free = (free_count * 2) > total_count

    # --- 3b. SLOW engine ----------------------------------------------
    slow_counts = traversal.march_free_counts(
        grid, B.origin, B.points, target_keys
    )
    slow_free = slow_counts >= SLOW_MIN_RAYS

    # --- 4. reporting --------------------------------------------------
    disagree = fast_free != slow_free
    n_dis = int(disagree.sum())
    fast_yes_slow_no = int((fast_free & ~slow_free).sum())
    slow_yes_fast_no = int((~fast_free & slow_free).sum())

    print()
    print("=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  target voxels                : {n_targets:,}")
    print(f"  FAST says free               : {int(fast_free.sum()):,} "
          f"({100.0 * fast_free.mean():.2f}%)")
    print(f"  SLOW says free               : {int(slow_free.sum()):,} "
          f"({100.0 * slow_free.mean():.2f}%)")
    print(f"  disagreements                : {n_dis:,} "
          f"({100.0 * n_dis / max(n_targets, 1):.3f}%)")
    print(f"    FAST free / SLOW not-free  : {fast_yes_slow_no:,}")
    print(f"    SLOW free / FAST not-free  : {slow_yes_fast_no:,}")

    centres = grid.centre(target_keys)
    d_from_B = np.linalg.norm(centres - B.origin, axis=1)

    if n_dis:
        print(f"  disagreeing voxels, median distance from B.origin : "
              f"{np.median(d_from_B[disagree]):.2f} m")
        print(f"    (all target voxels, for comparison             : "
              f"{np.median(d_from_B):.2f} m)")

        # cheap angle of incidence: PCA normal from A's points in the
        # disagreeing voxel plus its 26 neighbours (a single voxel at range
        # rarely holds enough points), vs. the B->voxel view direction.
        s_order = np.argsort(keys_A, kind="stable")
        keys_A_sorted = keys_A[s_order]
        idx3 = grid.unkey(target_keys)  # (n_targets, 3) voxel indices
        offs = np.array([[dx, dy, dz]
                         for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1)
                         for dz in (-1, 0, 1)], dtype=np.int64)
        dir_B = centres - B.origin
        dir_B /= np.linalg.norm(dir_B, axis=1, keepdims=True)

        inc = []
        for vi in np.flatnonzero(disagree):
            nbr_idx = idx3[vi] + offs
            nbr_keys = grid.key(nbr_idx)
            lo_i = np.searchsorted(keys_A_sorted, nbr_keys, side="left")
            hi_i = np.searchsorted(keys_A_sorted, nbr_keys, side="right")
            pidx = np.concatenate([s_order[a:b] for a, b in zip(lo_i, hi_i)])
            nrm = voxel_normal_pca(A.points[pidx])
            if nrm is None:
                continue
            ca = abs(float(np.dot(dir_B[vi], nrm)))
            inc.append(np.degrees(np.arccos(np.clip(ca, 0.0, 1.0))))
        if inc:
            print(f"  disagreeing voxels, median angle of incidence     : "
                  f"{np.median(inc):.1f} deg  "
                  f"(PCA normal over voxel+26 nbrs, {len(inc)}/{n_dis} voxels; "
                  f"0 deg = B along surface normal, 90 deg = grazing)")
        else:
            print("  angle of incidence: too few points per voxel to estimate")
    else:
        print("  no disagreements.")

    # --- 5. plan-view scatter ---------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ag = ~disagree
    ax.scatter(centres[ag, 0], centres[ag, 1], s=3, c=GREY, lw=0,
               rasterized=True, label=f"agree ({int(ag.sum()):,})")
    ax.scatter(centres[disagree, 0], centres[disagree, 1], s=9, c=RED, lw=0,
               label=f"disagree ({n_dis:,})")
    ax.plot(*B.origin[:2], "^", color=BLUE, ms=12, mec="white", mew=1.5,
            zorder=6, label=f"B origin ({B.name})")
    ax.plot(*A.origin[:2], "o", color=INK, ms=8, mec="white", mew=1.3,
            zorder=6, label=f"A origin ({A.name})")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Engine cross-validation, plan view\n"
        f"{A.name} occupied voxels, is-{B.name}-free?  "
        f"FAST vs SLOW  |  disagreement {100.0 * n_dis / max(n_targets, 1):.2f}%",
        fontsize=11, loc="left", weight="bold",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.14)
    fig.tight_layout()
    out = "out/fig5_crossval.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
