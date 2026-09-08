#!/usr/bin/env python3
"""Adjudicate the two visibility engines against EXACT ground truth.

The earlier head-to-head (`cross_validate.py`) pitted the two engines against
each other at voxel granularity. That was inconclusive: the range-image path
and the ray-march path use different decision rules, and
`traversal.march_free_counts` applies a range-dependent threshold (a `counts
>= 2` rule means one thing at 5 m and something else at 40 m, because beam
spacing overtakes voxel size at range). Neither engine is "truth" in that
comparison.

The synthetic plaza has exact per-point labels by construction, so here the
labels are the arbiter. We run the FULL carve twice on the SAME scene with an
identical `CarveConfig`, changing exactly one thing:

  RUN A : the production range-image path            (carve.carve, unmodified)
  RUN B : identical conflict logic, but the free-space evidence step swapped
          for traversal.march_free_mask               (carve_march, below)

Both runs are scored with metrics.evaluate / metrics.report against the
three-class ground truth (0 static, 1 dynamic, 2 mixed-pixel noise). Then, for
the points the two runs disagree on, we ask the ground truth what those points
actually are.

Nothing is tuned. carve.py, visibility.py and traversal.py are untouched.

Run:  python adjudicate.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from scan2twin import CarveConfig, CarveResult, carve, metrics, traversal
from scan2twin.grid import VoxelGrid, sorted_membership
from scan2twin.io.synthetic import build_plaza_scene


# --------------------------------------------------------------------------
# RUN B: a carve that reuses carve.py's structure but swaps the evidence step
# --------------------------------------------------------------------------
def carve_march(scene, config: CarveConfig) -> CarveResult:
    """carve.carve with phase 3's free-space evidence taken from ray marching.

    Everything else - the occupancy bookkeeping, the `min_witnesses` gate, the
    strict-majority rule, the reachability cull - is copied verbatim from
    carve.carve so the only variable is HOW "did station B see through here?"
    gets answered:

        carve.carve   ->  visibility.free_space_evidence_dilated  (range image)
        carve_march   ->  traversal.march_free_mask               (ray march)

    The range-image path answers per query POINT. The march path answers per
    VOXEL. We march each station's rays once against the occupied-voxel set,
    then a point counts as "seen through" by station j iff the voxel it lives
    in was traversed by at least `min_rays` (march_free_mask default = 2) of
    station j's rays. march_free_mask keeps its own defaults - it is the
    untouched reference and there is nothing to tune.
    """
    cfg = config
    if len(scene) < 2:
        raise ValueError("cross-station carving needs at least 2 stations")

    t = {}
    names = [s.name for s in scene]

    # --- phase 2: occupancy bookkeeping (verbatim from carve.carve) --------
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

    # --- phase 2b: per-station free-voxel evidence, via ray marching ------
    # Marched once per station against the shared occupied set and reused for
    # every ordered pair. This is the step that replaces the range images.
    t0 = time.perf_counter()
    free_vox = {}  # station name -> bool array over `occupied`
    for s in scene:
        free_vox[s.name] = traversal.march_free_mask(
            grid, s.origin, s.points, occupied
        )
    t["march_free_mask"] = time.perf_counter() - t0

    # --- phase 3: point-level conflict (verbatim logic, swapped evidence) --
    t0 = time.perf_counter()
    masks, wit_out, info = {}, {}, {}
    for i, s in enumerate(scene):
        witnesses = np.zeros(len(s), dtype=np.int16)
        slot_all = sorted_membership(point_keys[s.name], occupied)
        for j, other in enumerate(scene):
            if i == j:
                continue
            # identical reachability cull to carve.carve; there the bound is
            # info[other]["max_recorded_range"] = float(other.ranges().max())
            reach = float(other.ranges().max())
            d = np.linalg.norm(s.points - other.origin, axis=1)
            near = d <= reach
            if not near.any():
                continue
            ev = np.zeros(len(s), dtype=bool)
            ok = near & (slot_all >= 0)
            ev[ok] = free_vox[other.name][slot_all[ok]]
            witnesses[ev] += 1

        is_dyn = witnesses >= cfg.min_witnesses
        if cfg.strict_majority:
            other_occ = np.zeros(len(s), dtype=np.int16)
            ok = slot_all >= 0
            other_occ[ok] = (
                n_occ_total[slot_all[ok]] - occ[slot_all[ok], i]
            ).astype(np.int16)
            is_dyn &= witnesses >= other_occ

        masks[s.name] = is_dyn
        wit_out[s.name] = witnesses
        info[s.name] = {"removed": int(is_dyn.sum())}
    t["visibility"] = time.perf_counter() - t0

    return CarveResult(
        masks=masks,
        witnesses=wit_out,
        config=cfg,
        n_occupied_voxels=n_occ,
        timings=t,
        per_station=info,
    )


# --------------------------------------------------------------------------
def _flat(scene, masks: dict) -> np.ndarray:
    return np.concatenate([np.asarray(masks[s.name], dtype=bool) for s in scene])


def _flat_labels(scene) -> np.ndarray:
    return np.concatenate([np.asarray(s.labels) for s in scene])


def main() -> None:
    STATIC, DYNAMIC, NOISE = metrics.STATIC, metrics.DYNAMIC, metrics.NOISE

    t0 = time.perf_counter()
    scene = build_plaza_scene(az_steps=900, el_steps=180, n_walkers=12)
    print(f"[capture simulated in {time.perf_counter() - t0:.1f} s]\n")
    print(scene.describe())
    print()

    gt = _flat_labels(scene)
    n_total = len(gt)
    print(f"  ground truth composition ({n_total:,} points)")
    print(f"    static (0) : {int((gt == STATIC).sum()):>10,}")
    print(f"    dynamic(1) : {int((gt == DYNAMIC).sum()):>10,}")
    print(f"    noise  (2) : {int((gt == NOISE).sum()):>10,}")
    print()

    cfg = CarveConfig(
        voxel_size=0.12,
        margin_m=0.20,
        min_witnesses=1,
        strict_majority=True,
        neighbourhood=1,
    )
    print(f"  identical config for both runs: {cfg.describe()}")
    print("  (neighbourhood applies only to the range-image path; the march")
    print("   path uses traversal.march_free_mask with its own untouched")
    print("   defaults: min_rays=2, margin_voxels=1.5)\n")

    # --- RUN A: production range-image path -------------------------------
    print("running RUN A  (range-image path, carve.carve as-is) ...", flush=True)
    tA = time.perf_counter()
    resA = carve(scene, cfg)
    wallA = time.perf_counter() - tA
    totA, perA = metrics.evaluate(scene, resA.masks)
    print(f"  done in {wallA:.2f} s\n")

    # --- RUN B: ray-march evidence --------------------------------------
    print("running RUN B  (ray-march evidence, traversal.march_free_mask) ...",
          flush=True)
    tB = time.perf_counter()
    resB = carve_march(scene, cfg)
    wallB = time.perf_counter() - tB
    totB, perB = metrics.evaluate(scene, resB.masks)
    print(f"  done in {wallB:.2f} s\n")

    # --- full metrics.report for each ----------------------------------
    print(metrics.report(totA, perA, "RUN A  -  range-image path"))
    print()
    print(metrics.report(totB, perB, "RUN B  -  ray-march evidence"))
    print()

    # --- side-by-side table ------------------------------------------
    print("=" * 74)
    print("SIDE BY SIDE  (arbiter: exact three-class ground truth)")
    print("=" * 74)
    print(f"  {'metric':<26s} {'RUN A range-image':>18s} {'RUN B ray-march':>18s}")
    print(f"  {'-' * 26} {'-' * 18} {'-' * 18}")
    rows = [
        ("ghost recall",         f"{100 * totA.ghost_recall:.2f} %",
                                 f"{100 * totB.ghost_recall:.2f} %"),
        ("surface damage rate",  f"{100 * totA.surface_damage_rate:.4f} %",
                                 f"{100 * totB.surface_damage_rate:.4f} %"),
        ("ghost precision",      f"{100 * totA.ghost_precision:.2f} %",
                                 f"{100 * totB.ghost_precision:.2f} %"),
        ("F1 (dynamic class)",   f"{totA.f1:.4f}",
                                 f"{totB.f1:.4f}"),
        ("wall clock",           f"{wallA:.2f} s",
                                 f"{wallB:.2f} s"),
    ]
    for name, a, b in rows:
        print(f"  {name:<26s} {a:>18s} {b:>18s}")
    print()
    print(f"  supporting counts")
    print(f"    {'real surface deleted (fp)':<28s} "
          f"{totA.fp:>10,} {totB.fp:>10,}")
    print(f"    {'ghosts removed (tp)':<28s} "
          f"{totA.tp:>10,} {totB.tp:>10,}")
    print(f"    {'ghosts missed (fn)':<28s} "
          f"{totA.fn:>10,} {totB.fn:>10,}")
    print(f"    {'noise removed':<28s} "
          f"{totA.noise_removed:>10,} {totB.noise_removed:>10,}")
    print(f"    {'total points removed':<28s} "
          f"{resA.n_removed:>10,} {resB.n_removed:>10,}")
    print()

    # --- what does ground truth say about the disagreements? ----------
    mA = _flat(scene, resA.masks)
    mB = _flat(scene, resB.masks)
    dis = mA != mB
    n_dis = int(dis.sum())

    print("=" * 74)
    print("DISAGREEMENT, ADJUDICATED BY GROUND TRUTH")
    print("=" * 74)
    print(f"  points where RUN A and RUN B differ : {n_dis:,} of {n_total:,} "
          f"({100.0 * n_dis / n_total:.4f} %)")
    print()

    def breakdown(sel: np.ndarray, label: str) -> None:
        n = int(sel.sum())
        d = int((sel & (gt == DYNAMIC)).sum())
        s = int((sel & (gt == STATIC)).sum())
        z = int((sel & (gt == NOISE)).sum())
        print(f"  {label}  (n = {n:,})")
        if n:
            print(f"    truly dynamic : {d:>8,}  ({100.0 * d / n:5.1f} %)")
            print(f"    truly static  : {s:>8,}  ({100.0 * s / n:5.1f} %)")
            print(f"    truly noise   : {z:>8,}  ({100.0 * z / n:5.1f} %)")
        else:
            print("    (none)")
        print()

    breakdown(dis, "ALL disagreements")
    breakdown(mA & ~mB, "RUN A removed, RUN B kept")
    breakdown(~mA & mB, "RUN B removed, RUN A kept")

    print("  reading: a disagreement over a 'truly dynamic' point means one")
    print("  engine caught a ghost the other missed; over a 'truly static'")
    print("  point it means one engine damaged surface the other preserved;")
    print("  over 'truly noise' it is a wash for the deliverable either way.")
    print("=" * 74)


if __name__ == "__main__":
    main()
