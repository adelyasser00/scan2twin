#!/usr/bin/env python3
"""End-to-end run: synthetic plaza -> carve -> measure -> report.

Everything printed here is measured, not asserted. Where a number is an
estimate it is labelled as one in the output itself.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from scan2twin import CarveConfig, apply, carve, metrics
from scan2twin.io.synthetic import build_plaza_scene, estimate_angular_resolution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=int, default=1600)
    ap.add_argument("--el", type=int, default=300)
    ap.add_argument("--walkers", type=int, default=14)
    ap.add_argument("--voxel", type=float, default=0.12)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--nbhd", type=int, default=1)
    ap.add_argument("--witnesses", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    scene = build_plaza_scene(
        seed=args.seed, az_steps=args.az, el_steps=args.el, n_walkers=args.walkers
    )
    print(f"[capture simulated in {time.perf_counter() - t0:.1f} s]\n")
    print(scene.describe())
    print()

    ang = float(np.median([estimate_angular_resolution(s) for s in scene]))
    print(f"  estimated scan angular resolution: {ang:.3f} deg")
    print("  (range image bins are set from this, not from a hardcoded default)\n")

    def run(voxel, margin, nbhd, witnesses, verbose=True):
        cfg = CarveConfig(
            voxel_size=voxel,
            margin_m=margin,
            min_witnesses=witnesses,
            strict_majority=True,
            angular_res_deg=ang * 1.05,
            neighbourhood=nbhd,
        )
        t = time.perf_counter()
        res = carve(scene, cfg)
        wall = time.perf_counter() - t
        total, per = metrics.evaluate(scene, res.masks)
        if verbose:
            print(res.describe(scene))
            print()
            print(metrics.report(total, per, "Ghost removal, synthetic plaza"))
            print()
            print(metrics.miss_analysis(scene, res.masks))
            print()
            print(metrics.witness_analysis(scene, res.masks))
            print()
            print(f"  wall clock: {wall:.2f} s for {scene.n_points:,} points "
                  f"({scene.n_points / wall / 1e6:.2f} M pts/s)")
            print(f"  {metrics.hours_estimate(res.n_removed)}")
        return res, total, wall

    res, total, wall = run(args.voxel, args.margin, args.nbhd, args.witnesses)

    if args.sweep:
        print("\n" + "=" * 72)
        print("PARAMETER SWEEP  (the plot that goes in the case study)")
        print("=" * 72)
        print(f"{'voxel':>7s} {'margin':>7s} {'nbhd':>5s} "
              f"{'recall':>9s} {'damage':>10s} {'F1':>7s} {'sec':>6s}")
        for voxel in (0.08, 0.12, 0.20, 0.30):
            for margin in (0.10, 0.20, 0.35):
                for nbhd in (0, 1):
                    _, c, w = run(voxel, margin, nbhd, args.witnesses, verbose=False)
                    print(f"{voxel:7.2f} {margin:7.2f} {nbhd:5d} "
                          f"{100 * c.ghost_recall:8.2f}% "
                          f"{100 * c.surface_damage_rate:9.4f}% "
                          f"{c.f1:7.4f} {w:6.1f}")

    cleaned = apply(scene, res)
    print(f"\n  cleaned scene: {cleaned.n_points:,} points "
          f"(was {scene.n_points:,})")


if __name__ == "__main__":
    main()
