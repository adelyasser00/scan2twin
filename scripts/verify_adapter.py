#!/usr/bin/env python3
"""verify_adapter.py -- exercise the lecturehall adapter on a real subset.

Loads both stations at 2 M points each, prints the scene, checks the scanner
separation, and re-runs the overlap nearest-neighbour registration check that
was used to validate the pose-2 transform in the first place. That last step
is a regression test on the adapter: if a future edit breaks the transform,
the overlap median jumps off ~0.05 m and this script says so.

Does NOT carve. It only confirms the data comes in correctly.
"""

from __future__ import annotations

import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan2twin.io.readers import load_lecturehall

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "lecturehall")
OUT_DIR = os.path.join(HERE, "out")

MAX_POINTS = 2_000_000
EXPECTED_SEPARATION_M = 6.71
SEPARATION_TOL_M = 0.2
OVERLAP_RADIUS_M = 0.5
NN_SAMPLE = 300_000
EXPECTED_OVERLAP_MEDIAN_M = 0.05
RNG = np.random.default_rng(42)


def subsample(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[RNG.choice(len(a), n, replace=False)]


# ---------------------------------------------------------------------------
print("=" * 78)
print(f"Loading lecturehall, max_points={MAX_POINTS:,} per station")
print("=" * 78)
t0 = time.perf_counter()
scene = load_lecturehall(DATA_DIR, max_points=MAX_POINTS)
print(f"  loaded in {time.perf_counter() - t0:.1f} s\n")

# ---------------------------------------------------------------------------
print("=" * 78)
print("scene.describe()")
print("=" * 78)
print(scene.describe())
print()

# ---------------------------------------------------------------------------
print("=" * 78)
print("Scanner origins and separation")
print("=" * 78)
s1, s2 = scene.stations
for s in scene.stations:
    o = s.origin
    print(f"  {s.name:<22s} origin = ({o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}) m")
separation = float(np.linalg.norm(s2.origin - s1.origin))
ok_sep = abs(separation - EXPECTED_SEPARATION_M) <= SEPARATION_TOL_M
print(f"\n  separation                 : {separation:.3f} m")
print(f"  expected                    : ~{EXPECTED_SEPARATION_M} m "
      f"(+/- {SEPARATION_TOL_M} m)")
print(f"  check                       : {'PASS' if ok_sep else 'FAIL'}")
print()

# ---------------------------------------------------------------------------
print("=" * 78)
print("Regression check: overlap nearest-neighbour registration")
print("=" * 78)
print(f"  Stations are already in the shared frame (adapter applied both poses).")
print(f"  Subsample {NN_SAMPLE:,} pts/station, build KD-tree on {s1.name},")
print(f"  gate {s2.name} points to those within {OVERLAP_RADIUS_M} m, report median NN.\n")

from scipy.spatial import cKDTree

p1 = subsample(s1.points, NN_SAMPLE)
p2 = subsample(s2.points, NN_SAMPLE)
tree = cKDTree(p1)
d, _ = tree.query(p2, workers=-1)
gate = d <= OVERLAP_RADIUS_M
d_overlap = d[gate]

median_nn = float(np.median(d_overlap))
p25_nn = float(np.percentile(d_overlap, 25))
ok_nn = median_nn <= 0.08  # comfortably above the validated 0.046 m, well below identity's 0.229 m

print(f"  overlap gate survivors      : {gate.sum():,} of {len(d):,} "
      f"({100 * gate.mean():.1f} %)")
print(f"  overlap median NN           : {median_nn:.4f} m   "
      f"(validated value ~0.046 m)")
print(f"  overlap 25th pct NN         : {p25_nn:.4f} m   (validated ~0.030 m)")
print(f"  check (<= 0.08 m)           : {'PASS' if ok_nn else 'FAIL'}")
print()

# ---------------------------------------------------------------------------
print("=" * 78)
print("Figure: out/fig9_adapter_check.png")
print("=" * 78)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(OUT_DIR, exist_ok=True)

plot1 = subsample(s1.points, 200_000)
plot2 = subsample(s2.points, 200_000)
dyn1 = s1.points[s1.labels == 1]
plot_dyn = subsample(dyn1, 60_000)

fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(plot1[:, 0], plot1[:, 1], s=1.0, c="0.62", alpha=0.30, linewidths=0,
           label=f"{s1.name} static ({len(plot1):,} shown)")
ax.scatter(plot2[:, 0], plot2[:, 1], s=1.0, c="tab:blue", alpha=0.30, linewidths=0,
           label=f"{s2.name} ({len(plot2):,} shown)")
ax.scatter(plot_dyn[:, 0], plot_dyn[:, 1], s=1.5, c="red", alpha=0.55, linewidths=0,
           label=f"{s1.name} dynamic ({len(plot_dyn):,} of {len(dyn1):,} shown)")
for s, mk in ((s1, "^"), (s2, "v")):
    ax.plot(s.origin[0], s.origin[1], mk, ms=13, mfc="yellow", mec="k", mew=1.5)
    ax.annotate(f"  {s.name} scanner", (s.origin[0], s.origin[1]),
                fontsize=8, va="center")
ax.set_aspect("equal")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title(
    "Lecturehall adapter check -- plan view, metres, shared frame\n"
    f"separation {separation:.2f} m, overlap median NN {median_nn:.3f} m")
ax.grid(True, alpha=0.3)
leg = ax.legend(loc="upper right", framealpha=0.9, markerscale=5)
for h in leg.legend_handles:
    try:
        h.set_alpha(1.0)
    except Exception:
        pass

out_path = os.path.join(OUT_DIR, "fig9_adapter_check.png")
fig.savefig(out_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  saved {out_path}")
print()

# ---------------------------------------------------------------------------
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  {s1.name}: {len(s1):,} pts, {int((s1.labels == 1).sum()):,} dynamic, "
      f"{int((s1.labels == 0).sum()):,} static")
print(f"  {s2.name}: {len(s2):,} pts, {int((s2.labels == 1).sum()):,} dynamic, "
      f"{int((s2.labels == 0).sum()):,} static")
print(f"  label values present        : "
      f"{sorted(set(np.unique(s1.labels).tolist()) | set(np.unique(s2.labels).tolist()))}")
print(f"  scanner separation          : {separation:.3f} m  "
      f"[{'PASS' if ok_sep else 'FAIL'}]")
print(f"  overlap median NN           : {median_nn:.4f} m  "
      f"[{'PASS' if ok_nn else 'FAIL'}]")
print(f"  figure                      : {out_path}")

all_ok = ok_sep and ok_nn
print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
sys.exit(0 if all_ok else 1)
