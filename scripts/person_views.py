#!/usr/bin/env python3
"""person_views.py -- render the lecturehall ground-truth person.

Loads station 1 of the lecturehall dataset, keeps only the dynamic-labelled
points (label == 1), and tries to render a recognisable human silhouette.

No tuning, no analysis beyond what is asked for. Package files untouched.
"""

from __future__ import annotations

import os
import sys

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
MAX_POINTS = 4_000_000

EPS = 0.15
MIN_SAMPLES = 30


def dbscan(P: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Plain DBSCAN via a KD-tree. Returns integer labels, -1 for noise."""
    from scipy.spatial import cKDTree

    n = len(P)
    tree = cKDTree(P)
    neighbours = tree.query_ball_point(P, r=eps, workers=-1)
    n_nb = np.fromiter((len(x) for x in neighbours), dtype=np.int64, count=n)
    is_core = n_nb >= min_samples  # neighbour count includes the point itself

    labels = np.full(n, -1, dtype=np.int64)
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        # breadth-first expansion from this core point
        labels[i] = cluster
        stack = [i]
        while stack:
            j = stack.pop()
            for k in neighbours[j]:
                if labels[k] == -1:
                    labels[k] = cluster
                    if is_core[k]:
                        stack.append(k)
                elif labels[k] == -2:  # previously visited as noise border
                    labels[k] = cluster
        cluster += 1
    return labels


def extent(P: np.ndarray) -> np.ndarray:
    return P.max(axis=0) - P.min(axis=0)


# ---------------------------------------------------------------------------
print("=" * 78)
print(f"Loading lecturehall station 1, max_points={MAX_POINTS:,}")
print("=" * 78)
scene = load_lecturehall(DATA_DIR, max_points=MAX_POINTS)
s1 = scene.stations[0]
person = s1.points[s1.labels == 1]
print(f"  station 1 total points : {len(s1):,}")
print(f"  dynamic (label == 1)   : {len(person):,}")
print()

# ---------------------------------------------------------------------------
# 1. bounding box
# ---------------------------------------------------------------------------
print("=" * 78)
print("1. Bounding box of the dynamic points")
print("=" * 78)
lo, hi = person.min(axis=0), person.max(axis=0)
ext = hi - lo
for ax, l, h, e in zip("xyz", lo, hi, ext):
    print(f"  {ax}: {l:+8.3f} .. {h:+8.3f} m    extent {e:6.3f} m")
print()
print(f"  A single standing adult is roughly 0.5 x 0.5 x 1.8 m.")
horiz = max(ext[0], ext[1])
if horiz > 1.5 or ext[2] > 2.4:
    print(f"  This bounding box ({ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m) is much")
    print(f"  larger than one person. The horizontal extent alone ({horiz:.2f} m) rules")
    print(f"  out a single stationary body: this is several people, or one person")
    print(f"  smeared across a walked path during the sweep.")
else:
    print(f"  This bounding box ({ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m) is")
    print(f"  consistent with one standing adult.")
print()

# ---------------------------------------------------------------------------
# 2. DBSCAN
# ---------------------------------------------------------------------------
print("=" * 78)
print(f"2. DBSCAN  (eps={EPS}, min_samples={MIN_SAMPLES})")
print("=" * 78)
labels = dbscan(person, EPS, MIN_SAMPLES)
uniq = [c for c in sorted(set(labels.tolist())) if c != -1]
n_noise = int((labels == -1).sum())
print(f"  clusters found : {len(uniq)}")
print(f"  noise points   : {n_noise:,} of {len(person):,}")
print()

sizes = [(c, int((labels == c).sum())) for c in uniq]
sizes.sort(key=lambda t: -t[1])
print(f"  clusters larger than 500 points:")
print(f"    {'id':>4}  {'points':>9}   extent x,y,z (m)")
big = [c for c, sz in sizes if sz > 500]
for c in big:
    Pc = person[labels == c]
    e = extent(Pc)
    print(f"    {c:>4}  {len(Pc):>9,}   {e[0]:5.2f} x {e[1]:5.2f} x {e[2]:5.2f}")
print()

if len(big) <= 1:
    print("  One dominant cluster -> a single connected body of points.")
else:
    print(f"  {len(big)} large clusters -> either several people or one person")
    print(f"  captured at several separated positions during the sweep.")
print()

# ---------------------------------------------------------------------------
# 3. render the largest cluster
# ---------------------------------------------------------------------------
print("=" * 78)
print("3. Rendering largest cluster -> out/fig12_person_views.png")
print("=" * 78)

if not big:
    largest = person
    print("  no cluster over 500 points; rendering all dynamic points instead")
else:
    largest_id = big[0]
    largest = person[labels == largest_id]
    print(f"  cluster {largest_id}: {len(largest):,} points")

e = extent(largest)
print(f"  extent: {e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f} m  (x, y, z)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(OUT_DIR, exist_ok=True)

X, Y, Z = largest[:, 0], largest[:, 1], largest[:, 2]
panels = [
    ("front elevation", X, Z, "X (m)", "Z (m)"),
    ("side elevation", Y, Z, "Y (m)", "Z (m)"),
    ("plan", X, Y, "X (m)", "Y (m)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="black")
for ax, (title, a, b, la, lb) in zip(axes, panels):
    ax.set_facecolor("black")
    ax.scatter(a, b, s=0.5, c="#63d2ff", alpha=0.5, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title, color="white")
    ax.set_xlabel(la, color="white")
    ax.set_ylabel(lb, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

fig.suptitle(
    f"Lecturehall ground-truth person -- largest DBSCAN cluster "
    f"({len(largest):,} pts)",
    color="white",
)
out_path = os.path.join(OUT_DIR, "fig12_person_views.png")
fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="black")
plt.close(fig)
print(f"  saved {out_path}")
print()

# ---------------------------------------------------------------------------
print("=" * 78)
print("VERDICT")
print("=" * 78)
if e[2] < 1.2:
    print(f"  The largest cluster is only {e[2]:.2f} m tall. That is not a standing")
    print(f"  human. A standing adult is NOT recognisable in these views.")
elif max(e[0], e[1]) > 1.0:
    print(f"  The largest cluster is {e[2]:.2f} m tall but {max(e[0], e[1]):.2f} m wide")
    print(f"  horizontally -- wider than a person. Likely a motion smear, not a")
    print(f"  clean standing silhouette.")
else:
    print(f"  The largest cluster is {e[2]:.2f} m tall and {e[0]:.2f} x {e[1]:.2f} m")
    print(f"  in plan -- consistent with a standing adult. Check the elevation")
    print(f"  panels of {os.path.basename(out_path)} for a recognisable silhouette.")
