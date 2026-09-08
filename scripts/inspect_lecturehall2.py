"""
inspect_lecturehall2.py

Revisit of the lecturehall alignment conclusion.

Part 1 of inspect_lecturehall.py (ground-truth label check) passed and is not
repeated here. Parts 2 and 3 of that script applied pose2 to scan 2 and then
judged the alignment "broken". This script tests the competing hypothesis
first, and applies NO pose transform anywhere:

    The Wuerzburg README says the two scans are ALREADY REGISTERED. If that is
    true, applying pose2 to scan 2 is wrong - it shoves an already-aligned
    scan ~645 units away. The pose files would then be metadata describing
    where the scanner stood, not a transform to apply.

Every number below is computed on the raw coordinates exactly as they sit in
the CSV files. No rotation, no translation, ever.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = "data/lecturehall"
SCAN1_FILE = f"{DATA}/lecturehall1.pose1.object1.label.csv"
SCAN2_FILE = f"{DATA}/lecturehall1.pose2.object2.label.csv"
POSE1_FILE = f"{DATA}/lecturehall1.pose1.csv"
POSE2_FILE = f"{DATA}/lecturehall1.pose2.csv"

CHUNKSIZE = 1_000_000
N_SUB = 200_000
N_TREE_DENSE = 3_000_000     # denser tree to separate sampling noise from real error
ROOM_RADIUS = 30.0           # the prescribed "within 30 m of the median"
RADII = [30, 100, 300, 1000, 3000, 10000]
NN_BUCKETS = [0.5, 1, 2, 5, 10, 20, 50, 100]
PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
RNG = np.random.default_rng(42)


def load_scan(path):
    """Load x, y, z, label into an (N, 4) float32 array. No transform."""
    parts = []
    for chunk in pd.read_csv(path, header=None, usecols=[0, 1, 2, 4],
                             names=["x", "y", "z", "label"],
                             chunksize=CHUNKSIZE):
        parts.append(chunk.to_numpy(dtype=np.float32))
    return np.vstack(parts)


def subsample(xyz, n, rng):
    if len(xyz) <= n:
        return xyz
    return xyz[rng.choice(len(xyz), n, replace=False)]


def nn_dists(ref_xyz, qry_xyz):
    tree = cKDTree(ref_xyz)
    d, _ = tree.query(qry_xyz, workers=-1)
    return d


def report_nn(d, tag):
    p10, p25, p50, p75 = np.percentile(d, [10, 25, 50, 75])
    print(f"  [{tag}]")
    print(f"     10th pct : {p10:9.4f}     25th pct : {p25:9.4f}")
    print(f"     MEDIAN   : {p50:9.4f}     75th pct : {p75:9.4f}     mean : {d.mean():9.4f}")
    frac = "  ".join(f"<{b:g}:{np.count_nonzero(d <= b) / len(d) * 100:5.1f}%"
                     for b in NN_BUCKETS)
    print(f"     fraction of query pts within N units of a ref pt:\n       {frac}")
    return p50


def pct_table(xyz, name):
    print(f"\n{name}: percentiles of x, y, z (raw coords, no transform)")
    header = "  axis | " + " ".join(f"{p:>10}" for p in PCTS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ax, col in zip("xyz", range(3)):
        vals = np.percentile(xyz[:, col], PCTS)
        print(f"  {ax:>4} | " + " ".join(f"{v:10.2f}" for v in vals))
    med = np.median(xyz[:, :3], axis=0)
    print(f"  median (x, y, z)   = ({med[0]:10.3f}, {med[1]:10.3f}, {med[2]:10.3f})")
    mean = xyz[:, :3].mean(axis=0)
    print(f"  mean   (x, y, z)   = ({mean[0]:10.3f}, {mean[1]:10.3f}, {mean[2]:10.3f})")
    return med


def dist_from(xyz, pt):
    return np.sqrt(((xyz[:, :3] - pt) ** 2).sum(axis=1))


def extent(xyz):
    lo = xyz[:, :3].min(axis=0)
    hi = xyz[:, :3].max(axis=0)
    return lo, hi, hi - lo


def show_extent(xyz, tag):
    lo, hi, ext = extent(xyz)
    print(f"  {tag}: {len(xyz):,} points")
    for ax, i in zip("xyz", range(3)):
        print(f"     {ax}: [{lo[i]:11.3f}, {hi[i]:11.3f}]   extent = {ext[i]:10.3f}")


# ---------------------------------------------------------------------------
print("=" * 78)
print("Poses (raw, for reference only - NOT applied)")
print("=" * 78)
with open(POSE1_FILE) as f:
    p1 = f.read().strip()
with open(POSE2_FILE) as f:
    p2 = f.read().strip()
print(f"  pose1: {p1}")
print(f"  pose2: {p2}")
pose1 = np.array([float(v) for v in p1.split(",")])
pose2 = np.array([float(v) for v in p2.split(",")])
print(f"  pose1 translation = {pose1[:3]}   rot(rad) = {pose1[3:]}")
print(f"  pose2 translation = {pose2[:3]}   rot(rad) = {pose2[3:]}"
      f"   rot(deg) = {np.degrees(pose2[3:]).round(2)}")

print("\n" + "=" * 78)
print("Loading raw coordinates (no transform applied)")
print("=" * 78)
scan1 = load_scan(SCAN1_FILE)
print(f"  scan 1: {len(scan1):,} points")
scan2 = load_scan(SCAN2_FILE)
print(f"  scan 2: {len(scan2):,} points")
dyn_mask1 = scan1[:, 3] == 1
print(f"  scan 1 dynamic (label==1): {np.count_nonzero(dyn_mask1):,}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART A - alignment with NO transform at all")
print("=" * 78)
print(f"Subsampling {N_SUB:,} points from each scan (uniform random, seed 42).")
sub1 = subsample(scan1[:, :3], N_SUB, RNG)
sub2 = subsample(scan2[:, :3], N_SUB, RNG)

print("\n(prescribed) tree on scan 1 subsample, query scan 2 subsample, AS-IS:")
dA = nn_dists(sub1, sub2)
medA = report_nn(dA, "200k <- 200k, raw")

print(f"\n(denser) tree on {N_TREE_DENSE:,} scan-1 points, query {N_SUB:,} scan-2 points,")
print("AS-IS  - shrinks the nearest-neighbour inflation caused by 1% subsampling:")
treeref = subsample(scan1[:, :3], N_TREE_DENSE, RNG)
dA2 = nn_dists(treeref, sub2)
medA2 = report_nn(dA2, f"{N_TREE_DENSE // 1_000_000}M <- 200k, raw")

print("\n  >>> The two raw clouds are NOT tightly coincident point-for-point,")
print("      but check whether they share a coordinate frame:")
med1 = np.median(scan1[:, :3], axis=0)
med2 = np.median(scan2[:, :3], axis=0)
c1 = scan1[:, :3].mean(axis=0)
c2 = scan2[:, :3].mean(axis=0)
print(f"      scan1 median {med1.round(2)}   scan2 median {med2.round(2)}   "
      f"|diff| = {np.linalg.norm(med1 - med2):.2f}")
print(f"      scan1 mean   {c1.round(2)}   scan2 mean   {c2.round(2)}   "
      f"|diff| = {np.linalg.norm(c1 - c2):.2f}")
print(f"      pose2 translation magnitude = {np.linalg.norm(pose2[:3]):.2f}  "
      f"(what old Part 3 would have added)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART B - percentiles, not min/max")
print("=" * 78)
med1 = pct_table(scan1, "Scan 1")
med2 = pct_table(scan2, "Scan 2")

d1_all = dist_from(scan1, med1)
d2_all = dist_from(scan2, med2)
print(f"\nFraction of points within {ROOM_RADIUS:.0f} units of the median (x,y,z):")
print(f"  scan 1 : {np.count_nonzero(d1_all <= ROOM_RADIUS) / len(d1_all) * 100:.3f} %"
      f"   ({np.count_nonzero(d1_all <= ROOM_RADIUS):,} of {len(d1_all):,})")
print(f"  scan 2 : {np.count_nonzero(d2_all <= ROOM_RADIUS) / len(d2_all) * 100:.3f} %"
      f"   ({np.count_nonzero(d2_all <= ROOM_RADIUS):,} of {len(d2_all):,})")

print("\ndistance-from-median percentiles (units as stored):")
for tag, d in (("scan 1", d1_all), ("scan 2", d2_all)):
    dp = np.percentile(d, PCTS)
    print(f"  {tag}: " + " ".join(f"{p}={v:.1f}" for p, v in zip(PCTS, dp)))

print("\nfraction within a growing ball around the median (finding the room):")
print("  radius |    scan 1    |    scan 2")
for r in RADII:
    f1 = np.count_nonzero(d1_all <= r) / len(d1_all) * 100
    f2 = np.count_nonzero(d2_all <= r) / len(d2_all) * 100
    print(f"  {r:6d} | {f1:9.3f} %  | {f2:9.3f} %")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART C - the room itself")
print("=" * 78)
print(f"(prescribed) filter to within {ROOM_RADIUS:.0f} units of the median:")
room1 = scan1[d1_all <= ROOM_RADIUS]
room2 = scan2[d2_all <= ROOM_RADIUS]
print(f"  scan 1: {len(room1):,} points     scan 2: {len(room2):,} points")
if len(room1) == 0 or len(room2) == 0:
    print("  -> empty. A 30-unit ball at the centroid of a room-shaped shell is")
    print("     mid-air and empty. This is expected geometry, not corruption;")
    print("     combined with the percentile spread it says the stored unit is")
    print("     not metres (README section 6: 'some 3DTK data is centimetres').")

# pick a radius that actually contains the bulk, per scan
def room_radius(d, target=0.90):
    for r in RADII:
        if np.count_nonzero(d <= r) / len(d) >= target:
            return r
    return RADII[-1]

R1 = room_radius(d1_all)
R2 = room_radius(d2_all)
R = max(R1, R2)
print(f"\n(adaptive) smallest listed radius holding >=90% of points: "
      f"scan1={R1}, scan2={R2}  ->  using R={R} for both")
room1 = scan1[d1_all <= R]
room2 = scan2[d2_all <= R]
show_extent(room1, "scan 1 room subset")
show_extent(room2, "scan 2 room subset")
if R >= 100:
    print(f"  (if the CSV is in centimetres, R={R} units = {R/100:.1f} m,")
    print(f"   scan 1 room diagonal extent ~ "
          f"{np.linalg.norm(extent(room1)[2]) / 100:.1f} m)")

print("\nPart A alignment score on the room-sized subset only (no transform):")
rsub1 = subsample(room1[:, :3], N_SUB, RNG)
rsub2 = subsample(room2[:, :3], N_SUB, RNG)
dC = nn_dists(rsub1, rsub2)
medC = report_nn(dC, "room 200k <- room 200k, raw")
rtree = subsample(room1[:, :3], N_TREE_DENSE, RNG)
dC2 = nn_dists(rtree, rsub2)
medC2 = report_nn(dC2, f"room {len(rtree)//1_000_000 or 1}M <- room 200k, raw")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART D - where are the dynamic points (scan 1, label==1)")
print("=" * 78)
dyn1 = scan1[dyn_mask1]
print(f"  count: {len(dyn1):,}")
_ = pct_table(dyn1, "Scan 1 dynamic")
d_dyn = dist_from(dyn1, med1)
dp = np.percentile(d_dyn, PCTS)
print(f"\n  distance from scan 1 median {tuple(round(v, 2) for v in med1)}:")
print("  " + " ".join(f"{p}={v:.1f}" for p, v in zip(PCTS, dp)))
print(f"  min / max distance from median: {d_dyn.min():.2f} / {d_dyn.max():.2f}")
for r in RADII:
    w = np.count_nonzero(d_dyn <= r)
    print(f"  within {r:6d} units of median: {w:,} ({w / len(d_dyn) * 100:.2f} %)")
# also relative to the dynamic cluster's own centre
dm = np.median(dyn1[:, :3], axis=0)
dd = dist_from(dyn1, dm)
print(f"\n  dynamic-cluster own median = {dm.round(2)}")
print(f"  spread about own median: p50={np.percentile(dd, 50):.1f}  "
      f"p95={np.percentile(dd, 95):.1f}  max={dd.max():.1f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART F - orientation of the two rooms (measurement only, NO transform)")
print("=" * 78)
print("If the scans shared a fully common frame the walls would be parallel.")
print("Method: 2-D histogram of edge-normal directions is overkill; use the")
print("angle that maximises axis-aligned point mass (wall-fit) on x-y.\n")


def dominant_yaw(xy):
    """Angle (deg, 0..90) that best aligns the point cloud to axis-parallel
    walls: pick theta minimising the product of rotated-x and rotated-y IQRs
    is unstable; instead maximise how peaked the rotated coordinate
    histograms are (walls -> sharp peaks). Pure measurement, no pose."""
    best_t, best_score = 0.0, -np.inf
    c = xy - np.median(xy, axis=0)
    for t in np.arange(0.0, 90.0, 0.5):
        r = np.radians(t)
        rot = c @ np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).T
        # sharpness = sum of squared histogram densities (higher = more wall-like)
        hx, _ = np.histogram(rot[:, 0], bins=400)
        hy, _ = np.histogram(rot[:, 1], bins=400)
        score = ((hx / hx.sum()) ** 2).sum() + ((hy / hy.sum()) ** 2).sum()
        if score > best_score:
            best_score, best_t = score, t
    return best_t


s1_xy = subsample(room1[:, :2], 400_000, RNG)
s2_xy = subsample(room2[:, :2], 400_000, RNG)
y1 = dominant_yaw(s1_xy)
y2 = dominant_yaw(s2_xy)
diff = (y2 - y1) % 90.0
diff = min(diff, 90.0 - diff)
print(f"  scan 1 wall yaw (mod 90): {y1:.1f} deg")
print(f"  scan 2 wall yaw (mod 90): {y2:.1f} deg")
print(f"  angle between the two rooms' walls: {diff:.1f} deg")
print(f"  pose2 rz component               : {np.degrees(pose2[5]):.2f} deg")
print(f"  pose2 full rotation angle         : "
      f"{np.degrees(np.linalg.norm(pose2[3:])):.2f} deg")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART E - figure: out/fig7_lecturehall_room.png")
print("=" * 78)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("out", exist_ok=True)
room1_static = room1[room1[:, 3] != 1]
room1_dyn = room1[room1[:, 3] == 1]
# dynamic within the adaptive room may be empty; fall back to all dynamic if so
dyn_for_fig = room1_dyn if len(room1_dyn) else dyn1
pf_static = subsample(room1_static[:, :3], 150_000, RNG)
pf_dyn = subsample(dyn_for_fig[:, :3], 150_000, RNG)
pf_scan2 = subsample(room2[:, :3], 150_000, RNG)

fig, ax = plt.subplots(figsize=(13, 12))
ax.scatter(pf_static[:, 0], pf_static[:, 1], c="0.6", s=1.5, alpha=0.25,
           linewidths=0, label=f"scan 1 static ({len(room1_static):,})")
ax.scatter(pf_scan2[:, 0], pf_scan2[:, 1], c="tab:blue", s=1.5, alpha=0.25,
           linewidths=0, label=f"scan 2 ({len(room2):,})")
ax.scatter(pf_dyn[:, 0], pf_dyn[:, 1], c="tab:red", s=2.5, alpha=0.4,
           linewidths=0,
           label=f"scan 1 dynamic ({len(room1_dyn):,} in room"
                 f"{'' if len(room1_dyn) else ' - showing all ' + format(len(dyn1), ',')})")
ax.set_xlabel("X (stored units)")
ax.set_ylabel("Y (stored units)")
ax.set_aspect("equal")
ax.set_title(f"Lecturehall room subset (radius {R} units of median), plan view x-y\n"
             "NO transform applied   |   grey = scan1 static, red = scan1 dynamic, blue = scan2")
leg = ax.legend(loc="upper right", framealpha=0.9)
for h in leg.legend_handles:
    h.set_alpha(1.0)
ax.grid(True, alpha=0.3)
fig.savefig("out/fig7_lecturehall_room.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved out/fig7_lecturehall_room.png")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY OF NUMBERS")
print("=" * 78)
print(f"  Part A  median NN, raw, 200k<-200k            : {medA:9.4f}")
print(f"  Part A  median NN, raw, 3M<-200k              : {medA2:9.4f}")
print(f"  Part C  median NN, raw room, 200k<-200k       : {medC:9.4f}")
print(f"  Part C  median NN, raw room, dense<-200k      : {medC2:9.4f}")
print(f"  |scan1 median - scan2 median|                 : {np.linalg.norm(med1 - med2):9.2f}")
print(f"  pose2 translation magnitude (NOT applied)     : {np.linalg.norm(pose2[:3]):9.2f}")
print(f"  scan1 pts within 30 units of median           : "
      f"{np.count_nonzero(d1_all <= 30) / len(d1_all) * 100:.3f} %")
print(f"  scan1 pts within {R} units of median{' ' * (10 - len(str(R)))}       : "
      f"{np.count_nonzero(d1_all <= R) / len(d1_all) * 100:.3f} %")
print(f"  dynamic pts within {R} units of median          : "
      f"{np.count_nonzero(d_dyn <= R) / len(d_dyn) * 100:.3f} %")
