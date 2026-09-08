"""
inspect_lecturehall3.py

Settle the residual-yaw question for the Wuerzburg lecturehall dataset by
DIRECT MEASUREMENT, not by the coarse wall-mass estimator used in
inspect_lecturehall2.py Part F.

Established facts (not re-derived here):
  - stored coordinates are centimetres -> multiply by 0.01 for metres
  - the two label CSVs share a single translation origin
  - pose translations (cm): pose1 ~ (0.9, 4.9, 3.1), pose2 = (169.78, -645.81, -69.07)
    -> pose2 move in metres ~ 6.7 m, a physically sensible tripod relocation
  - pose rotations are radians: pose2 = (-0.0551, -0.0162, 0.3604), rz = 20.65 deg
  - z is up, right handed

Everything below works in METRES. 300,000 points are subsampled per scan.
No package files are touched.
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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "lecturehall")
SCAN1_FILE = os.path.join(DATA, "lecturehall1.pose1.object1.label.csv")
SCAN2_FILE = os.path.join(DATA, "lecturehall1.pose2.object2.label.csv")
POSE1_FILE = os.path.join(DATA, "lecturehall1.pose1.csv")
POSE2_FILE = os.path.join(DATA, "lecturehall1.pose2.csv")
OUT_DIR = os.path.join(HERE, "out")

CM_TO_M = 0.01
CHUNKSIZE = 1_000_000
N_SUB = 300_000
OVERLAP_RADIUS_M = 0.5
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
def load_scan_m(path):
    """Load x, y, z (metres) into an (N, 3) float64 array. No transform."""
    parts = []
    for chunk in pd.read_csv(path, header=None, usecols=[0, 1, 2],
                             names=["x", "y", "z"], chunksize=CHUNKSIZE):
        parts.append(chunk.to_numpy(dtype=np.float64))
    xyz = np.vstack(parts)
    xyz *= CM_TO_M
    return xyz


def subsample(xyz, n, rng):
    if len(xyz) <= n:
        return xyz.copy()
    return xyz[rng.choice(len(xyz), n, replace=False)]


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def nn_stats(tree, pts):
    d, _ = tree.query(pts, workers=-1)
    return d


def pct(d, q):
    return float(np.percentile(d, q))


# ---------------------------------------------------------------------------
print("=" * 78)
print("Poses (raw file values, for reference)")
print("=" * 78)
with open(POSE1_FILE) as f:
    p1 = np.array([float(v) for v in f.read().strip().split(",")])
with open(POSE2_FILE) as f:
    p2 = np.array([float(v) for v in f.read().strip().split(",")])
print(f"  pose1 raw: t = {p1[:3]}   rot(rad) = {p1[3:]}")
print(f"  pose2 raw: t = {p2[:3]}   rot(rad) = {p2[3:]}")
pose2_t_m = p2[:3] * CM_TO_M
pose2_rot = p2[3:]
print(f"  pose2 translation in metres            : {pose2_t_m.round(4)}  "
      f"|t| = {np.linalg.norm(pose2_t_m):.3f} m")
print(f"  pose2 rotation in degrees (rx, ry, rz)  : {np.degrees(pose2_rot).round(4)}")
print(f"  pose2 rz                                : {np.degrees(pose2_rot[2]):.4f} deg")

R2 = rot_z(pose2_rot[2]) @ rot_y(pose2_rot[1]) @ rot_x(pose2_rot[0])
print(f"  R2 = Rz @ Ry @ Rx  (det = {np.linalg.det(R2):.6f})")

print("\n" + "=" * 78)
print("Loading raw coordinates and converting cm -> m (no rotation/translation)")
print("=" * 78)
scan1 = load_scan_m(SCAN1_FILE)
print(f"  scan 1: {len(scan1):,} points")
scan2 = load_scan_m(SCAN2_FILE)
print(f"  scan 2: {len(scan2):,} points")

for tag, s in (("scan 1", scan1), ("scan 2", scan2)):
    lo = s.min(axis=0)
    hi = s.max(axis=0)
    med = np.median(s, axis=0)
    print(f"  {tag}: x[{lo[0]:8.2f},{hi[0]:8.2f}] y[{lo[1]:8.2f},{hi[1]:8.2f}] "
          f"z[{lo[2]:8.2f},{hi[2]:8.2f}]  median {med.round(2)}  (metres)")

print(f"\n  Subsampling {N_SUB:,} points from each scan (uniform, seed 42).")
sub1 = subsample(scan1, N_SUB, RNG)
sub2_full = subsample(scan2, N_SUB, RNG)

tree1 = cKDTree(sub1)

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 1 - restrict to genuine overlap (identity transform baseline)")
print("=" * 78)
print(f"  cKDTree built on the FULL scan-1 subsample ({len(sub1):,} pts).")
print(f"  Keeping scan-2 points whose nearest scan-1 neighbour is <= "
      f"{OVERLAP_RADIUS_M} m (identity transform).")

d2_identity_full = nn_stats(tree1, sub2_full)
overlap_mask = d2_identity_full <= OVERLAP_RADIUS_M
sub2 = sub2_full[overlap_mask]
n_overlap = len(sub2)
print(f"\n  scan-2 subsample points surviving the {OVERLAP_RADIUS_M} m overlap gate: "
      f"{n_overlap:,} of {len(sub2_full):,}  ({n_overlap / len(sub2_full) * 100:.1f} %)")

d_base = d2_identity_full[overlap_mask]
print(f"\n  BASELINE (identity, overlap subset only):")
print(f"     median NN   : {pct(d_base, 50):.4f} m")
print(f"     25th pct NN : {pct(d_base, 25):.4f} m")
print(f"     10th pct NN : {pct(d_base, 10):.4f} m")
print(f"     mean NN     : {d_base.mean():.4f} m")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 2 - four candidate transforms, all scored on the Part 1 overlap subset")
print("=" * 78)
print("  Rotations are about the shared coordinate origin (0,0,0). Metres throughout.")
print("  Convention: R = Rz @ Ry @ Rx from pose2.\n")

transforms = {
    "T0  identity (no change)":          lambda p: p,
    "T1  R2 rotation only, no translate": lambda p: p @ R2.T,
    "T2  R2 rotation + pose2 translate":  lambda p: p @ R2.T + pose2_t_m,
    "T3  R2^-1 (inverse) rotation only":  lambda p: p @ R2,
}

results = {}
for name, fn in transforms.items():
    moved = fn(sub2)
    d = nn_stats(tree1, moved)
    results[name] = (pct(d, 50), pct(d, 25), d)
    print(f"  {name}")
    print(f"       median NN   : {pct(d, 50):.4f} m")
    print(f"       25th pct NN : {pct(d, 25):.4f} m")
    print(f"       10th pct NN : {pct(d, 10):.4f} m     mean : {d.mean():.4f} m")
    print()

best = min(results.items(), key=lambda kv: kv[1][0])
print(f"  lowest median NN among the four: {best[0]}  ->  {best[1][0]:.4f} m")
if best[1][0] > 0.05:
    print(f"  NOTE: even the best transform leaves the overlap median at "
          f"{best[1][0]:.4f} m, well above ~0.05 m.")
    print("        No candidate transform brings the clouds into tight registration.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 3 - independent yaw measurement, transform free")
print("=" * 78)
print("  Wall band: points with 1.0 m <= z <= 2.5 m (above furniture, below ceiling).")
print("  Project to x-y, estimate local 2-D normals by PCA on k-NN neighbourhoods,")
print("  keep strongly linear neighbourhoods, histogram normal orientation mod 90 deg.\n")


def wall_yaw(xyz, rng, k=12, n_pts=60_000, linearity_min=0.6):
    band = xyz[(xyz[:, 2] >= 1.0) & (xyz[:, 2] <= 2.5)]
    xy_all = band[:, :2]
    xy = subsample(xy_all, n_pts, rng)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k, workers=-1)
    angles = []
    weights = []
    for i in range(len(xy)):
        nb = xy[idx[i]]
        nb = nb - nb.mean(axis=0)
        cov = nb.T @ nb / len(nb)
        w, v = np.linalg.eigh(cov)          # ascending
        lam_small, lam_big = w[0], w[1]
        if lam_big <= 1e-12:
            continue
        linearity = 1.0 - lam_small / lam_big
        if linearity < linearity_min:
            continue
        normal = v[:, 0]                    # smallest-variance direction
        a = np.degrees(np.arctan2(normal[1], normal[0])) % 180.0
        angles.append(a % 90.0)            # two dominant wall sets are ~orthogonal
        weights.append(linearity)
    angles = np.asarray(angles)
    weights = np.asarray(weights)
    # circular-ish peak on [0,90): use a fine weighted histogram with wrap
    bins = np.arange(0, 90.0 + 0.5, 0.5)
    hist, edges = np.histogram(angles, bins=bins, weights=weights)
    # smooth with a small circular window
    kern = np.ones(5) / 5.0
    hist_s = np.convolve(np.r_[hist[-2:], hist, hist[:2]], kern, mode="same")[2:-2]
    peak = edges[np.argmax(hist_s)] + 0.25
    # weighted circular mean on doubled angle over 90-deg period for a refined value
    theta = np.radians(angles * 4.0)       # period 90 -> *4 maps to 2*pi*2; use *4 for [0,90)->[0,360)
    cmean = np.degrees(np.arctan2((weights * np.sin(theta)).sum(),
                                  (weights * np.cos(theta)).sum())) / 4.0
    cmean = cmean % 90.0
    return peak, cmean, len(angles), band.shape[0]


y1_peak, y1_cm, n1, nb1 = wall_yaw(scan1, RNG)
y2_peak, y2_cm, n2, nb2 = wall_yaw(scan2, RNG)


def circ_diff_90(a, b):
    d = (a - b) % 90.0
    return min(d, 90.0 - d)


diff_peak = circ_diff_90(y1_peak, y2_peak)
diff_cm = circ_diff_90(y1_cm, y2_cm)

print(f"  scan 1 wall band points (1.0-2.5 m z) : {nb1:,}   linear nbhds used: {n1:,}")
print(f"  scan 2 wall band points (1.0-2.5 m z) : {nb2:,}   linear nbhds used: {n2:,}")
print(f"\n  scan 1 dominant wall orientation (mod 90):  histogram peak {y1_peak:.2f} deg"
      f"   circular mean {y1_cm:.2f} deg")
print(f"  scan 2 dominant wall orientation (mod 90):  histogram peak {y2_peak:.2f} deg"
      f"   circular mean {y2_cm:.2f} deg")
print(f"\n  wall-orientation difference (histogram peak): {diff_peak:.2f} deg")
print(f"  wall-orientation difference (circular mean) : {diff_cm:.2f} deg")
print(f"  pose2 rz for comparison                     : {np.degrees(pose2_rot[2]):.2f} deg")

# also: what yaw best aligns scan2 walls onto scan1 walls? brute force on the band
def band_xy(xyz, rng, n=120_000):
    b = xyz[(xyz[:, 2] >= 1.0) & (xyz[:, 2] <= 2.5)]
    return subsample(b[:, :2], n, rng)


b1 = band_xy(scan1, RNG)
b2 = band_xy(scan2, RNG)
b1c = b1 - b1.mean(axis=0)
b2c = b2 - b2.mean(axis=0)
tb1 = cKDTree(b1c)
print("\n  brute-force: rotate scan-2 wall band about its own centroid, median NN to scan-1 band")
best_ang, best_med = None, np.inf
for deg in np.arange(-30, 30.01, 1.0):
    r = np.radians(deg)
    M = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    dd, _ = tb1.query(b2c @ M.T, workers=-1)
    m = np.median(dd)
    if m < best_med:
        best_med, best_ang = m, deg
print(f"    best-fit in-plane rotation scan2->scan1 walls: {best_ang:.1f} deg "
      f"(median NN {best_med:.3f} m at that angle)")
print(f"    (compare to pose2 rz = {np.degrees(pose2_rot[2]):.2f} deg)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 4 - figure out/fig8_lecturehall_transforms.png")
print("=" * 78)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(OUT_DIR, exist_ok=True)
sub1_plot = subsample(sub1, 120_000, RNG)

fig, axes = plt.subplots(2, 2, figsize=(15, 14))
for ax, (name, fn) in zip(axes.ravel(), transforms.items()):
    moved = fn(sub2)
    med = results[name][0]
    ax.scatter(sub1_plot[:, 0], sub1_plot[:, 1], s=1.0, c="0.65", alpha=0.30,
               linewidths=0, label=f"scan 1 subsample ({len(sub1_plot):,})")
    ax.scatter(moved[:, 0], moved[:, 1], s=1.0, c="tab:blue", alpha=0.30,
               linewidths=0, label=f"scan 2 overlap subset ({len(moved):,})")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{name}\nmedian NN = {med:.4f} m   (25th pct {results[name][1]:.4f} m)")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="upper right", framealpha=0.9, markerscale=6)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

fig.suptitle("Lecturehall: scan-2 overlap subset under four candidate transforms "
             "(plan view, metres)\nscan 1 grey, scan 2 blue, overlap subset only, "
             "rotations about shared origin", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(OUT_DIR, "fig8_lecturehall_transforms.png")
fig.savefig(out_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  saved {out_path}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY OF NUMBERS")
print("=" * 78)
print(f"  overlap gate survivors (identity, 0.5 m)     : {n_overlap:,} / {len(sub2_full):,}")
print(f"  baseline (identity) overlap median NN        : {pct(d_base, 50):.4f} m")
print(f"  baseline (identity) overlap 25th pct NN      : {pct(d_base, 25):.4f} m")
for name, (m, q, _) in results.items():
    print(f"  {name:36s} median {m:.4f} m   25th {q:.4f} m")
print(f"  scan1 wall orientation (peak / circ mean)    : {y1_peak:.2f} / {y1_cm:.2f} deg")
print(f"  scan2 wall orientation (peak / circ mean)    : {y2_peak:.2f} / {y2_cm:.2f} deg")
print(f"  wall-orientation difference (peak / circ)    : {diff_peak:.2f} / {diff_cm:.2f} deg")
print(f"  best-fit in-plane rotation scan2->scan1      : {best_ang:.1f} deg (median NN {best_med:.3f} m)")
print(f"  pose2 rz                                     : {np.degrees(pose2_rot[2]):.2f} deg")
best_name, (best_m, best_q, _) = min(results.items(), key=lambda kv: kv[1][0])
print(f"\n  best transform by overlap median NN          : {best_name}  ({best_m:.4f} m)")
if best_m > 0.05:
    print("  -> No transform gets the overlap median under ~0.05 m. See discussion above.")
