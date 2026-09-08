import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

# File paths
scan1_file = "data/lecturehall/lecturehall1.pose1.object1.label.csv"
scan2_file = "data/lecturehall/lecturehall1.pose2.object2.label.csv"
pose1_file = "data/lecturehall/lecturehall1.pose1.csv"
pose2_file = "data/lecturehall/lecturehall1.pose2.csv"

CHUNKSIZE = 500_000

print("=" * 80)
print("PART 1: GROUND TRUTH CHECK")
print("=" * 80)

# Count col5 values for scan1
print("\nScan 1 (lecturehall1.pose1.object1.label.csv):")
col5_counts_scan1 = {}
total_scan1 = 0
for chunk in pd.read_csv(scan1_file, header=None, usecols=[4], chunksize=CHUNKSIZE):
    total_scan1 += len(chunk)
    for val in chunk[4].unique():
        col5_counts_scan1[val] = col5_counts_scan1.get(val, 0) + (chunk[4] == val).sum()

print(f"  Total rows: {total_scan1:,}")
print(f"  Distinct values in col5: {sorted(col5_counts_scan1.keys())}")
for val in sorted(col5_counts_scan1.keys()):
    count = col5_counts_scan1[val]
    pct = 100.0 * count / total_scan1
    print(f"    Value {int(val)}: {count:,} rows ({pct:.2f}%)")

# Count col5 values for scan2
print("\nScan 2 (lecturehall1.pose2.object2.label.csv):")
col5_counts_scan2 = {}
total_scan2 = 0
for chunk in pd.read_csv(scan2_file, header=None, usecols=[4], chunksize=CHUNKSIZE):
    total_scan2 += len(chunk)
    for val in chunk[4].unique():
        col5_counts_scan2[val] = col5_counts_scan2.get(val, 0) + (chunk[4] == val).sum()

print(f"  Total rows: {total_scan2:,}")
print(f"  Distinct values in col5: {sorted(col5_counts_scan2.keys())}")
for val in sorted(col5_counts_scan2.keys()):
    count = col5_counts_scan2[val]
    pct = 100.0 * count / total_scan2
    print(f"    Value {int(val)}: {count:,} rows ({pct:.2f}%)")

# Check ground truth
print("\nGround truth check:")
if 1 not in col5_counts_scan1:
    print("❌ FAILED: Scan 1 has no value-1 (dynamic) rows. Dataset has no usable ground truth.")
    exit(1)

value_1_count = col5_counts_scan1[1]
print(f"  ✓ Scan 1 has {value_1_count:,} dynamic points")
if abs(value_1_count - 500_000) > 50_000:
    print(f"    ⚠️  Expected ~500,000, got {value_1_count:,}")
else:
    print(f"    ✓ Matches expected ~500,000")

if set(col5_counts_scan2.keys()) == {0}:
    print(f"  ✓ Scan 2 is all static (all zeros)")
else:
    print(f"  ⚠️  Scan 2 unexpected non-zero values: {sorted(col5_counts_scan2.keys())}")

print("\n✓ PART 1 PASSED - Ground truth confirmed\n")

print("=" * 80)
print("PART 2: COORDINATE SANITY")
print("=" * 80)

def compute_stats(filepath):
    """Compute min/max/mean for x, y, z chunkwise"""
    mins = {'x': np.inf, 'y': np.inf, 'z': np.inf}
    maxs = {'x': -np.inf, 'y': -np.inf, 'z': -np.inf}
    sums = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    counts = {'x': 0, 'y': 0, 'z': 0}

    for chunk in pd.read_csv(filepath, header=None, usecols=[0, 1, 2],
                            names=['x', 'y', 'z'], chunksize=CHUNKSIZE):
        for col in ['x', 'y', 'z']:
            mins[col] = min(mins[col], chunk[col].min())
            maxs[col] = max(maxs[col], chunk[col].max())
            sums[col] += chunk[col].sum()
            counts[col] += len(chunk)

    result = {}
    for col in ['x', 'y', 'z']:
        result[col] = {
            'min': mins[col],
            'max': maxs[col],
            'mean': sums[col] / counts[col],
            'range': maxs[col] - mins[col],
        }

    return result

print("\nComputing statistics for Scan 1...")
stats1 = compute_stats(scan1_file)

print("Computing statistics for Scan 2...")
stats2 = compute_stats(scan2_file)

print("\nScan 1 coordinate ranges:")
for axis in ['x', 'y', 'z']:
    s = stats1[axis]
    print(f"  {axis.upper()}: [{s['min']:10.4f}, {s['max']:10.4f}]  mean={s['mean']:10.4f}  range={s['range']:8.4f}m")

print("\nScan 2 coordinate ranges:")
for axis in ['x', 'y', 'z']:
    s = stats2[axis]
    print(f"  {axis.upper()}: [{s['min']:10.4f}, {s['max']:10.4f}]  mean={s['mean']:10.4f}  range={s['range']:8.4f}m")

# Determine which axis is vertical
ranges_scan1 = {axis: stats1[axis]['range'] for axis in ['x', 'y', 'z']}
ranges_scan2 = {axis: stats2[axis]['range'] for axis in ['x', 'y', 'z']}

vertical_axis_1 = min(ranges_scan1, key=ranges_scan1.get)
vertical_axis_2 = min(ranges_scan2, key=ranges_scan2.get)

print(f"\nScan 1: smallest range (likely vertical) → {vertical_axis_1.upper()} ({ranges_scan1[vertical_axis_1]:.4f}m)")
print(f"Scan 2: smallest range (likely vertical) → {vertical_axis_2.upper()} ({ranges_scan2[vertical_axis_2]:.4f}m)")

print("\n" + "=" * 80)
print("PART 3: ALIGNMENT CHECK")
print("=" * 80)

# Read poses
with open(pose1_file) as f:
    pose1_line = f.read().strip()
pose1 = np.array([float(x) for x in pose1_line.split(',')])

with open(pose2_file) as f:
    pose2_line = f.read().strip()
pose2 = np.array([float(x) for x in pose2_line.split(',')])

print(f"\nPose 1 (raw from file):\n  {pose1_line}")
print(f"\nPose 2 (raw from file):\n  {pose2_line}")

# Extract poses
t1 = pose1[:3]
r1_angles = pose1[3:]
t2 = pose2[:3]
r2_angles = pose2[3:]

print(f"\nPose 1:")
print(f"  Translation: {t1}")
print(f"  Rotation (rx, ry, rz) rad: {r1_angles}")
print(f"  Rotation deg: {np.degrees(r1_angles)}")

print(f"\nPose 2:")
print(f"  Translation: {t2}")
print(f"  Rotation (rx, ry, rz) rad: {r2_angles}")
print(f"  Rotation deg: {np.degrees(r2_angles)}")

# Subsample 200k points
def subsample_points(filepath, n_samples=200_000):
    """Subsample using systematic skip"""
    total_rows = 0
    for chunk in pd.read_csv(filepath, header=None, usecols=[0, 1, 2], chunksize=CHUNKSIZE):
        total_rows += len(chunk)

    skip = max(1, total_rows // n_samples)
    print(f"    (total rows: {total_rows:,}, skip: {skip})")

    all_points = []
    row_idx = 0
    for chunk in pd.read_csv(filepath, header=None, usecols=[0, 1, 2], chunksize=CHUNKSIZE):
        for row in chunk.values:
            if row_idx % skip == 0:
                all_points.append(row)
            row_idx += 1

    return np.array(all_points[:n_samples])

print(f"\nSubsampling 200,000 points from Scan 1...")
scan1_pts = subsample_points(scan1_file, 200_000)
print(f"  Got {len(scan1_pts)} points")

print(f"Subsampling 200,000 points from Scan 2...")
scan2_pts = subsample_points(scan2_file, 200_000)
print(f"  Got {len(scan2_pts)} points")

# Rotation matrix functions
def rotation_matrix_from_euler(rx, ry, rz, order='ZYX'):
    """Create rotation matrix from Euler angles"""
    # Individual rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    if order == 'ZYX':
        return Rz @ Ry @ Rx
    else:  # 'XYZ'
        return Rx @ Ry @ Rz

def apply_pose(points, translation, euler_angles_rad, order='ZYX'):
    """Apply rotation and translation: p_transformed = p @ R.T + t"""
    rx, ry, rz = euler_angles_rad
    R = rotation_matrix_from_euler(rx, ry, rz, order)
    transformed = points @ R.T + translation
    return transformed

print(f"\nApplying pose 2 to scan 2 (Rz @ Ry @ Rx order)...")
scan2_transformed_zyx = apply_pose(scan2_pts, t2, r2_angles, 'ZYX')

print("Building KDTree and computing nearest-neighbor distances...")
tree = cKDTree(scan1_pts)
distances_zyx, _ = tree.query(scan2_transformed_zyx)

median_dist_zyx = np.median(distances_zyx)
mean_dist_zyx = np.mean(distances_zyx)
p95_dist_zyx = np.percentile(distances_zyx, 95)

print(f"\nAlignment score (Rz @ Ry @ Rx):")
print(f"  Median NN distance: {median_dist_zyx:.6f} m")
print(f"  Mean NN distance:   {mean_dist_zyx:.6f} m")
print(f"  95th percentile:    {p95_dist_zyx:.6f} m")

best_order = 'ZYX'
best_median = median_dist_zyx
scan2_transformed = scan2_transformed_zyx

# Try other rotation order if poor alignment
if median_dist_zyx > 0.30:
    print(f"\nAlignment poor (> 0.30m). Trying Rx @ Ry @ Rz order...")
    scan2_transformed_xyz = apply_pose(scan2_pts, t2, r2_angles, 'XYZ')
    distances_xyz, _ = tree.query(scan2_transformed_xyz)

    median_dist_xyz = np.median(distances_xyz)
    mean_dist_xyz = np.mean(distances_xyz)
    p95_dist_xyz = np.percentile(distances_xyz, 95)

    print(f"\nAlignment score (Rx @ Ry @ Rz):")
    print(f"  Median NN distance: {median_dist_xyz:.6f} m")
    print(f"  Mean NN distance:   {mean_dist_xyz:.6f} m")
    print(f"  95th percentile:    {p95_dist_xyz:.6f} m")

    if median_dist_xyz < median_dist_zyx:
        print(f"\n  → Rx @ Ry @ Rz is better ({median_dist_xyz:.6f} < {median_dist_zyx:.6f})")
        best_order = 'XYZ'
        best_median = median_dist_xyz
        scan2_transformed = scan2_transformed_xyz
    else:
        print(f"\n  → Rz @ Ry @ Rx is still better ({median_dist_zyx:.6f} < {median_dist_xyz:.6f})")

if best_median < 0.10:
    print(f"\n✓ Alignment: GOOD (< 0.10 m)")
elif best_median < 0.30:
    print(f"\n✓ Alignment: ACCEPTABLE (0.10–0.30 m)")
else:
    print(f"\n❌ Alignment: POOR (> 0.30 m)")

print("\n" + "=" * 80)
print("PART 4: OVERLAY FIGURE")
print("=" * 80)

print("Creating plan-view scatter plot...")
np.random.seed(42)

n_fig = 100_000
if len(scan1_pts) > n_fig:
    idx1 = np.random.choice(len(scan1_pts), n_fig, replace=False)
    scan1_fig = scan1_pts[idx1]
else:
    scan1_fig = scan1_pts

if scan2_transformed.shape[0] > n_fig:
    idx2 = np.random.choice(scan2_transformed.shape[0], n_fig, replace=False)
    scan2_fig = scan2_transformed[idx2]
else:
    scan2_fig = scan2_transformed

print(f"  Scan 1: {scan1_fig.shape[0]:,} points")
print(f"  Scan 2 (transformed): {scan2_fig.shape[0]:,} points")

fig, ax = plt.subplots(figsize=(14, 12))
ax.scatter(scan1_fig[:, 0], scan1_fig[:, 1], c='grey', alpha=0.2, s=2, label='Scan 1')
ax.scatter(scan2_fig[:, 0], scan2_fig[:, 1], c='blue', alpha=0.2, s=2, label='Scan 2 (transformed)')
ax.set_xlabel('X (metres)', fontsize=12)
ax.set_ylabel('Y (metres)', fontsize=12)
ax.set_title('Lecturehall Overlay – Plan View (X-Y projection)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

os.makedirs('out', exist_ok=True)
fig.savefig('out/fig6_lecturehall_overlay.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"✓ Figure saved to: out/fig6_lecturehall_overlay.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
