#!/usr/bin/env python3
"""export_rgb_inspection.py -- carve results as LAS with z-intensity and classification.

Writes three LAS files from station 1:
  - lh_kept.las: points KEPT (static kept + ghosts missed)
  - lh_removed.las: points REMOVED (ghosts caught + static damage)
  - lh_person_gt.las: ALL points with label==1 (ground truth dynamic)

For each file:
  - Intensity field: z-coordinate scaled to 0-65535 range
  - Classification field: carve verdict (0=static kept, 1=ghost removed,
    2=ghost missed, 3=static removed)
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan2twin import CarveConfig, carve
from scan2twin.io.readers import load_lecturehall

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "lecturehall")
OUT_DIR = os.path.join(HERE, "out")

MAX_POINTS = 4_000_000

# Classification codes
CLS_STATIC_KEPT = 0
CLS_GHOST_REMOVED = 1
CLS_GHOST_MISSED = 2
CLS_STATIC_REMOVED = 3

CLASS_NAMES = {
    CLS_STATIC_KEPT: "static kept",
    CLS_GHOST_REMOVED: "ghost removed",
    CLS_GHOST_MISSED: "ghost missed",
    CLS_STATIC_REMOVED: "static removed",
}


def write_las(
    path: str,
    points: np.ndarray,
    classification: np.ndarray,
    intensity: np.ndarray,
) -> None:
    """Write point cloud with classification and intensity fields."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = classification.astype(np.uint8)
    las.intensity = intensity.astype(np.uint16)
    las.write(path)


# Load data
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Loading lecturehall, max_points={MAX_POINTS:,} per station")
scene = load_lecturehall(DATA_DIR, max_points=MAX_POINTS)
print(scene.describe())
print()

s1 = scene.stations[0]

# Carve
print(f"Carving station 1 with angular_res_deg=0.180, margin_m=0.20, neighbourhood=1")
cfg = CarveConfig(angular_res_deg=0.180, margin_m=0.20, neighbourhood=1)
result = carve(scene, cfg)
print()

# Get masks and labels
mask_removed = result.masks[s1.name]  # True = removed by carver
lab1 = s1.labels
is_static = lab1 == 0
is_dyn = lab1 == 1

# Compute intensity field: scale z to 0-65535
z_min = s1.points[:, 2].min()
z_max = s1.points[:, 2].max()
z_range = z_max - z_min
if z_range > 0:
    intensity_all = ((s1.points[:, 2] - z_min) / z_range * 65535).astype(np.uint16)
else:
    intensity_all = np.zeros(len(s1), dtype=np.uint16)

# Build classification arrays for each file
# File 1: lh_kept.las -- points kept (static kept + ghosts missed)
kept_mask = ~mask_removed
kept_indices = np.where(kept_mask)[0]
kept_points = s1.points[kept_mask]
kept_intensity = intensity_all[kept_mask]

kept_cls = np.empty(len(kept_indices), dtype=np.uint8)
for i, idx in enumerate(kept_indices):
    if is_static[idx]:
        kept_cls[i] = CLS_STATIC_KEPT
    else:  # is_dyn[idx]
        kept_cls[i] = CLS_GHOST_MISSED

# File 2: lh_removed.las -- points removed (ghosts caught + static damage)
removed_indices = np.where(mask_removed)[0]
removed_points = s1.points[mask_removed]
removed_intensity = intensity_all[mask_removed]

removed_cls = np.empty(len(removed_indices), dtype=np.uint8)
for i, idx in enumerate(removed_indices):
    if is_dyn[idx]:
        removed_cls[i] = CLS_GHOST_REMOVED
    else:  # is_static[idx]
        removed_cls[i] = CLS_STATIC_REMOVED

# File 3: lh_person_gt.las -- ALL points with label==1 (ground truth dynamic)
gt_mask = is_dyn
gt_indices = np.where(gt_mask)[0]
gt_points = s1.points[gt_mask]
gt_intensity = intensity_all[gt_mask]

gt_cls = np.empty(len(gt_indices), dtype=np.uint8)
for i, idx in enumerate(gt_indices):
    if mask_removed[idx]:
        gt_cls[i] = CLS_GHOST_REMOVED
    else:
        gt_cls[i] = CLS_GHOST_MISSED

# Write LAS files
kept_path = os.path.join(OUT_DIR, "lh_kept.las")
write_las(kept_path, kept_points, kept_cls, kept_intensity)

removed_path = os.path.join(OUT_DIR, "lh_removed.las")
write_las(removed_path, removed_points, removed_cls, removed_intensity)

gt_path = os.path.join(OUT_DIR, "lh_person_gt.las")
write_las(gt_path, gt_points, gt_cls, gt_intensity)

# Print statistics
print("=" * 78)
print("Point counts by file and classification")
print("=" * 78)
print()

print(f"lh_kept.las ({len(kept_points):,} points):")
for c in (CLS_STATIC_KEPT, CLS_GHOST_MISSED):
    n = int((kept_cls == c).sum())
    if n > 0:
        pct = 100.0 * n / len(kept_points)
        print(f"  class {c} ({CLASS_NAMES[c]:<20s}): {n:>12,}  ({pct:6.3f}%)")
print()

print(f"lh_removed.las ({len(removed_points):,} points):")
for c in (CLS_GHOST_REMOVED, CLS_STATIC_REMOVED):
    n = int((removed_cls == c).sum())
    if n > 0:
        pct = 100.0 * n / len(removed_points)
        print(f"  class {c} ({CLASS_NAMES[c]:<20s}): {n:>12,}  ({pct:6.3f}%)")
print()

print(f"lh_person_gt.las ({len(gt_points):,} points):")
for c in (CLS_GHOST_REMOVED, CLS_GHOST_MISSED):
    n = int((gt_cls == c).sum())
    if n > 0:
        pct = 100.0 * n / len(gt_points)
        print(f"  class {c} ({CLASS_NAMES[c]:<20s}): {n:>12,}  ({pct:6.3f}%)")
print()

print("=" * 78)
print(f"Files written to {os.path.relpath(OUT_DIR, HERE)}/:")
print(f"  - lh_kept.las")
print(f"  - lh_removed.las")
print(f"  - lh_person_gt.las")
print("=" * 78)
