#!/usr/bin/env python3
"""export_inspection.py -- carve results as LAS for CloudCompare inspection.

The plan-view matplotlib figures cannot show whether the cleaning is doing the
right thing in 3D. This script writes one LAS per carve config for station 1,
with a per-point classification code so the four outcomes (kept correctly,
removed correctly, missed ghost, destroyed surface) can be coloured and rotated
in CloudCompare. Station 2 is written once, all-static, so both stations can be
loaded into the same viewer in the same registered frame.

No tuning. No analysis beyond the per-class point counts. Two configs only.
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
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan2twin import CarveConfig, carve
from scan2twin.io.readers import load_lecturehall

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "lecturehall")
OUT_DIR = os.path.join(HERE, "out")

MAX_POINTS = 4_000_000

# classification codes written into the LAS classification field
CLS_STATIC_KEPT = 0      # correct   : static point, not removed
CLS_GHOST_REMOVED = 1    # correct   : dynamic point, removed
CLS_GHOST_MISSED = 2     # failure   : dynamic point, left in place
CLS_STATIC_REMOVED = 3   # failure   : static point, removed (destroyed geometry)

CLASS_NAMES = {
    CLS_STATIC_KEPT: "static, kept        (correct)",
    CLS_GHOST_REMOVED: "ghost,  removed     (correct)",
    CLS_GHOST_MISSED: "ghost,  missed      (failure: leftover)",
    CLS_STATIC_REMOVED: "static, removed     (failure: destroyed geometry)",
}

CONFIGS = {
    "A": CarveConfig(angular_res_deg=None, margin_m=0.20, neighbourhood=1),
    "B": CarveConfig(angular_res_deg=0.180, margin_m=0.20, neighbourhood=1),
}


def rule(c: str = "=") -> None:
    print(c * 78)


def write_las(path: str, points: np.ndarray, classification: np.ndarray) -> None:
    """Write an unstructured point cloud with a classification code per point."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = classification.astype(np.uint8)
    las.write(path)


# ---------------------------------------------------------------------------
# 1. load
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

rule()
print(f"1. Loading lecturehall, max_points={MAX_POINTS:,} per station")
rule()
t0 = time.perf_counter()
scene = load_lecturehall(DATA_DIR, max_points=MAX_POINTS)
print(f"  loaded in {time.perf_counter() - t0:.1f} s\n")
print(scene.describe())
print()

s1, s2 = scene.stations

# ---------------------------------------------------------------------------
# 2. carve at both configs, write station-1 LAS for each
# ---------------------------------------------------------------------------
for tag, cfg in CONFIGS.items():
    rule()
    print(f"2{tag}. Carve config {tag}: {cfg.describe()}")
    rule()
    t0 = time.perf_counter()
    result = carve(scene, cfg)
    print(f"  carved in {time.perf_counter() - t0:.1f} s")

    mask1 = result.masks[s1.name]          # True = flagged ghost / removed
    lab1 = s1.labels
    is_static = lab1 == 0
    is_dyn = lab1 == 1

    code = np.empty(len(s1), dtype=np.uint8)
    code[is_static & ~mask1] = CLS_STATIC_KEPT
    code[is_dyn & mask1] = CLS_GHOST_REMOVED
    code[is_dyn & ~mask1] = CLS_GHOST_MISSED
    code[is_static & mask1] = CLS_STATIC_REMOVED

    print(f"\n  station 1 ({s1.name}), {len(s1):,} points:")
    for c in (CLS_STATIC_KEPT, CLS_GHOST_REMOVED, CLS_GHOST_MISSED, CLS_STATIC_REMOVED):
        n = int((code == c).sum())
        print(f"    class {c}  {CLASS_NAMES[c]:<48s} {n:>12,}  "
              f"({100.0 * n / len(s1):6.3f} %)")

    out_path = os.path.join(OUT_DIR, f"inspect_{tag}.las")
    write_las(out_path, s1.points, code)
    print(f"\n  wrote {os.path.relpath(out_path, HERE)}")
    print()

# ---------------------------------------------------------------------------
# 3. station 2, all static, once
# ---------------------------------------------------------------------------
rule()
print("3. Station 2 -- all classification 0, for shared-frame context")
rule()
code2 = np.zeros(len(s2), dtype=np.uint8)
out2 = os.path.join(OUT_DIR, "inspect_station2.las")
write_las(out2, s2.points, code2)
print(f"  station 2 ({s2.name}), {len(s2):,} points, all class 0")
print(f"  wrote {os.path.relpath(out2, HERE)}")
print()

# ---------------------------------------------------------------------------
# 4. CloudCompare instructions
# ---------------------------------------------------------------------------
rule()
print("4. Colour by classification in CloudCompare")
rule()
print("""
  1. File > Open, and select out/inspect_A.las (or inspect_B.las). In the
     LAS import dialog, keep the "Classification" field ticked so it is
     loaded as a scalar field. Repeat File > Open for out/inspect_station2.las
     to see both scanner setups in the same registered frame.

  2. In the DB Tree (left panel), click the inspect_A cloud to select it.

  3. In the Properties panel (bottom left):
       - set "Colors" to "Scalar field"
       - under "SF display params", set "Current" to "Classification"

  4. Open the scalar-field colour ramp editor (the small colour-bar / "Edit"
     button next to the SF name, or Edit > Scalar fields > Edit ...). Set:
       - "Steps" / colour scale steps to 4
       - display range 0 to 3
     so each integer class gets one flat colour rather than a gradient.
     Optionally switch the colour scale to a categorical one
     ("Edit > Scalar fields > ...") for clearer separation.

  5. Class meaning (classification field value):
       0  static, kept       -> correct   (leave these; they are the surface)
       1  ghost,  removed     -> correct   (the cleaning working)
       2  ghost,  missed      -> failure   (dynamic object left behind)
       3  static, removed     -> failure   (real geometry destroyed)

  6. To isolate one class: Edit > Scalar fields > Filter by value, set the
     range to e.g. 3 to 3, and "Export" to pull just the destroyed-surface
     points into their own cloud for a close look.
""")
rule()
print("Done. LAS files in out/: inspect_A.las, inspect_B.las, inspect_station2.las")
rule()
