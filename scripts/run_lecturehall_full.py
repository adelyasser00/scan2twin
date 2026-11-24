#!/usr/bin/env python3
"""Carve the complete lecturehall scans and record the numbers.

Every earlier real-data run used a subset of each scan. This one loads all
44.6 M points, so nothing depends on how a subset was drawn. It runs the
default (auto) bins and a fixed list of bin sizes, and writes every row to
results/lecturehall_full.json. No row is picked here.

0.180 degrees is marked because the 6 Sep 2026 subset sweep already held it as
its highest-recall row, before any full-data run existed. It was still chosen
on these same labels, so it is not a held-out result.

Run from anywhere:  .venv/Scripts/python scripts/run_lecturehall_full.py
Needs about 6 GB of free memory and a minute per row.
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scan2twin import CarveConfig, carve, metrics  # noqa: E402
from scan2twin.io.readers import load_lecturehall  # noqa: E402

OUT = os.path.join(ROOT, "results", "lecturehall_full.json")
BINS = [None, 0.180, 0.060, 0.100, 0.140, 0.250]  # None = auto, the default


def main() -> None:
    scene = load_lecturehall(os.path.join(ROOT, "data", "lecturehall"), max_points=None)
    print(scene.describe(), flush=True)
    rec = {
        "dataset": "Wuerzburg lecturehall, all points, both stations",
        "points": {s.name: len(s) for s in scene},
        "moving_points": {s.name: int((s.labels == 1).sum()) for s in scene},
        "config": "voxel 0.10 m, margin 0.20 m, min_witnesses 1, strict majority, neighbourhood 1",
        "labels": "two classes, no noise class: surface damage is an upper bound under these labels",
        "rows": [],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ang in BINS:
        t0 = time.perf_counter()
        res = carve(scene, CarveConfig(voxel_size=0.10, margin_m=0.20, angular_res_deg=ang))
        secs = time.perf_counter() - t0
        c, _ = metrics.evaluate(scene, res.masks)
        row = {
            "bins": "auto" if ang is None else f"{ang:.3f}",
            "bins_used_deg": [round(res.per_station[s.name]["angular_res_deg"], 4) for s in scene],
            "tp": c.tp, "fn": c.fn, "fp": c.fp, "tn": c.tn,
            "recall_pct": round(100 * c.ghost_recall, 2),
            "damage_upper_bound_pct": round(100 * c.surface_damage_rate, 4),
            "precision_pct": round(100 * c.ghost_precision, 1),
            "seconds": round(secs, 1),
        }
        rec["rows"].append(row)
        print(row, flush=True)
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
