#!/usr/bin/env python3
"""sweep_lecturehall.py -- parameter sweep on the real Würzburg lecturehall data.

Follows run_lecturehall.py (the first honest single-config carve). That run
established a baseline at voxel 0.10 m / margin 0.20 m / neighbourhood 1:

    ghost recall            18.19 %
    surface damage rate      0.033 %   (UPPER BOUND, see below)
    misses with no sightline  6.5 %

Hypothesis under test: recall there is limited by CONFIG, not by the capture
geometry. Three named suspects:

  (a) voxel_size 0.10 m holds hundreds of points at this indoor density, so
      the strict_majority rule is far stricter here than it was on the
      synthetic plaza it was tuned on.
  (b) margin_m 0.20 m was tuned on a ~60 m outdoor plaza and is large next to
      a ~12 m room.
  (c) the data-driven angular_res estimate runs ~13 % coarse.

TWO-CLASS CAVEAT (identical to run_lecturehall.py, repeated because it changes
how every number below reads)
---------------------------------------------------------------------------
This dataset labels points 0 = static / 1 = dynamic and nothing else. There is
no mixed-pixel / sensor-noise class. Every genuine noise point the carver
correctly removes is scored here as destroyed surface geometry. So every
"surface damage" figure in this file is an UPPER BOUND on real damage, not a
measurement. It is labelled "damage UB" everywhere for that reason.

This script does NOT modify any package file. It only reads the adapter and
the carver and scores their output with metrics.evaluate, exactly as
run_lecturehall.py does.

No winner is picked. The sweep is reported in full and the final summary is
sorted by damage upper bound ascending, because recall is the flattering
number and damage is the one a surveyor will actually ask about.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan2twin import CarveConfig, carve, estimate_angular_resolution, metrics
from scan2twin.io.readers import load_lecturehall

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "lecturehall")
OUT_DIR = os.path.join(HERE, "out")
FIG_PATH = os.path.join(OUT_DIR, "fig11_lecturehall_sweep.png")
CSV_PATH = os.path.join(OUT_DIR, "fig11_lecturehall_sweep.csv")

MAX_POINTS = 2_000_000  # per station, same as run_lecturehall.py, for comparability

# --- sweep axes (exactly as specified in the task) -------------------------
VOXELS = [0.03, 0.05, 0.08, 0.12]
MARGINS = [0.05, 0.10, 0.20]
STRICT = [True, False]
NEIGHBOURHOOD = 1  # 1 only. neighbourhood 0 is NOT tested: it is documented in
#                    carve.py to raise synthetic damage from ~0.002 % to 9-50 %.

ANGULAR_FORCED = [0.10, 0.124, 0.14, 0.18]  # second table, at the best config


def rule(c: str = "=") -> None:
    print(c * 100)


# ---------------------------------------------------------------------------
# 1. load once, reuse for every run (carve does not mutate the scene)
# ---------------------------------------------------------------------------
rule()
print(f"1. Loading lecturehall, max_points={MAX_POINTS:,} per station "
      f"(systematic stride, same subset as the baseline run)")
rule()
t0 = time.perf_counter()
scene = load_lecturehall(DATA_DIR, max_points=MAX_POINTS)
t_load = time.perf_counter() - t0
print(f"  loaded in {t_load:.1f} s\n")
print(scene.describe())
print()

s1, s2 = scene.stations
sep = float(np.linalg.norm(s2.origin - s1.origin))
print(f"  scanner separation : {sep:.3f} m")
print()
print("  data-driven angular resolution estimate (what 'auto' feeds the carver,")
print("  before the x1.05 safety factor carve.py applies):")
for s in scene.stations:
    raw = estimate_angular_resolution(s.points, s.origin)
    print(f"    {s.name:<22s} raw {raw:.4f} deg   x1.05 used {raw * 1.05:.4f} deg")
print()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def score_config(cfg: CarveConfig) -> dict:
    """Run one carve, score it with metrics.evaluate, return a flat row."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = carve(scene, cfg)
    secs = time.perf_counter() - t0
    total, per = metrics.evaluate(scene, result.masks)
    ang_used = np.mean([
        result.per_station[n]["angular_res_deg"] for n in result.per_station
    ])
    return {
        "voxel": cfg.voxel_size,
        "margin": cfg.margin_m,
        "strict": cfg.strict_majority,
        "nbhd": cfg.neighbourhood,
        "angular_cfg": ("auto" if cfg.angular_res_deg is None
                        else f"{cfg.angular_res_deg:.3f}"),
        "angular_used_mean": float(ang_used),
        "recall": 100.0 * total.ghost_recall,
        "damage_ub": 100.0 * total.surface_damage_rate,
        "ghost_prec": 100.0 * total.ghost_precision,
        "f1": total.f1,
        "iou": total.iou,
        "tp": total.tp,
        "fp": total.fp,
        "fn": total.fn,
        "n_removed": result.n_removed,
        "secs": secs,
    }


def print_main_table(rows: list[dict]) -> None:
    print(f"  {'voxel':>6s} {'margin':>7s} {'strict':>7s} "
          f"{'recall %':>9s} {'damage UB %':>12s} {'ghost prec %':>13s} "
          f"{'F1':>8s} {'seconds':>9s}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  {r['voxel']:>6.2f} {r['margin']:>7.2f} {str(r['strict']):>7s} "
              f"{r['recall']:>9.2f} {r['damage_ub']:>12.4f} {r['ghost_prec']:>13.2f} "
              f"{r['f1']:>8.4f} {r['secs']:>9.2f}")


# ---------------------------------------------------------------------------
# 2. main sweep: voxel x margin x strict, neighbourhood 1, angular auto
# ---------------------------------------------------------------------------
rule()
print("2. Main sweep  --  voxel x margin x strict_majority")
print(f"   neighbourhood = {NEIGHBOURHOOD} throughout, angular_res = auto throughout")
print(f"   {len(VOXELS)} x {len(MARGINS)} x {len(STRICT)} = "
      f"{len(VOXELS) * len(MARGINS) * len(STRICT)} runs")
rule()

main_rows: list[dict] = []
for strict in STRICT:
    for voxel in VOXELS:
        for margin in MARGINS:
            cfg = CarveConfig(
                voxel_size=voxel,
                margin_m=margin,
                min_witnesses=1,
                strict_majority=strict,
                angular_res_deg=None,
                neighbourhood=NEIGHBOURHOOD,
            )
            row = score_config(cfg)
            main_rows.append(row)
            print(f"  done: voxel {voxel:.2f}  margin {margin:.2f}  "
                  f"strict {str(strict):<5s}  ->  recall {row['recall']:6.2f} %  "
                  f"damage UB {row['damage_ub']:7.4f} %  F1 {row['f1']:.4f}  "
                  f"({row['secs']:.1f} s)")

print()
print("  TABLE 1 -- main sweep, in run order")
print()
print_main_table(main_rows)
print()
print("  Baseline for reference (from run_lecturehall.py, voxel 0.10 / margin 0.20 /")
print("  strict True / nbhd 1): recall 18.19 %, damage UB 0.033 %. Not re-run here")
print("  because voxel 0.10 is not on the sweep grid; the closest point is voxel 0.12.")
print()


# ---------------------------------------------------------------------------
# 3. pick the config for the angular-resolution isolation
# ---------------------------------------------------------------------------
# "best config from above" is read here as the highest F1 (dynamic class):
# F1 is the single number that balances the two error types this project keeps
# separate, so it is the least arbitrary way to choose one config to hold
# fixed while sweeping angular_res. The choice is reported, not hidden, and it
# is NOT a recommendation -- the full table above stands on its own.
best = max(main_rows, key=lambda r: (r["f1"], -r["damage_ub"]))

rule()
print("3. Angular-resolution isolation")
rule()
print(f"   Config held fixed = highest F1 in the main sweep:")
print(f"     voxel {best['voxel']:.2f} m,  margin {best['margin']:.2f} m,  "
      f"strict {best['strict']},  neighbourhood {NEIGHBOURHOOD}")
print(f"     (that config on auto angular_res: recall {best['recall']:.2f} %, "
      f"damage UB {best['damage_ub']:.4f} %, F1 {best['f1']:.4f}, "
      f"angular used ~{best['angular_used_mean']:.4f} deg)")
print()
print("   Now forcing angular_res_deg to fixed values. Lower = finer bins =")
print("   more empty bins = less free-space evidence = lower recall, but never")
print("   lower correctness. Higher = coarser = more conservative near depth")
print("   edges. See build_range_image docstring.")
print()

ang_rows: list[dict] = []
for ang in ANGULAR_FORCED:
    cfg = CarveConfig(
        voxel_size=best["voxel"],
        margin_m=best["margin"],
        min_witnesses=1,
        strict_majority=best["strict"],
        angular_res_deg=ang,
        neighbourhood=NEIGHBOURHOOD,
    )
    row = score_config(cfg)
    ang_rows.append(row)
    print(f"  done: angular_res {ang:.3f} deg  ->  recall {row['recall']:6.2f} %  "
          f"damage UB {row['damage_ub']:7.4f} %  F1 {row['f1']:.4f}  "
          f"({row['secs']:.1f} s)")

print()
print("  TABLE 2 -- angular_res_deg forced, all else fixed at the config above")
print()
print(f"  {'angular_res deg':>15s} {'recall %':>10s} {'damage UB %':>13s} "
      f"{'ghost prec %':>13s} {'F1':>8s} {'seconds':>9s}")
print("  " + "-" * 74)
for r in ang_rows:
    print(f"  {float(r['angular_cfg']):>15.3f} {r['recall']:>10.2f} "
          f"{r['damage_ub']:>13.4f} {r['ghost_prec']:>13.2f} "
          f"{r['f1']:>8.4f} {r['secs']:>9.2f}")
print()
print(f"  For comparison, 'auto' on the same fixed config used "
      f"~{best['angular_used_mean']:.4f} deg and gave recall {best['recall']:.2f} %, "
      f"damage UB {best['damage_ub']:.4f} %.")
print()


# ---------------------------------------------------------------------------
# 4. final summary -- sorted by damage upper bound ascending
# ---------------------------------------------------------------------------
rule()
print("4. FINAL SUMMARY  --  all main-sweep configs, sorted by damage upper bound "
      "ascending")
print("   (recall is the flattering number, so it is not the sort key)")
rule()
summary = sorted(main_rows, key=lambda r: (r["damage_ub"], -r["recall"]))
print_main_table(summary)
print()
print("  Reminder: every 'damage UB' value is an UPPER BOUND. This ground truth")
print("  has no sensor-noise class, so genuine noise the carver correctly removes")
print("  is counted here as destroyed geometry. True damage is lower by whatever")
print("  fraction of those deletions were real noise. State this with any number.")
print()


# ---------------------------------------------------------------------------
# 5. CSV dump of everything
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
import csv

fields = ["voxel", "margin", "strict", "nbhd", "angular_cfg", "angular_used_mean",
          "recall", "damage_ub", "ghost_prec", "f1", "iou", "tp", "fp", "fn",
          "n_removed", "secs"]
with open(CSV_PATH, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in main_rows + ang_rows:
        w.writerow({k: r[k] for k in fields})
print(f"  wrote {os.path.relpath(CSV_PATH, HERE)}  "
      f"({len(main_rows)} main + {len(ang_rows)} angular rows)")
print()


# ---------------------------------------------------------------------------
# 6. figure: recall vs damage upper bound, one marker per config
#    same idea as fig1_sweep.png -- damage on a log x axis, recall on y
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#14181d"
RED = "#d1344a"
GREEN = "#2c8a5a"
GREY = "#5a636e"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

from matplotlib.lines import Line2D

DAMAGE_FLOOR = 1e-4  # so a zero-damage config still plots on the log axis


def render_sweep_figure(main_rows: list[dict], ang_rows: list[dict],
                        best: dict, out_path: str) -> None:
    """recall vs damage-upper-bound, one marker per config, fig1_sweep idiom.

    Two series, exactly as fig1_sweep.png has two:
      * the main grid: 24 configs. voxel_size and strict_majority turned out
        to be inert on this data, so all 24 collapse onto 3 points, one per
        margin_m value. Plotted as one marker each with the count called out,
        because drawing 8 identical markers on top of each other would imply
        a spread that is not there.
      * the angular_res_deg sweep at the held-fixed config, which is where the
        recall actually moves.
    """
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.subplots_adjust(top=0.78)

    # --- series 1: the main grid, collapsed by its only live axis (margin) --
    by_margin: dict[float, list[dict]] = {}
    for r in main_rows:
        by_margin.setdefault(r["margin"], []).append(r)
    m_pts = sorted(
        ((m, rs[0]["damage_ub"], rs[0]["recall"], len(rs)) for m, rs in by_margin.items()),
        key=lambda t: t[1],
    )
    ax.plot([max(d, DAMAGE_FLOOR) for _, d, _, _ in m_pts],
            [rc for _, _, rc, _ in m_pts],
            "s-", color=GREEN, lw=2.0, ms=10, mec="white", mew=1.3, zorder=4,
            label="main grid (angular_res = auto)")
    for i, (m, d, rc, n) in enumerate(m_pts):
        ax.annotate(f"margin {m:g} m  ({n} configs)",
                    (max(d, DAMAGE_FLOOR), rc), textcoords="offset points",
                    xytext=(6, 10 + 12 * i), fontsize=7.8, color=GREEN,
                    va="bottom", ha="left")

    # --- series 2: angular_res_deg forced, all else = best config ----------
    a_sorted = sorted(ang_rows, key=lambda r: float(r["angular_cfg"]))
    ax.plot([max(r["damage_ub"], DAMAGE_FLOOR) for r in a_sorted],
            [r["recall"] for r in a_sorted],
            "o-", color=RED, lw=2.0, ms=8, mec="white", mew=1.3, zorder=5,
            label=(f"angular_res forced  (voxel {best['voxel']:g} / "
                   f"margin {best['margin']:g} / strict {best['strict']})"))
    for r in a_sorted:
        ax.annotate(f"{float(r['angular_cfg']):.3f}\N{DEGREE SIGN}",
                    (max(r["damage_ub"], DAMAGE_FLOOR), r["recall"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.8,
                    color=RED)

    # the auto estimate, and the run_lecturehall.py baseline, are the same
    # point: margin 0.20 / auto angular_res
    ax.scatter([max(best["damage_ub"], DAMAGE_FLOOR)], [best["recall"]],
               s=230, marker="*", c="#f4c20d", edgecolors=INK, linewidths=1.0,
               zorder=6,
               label=(f"auto angular_res ≈ {best['angular_used_mean']:.3f}\N{DEGREE SIGN}"
                      "  =  run_lecturehall.py baseline"))

    ax.set_xscale("log")
    ax.set_xlabel("surface points destroyed  --  UPPER BOUND  (%, log scale)",
                  fontsize=10)
    ax.set_ylabel("ghosts removed  (%)", fontsize=10)
    ax.grid(alpha=.15, which="both")
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    fig.text(0.045, 0.965,
             "Lecturehall parameter sweep: recall vs damage upper bound",
             fontsize=14, weight="bold", va="top")
    fig.text(0.045, 0.895,
             f"Neighbourhood 1, {MAX_POINTS:,} pts/station. Damage is an UPPER "
             "BOUND: this ground truth has no sensor-noise class, so real noise "
             "the carver removes\ncounts as destroyed geometry. voxel_size "
             "(0.03–0.12 m) and strict_majority moved nothing — the "
             "24-config grid is 3 points. The lever is angular_res.\n"
             "No config is endorsed.",
             fontsize=9.0, color=GREY, va="top")

    fig.savefig(out_path)
    plt.close(fig)


render_sweep_figure(main_rows, ang_rows, best, FIG_PATH)
print(f"  wrote {os.path.relpath(FIG_PATH, HERE)}")
print()

rule()
print("Sweep complete. Full tables above, CSV in out/, figure in out/. No winner "
      "picked; the F1-max config used for table 2 is stated, not recommended.")
rule()
