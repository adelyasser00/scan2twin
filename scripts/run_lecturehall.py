#!/usr/bin/env python3
"""run_lecturehall.py -- first carve on real data.

Würzburg lecturehall, two terrestrial Riegl VZ-400 setups ~6.7 m apart.
This is the first time the cross-station carver has been pointed at real
hardware rather than the synthetic generator. One config, one honest number.
No sweep, no tuning, no scale-up. See the task note at the bottom.

TWO-CLASS CAVEAT (repeated here because it changes how the output reads)
----------------------------------------------------------------------
This dataset labels points 0=static / 1=dynamic and nothing else. There is
no mixed-pixel / noise class. Every genuine sensor-noise point the carver
correctly removes is scored here as destroyed surface geometry, because the
ground truth has no way to say otherwise. The reported surface-damage rate is
therefore an UPPER BOUND on real damage, not a measurement. This is stated
explicitly in the output.
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

from scan2twin import CarveConfig, carve, apply, estimate_angular_resolution, metrics
from scan2twin.io.readers import load_lecturehall

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "lecturehall")
OUT_DIR = os.path.join(HERE, "out")
FIG_PATH = os.path.join(OUT_DIR, "fig10_lecturehall_carve.png")

MAX_POINTS = 2_000_000


def rule(c: str = "=") -> None:
    print(c * 78)


# ---------------------------------------------------------------------------
# 1. load
# ---------------------------------------------------------------------------
rule()
print(f"1. Loading lecturehall, max_points={MAX_POINTS:,} per station")
rule()
t0 = time.perf_counter()
scene = load_lecturehall(DATA_DIR, max_points=MAX_POINTS)
t_load = time.perf_counter() - t0
print(f"  loaded in {t_load:.1f} s\n")
print(scene.describe())
print()

s1, s2 = scene.stations
sep = float(np.linalg.norm(s2.origin - s1.origin))
print(f"  scanner separation          : {sep:.3f} m")
print()

# ---------------------------------------------------------------------------
# 2. angular resolution -- estimated from the data, tested against real hw
# ---------------------------------------------------------------------------
rule()
print("2. Angular resolution estimate (CarveConfig angular_res_deg = None)")
rule()
print("   Real Riegl VZ-400 scan. This is the first time the data-driven")
print("   estimator in carve.estimate_angular_resolution runs against real")
print("   hardware rather than the synthetic generator.\n")
print(f"   {'station':<22s} {'raw estimate':>14s} {'x1.05 (used)':>14s}")
for s in scene.stations:
    raw = estimate_angular_resolution(s.points, s.origin)
    print(f"   {s.name:<22s} {raw:>13.4f}\N{DEGREE SIGN} {raw * 1.05:>13.4f}\N{DEGREE SIGN}")
print()
print("   (fill ratio of the range image built at that resolution is reported")
print("    per station in the carve result below.)")
print()

# ---------------------------------------------------------------------------
# 3. carve
# ---------------------------------------------------------------------------
cfg = CarveConfig(
    voxel_size=0.10,
    margin_m=0.20,
    min_witnesses=1,
    strict_majority=True,
    angular_res_deg=None,
    neighbourhood=1,
)

rule()
print("3. Carve")
rule()
print(f"   config: {cfg.describe()}\n")

t0 = time.perf_counter()
result = carve(scene, cfg)
t_carve = time.perf_counter() - t0

print(result.describe(scene))
print()

print("   Estimated angular resolution and range-image fill ratio, per station:")
print(f"   {'station':<22s} {'ang_res used':>14s} {'fill ratio':>12s} "
      f"{'max range':>12s}")
for name in (s1.name, s2.name):
    i = result.per_station[name]
    print(f"   {name:<22s} {i['angular_res_deg']:>13.4f}\N{DEGREE SIGN} "
          f"{i['fill_ratio']:>12.3f} {i['max_recorded_range']:>10.2f} m")
print()

# ---------------------------------------------------------------------------
# 4. score
# ---------------------------------------------------------------------------
rule()
print("4. Metrics")
rule()
total, per = metrics.evaluate(scene, result.masks)
print(metrics.report(total, per, title="Lecturehall -- first carve, stated config"))
print()
print("  " + "-" * 74)
print("  READ THIS BEFORE QUOTING THE SURFACE-DAMAGE NUMBER")
print("  " + "-" * 74)
print("  This dataset is TWO-CLASS: 0=static, 1=dynamic. There is NO noise")
print("  label. metrics.noise_removed / noise_kept are structurally zero:")
print(f"      noise_removed = {total.noise_removed} , noise_kept = {total.noise_kept}")
print("  Every real mixed-pixel / range-noise point the carver correctly")
print("  removes is counted above as a destroyed static surface point.")
print()
print(f"  => The reported surface-damage rate of "
      f"{100 * total.surface_damage_rate:.4f} % is an UPPER BOUND,")
print("     not a measurement. True surface damage is lower by whatever")
print("     fraction of those deletions were genuine sensor noise.")
print("  " + "-" * 74)
print()

# ---------------------------------------------------------------------------
# 5. miss and witness analysis
# ---------------------------------------------------------------------------
rule()
print("5. Miss analysis")
rule()
print(metrics.miss_analysis(scene, result.masks))
print()
rule()
print("5b. Witness analysis")
rule()
print("   Only two stations here, so a large share of surviving ghosts may")
print("   simply have had no second sightline. This separates 'algorithm")
print("   missed it' from 'capture geometry made it impossible'.\n")
print(metrics.witness_analysis(scene, result.masks))
print()

# ---------------------------------------------------------------------------
# 6. figure: plan view of station 1, as-captured vs after carving
# ---------------------------------------------------------------------------
rule()
print(f"6. Figure -> {os.path.relpath(FIG_PATH, HERE)}")
rule()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(0)


def sub(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[RNG.choice(len(a), n, replace=False)]


mask1 = result.masks[s1.name]          # True = flagged ghost / removed
lab1 = s1.labels
P = s1.points

is_static = lab1 == 0
is_dyn = lab1 == 1

static_kept = is_static & ~mask1
static_removed = is_static & mask1     # <- the damage (upper bound), shown honestly
ghost_removed = is_dyn & mask1
ghost_missed = is_dyn & ~mask1

p_static_kept = sub(P[static_kept], 250_000)
p_static_removed = P[static_removed]
p_ghost_removed = P[ghost_removed]
p_ghost_missed = P[ghost_missed]
p_dyn_all = sub(P[is_dyn], 80_000)
p_static_all = sub(P[is_static], 250_000)

fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), sharex=True, sharey=True)

# --- panel A: as captured -------------------------------------------------
axA = axes[0]
axA.scatter(p_static_all[:, 0], p_static_all[:, 1], s=1.0, c="0.62",
            alpha=0.30, linewidths=0, label=f"static ({int(is_static.sum()):,})")
axA.scatter(p_dyn_all[:, 0], p_dyn_all[:, 1], s=2.0, c="tab:orange",
            alpha=0.55, linewidths=0,
            label=f"dynamic, ground truth ({int(is_dyn.sum()):,})")
axA.set_title(f"{s1.name}: as captured")

# --- panel B: after carving --------------------------------------------
axB = axes[1]
AMBER = "#ffb000"
axB.scatter(p_static_kept[:, 0], p_static_kept[:, 1], s=1.0, c="0.62",
            alpha=0.30, linewidths=0,
            label=f"static kept ({int(static_kept.sum()):,})")
axB.scatter(p_ghost_missed[:, 0], p_ghost_missed[:, 1], s=2.5, c=AMBER,
            alpha=0.75, linewidths=0,
            label=f"ghost missed ({int(ghost_missed.sum()):,})")
if len(p_static_removed):
    axB.scatter(p_static_removed[:, 0], p_static_removed[:, 1], s=3.0,
                c="magenta", alpha=0.60, linewidths=0,
                label=f"static removed — damage upper bound "
                      f"({int(static_removed.sum()):,})")
axB.scatter(p_ghost_removed[:, 0], p_ghost_removed[:, 1], s=3.0, c="red",
            alpha=0.75, linewidths=0, zorder=4,
            label=f"ghost removed ({int(ghost_removed.sum()):,})")
axB.set_title(f"{s1.name}: after carving")

for ax in axes:
    for s, mk in ((s1, "^"), (s2, "v")):
        ax.plot(s.origin[0], s.origin[1], mk, ms=13, mfc="yellow", mec="k",
                mew=1.5, zorder=5)
        ax.annotate(f"  {s.name}", (s.origin[0], s.origin[1]), fontsize=8,
                    va="center", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Y (m)")

fig.suptitle(
    "Lecturehall station 1 -- plan view (metres, shared frame)\n"
    f"voxel {cfg.voxel_size} m, margin {cfg.margin_m} m, min_witnesses "
    f"{cfg.min_witnesses}, strict_majority {cfg.strict_majority}, "
    f"neighbourhood {cfg.neighbourhood}   |   "
    f"ghost recall {100 * per[s1.name].ghost_recall:.1f} %, "
    f"surface-damage upper bound {100 * per[s1.name].surface_damage_rate:.4f} %",
    fontsize=11)

for ax in axes:
    leg = ax.legend(loc="upper right", framealpha=0.9, markerscale=5)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass

fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  saved {FIG_PATH}")
print()

# ---------------------------------------------------------------------------
# 7. wall clock and throughput
# ---------------------------------------------------------------------------
rule()
print("7. Wall clock and throughput")
rule()
n_pts = scene.n_points
print(f"  scene points (both stations): {n_pts:,}")
print(f"  load wall clock             : {t_load:8.2f} s")
print(f"  carve wall clock            : {t_carve:8.2f} s")
for k, v in result.timings.items():
    print(f"      t[{k:<13s}]        : {v:8.2f} s")
print(f"  carve throughput            : {n_pts / t_carve:,.0f} points / s")
print(f"  end-to-end (load + carve)   : {t_load + t_carve:8.2f} s  "
      f"({n_pts / (t_load + t_carve):,.0f} pts/s)")
print()
rule()
print("STOP. First honest number recorded at the stated config. No sweep,")
print("no tuning, no scale-up, as instructed.")
rule()
