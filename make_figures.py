#!/usr/bin/env python3
"""Figures for the case study.

Three assets, each doing a different job:

  fig1_sweep      the money chart. Recall vs surface damage, with and without
                  the angular agreement check. Proves the thesis: a demo that
                  reports only recall looks better and is catastrophically
                  wrong.

  fig2_plan       the eye candy. Top-down, ghosts in red, misses in orange.
                  Crop tight on the artifacts, never show an overview.

  fig3_mechanism  the explainer. Why one station's laser proves another
                  station's point was not there. This is the drawing that
                  makes a surveyor say "oh".
"""

from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

sys.path.insert(0, ".")

from scan2twin import CarveConfig, carve, metrics
from scan2twin.io.synthetic import build_plaza_scene

INK = "#14181d"
GREY = "#c2c8d0"
RED = "#d1344a"
AMBER = "#e08a1e"
BLUE = "#2f6fb5"
GREEN = "#2c8a5a"

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


def fig_sweep(scene, out="out/fig1_sweep.png"):
    margins = [0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
    rows = {0: [], 1: []}
    for nbhd in (0, 1):
        for m in margins:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = carve(scene, CarveConfig(margin_m=m, neighbourhood=nbhd))
            c, _ = metrics.evaluate(scene, res.masks)
            rows[nbhd].append((100 * c.surface_damage_rate, 100 * c.ghost_recall, m))

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.subplots_adjust(top=0.80)
    for nbhd, colour, label, mk in (
        (0, RED, "no angular agreement check", "o"),
        (1, GREEN, "angular agreement check on", "s"),
    ):
        d = np.array([r[0] for r in rows[nbhd]])
        r = np.array([r[1] for r in rows[nbhd]])
        ax.plot(np.maximum(d, 1e-4), r, mk + "-", color=colour, label=label,
                lw=2.2, ms=8, mec="white", mew=1.3, zorder=4)
        for k, (dd, rr, m) in enumerate(rows[nbhd]):
            dy = 9 if k % 2 == 0 else -14
            ax.annotate(f"{m:g} m", (max(dd, 1e-4), rr), textcoords="offset points",
                        xytext=(6, dy), fontsize=7.8, color=colour, alpha=.9)

    ax.set_xscale("log")
    ax.set_xlim(6e-5, 120)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y - 1.2, hi_y + 1.4)
    ax.set_xlabel("real surface points destroyed   (%, log scale)", fontsize=10)
    ax.set_ylabel("ghosts removed   (%)", fontsize=10)
    ax.axvspan(6e-5, 0.01, color=GREEN, alpha=.07, zorder=0)
    ax.text(7e-5, ax.get_ylim()[1] - 0.45, "usable for a survey deliverable",
            fontsize=8.4, color=GREEN, va="top")
    ax.text(0.6, ax.get_ylim()[1] - 0.45, "unusable at any recall",
            fontsize=8.4, color=RED, va="top")
    ax.legend(frameon=False, loc="center left", fontsize=9.5,
              bbox_to_anchor=(0.02, 0.55))
    ax.grid(alpha=.15, which="both")

    fig.text(0.045, 0.955, "Recall is the flattering number. Damage is the honest one.",
             fontsize=14, weight="bold", va="top")
    fig.text(0.045, 0.895,
             "Each marker is one free-space margin setting on the same 4-station synthetic capture.\n"
             "Dropping the angular agreement check buys 8 points of recall and destroys up to a third of the real scene.",
             fontsize=9.2, color="#5a636e", va="top")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_plan(scene, res, out="out/fig2_plan.png", station_idx=1):
    s = scene.stations[station_idx]
    pred = res.masks[s.name]
    gt = s.labels
    P = s.points

    kept = ~pred & (gt == 0)
    hit = pred & (gt == 1)
    miss = ~pred & (gt == 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9))
    for ax, show_ghosts, title in (
        (axes[0], True, "as captured"),
        (axes[1], False, "after cross-station carving"),
    ):
        sub = np.random.default_rng(1).permutation(np.flatnonzero(kept))[:150000]
        ax.scatter(P[sub, 0], P[sub, 1], s=.045, c=GREY, lw=0, rasterized=True)
        if show_ghosts:
            ax.scatter(P[hit, 0], P[hit, 1], s=2.6, c=RED, lw=0,
                       label=f"ghosts ({hit.sum():,})")
            ax.scatter(P[miss, 0], P[miss, 1], s=2.6, c=AMBER, lw=0,
                       label=f"missed ({miss.sum():,})")
        else:
            ax.scatter(P[miss, 0], P[miss, 1], s=2.6, c=AMBER, lw=0,
                       label=f"residue left for hand cleanup ({miss.sum():,})")
        ax.plot(*s.origin[:2], "^", color=BLUE, ms=10, mec="white", mew=1.4, zorder=6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  bbox_to_anchor=(0.0, -0.09), markerscale=6, ncol=2,
                  handletextpad=.4, columnspacing=1.6)
        ax.set_xlabel("x (m)")
        ax.grid(alpha=.14)
    axes[0].set_ylabel("y (m)")
    fig.suptitle(f"{s.name}: plan view, {len(s):,} points",
                 fontsize=12.5, x=.09, ha="left", weight="bold")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_mechanism(out="out/fig3_mechanism.png"):
    """Schematic. Deliberately drawn, not plotted: it is an explainer."""
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    A = np.array([1.0, 1.2])
    B = np.array([13.5, 1.6])
    wall_x = 9.0
    ghost = np.array([5.6, 3.3])

    ax.plot([wall_x, wall_x], [-0.4, 7.4], color=INK, lw=6, solid_capstyle="butt")
    ax.text(wall_x + .25, 7.0, "facade", fontsize=9.5, color=INK)

    for t in np.linspace(-0.5, 0.55, 15):
        tgt = ghost + np.array([0, t])
        ax.plot([A[0], tgt[0]], [A[1], tgt[1]], color=RED, lw=.7, alpha=.5)
    ax.add_patch(Circle(ghost, .55, color=RED, alpha=.30, lw=0))
    ax.add_patch(Circle(ghost, .30, color=RED, alpha=.85, lw=0))
    ax.text(ghost[0] - .1, ghost[1] + .95,
            "person, present only\nwhen A scanned",
            fontsize=9, color=RED, ha="center")

    for t in np.linspace(-0.9, 1.0, 11):
        d = (ghost + np.array([0, t])) - B
        d = d / np.linalg.norm(d)
        end = B + d * ((wall_x - B[0]) / d[0])
        ax.plot([B[0], end[0]], [B[1], end[1]], color=BLUE, lw=.8, alpha=.55)

    ax.plot(*A, "^", color=RED, ms=15, mec="white", mew=1.6, zorder=6)
    ax.text(A[0], A[1] - .75, "station A", fontsize=10, color=RED,
            ha="center", weight="bold")
    ax.plot(*B, "^", color=BLUE, ms=15, mec="white", mew=1.6, zorder=6)
    ax.text(B[0], B[1] - .75, "station B", fontsize=10, color=BLUE,
            ha="center", weight="bold")

    ax.annotate("", xy=(wall_x - .2, 4.6), xytext=(ghost[0] + .7, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5))
    ax.text(6.6, 5.5,
            "B's beams reach the wall.\nThey passed through the space\n"
            "where A recorded a person.",
            fontsize=9.5, color=BLUE)
    ax.text(0.4, 6.6,
            "No timestamps. No tracking. No model.\n"
            "Two viewpoints and their origins is the whole dependency list.",
            fontsize=9.5, color=INK, weight="bold")

    ax.set_xlim(-.5, 15.5)
    ax.set_ylim(-1.4, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Why a second tripod position is proof",
                 fontsize=13, loc="left", weight="bold", pad=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_grazing(out="out/fig4_grazing.png"):
    """The bug that ate the ground plane, drawn as a chart."""
    ang = np.linspace(0.6, 45, 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for off, c in ((0.03, BLUE), (0.07, AMBER), (0.15, RED)):
        ax.plot(ang, off / np.sin(np.radians(ang)), color=c, lw=2,
                label=f"{off*100:.0f} cm positional offset")
    ax.axhline(0.20, color=INK, ls="--", lw=1.2)
    ax.text(30, 0.23, "typical 0.20 m free-space margin", fontsize=8.5)
    ax.axvline(2.3, color="#8a929c", ls=":", lw=1.2)
    ax.text(2.6, 6, "ground at 40 m from a\n1.6 m tripod = 2.3°",
            fontsize=8.5, color="#5a636e")
    ax.set_yscale("log")
    ax.set_xlabel("angle of incidence  (degrees)")
    ax.set_ylabel("apparent range error  (m, log scale)")
    ax.set_title("Why querying a voxel centre destroys floors",
                 fontsize=12.5, loc="left", weight="bold", pad=12)
    ax.text(0, 1.015,
            "Sub-voxel offset divided by sin(incidence). On a wall it is "
            "nothing. On the ground at range it is metres.",
            transform=ax.transAxes, fontsize=8.6, color="#5a636e", va="bottom")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.16, which="both")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    scene = build_plaza_scene(az_steps=1100, el_steps=220, n_walkers=14)
    res = carve(scene, CarveConfig())
    outs = [
        fig_sweep(scene),
        fig_plan(scene, res),
        fig_mechanism(),
        fig_grazing(),
    ]
    for o in outs:
        print("wrote", o)
