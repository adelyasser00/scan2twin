"""Honest metrics.

The whole differentiator of this project is that it publishes what the
pipeline FAILED to do, and what it did by accident. So the reporting here is
deliberately asymmetric, because the error types are not equally bad for the
person paying:

    FALSE POSITIVE  - a real surface point was deleted.
                      Catastrophic. The deliverable is now wrong and the
                      client cannot tell by looking. Report this first.

    FALSE NEGATIVE  - a ghost survived.
                      Annoying. The technician cleans it by hand, which is
                      what they were doing anyway. Report it without apology.

Any tool reporting a single "accuracy" for this task is hiding the first
behind the second: static points outnumber ghosts by two orders of magnitude,
so accuracy reads 99% while the pipeline eats a wall.

THREE CLASSES, NOT TWO
----------------------
Ground truth here is 0=static, 1=dynamic, 2=mixed-pixel noise. The third
class is not decoration. On the first run of this pipeline the reported
surface damage rate was 0.27%, which turned out to be almost exactly the
noise fraction that had been injected: the carver was correctly deleting
noise, and a two-class ground truth was scoring every one of those deletions
as destroyed geometry. A benchmark that cannot tell "removed a bad point"
from "removed a good point" does not measure cleanup quality. If a real
dataset only offers two classes, that limitation belongs in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Scene

STATIC, DYNAMIC, NOISE = 0, 1, 2


@dataclass
class Confusion:
    tp: int = 0  # ghost, removed             -> the job
    fp: int = 0  # static surface, removed    -> DAMAGE
    fn: int = 0  # ghost, kept                -> missed
    tn: int = 0  # static surface, kept       -> the job
    noise_removed: int = 0  # noise, removed  -> free bonus
    noise_kept: int = 0  # noise, kept        -> left for SOR

    @property
    def n(self) -> int:
        return (self.tp + self.fp + self.fn + self.tn
                + self.noise_removed + self.noise_kept)

    @property
    def ghost_recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def precision(self) -> float:
        """Of everything deleted, how much SHOULD have gone (ghost or noise)."""
        d = self.tp + self.fp + self.noise_removed
        return (self.tp + self.noise_removed) / d if d else float("nan")

    @property
    def ghost_precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.ghost_precision, self.ghost_recall
        return 2 * p * r / (p + r) if (p + r) else float("nan")

    @property
    def iou(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def surface_damage_rate(self) -> float:
        """Fraction of genuinely static surface points that were destroyed."""
        d = self.fp + self.tn
        return self.fp / d if d else float("nan")

    @property
    def noise_recall(self) -> float:
        d = self.noise_removed + self.noise_kept
        return self.noise_removed / d if d else float("nan")

    def __add__(self, o: "Confusion") -> "Confusion":
        return Confusion(
            self.tp + o.tp,
            self.fp + o.fp,
            self.fn + o.fn,
            self.tn + o.tn,
            self.noise_removed + o.noise_removed,
            self.noise_kept + o.noise_kept,
        )


def confusion(pred_remove: np.ndarray, gt: np.ndarray) -> Confusion:
    pred = np.asarray(pred_remove, dtype=bool)
    gt = np.asarray(gt)
    dyn = gt == DYNAMIC
    sta = gt == STATIC
    noi = gt == NOISE
    return Confusion(
        tp=int((pred & dyn).sum()),
        fp=int((pred & sta).sum()),
        fn=int((~pred & dyn).sum()),
        tn=int((~pred & sta).sum()),
        noise_removed=int((pred & noi).sum()),
        noise_kept=int((~pred & noi).sum()),
    )


def evaluate(scene: Scene, masks: dict) -> tuple[Confusion, dict]:
    total = Confusion()
    per: dict[str, Confusion] = {}
    for s in scene:
        if s.labels is None:
            continue
        c = confusion(masks[s.name], s.labels)
        per[s.name] = c
        total = total + c
    return total, per


def report(total: Confusion, per: dict, title: str = "Ghost removal") -> str:
    L = ["=" * 74, title, "=" * 74, ""]
    L.append("  WHAT IT DESTROYED   (the only number a surveyor will ask about)")
    L.append(f"    real surface points deleted : {total.fp:,} of "
             f"{total.fp + total.tn:,} static")
    L.append(f"    surface damage rate         : "
             f"{100 * total.surface_damage_rate:.4f} %")
    L.append("")
    L.append("  WHAT IT CAUGHT")
    L.append(f"    ghosts removed              : {total.tp:,} of "
             f"{total.tp + total.fn:,}")
    L.append(f"    ghost recall                : {100 * total.ghost_recall:.2f} %")
    L.append("")
    L.append("  WHAT IT MISSED      (still hand work, and some always will be)")
    L.append(f"    ghosts left behind          : {total.fn:,}")
    L.append("")
    L.append("  WHAT IT CAUGHT BY ACCIDENT")
    L.append(f"    mixed-pixel noise removed   : {total.noise_removed:,} of "
             f"{total.noise_removed + total.noise_kept:,} "
             f"({100 * total.noise_recall:.1f} %)")
    L.append("    (noise sits in space another station saw through, so the same")
    L.append("     visibility test that finds ghosts finds it. Not claimed as a")
    L.append("     feature. It is a side effect, and it is worth reporting.)")
    L.append("")
    L.append("  Combined")
    L.append(f"    ghost precision             : "
             f"{100 * total.ghost_precision:.2f} %")
    L.append(f"    precision incl. noise       : {100 * total.precision:.2f} %")
    L.append(f"    F1  (dynamic class)         : {total.f1:.4f}")
    L.append(f"    IoU (dynamic class)         : {total.iou:.4f}")
    if per:
        L += ["", "  Per station",
              f"    {'station':<20s} {'recall':>8s} {'damage':>9s} "
              f"{'ghosts':>9s} {'missed':>8s}"]
        for name, c in per.items():
            L.append(f"    {name:<20s} {100 * c.ghost_recall:7.2f}% "
                     f"{100 * c.surface_damage_rate:8.4f}% "
                     f"{c.tp + c.fn:>9,} {c.fn:>8,}")
    L.append("=" * 74)
    return "\n".join(L)


def miss_analysis(scene: Scene, masks: dict, n_bins: int = 6) -> str:
    """Why did the misses survive? Break down by range from the scanner.

    Ghosts far from the scanner are harder: wider beam footprint, fewer
    returns on the moving object, less chance another station had a clean
    line of sight. If misses concentrate at range, that is a capture geometry
    story, not a code story, and should be told that way.
    """
    ranges, missed = [], []
    for s in scene:
        if s.labels is None:
            continue
        gt = s.labels == DYNAMIC
        if not gt.any():
            continue
        ranges.append(s.ranges()[gt])
        missed.append(~masks[s.name][gt])
    if not ranges:
        return "  (no ground truth available for miss analysis)"

    r = np.concatenate(ranges)
    m = np.concatenate(missed)
    edges = np.linspace(r.min(), r.max(), n_bins + 1)
    L = ["  Missed ghosts by range from scanner", "",
         f"    {'range (m)':<20s} {'ghosts':>9s} {'missed':>9s} {'miss rate':>10s}"]
    for i in range(n_bins):
        hi_ok = r < edges[i + 1] if i < n_bins - 1 else r <= edges[i + 1]
        sel = (r >= edges[i]) & hi_ok
        n = int(sel.sum())
        if n == 0:
            continue
        mm = int(m[sel].sum())
        L.append(f"    {edges[i]:7.1f} - {edges[i+1]:7.1f} {n:>9,} {mm:>9,} "
                 f"{100.0 * mm / n:9.1f}%")
    return "\n".join(L)


def witness_analysis(scene: Scene, masks: dict, margin_m: float = 0.6) -> str:
    """How many OTHER stations had any sightline through each missed ghost.

    This separates "the algorithm failed" from "the capture geometry made it
    impossible". A ghost that no second station could ever have seen through
    is not a software defect, and pretending otherwise leads to tuning
    parameters until real walls start disappearing.
    """
    from .visibility import build_range_image, free_space_evidence

    ris = {s.name: build_range_image(s.points, s.origin, 0.5) for s in scene}
    L = ["  Missed ghosts: did any other station even have a sightline?", ""]
    tot_no_wit = tot_missed = 0
    for s in scene:
        if s.labels is None:
            continue
        gt = s.labels == DYNAMIC
        miss = gt & ~masks[s.name]
        if not miss.any():
            continue
        q = s.points[miss]
        wit = np.zeros(len(q), dtype=np.int64)
        for other in scene:
            if other.name == s.name:
                continue
            wit += free_space_evidence(ris[other.name], q, margin_m).astype(np.int64)
        no_wit = int((wit == 0).sum())
        tot_no_wit += no_wit
        tot_missed += len(q)
        L.append(f"    {s.name:<20s} missed {len(q):>7,}   "
                 f"no sightline at all: {no_wit:>7,} "
                 f"({100.0 * no_wit / len(q):5.1f}%)")
    if tot_missed:
        L += ["", f"    {100.0 * tot_no_wit / tot_missed:.1f}% of all misses were "
                  f"geometrically unrecoverable from this station layout.",
              "    That fraction is a capture-planning number, not a code-quality one."]
    return "\n".join(L)


def hours_estimate(n_removed: int, points_per_hour: float = 1_200_000) -> str:
    """Rough manual-cleanup time equivalent. THIS IS AN ESTIMATE and says so.

    `points_per_hour` is a placeholder, not a measurement. Until it is
    calibrated against a real technician on a real job, any number this
    produces must carry the caveat. Do not put a bare hours figure on a slide.
    """
    hrs = n_removed / points_per_hour
    return (f"~{hrs:.2f} h equivalent manual cleanup  [ESTIMATE: assumes "
            f"{points_per_hour:,.0f} pts/h, an uncalibrated placeholder rate, "
            f"not a measurement]")
