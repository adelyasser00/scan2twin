"""Free-space evidence via spherical range images.

THE IDEA
--------
A terrestrial scanner sweeps a regular grid of azimuth / elevation directions.
That means its returns are naturally a *range image*: for each direction, the
distance to the first thing the beam hit.

To ask "did station B see through the place where station A recorded a
person?", we do not need to march a ray. We project the candidate location
into B's spherical frame, look up the range B recorded in that direction, and
compare. If B measured something further away than the candidate, then B's
beam passed straight through the candidate location, and nothing was there
when B fired.

Cost is O(candidate voxels) per station instead of O(rays x ray length).
That difference is what makes this survivable on a 500-scan project.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
An empty angular bin is NOT treated as free space. A direction with no return
could be sky, or it could be outside the scan's field of view, or a shadow
behind an occluder. Treating "no data" as "nothing there" is the single
easiest way to carve holes in real walls, and several published pipelines get
this wrong. Here, no data means no evidence, full stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RangeImage:
    """Nearest-return range per angular bin, in a station's local frame."""

    origin: np.ndarray
    az_bins: int
    el_bins: int
    el_min: float
    el_max: float
    data: np.ndarray  # (az_bins, el_bins) float32, +inf where no return
    n_filled: int

    @property
    def fill_ratio(self) -> float:
        return self.n_filled / float(self.az_bins * self.el_bins)


def _spherical(pts: np.ndarray, origin: np.ndarray):
    d = pts - origin
    r = np.linalg.norm(d, axis=1)
    safe = np.maximum(r, 1e-12)
    az = np.arctan2(d[:, 1], d[:, 0])  # [-pi, pi]
    el = np.arcsin(np.clip(d[:, 2] / safe, -1.0, 1.0))  # [-pi/2, pi/2]
    return r, az, el


def build_range_image(
    points: np.ndarray,
    origin: np.ndarray,
    angular_res_deg: float = 0.06,
    el_pad_deg: float = 1.0,
) -> RangeImage:
    """Build the nearest-return range image for one station.

    Parameters
    ----------
    angular_res_deg:
        Bin size. Should be at, or slightly coarser than, the scan's angular
        resolution. Too fine and most bins are empty, so we lose evidence and
        recall drops. Too coarse and the nearest-return rule makes us
        over-conservative near depth edges, which costs recall but never
        costs correctness. When unsure, go coarser. Missing a ghost is a
        recoverable disappointment; deleting a real wall is not.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    r, az, el = _spherical(points, origin)

    el_pad = np.radians(el_pad_deg)
    el_min = float(el.min()) - el_pad
    el_max = float(el.max()) + el_pad
    el_span = max(el_max - el_min, 1e-6)

    res = np.radians(angular_res_deg)
    az_bins = int(np.ceil(2 * np.pi / res))
    el_bins = int(np.ceil(el_span / res))
    az_bins = max(az_bins, 4)
    el_bins = max(el_bins, 4)

    ai = np.clip(((az + np.pi) / (2 * np.pi) * az_bins).astype(np.int64), 0, az_bins - 1)
    ei = np.clip(((el - el_min) / el_span * el_bins).astype(np.int64), 0, el_bins - 1)
    flat = ai * el_bins + ei

    img = np.full(az_bins * el_bins, np.inf, dtype=np.float32)
    # Nearest return wins. np.minimum.at is the correct scatter-min here;
    # a plain fancy-index assignment would silently keep an arbitrary point.
    np.minimum.at(img, flat, r.astype(np.float32))

    n_filled = int(np.isfinite(img).sum())
    return RangeImage(
        origin=origin,
        az_bins=az_bins,
        el_bins=el_bins,
        el_min=el_min,
        el_max=el_max,
        data=img.reshape(az_bins, el_bins),
        n_filled=n_filled,
    )


def free_space_evidence(
    ri: RangeImage,
    query: np.ndarray,
    margin_m: float,
    max_range: float | None = None,
) -> np.ndarray:
    """Boolean mask: which query points this station demonstrably saw through.

    A query point counts as free space if, in its direction, the station
    recorded a return that is further away by more than `margin_m`.

    `margin_m` absorbs registration error between stations, the beam
    footprint at range, and half a voxel diagonal. Set it too small and
    correctly-registered surfaces start carving each other. It is the
    conservatism dial and it should be exposed in the UI.
    """
    r, az, el = _spherical(query, ri.origin)

    el_span = ri.el_max - ri.el_min
    ai = ((az + np.pi) / (2 * np.pi) * ri.az_bins).astype(np.int64)
    ei = ((el - ri.el_min) / el_span * ri.el_bins).astype(np.int64)

    inside = (ei >= 0) & (ei < ri.el_bins)
    ai = np.clip(ai, 0, ri.az_bins - 1)
    ei = np.clip(ei, 0, ri.el_bins - 1)

    recorded = ri.data[ai, ei]
    free = inside & np.isfinite(recorded) & (r < (recorded - margin_m))

    if max_range is not None:
        free &= r <= max_range
    return free


def free_space_evidence_dilated(
    ri: RangeImage,
    query: np.ndarray,
    margin_m: float,
    neighbourhood: int = 1,
    max_range: float | None = None,
) -> np.ndarray:
    """Stricter variant: require the whole angular neighbourhood to agree.

    Near a depth discontinuity, a query point can land in a bin whose nearest
    return is far away while the bin next door is a wall edge. Requiring the
    (2n+1)^2 neighbourhood to all report free space kills that failure mode.

    This lowers recall and raises precision. For a survey deliverable that is
    the correct direction to be wrong in.
    """
    if neighbourhood <= 0:
        return free_space_evidence(ri, query, margin_m, max_range)

    r, az, el = _spherical(query, ri.origin)
    el_span = ri.el_max - ri.el_min
    ai0 = ((az + np.pi) / (2 * np.pi) * ri.az_bins).astype(np.int64)
    ei0 = ((el - ri.el_min) / el_span * ri.el_bins).astype(np.int64)
    inside = (ei0 >= 0) & (ei0 < ri.el_bins)

    ok = inside.copy()
    for da in range(-neighbourhood, neighbourhood + 1):
        for de in range(-neighbourhood, neighbourhood + 1):
            ai = (ai0 + da) % ri.az_bins  # azimuth wraps, elevation does not
            ei = ei0 + de
            valid = (ei >= 0) & (ei < ri.el_bins)
            ei_c = np.clip(ei, 0, ri.el_bins - 1)
            recorded = ri.data[ai, ei_c]
            this = valid & np.isfinite(recorded) & (r < (recorded - margin_m))
            ok &= this
            if not ok.any():
                break
    if max_range is not None:
        ok &= r <= max_range
    return ok
