"""Dataset adapters.

Every reader here produces `Station` objects and nothing else. The core
package never learns what a Semantic3D .txt or a Riegl .3d looks like. That
boundary is the whole reason this can be pointed at a client's own scanner
output later without touching the algorithm.

STATUS, STATED PLAINLY
----------------------
`load_semantic3d` and `load_uosr` are written from the published format
descriptions and have NOT been run against the real files, because the
machine this was built on has no network route to those hosts. Treat them as
a first draft to be verified against the actual download, not as tested code.
The synthetic path is the only one that has been executed end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..types import Scene, Station


# --------------------------------------------------------------------------
# Semantic3D
# --------------------------------------------------------------------------

SEMANTIC3D_CLASSES = {
    0: "unlabelled",
    1: "man-made terrain",
    2: "natural terrain",
    3: "high vegetation",
    4: "low vegetation",
    5: "buildings",
    6: "hard scape",
    7: "scanning artefacts",
    8: "cars and trucks",
}
SEMANTIC3D_DYNAMIC = {7}  # the class this pipeline is judged on


def load_semantic3d(
    txt_path: str | Path,
    origin: np.ndarray | None = None,
    labels_path: str | Path | None = None,
    max_points: int | None = None,
    chunk: int = 2_000_000,
) -> Station:
    """Read one Semantic3D station.

    Format: whitespace-separated ASCII, `x y z intensity r g b`, one point per
    line. Labels, when present, are a separate file with one integer per line
    in the same order.

    Two things to check before trusting the result, both of which can quietly
    invalidate everything downstream:

    1. LABELS SHIP WITH THE TRAINING SPLIT ONLY. Test split labels are held
       back for the online benchmark. If no .labels file sits next to the
       .txt, you have no ground truth and the metrics harness has nothing to
       measure against.

    2. ORIGIN. If the published file is in scanner-local coordinates the
       origin is (0, 0, 0) and everything works. If it has been registered
       into a site frame, the origin is somewhere else and is not in the
       file. Run `looks_station_local` and, if it says no, `recover_origin`.
       Do not guess. A wrong origin does not fail loudly, it produces
       confident nonsense.

    Do not read these with np.loadtxt. A 30 M point station is roughly 1.2 GB
    of text and np.loadtxt will take minutes and several times that in RAM.
    Convert once to LAZ and never touch the text again.
    """
    txt_path = Path(txt_path)
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required to read Semantic3D text files") from e

    cols = ["x", "y", "z", "intensity", "r", "g", "b"]
    frames, n = [], 0
    reader = pd.read_csv(
        txt_path, sep=r"\s+", header=None, names=cols, chunksize=chunk,
        dtype={"x": "float64", "y": "float64", "z": "float64",
               "intensity": "float32", "r": "uint8", "g": "uint8", "b": "uint8"},
    )
    for part in reader:
        frames.append(part)
        n += len(part)
        if max_points and n >= max_points:
            break
    df = frames[0] if len(frames) == 1 else __import__("pandas").concat(frames)
    if max_points:
        df = df.iloc[:max_points]

    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    labels = None
    if labels_path is None:
        guess = txt_path.with_suffix(".labels")
        labels_path = guess if guess.exists() else None
    if labels_path is not None:
        raw = np.loadtxt(labels_path, dtype=np.int16)
        if len(raw) != len(df) and max_points is None:
            raise ValueError(
                f"label count {len(raw):,} does not match point count "
                f"{len(df):,}. The files are not the pair they look like."
            )
        raw = raw[: len(pts)]
        # collapse to this package's 0=static / 1=dynamic / -1=unknown
        labels = np.where(np.isin(raw, list(SEMANTIC3D_DYNAMIC)), 1, 0).astype(np.int8)
        labels[raw == 0] = -1

    if origin is None:
        origin = np.zeros(3)

    return Station(
        name=txt_path.stem,
        points=pts,
        origin=np.asarray(origin, dtype=np.float64),
        rgb=df[["r", "g", "b"]].to_numpy(dtype=np.uint8),
        intensity=df["intensity"].to_numpy(dtype=np.float32),
        labels=labels,
        meta={
            "source": "semantic3d",
            "licence": "CC BY-NC-SA 3.0",
            "licence_note": (
                "Non-commercial, and ShareAlike: any cleaned cloud or tileset "
                "published from this data inherits the same licence. Your code "
                "does not. Do not use this dataset for a client-facing demo."
            ),
            "origin_provenance": "assumed (0,0,0)" if origin is None else "supplied",
        },
    )


# --------------------------------------------------------------------------
# Würzburg / Osnabrück Robotic 3D Scan Repository
# --------------------------------------------------------------------------


def load_uosr(
    scan_path: str | Path,
    pose_path: str | Path | None = None,
    scale: float = 1.0,
) -> Station:
    """Read one scan from the Robotic 3D Scan Repository.

    Format: `scanXXX.3d` as ASCII `x y z [reflectance]`, with a matching
    `scanXXX.pose` giving three translations and three Euler angles.

    THREE THINGS TO VERIFY AGAINST THAT DATASET'S README BEFORE TRUSTING THIS.
    The repository is not consistent across datasets and the differences are
    silent:

      * Handedness. The repository lists some sets as left-handed and some as
        right-handed. Getting this wrong mirrors the scene and the carving
        will delete surfaces at random.
      * Units. Some 3DTK data is in centimetres. Pass `scale=0.01` if so.
      * Euler convention and order. The rotation below is a placeholder.

    A wrong transform here is worse than no transform, because the scans will
    still look plausible when overlaid and the ghosts will be real geometry.
    Load two stations, overlay them, and eyeball a wall before you carve.

    Licence: attribution to the recording authors and their institution.
    No non-commercial clause, which makes this the safer choice of the two
    public options for anything attached to a business.
    """
    scan_path = Path(scan_path)
    raw = np.loadtxt(scan_path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    pts = raw[:, :3] * scale
    refl = raw[:, 3].astype(np.float32) if raw.shape[1] > 3 else None

    origin = np.zeros(3)
    if pose_path is None:
        guess = scan_path.with_suffix(".pose")
        pose_path = guess if guess.exists() else None
    if pose_path is not None:
        pose = np.loadtxt(pose_path, dtype=np.float64).ravel()
        t = pose[:3] * scale
        if len(pose) >= 6:
            rx, ry, rz = np.radians(pose[3:6])
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx  # PLACEHOLDER ORDER, verify against the README
            pts = pts @ R.T
        pts = pts + t
        origin = t

    return Station(
        name=scan_path.stem,
        points=pts,
        origin=origin,
        intensity=refl,
        meta={
            "source": "robotic_3d_scan_repository",
            "licence": "attribution to recorders and institution; no NC clause",
            "warning": "handedness, units and Euler order unverified",
        },
    )


def load_uosr_scene(directory: str | Path, name: str = "uosr", **kw) -> Scene:
    d = Path(directory)
    scans = sorted(p for p in d.glob("scan*.3d"))
    if not scans:
        raise FileNotFoundError(f"no scan*.3d files in {d}")
    return Scene(name=name, stations=[load_uosr(p, **kw) for p in scans])


# --------------------------------------------------------------------------
# Würzburg lecturehall: two terrestrial setups, dynamic ground truth
# --------------------------------------------------------------------------

LECTUREHALL_CM_TO_M = 0.01

_LECTUREHALL_FILES = {
    "pose1": ("lecturehall1.pose1.object1.label.csv", "lecturehall1.pose1.csv"),
    "pose2": ("lecturehall1.pose2.object2.label.csv", "lecturehall1.pose2.csv"),
}


def _euler_zyx(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix R = Rz @ Ry @ Rx, right-handed, angles in radians.

    This is the order verified against the lecturehall pose files, not a guess.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _count_lines(path: Path, block: int = 1 << 24) -> int:
    """Fast newline count. Used to size a systematic stride over the file."""
    n = 0
    last = b"\n"
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            n += b.count(b"\n")
            last = b[-1:]
    if last not in (b"\n", b""):
        n += 1  # final line has no trailing newline
    return n


def _read_lecturehall_points(
    points_csv: Path,
    R: np.ndarray,
    t_m: np.ndarray,
    max_points: int | None,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream one lecturehall point CSV, transform to the shared frame in metres.

    Columns are `x, y, z, scan_num, label` with no header. Coordinates are
    centimetres. The file is read in `chunk`-row blocks and never held in full.

    When `max_points` is set, a single systematic stride is taken across the
    ENTIRE file (every k-th row, k = total // max_points) rather than the first
    N rows, so a development subset still spans the whole sweep instead of one
    contiguous arc of it.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required to read lecturehall CSVs") from e

    names = ["x", "y", "z", "scan_num", "label"]
    dtypes = {"x": "float64", "y": "float64", "z": "float64",
              "scan_num": "int32", "label": "int8"}

    stride = 1
    if max_points is not None:
        total = _count_lines(points_csv)
        stride = max(1, total // int(max_points))

    reader = pd.read_csv(
        points_csv, header=None, names=names, usecols=["x", "y", "z", "label"],
        dtype=dtypes, chunksize=chunk,
    )

    xyz_parts: list[np.ndarray] = []
    lab_parts: list[np.ndarray] = []
    seen = 0
    for part in reader:
        m = len(part)
        if stride > 1:
            first = (-seen) % stride  # smallest local i with (seen + i) % stride == 0
            part = part.iloc[first::stride]
        seen += m
        xyz_parts.append(part[["x", "y", "z"]].to_numpy(dtype=np.float64))
        lab_parts.append(part["label"].to_numpy(dtype=np.int8))
        if max_points is not None and stride == 1 and sum(len(p) for p in lab_parts) >= max_points:
            break

    pts_cm = np.vstack(xyz_parts) if xyz_parts else np.zeros((0, 3))
    labels = np.concatenate(lab_parts) if lab_parts else np.zeros(0, dtype=np.int8)
    if max_points is not None and len(pts_cm) > max_points:
        pts_cm = pts_cm[:max_points]
        labels = labels[:max_points]

    pts_m = (pts_cm * LECTUREHALL_CM_TO_M) @ R.T + t_m
    return np.ascontiguousarray(pts_m), labels


def load_lecturehall(
    data_dir: str | Path,
    max_points: int | None = None,
    chunk: int = 2_000_000,
) -> Scene:
    """Read the Würzburg lecturehall dataset as a two-`Station` `Scene`.

    This is the first real-data path in the project. Everything the synthetic
    generator gave for free (units, frame, per-station origin) has to be
    reconstructed from the files here, and every one of the numbers below was
    checked against the data rather than assumed.

    LAYOUT
    ------
    `data_dir` holds four files:
        lecturehall1.pose1.object1.label.csv   (~22.3 M rows, ~790 MB)
        lecturehall1.pose2.object2.label.csv   (~22.3 M rows)
        lecturehall1.pose1.csv                 (pose, single line)
        lecturehall1.pose2.csv                 (pose, single line)
    Point CSVs are headerless `x, y, z, scan_num, label`. Pose CSVs are
    headerless `tx, ty, tz, rx, ry, rz`.

    UNITS AND FRAME (verified 6 Sep 2026)
    ------------------------------------
    Point coordinates and pose translations are CENTIMETRES; multiply by 0.01.
    Pose rotations are RADIANS. Right-handed, z up.

    TRANSFORM (validated, not assumed)
    ----------------------------------
    Each station is placed in the shared frame by its own pose:

        pts_m = (pts_cm * 0.01) @ R.T + (t_cm * 0.01),   R = Rz @ Ry @ Rx

    Pose 1 is effectively identity (t ~ (0.009, 0.049, 0.031) m, rotations
    ~1e-4 rad) and defines the frame; it is passed through the same code path
    for consistency. Pose 2 is a real 6.7 m tripod relocation with a 20.65 deg
    yaw (rz = 0.360 rad).

    The pose-2 transform was validated by nearest-neighbour distance on the
    station overlap, restricted to scan-2 points within 0.5 m of scan 1:

        identity                     : median NN 0.229 m
        R2 rotation, no translation  : median NN 0.362 m
        full pose-2 (R2 + t)         : median NN 0.046 m, 25th pct 0.030 m

    Only the full transform registers the clouds. `verify_adapter.py` re-runs
    this check on the loaded Stations as a regression test.

    LABELS
    ------
    Column 5 is 0 = static, 1 = dynamic, which is already this package's
    convention (0 static, 1 dynamic). It is copied through unchanged. There is
    NO noise class in this data and none is invented: `labels` is strictly
    two-valued. Ground truth is 521,923 dynamic points in scan 1 and zero in
    scan 2.

    Parameters
    ----------
    data_dir:
        Directory containing the four files above.
    max_points:
        Optional cap PER STATION for development on a subset. Sampling is a
        systematic stride across the whole file, not a head slice.
    chunk:
        Rows per read block. The full file is never loaded in one call.
    """
    d = Path(data_dir)

    stations: list[Station] = []
    for key in ("pose1", "pose2"):
        pts_name, pose_name = _LECTUREHALL_FILES[key]
        pts_path, pose_path = d / pts_name, d / pose_name
        if not pts_path.exists():
            raise FileNotFoundError(f"lecturehall point file missing: {pts_path}")
        if not pose_path.exists():
            raise FileNotFoundError(f"lecturehall pose file missing: {pose_path}")

        pose = np.loadtxt(pose_path, delimiter=",").ravel()
        if pose.size < 6:
            raise ValueError(f"{pose_path.name}: expected 6 values, got {pose.size}")
        t_m = pose[:3] * LECTUREHALL_CM_TO_M
        R = _euler_zyx(pose[3], pose[4], pose[5])

        pts_m, labels = _read_lecturehall_points(pts_path, R, t_m, max_points, chunk)

        stations.append(Station(
            name=f"lecturehall_{key}",
            points=pts_m,
            origin=t_m,
            labels=labels,
            meta={
                "source": "wuerzburg lecturehall (3DTK / robotic 3d scan repository)",
                "licence": "attribution to recorders and institution, no NC clause",
                "units_note": (
                    "source coordinates and pose translations are centimetres; "
                    "multiplied by 0.01 on load. pose rotations are radians. "
                    "right-handed, z up."
                ),
                "two_class_labels": (
                    "ground truth is static/dynamic only, with no label for "
                    "real sensor noise. any surface-damage figure from this "
                    "dataset is inflated by whatever genuine range noise and "
                    "mixed-pixel error the carver correctly removes, because "
                    "every such removal scores as destroyed geometry. this must "
                    "be stated with any published number."
                ),
                "pose_raw_cm_rad": pose.tolist(),
                "transform": "pts_m = (pts_cm * 0.01) @ R.T + t_m,  R = Rz @ Ry @ Rx",
                "alignment_validated": (
                    "overlap-restricted median NN 0.046 m (25th pct 0.030 m) "
                    "under the full pose-2 transform; 0.229 m under identity. "
                    "validated against the data, not assumed."
                ),
                "subsampled": max_points is not None,
            },
        ))

    return Scene(
        name="lecturehall",
        stations=stations,
        meta={
            "source": "wuerzburg lecturehall",
            "licence": "attribution to recorders and institution, no NC clause",
            "note": (
                "first real-data path. two terrestrial setups ~6.7 m apart, "
                "20.65 deg relative yaw. scan 1 defines the frame."
            ),
        },
    )


# --------------------------------------------------------------------------
# origin diagnostics: the part that decides whether a dataset is usable
# --------------------------------------------------------------------------


def looks_station_local(points: np.ndarray, tol_frac: float = 0.02) -> dict:
    """Does this cloud look like it is in scanner-local coordinates?

    Cheap, decisive check. In a scanner-local cloud the origin sits inside the
    point cloud with a small spherical void around it (nothing is measured at
    zero range), and points radiate outward in every direction. In a cloud
    registered into a site frame, (0,0,0) is arbitrary and usually nowhere
    near the middle.

    Returns a dict of evidence rather than a bare bool, because this decision
    determines whether the whole method is applicable and it deserves to be
    looked at rather than trusted.
    """
    r = np.linalg.norm(points, axis=1)
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = float(np.linalg.norm(hi - lo))
    inside = bool(np.all(lo < 0) and np.all(hi > 0))

    d = points / np.maximum(r, 1e-9)[:, None]
    az = np.arctan2(d[:, 1], d[:, 0])
    az_hist, _ = np.histogram(az, bins=72, range=(-np.pi, np.pi))
    az_coverage = float((az_hist > 0).mean())

    return {
        "origin_inside_bbox": inside,
        "min_range_m": float(r.min()),
        "min_range_frac_of_extent": float(r.min() / max(extent, 1e-9)),
        "azimuth_coverage": az_coverage,
        "verdict": bool(
            inside
            and az_coverage > 0.85
            and (r.min() / max(extent, 1e-9)) < tol_frac
        ),
        "note": (
            "verdict True means the cloud plausibly sits in scanner-local "
            "coordinates and origin=(0,0,0) is safe. False means you must "
            "supply or recover the origin. Do not proceed on a guess."
        ),
    }


def recover_origin(
    points: np.ndarray,
    coarse_steps: int = 9,
    refine_rounds: int = 5,
    sample: int = 200_000,
    angular_bins: int = 360,
    seed: int = 0,
    tripod_height: tuple[float, float] | None = (1.35, 1.95),
) -> tuple[np.ndarray, float]:
    """Estimate a TLS scanner's position from a registered cloud alone.

    THE OBJECTIVE
    -------------
    A single terrestrial scan records at most one return per beam direction.
    So from the TRUE origin, each (azimuth, elevation) bin contains points at
    essentially one range. From a WRONG origin, the same bin gathers points
    from surfaces at different depths, and the within-bin range spread blows
    up. Minimising the median within-bin range spread therefore recovers the
    origin, with no correspondence, no registration and no model.

    This exists because most public datasets ship registered clouds and throw
    the origins away, and without an origin this entire method is inert. It
    is the difference between "that dataset is unusable" and "that dataset is
    usable".

    THE VERTICAL PROBLEM, AND WHY THE TRIPOD PRIOR IS NOT A CHEAT
    -------------------------------------------------------------
    Measured on the synthetic plaza, the free 3D search recovers x and y to
    within 0.1-0.9 m and misses z by 4-10 m on every station. That is not a
    bug. A plaza is a horizontal ground plane and vertical facades, which is
    close to rotationally symmetric about the vertical axis, so sliding the
    candidate origin straight up barely changes which surfaces share an
    angular bin. The objective has a long flat valley in z and the optimiser
    settles anywhere in it.

    A wrong z is not a cosmetic error. Put the scanner 5 m up instead of 1.6 m
    and every grazing-angle relationship in the scene changes, which is the
    exact quantity the carving is most sensitive to.

    `tripod_height` constrains z to a band above the local ground beneath the
    recovered x, y. This is a real prior, not a fudge: terrestrial scans are
    shot off tripods at roughly chest height, and if a dataset was captured
    some other way you will know, because it will say so. Pass None to
    disable and reproduce the failure above.

    EXPERIMENTAL. Run only on synthetic data generated by this package, where
    it is arguably too easy. Validate on a real scan whose origin you already
    know before relying on it for one where you do not.

    Returns (origin, score). Lower score is better.
    """
    rng = np.random.default_rng(seed)
    P = points
    if len(P) > sample:
        P = P[rng.choice(len(P), sample, replace=False)]

    lo, hi = P.min(axis=0), P.max(axis=0)

    def score(o: np.ndarray) -> float:
        d = P - o
        r = np.linalg.norm(d, axis=1)
        ok = r > 1e-6
        if ok.sum() < 1000:
            return np.inf
        r = r[ok]
        d = d[ok]
        az = np.arctan2(d[:, 1], d[:, 0])
        el = np.arcsin(np.clip(d[:, 2] / r, -1, 1))
        ai = ((az + np.pi) / (2 * np.pi) * angular_bins).astype(np.int64) % angular_bins
        ei = ((el + np.pi / 2) / np.pi * angular_bins).astype(np.int64)
        key = ai * angular_bins + ei
        order = np.argsort(key, kind="stable")
        k, rr = key[order], r[order]
        edges = np.flatnonzero(np.diff(k)) + 1
        groups = np.split(rr, edges)
        spreads = [g.max() - g.min() for g in groups if len(g) > 2]
        return float(np.median(spreads)) if spreads else np.inf

    def local_ground(x: float, y: float, radius: float = 2.5) -> float:
        near = P[(np.abs(P[:, 0] - x) < radius) & (np.abs(P[:, 1] - y) < radius)]
        if len(near) < 20:
            return float(np.percentile(P[:, 2], 2))
        return float(np.percentile(near[:, 2], 5))

    best, best_s = None, np.inf
    span = hi - lo
    for _ in range(refine_rounds):
        gx = np.linspace(lo[0], hi[0], coarse_steps)
        gy = np.linspace(lo[1], hi[1], coarse_steps)
        for x in gx:
            for y in gy:
                if tripod_height is not None:
                    g = local_ground(x, y)
                    gz = np.linspace(g + tripod_height[0], g + tripod_height[1], 4)
                else:
                    gz = np.linspace(lo[2], hi[2], max(coarse_steps // 2, 3))
                for z in gz:
                    o = np.array([x, y, z])
                    s = score(o)
                    if s < best_s:
                        best_s, best = s, o
        span = span / (coarse_steps - 1) * 2.0
        lo = np.array([best[0] - span[0] / 2, best[1] - span[1] / 2, lo[2]])
        hi = np.array([best[0] + span[0] / 2, best[1] + span[1] / 2, hi[2]])
    return best, best_s
