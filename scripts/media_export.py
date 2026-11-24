#!/usr/bin/env python3
"""Portfolio media from the recorded full-data lecturehall run. Writes media_export/.

Re-runs two rows of results/lecturehall_full.json (the auto default and 0.180
degrees), refuses to draw anything if the counts differ from the recorded ones,
and renders:

    lecturehall_before_after_1600x900.webp
    lecturehall_portrait_1080x1350.webp
    lecturehall_before_after_loop.mp4 / .gif
    lecturehall_moving_points_1600x900.webp
    FILES.md

Run from anywhere:  .venv/Scripts/python scripts/media_export.py
Needs data/lecturehall/ and ffmpeg (FFMPEG=path, or on PATH, or imageio-ffmpeg).
MEDIA_FONT_DIR may point at a folder with Barlow-Medium.ttf and
Barlow-SemiBold.ttf; otherwise matplotlib's DejaVu Sans is used.

DISPLAY AXIS: the stored z axis points DOWN in this dataset (elevations run
-60 to +40 degrees and the seat rows hang from the top if z is drawn up). The
carver does not care which way is up. For display the cloud is rotated 180
degrees about x, which keeps its handedness as stored.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scan2twin import CarveConfig, carve, metrics  # noqa: E402
from scan2twin.io.readers import load_lecturehall  # noqa: E402

OUT = os.path.join(ROOT, "media_export")
RECORD = os.path.join(ROOT, "results", "lecturehall_full.json")  # from run_lecturehall_full.py

# (key, row in the record, bin size)
CONFIGS = (("default", "auto", None), ("0.18", "0.180", 0.180))

BG = (13, 16, 22)
INK = (240, 242, 246)
DIM = (160, 168, 180)
STATIC, GHOST, LOST, GONE = 0, 1, 2, 3
PALETTE = np.array([[186, 194, 205], [255, 140, 40], [255, 70, 200]], float)
CAM = dict(cam=(1.30, 5.15, 0.62), target=(1.62, 1.90, 0.06), fov=56)


def font(size, bold=False):
    d = os.environ.get("MEDIA_FONT_DIR")
    if d:
        return ImageFont.truetype(os.path.join(d, "Barlow-SemiBold.ttf" if bold else "Barlow-Medium.ttf"), size)
    import matplotlib
    ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    return ImageFont.truetype(os.path.join(ttf, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), int(size * 0.86))


def render(points, kind, W, H, cam, target, fov, ss=2, world=0.006, maxpx=3, chunk=3_000_000):
    """Z-buffered point splats with eye-dome shading. Numpy only, in chunks."""
    W2, H2 = W * ss, H * ss
    c = np.array(cam, float)
    f = np.array(target, float) - c
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fp = W2 / 2 / np.tan(np.radians(fov) / 2)
    D = np.full(H2 * W2, np.inf, np.float32)
    K = np.full(H2 * W2, 255, np.uint8)
    for lo in range(0, len(points), chunk):
        k = kind[lo:lo + chunk]
        d = points[lo:lo + chunk] * np.array([1.0, -1.0, -1.0]) - c  # display axis, see the docstring
        z = d @ f
        m = (z > 0.25) & (k != GONE)
        d, z, k = d[m], z[m], k[m]
        x, y = W2 / 2 + fp * (d @ r) / z, H2 / 2 - fp * (d @ u) / z
        m = (x > -8) & (x < W2 + 8) & (y > -8) & (y < H2 + 8)
        x, y, z, k = x[m], y[m], z[m], k[m]
        o = np.argsort(-z, kind="stable")  # far to near, so the nearest write wins
        x, y, z, k = x[o], y[o], z[o].astype(np.float32), k[o]
        size = np.clip(np.round(world * fp / z), 1, maxpx).astype(int)
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        for dx in range(maxpx):
            for dy in range(maxpx):
                X, Y = xi + dx - size // 2, yi + dy - size // 2
                q = (size > dx) & (size > dy) & (X >= 0) & (X < W2) & (Y >= 0) & (Y < H2)
                if not q.any():
                    continue
                i = Y[q] * W2 + X[q]
                t = np.full(H2 * W2, np.inf, np.float32)
                tk = np.full(H2 * W2, 255, np.uint8)
                t[i], tk[i] = z[q], k[q]
                w = t < D
                D[w], K[w] = t[w], tk[w]
    D, K = D.reshape(H2, W2), K.reshape(H2, W2)
    far = float(D[np.isfinite(D)].max())
    Df = np.where(np.isfinite(D), D, far)
    Z = np.log2(np.where(np.isfinite(D), D, far * 1.5))
    s = np.zeros_like(Z)
    for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (2, -2), (-2, -2)):
        s += np.clip(Z - np.roll(Z, (dy, dx), (0, 1)), 0, 0.5)
    shade = np.exp(-4.5 * s)
    fog = np.clip(1.08 - 0.045 * Df, 0.45, 1)
    bg = np.array(BG, float)
    img = np.empty((H2, W2, 3))
    img[:] = bg
    for j in (STATIC, GHOST, LOST):
        q = K == j
        a = shade[q] * (fog[q] if j == STATIC else 1.0)
        if j == LOST:
            a = np.ones_like(a)
        a = np.clip(a, 0.18, 1)[:, None]
        img[q] = PALETTE[j] * a + bg * (1 - a)
    return Image.fromarray(img.astype(np.uint8)).resize((W, H), Image.LANCZOS)


def text(d, xy, s, size, fill=INK, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def legend(d, x, y, items, size, gap=26):
    """Coloured dot plus label, left to right."""
    f = font(size)
    for colour, label in items:
        r = size * 0.32
        cy = y + size * 0.62
        d.ellipse((x, cy - r, x + 2 * r, cy + r), fill=colour)
        x += 2 * r + size * 0.4
        d.text((x, y), label, font=f, fill=INK)
        x += d.textlength(label, font=f) + gap


ORANGE = tuple(int(v) for v in PALETTE[GHOST])
PINK = tuple(int(v) for v in PALETTE[LOST])
# The repository requires naming the recorder and the institution.
SOURCE = "Data: Johannes Schauer, University of Würzburg (Robotic 3D Scan Repository)"


def build_states():
    print("loading", flush=True)
    with open(RECORD, encoding="utf-8") as f:
        record = {r["bins"]: r for r in json.load(f)["rows"]}
    scene = load_lecturehall(os.path.join(ROOT, "data", "lecturehall"), max_points=None)
    L = np.concatenate([s.labels for s in scene])
    states = {"before": dict(kind=(L == 1).astype(np.uint8), moving=int((L == 1).sum()))}
    for key, row, ang in CONFIGS:
        tp, fp = record[row]["tp"], record[row]["fp"]
        res = carve(scene, CarveConfig(voxel_size=0.10, margin_m=0.20, angular_res_deg=ang))
        c, _ = metrics.evaluate(scene, res.masks)
        if (c.tp, c.fp) != (tp, fp):
            raise SystemExit(f"{key}: got tp={c.tp} fp={c.fp}, recorded tp={tp} fp={fp}. Not drawing.")
        rm = np.concatenate([res.masks[s.name] for s in scene])
        kind = (L == 1).astype(np.uint8)
        kind[rm & (L == 0)] = LOST  # removed fixed surface stays visible, in pink
        kind[rm & (L == 1)] = GONE  # removed moving points are not drawn
        states[key] = dict(kind=kind, tp=c.tp, fn=c.fn, fp=c.fp,
                           recall=100 * c.ghost_recall, damage=100 * c.surface_damage_rate)
        print(f"  {key}: removed {c.tp:,} of {c.tp + c.fn:,} moving points ({100 * c.ghost_recall:.2f}%), "
              f"fixed surface lost {c.fp:,} ({100 * c.surface_damage_rate:.4f}%)  matches the record", flush=True)
    P = np.concatenate([s.points for s in scene])
    for st in states.values():
        st["points"] = P
    states["total"] = len(P)
    states["origin"] = scene.stations[0].origin
    return states


def shot(st, W, H, **over):
    cam = {**CAM, **over}
    return render(st["points"], st["kind"], W, H, cam["cam"], cam["target"], cam["fov"])


def hero(S):
    W, H, pw, ph, top = 1600, 900, 788, 664, 96
    a, b = S["before"], S["0.18"]
    im = Image.new("RGB", (W, H), BG)
    im.paste(shot(a, pw, ph), (8, top))
    im.paste(shot(b, pw, ph), (W - 8 - pw, top))
    d = ImageDraw.Draw(im)
    for x, title, items in (
        (8, "As scanned", [(ORANGE, f"moving, {a['moving']:,} points")]),
        (W - 8 - pw, "After carving", [(ORANGE, f"moving, left behind: {b['fn']:,}"),
                                       (PINK, f"fixed surface removed: {b['fp']:,}")]),
    ):
        text(d, (x + 12, 10), title, 38, bold=True)
        legend(d, x + 12, 58, items, 25)
    y = top + ph + 16
    text(d, (20, y), f"{b['recall']:.1f}% of the moving points removed. Fixed surface lost: at most {b['damage']:.2f}%.", 34, bold=True)
    dd = S["default"]
    text(d, (20, y + 46), f"0.18° bins, a setting from an earlier sweep on the same labels. Default setting: {dd['recall']:.1f}% removed, at most {dd['damage']:.2f}% lost.", 24, DIM)
    text(d, (20, y + 78), f"Real scan, two stations, all {S['total'] / 1e6:.1f} M points.  {SOURCE}.", 24, DIM)
    return im


def portrait(S):
    W, H, ph = 1080, 1350, 520
    a, b = S["before"], S["0.18"]
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    over = dict(fov=60, target=(1.62, 1.90, 0.10))
    text(d, (28, 14), "As scanned", 38, bold=True)
    legend(d, 300, 24, [(ORANGE, f"moving, {a['moving']:,} points")], 27)
    im.paste(shot(a, W, ph, **over), (0, 66))
    y = 66 + ph + 12
    text(d, (28, y), "After carving", 38, bold=True)
    legend(d, 300, y + 10, [(ORANGE, f"left behind: {b['fn']:,}"), (PINK, f"fixed surface removed: {b['fp']:,}")], 27)
    im.paste(shot(b, W, ph, **over), (0, y + 52))
    y = y + 52 + ph + 14
    dd = S["default"]
    text(d, (28, y), f"{b['recall']:.1f}% of moving points removed, at most {b['damage']:.2f}% of surface lost", 33, bold=True)
    text(d, (28, y + 46), "0.18° bins, a setting from an earlier sweep on the same labels.", 25, DIM)
    text(d, (28, y + 78), f"Default setting: {dd['recall']:.1f}% removed, at most {dd['damage']:.2f}% lost. All {S['total'] / 1e6:.1f} M points.", 25, DIM)
    text(d, (28, y + 110), SOURCE, 25, DIM)
    return im


def loop_keys(S):
    W, H = 1280, 720
    rows = (
        ("before", "As scanned", f"{S['before']['moving']:,} moving points in orange"),
        ("default", "Default setting", "{recall:.1f}% of moving points removed, at most {damage:.2f}% of surface lost"),
        ("0.18", "0.18° bins", "{recall:.1f}% of moving points removed, at most {damage:.2f}% of surface lost"),
    )
    keys = []
    for i, (key, title, line) in enumerate(rows):
        st = S[key]
        im = shot(st, W, H, fov=66)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((0, H - 112, W, H), fill=(*BG, 225))
        text(d, (28, H - 104), title, 38, bold=True)
        text(d, (28, H - 58), line.format(**st), 28)
        text(d, (W - 120, H - 82), "Pink: fixed surface removed.  " + SOURCE, 18, DIM, anchor="rs")
        for j in range(3):
            x = W - 28 - (2 - j) * 28
            d.ellipse((x - 14, H - 96, x, H - 82), fill=INK if j == i else (80, 88, 100))
        keys.append(im)
    return keys


def moving(S):
    """The labelled moving points alone, upright: front, side and from above."""
    W, H = 1600, 900
    st = S["before"]
    g = st["points"][st["kind"] == GHOST]
    total = len(g)
    g = g[np.linalg.norm(g - S["origin"], axis=1) < 3.5]
    k = np.full(len(g), GHOST, np.uint8)
    disp = g * np.array([1.0, -1.0, -1.0])
    c = (np.percentile(disp, 1, axis=0) + np.percentile(disp, 99, axis=0)) / 2
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    text(d, (24, 12), "What moved during the scan: two people", 40, bold=True)
    for x, y, w, h, off, fov, label in (
        (8, 80, 1000, 680, (0, 5, 0), 29, "front"),
        (1024, 80, 568, 400, (5, 0, 0), 31, "side"),
        (1024, 496, 568, 264, (0, 0.5, 5), 28, "from above"),
    ):
        im.paste(render(g, k, w, h, c + np.array(off, float), c, fov), (x, y))
        d.rectangle((x, y, x + w - 1, y + h - 1), outline=(48, 54, 64))
        text(d, (x + 14, y + 8), label, 24, DIM)
    text(d, (24, 776), f"{len(g):,} of the {total:,} points labelled as moving, all within 3.5 m of scanner 1.", 30, bold=True)
    text(d, (24, 820), "The dataset page says one scan contains two people and the other does not.", 25, DIM)
    text(d, (24, 854), SOURCE, 25, DIM)
    return im


def ffmpeg():
    p = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not p:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
    return p


def encode(keys, name, hold=1.9, fade=0.5, fps=30):
    ff = ffmpeg()
    fr = [np.asarray(k).astype(np.float32) for k in keys]
    tmp = tempfile.mkdtemp(prefix="media-loop-")
    n = 0
    for i, a in enumerate(fr):
        b = fr[(i + 1) % len(fr)]  # the last key fades back into the first
        nh, nf = int(hold * fps), int(fade * fps)
        for j in range(nh + nf):
            t = 0.0 if j < nh else (j - nh + 1) / (nf + 1)
            t = t * t * (3 - 2 * t)
            Image.fromarray((a * (1 - t) + b * t).astype(np.uint8)).save(os.path.join(tmp, f"f{n:05d}.png"))
            n += 1
    mp4, gif = os.path.join(OUT, name + ".mp4"), os.path.join(OUT, name + ".gif")
    for crf in (24, 27, 30, 33):
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"),
                        "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                        "-color_trc", "bt709", "-movflags", "+faststart", "-an", mp4], check=True)
        if os.path.getsize(mp4) < 2.9e6:
            break
    for gw, gfps in ((800, 12), (720, 10), (640, 10)):
        vf = (f"fps={gfps},scale={gw}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, "-loop", "0", gif], check=True)
        if os.path.getsize(gif) < 3.9e6:
            break
    shutil.rmtree(tmp, ignore_errors=True)
    return n / fps, gw


def main():
    os.makedirs(OUT, exist_ok=True)
    S = build_states()
    d, c = S["default"], S["0.18"]
    rows = []

    def add(name, dims, what):
        rows.append((name, f"{dims}, {os.path.getsize(os.path.join(OUT, name)) / 1024:.0f} kB", what))
        print(" ", rows[-1][0], rows[-1][1], flush=True)

    for name, im, what in (
        ("lecturehall_before_after_1600x900.webp", hero(S),
         f"Before and after on a real two-station scan of a lecture hall. Left: the scan with its {S['before']['moving']:,} labelled "
         f"moving points in orange. Right: after carving at the 0.18 degree setting, {c['recall']:.1f}% of them removed; "
         f"{c['fn']:,} left behind (orange) and {c['fp']:,} fixed-surface points removed (pink, at most {c['damage']:.2f}%)"),
        ("lecturehall_portrait_1080x1350.webp", portrait(S),
         "The same before and after, stacked 4:5 for small screens"),
        ("lecturehall_moving_points_1600x900.webp", moving(S),
         "The points labelled as moving on their own, the right way up, from the front, the side and above: "
         "two people"),
    ):
        im.save(os.path.join(OUT, name), "WEBP", quality=86, method=6)
        add(name, f"{im.width}x{im.height}", what)
    secs, gw = encode(loop_keys(S), "lecturehall_before_after_loop")
    what = (f"Three states of the same view, fading in a loop with no cut: as scanned, after the default setting "
            f"({d['recall']:.1f}% of moving points removed, at most {d['damage']:.2f}% of surface lost) and after the 0.18 degree "
            f"setting ({c['recall']:.1f}%, at most {c['damage']:.2f}%)")
    add("lecturehall_before_after_loop.mp4", f"1280x720, {secs:.1f} s", what + ". H.264, no audio")
    add("lecturehall_before_after_loop.gif", f"{gw} px wide", "GIF fallback of the loop")
    with open(os.path.join(OUT, "FILES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# media_export\n\nRendered by scripts/media_export.py from results/lecturehall_full.json, "
                "all points, both stations merged. The script stops if the counts differ "
                "from the recorded run.\n\n" + SOURCE + ".\n\n| File | Size | What it shows |\n|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")


if __name__ == "__main__":
    main()
