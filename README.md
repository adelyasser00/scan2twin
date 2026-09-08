# scan2twin — Stage 1

Raw, messy terrestrial laser scan in. Clean, measurable point cloud out.
Every step shown, including the ones that did not work.

Built overnight, 5 September 2026. Everything below marked "measured" was
produced by running the code in this repo. Everything marked "untested" has
not been run against real data and says so.

---

## 1. The scope cut

The original Stage 1 was three problems wearing one coat:

| Problem | Method | Verdict |
|---|---|---|
| Ghost artefacts from moving objects | Cross-station free-space carving | **Deterministic geometry. Build this.** |
| Vegetation | Semantic classification | Research problem. Cut. |
| Noise, mixed pixels | Statistical outlier removal | Commodity. Include, label as commodity. |

Only the first is both hard for a human and solved in principle. Vegetation
carries research risk and does not differentiate anything, because vegetation
removal is mediocre in every survey package and nobody expects better. It is
cut from Stage 1 entirely.

**Why ghost removal is not a research problem.** If station B's laser passed
through the space where station A recorded a person, that person was not there
when B fired. That is a measurement, not an inference. The reason cleanup is
still 16 to 32 hours of hand work on a 250-scan job is that survey software
discards the per-station scanner origin at registration, deleting the one
signal that would automate it. That is a product gap, not an algorithm gap.

Prior art: Schauer & Nüchter, *The Peopleremover*, IEEE RA-L 3(3), 2018. The
authors state the method works on terrestrial scan data, not only mobile.

---

## 2. Two findings, both measured

### Finding 1 — the angular agreement check is not optional

Measured on a 4-station, 1.49 M point synthetic capture with exact ground
truth:

| angular agreement check | ghost recall | **real surface destroyed** |
|---|---|---|
| off | 98.1 – 99.5 % | **9.5 % – 50.1 %** |
| on | 88.8 – 91.2 % | **0.0000 %** |

Turning the check off buys about 8 points of recall and destroys up to half
the scene. A demo reporting only recall would look better and be worthless.
This is the entire honest-metrics thesis proving itself on the first run, and
it is `out/fig1_sweep.png`.

The library now emits a `RuntimeWarning` when the check is disabled.

### Finding 2 — grazing incidence eats floors, and it looks like a tuning problem

The first implementation queried visibility at **voxel centres**. It behaved
fine at most voxel sizes and destroyed 2.4 % of the scene at exactly one of
them, with 10,355 of 10,426 false positives on the ground plane.

Cause: a voxel centre sits up to half a voxel off the surface. At 0.20 m
voxels the grid happened to land with the ground's centre 7 cm high. A 1.6 m
tripod sees the ground at 40 m under a 2.3° depression angle, so

```
0.07 m / sin(2.3°) = 1.75 m of apparent range error
```

The 0.20 m margin never stood a chance. **Sub-voxel offset blows up as
offset / sin(incidence).** No fixed metric margin can absorb it on a floor,
and a margin large enough to try would hide a person at 5 m.

Fix: query at the measured points, which lie on the surface by construction.
After the fix, surface damage is 0.0000 % at every voxel size tested, and
F1 is 0.9488 / 0.9488 / 0.9485 / 0.9483 across 0.08 m to 0.30 m. That
flatness is the proof: the result no longer depends on where the grid lands.

Chart: `out/fig4_grazing.png`.

---

## 3. Measured headline, 4 stations, 1.49 M points

```
real surface points deleted : 8 of 437,141 static   (0.0018 %)
ghost recall                : 89.5 %
mixed-pixel noise removed   : 90.4 %   (free side effect, not a feature)
ghost precision             : 99.8 %
throughput                  : 2.26 M points / s on ONE CPU core
```

And the number that keeps the project honest:

```
40.4 % of missed ghosts had no second station with any sightline through them.
```

That fraction is a capture-planning number, not a code-quality one.
`metrics.witness_analysis` measures it, so residue can be attributed to
geometry rather than blamed on the algorithm — which is what stops you tuning
parameters until real walls disappear.

---

## 4. Structure

```
scan2twin/
  types.py       Station and Scene. Station REQUIRES an origin.
  grid.py        Sparse int64-keyed voxel grid. No dense arrays.
  visibility.py  Spherical range image. The fast path. O(queries) per station.
  traversal.py   Explicit ray marching. Slow reference to check the above.
  carve.py       Cross-station conflict resolution.
  sor.py         Commodity outlier removal.
  metrics.py     3-class scoring, asymmetric reporting, miss attribution.
  io/
    synthetic.py Ray-cast plaza with exact ground truth. Tested.
    readers.py   Semantic3D + Robotic 3D Scan Repository. UNTESTED.
```

The contract: **every adapter produces `Station`, and nothing in the core
knows what a file format is.** That is what makes this pointable at a
client's own scanner output later without touching the algorithm. Same
separation principle as a physics core behind a thin vendor wrapper.

### Ground truth is three classes, not two

`0 = static, 1 = dynamic, 2 = mixed-pixel noise.` This is not decoration. The
first run reported 0.27 % surface damage, which turned out to be almost
exactly the injected noise fraction: the carver was correctly deleting noise
and a two-class ground truth scored every deletion as destroyed geometry. A
benchmark that cannot tell "removed a bad point" from "removed a good point"
does not measure cleanup quality. **Semantic3D is two-class. That limitation
belongs in the write-up.**

---

## 5. Running it

```bash
python3 run_demo.py --az 1700 --el 320 --walkers 16          # full run
python3 run_demo.py --sweep                                   # parameter sweep
python3 make_figures.py                                       # case study figures
```

---

## 6. What is untested, explicitly

- `io/readers.py` — both real-dataset adapters are written from published
  format descriptions and have never been run against the real files, because
  the build machine had no network route to those hosts.
- `load_uosr` — handedness, units (some 3DTK data is centimetres) and Euler
  order are **unverified placeholders**. A wrong transform here is worse than
  none: the scans still look plausible overlaid and the "ghosts" become real
  geometry. Overlay two stations and eyeball a wall before carving.
- `traversal.py` — implemented as the reference but not yet cross-validated
  against `visibility.py` on the same scene. Do that before publishing any
  number, since it is the check that the fast path is not lying.
- `recover_origin` — see below.

---

## 7. `recover_origin`: the tool that decides which datasets are usable

Most public datasets ship registered clouds and throw the origins away.
Without an origin this whole method is inert. So: a single scan records one
return per beam direction, therefore from the true origin each angular bin
holds points at one range, and from a wrong origin the within-bin spread
blows up. Minimise the median spread.

Measured on synthetic data:

| | mean error | max error |
|---|---|---|
| free 3D search | **6.8 m** (all vertical) | 9.7 m |
| with tripod-height prior | **0.44 m** | 0.54 m |

The free search recovers x and y to 0.1–0.9 m and misses z by 4–10 m every
time. A plaza is close to rotationally symmetric about the vertical, so the
objective has a long flat valley in z. A wrong z is not cosmetic: it changes
every grazing-angle relationship, the exact quantity carving is most
sensitive to.

**Caveats you should not skip.** z pinned to the top of the prior band in 3 of
4 stations, so the band edge is binding and the objective still wants to
climb. 0.44 m of origin error is not nothing. And this has only been run on
data this package generated, which is arguably too easy. Validate it on a real
scan whose origin you already know before trusting it on one where you do not.

`looks_station_local` currently returns a **false positive** on the synthetic
scene, because the plaza happens to be centred near the world origin. The
heuristic is too weak. Correct fix: score `(0,0,0)` with the `recover_origin`
objective and compare against the global minimum, rather than using bounding
box and azimuth coverage. Not done.

---

## 8. Scenario 1 vs Scenario 2

They are the same build. The one real conflict is that a correct carve
removes 1–3 % of points, and **the honest before/after is visually subtle**.
`out/fig2_plan.png` proves it: 2,304 red dots on 162,000 grey ones. That is
the picture that tempts engineers to fake demos.

So the eye candy is not the cleaned cloud, it is the mechanism.
`out/fig3_mechanism.png` is the drawing that makes a surveyor say "oh", and
it cost nothing extra because the geometry had to be built anyway. Frame
tight on ghosts, never show an overview.

**Figure ranking for a post:** fig1 (sweep) > fig3 (mechanism) > fig4
(grazing) > fig2 (plan). The bug story in fig4 is more compelling content
than any success metric, because it is the thing nobody else publishes.

---

## 9. Order of operations, next session

1. Cross-validate `traversal.py` against `visibility.py` on the synthetic
   scene. If they disagree, the fast path is lying and everything above is
   suspect.
2. Download **lecturehall** from the Würzburg repository. 435 MiB, two scans,
   44.6 M points, ships per-point static/dynamic labels in CSV. Smallest real
   dataset with ground truth that exists for this task.
3. Run `looks_station_local` on it before anything else. That single check
   decides whether the dataset is usable.
4. Only then scale to `wue_city` (6 scans, 86.6 M points) or a Semantic3D
   training scene.
5. Stage 2 (meshing, 3D Tiles) only after Stage 1 publishes a number.

**Licence reminder.** Semantic3D is CC BY-NC-SA 3.0: non-commercial, and
ShareAlike means any cleaned cloud or tileset you publish from it inherits the
licence. Your code does not. Never point a paying client at a Semantic3D
demo. Build the client-facing version on the Würzburg data, which carries
attribution only and no NC clause.

**Cloudflare Pages limits for Stage 2.** 25 MiB per file, 20,000 files per
deployment on free. A before/after point-cloud tileset pair will approach that
file count. Count tiles before building the deploy; overflow means R2 with a
custom domain for tiles and Pages for the app.
