# Brief for the portfolio page: Scan to Twin (paste into the portfolio chat)

Build a portfolio section for this project. All media files below are final
(folder media_export/). Don't invent numbers or claims beyond this brief.
"Re-run" means I ran the code again on 23 Nov 2026.

## 1. In one sentence

Scan to Twin (repo name: scan2twin) removes things that moved during a laser
scan, such as people, by checking whether a second scanner position saw
straight through the place where they were recorded. It is an investigation
with measured results, not a product.

- Code: https://github.com/adelyasser00/scan2twin
- Live: none found in the repo
- Built September 2026, full-data run November 2026. Python (NumPy, SciPy, pandas, Matplotlib, laspy)

## 2. What the method does

1. Each scanner position (station) keeps its points and the place it fired from.
2. For every station it builds a range image: for each direction, the distance
   to the first thing hit (`scan2twin/visibility.py`).
3. Each point is looked up in the other stations' range images. If another
   station measured something further away in that direction, by more than a
   0.20 m margin, its beam passed through that point.
4. All nine neighbouring direction bins must agree (the angular agreement
   check). A direction with no return counts as no evidence, never as empty space.
5. A point is removed when at least one other station saw through it and those
   witnesses are not outvoted by stations that found the spot occupied
   (`scan2twin/carve.py`).

No machine learning, no shape recognition, no timestamps. It needs two or more
registered stations and their scanner positions.

## 3. Data it was tested on

| Data | Size | Ground truth |
|---|---|---|
| Synthetic plaza, ray cast by the repo (`io/synthetic.py`) | 4 stations, 442,365 points with the README command | Exact, three classes: fixed, moving, noise |
| Würzburg lecturehall, a real scan by Johannes Schauer, University of Würzburg (Riegl VZ-400, Robotic 3D Scan Repository) | 2 stations 6.71 m apart, 22,253,519 and 22,321,128 points, about 10.7 x 10.3 x 4.1 m | Two classes: fixed, moving. 521,923 moving points in scan 1, none in scan 2 |

## 4. What it outputs

- A keep or remove flag for every point and the number of stations that saw through it.
- A cleaned point cloud (`apply`), and LAS files with four classes for
  inspection in CloudCompare: fixed kept, moving removed, moving missed, fixed removed.
- A report that puts the surface it destroyed first, then what it caught and what it missed.
- It does not output a mesh, a BIM model, 3D Tiles or a browser viewer.

## 5. Results

Real scan, all 44.6 M points, every row of the run
(`results/lecturehall_full.json`). Recall is the share of moving points removed.

| Direction bins | Moving points removed | Fixed surface lost, upper bound | Share of removed points labelled moving |
|---|---|---|---|
| auto (0.042 / 0.043 deg), the default | 71.93% | 1.232% | 40.9% |
| 0.060 deg | 97.27% | 1.400% | 45.1% |
| 0.100 deg | 97.70% | 0.925% | 55.6% |
| 0.140 deg | 97.62% | 0.528% | 68.7% |
| 0.180 deg | 97.44% | 0.349% | 76.8% |
| 0.250 deg | 97.18% | 0.155% | 88.1% |

- With 0.18 degree bins: 508,564 of 521,923 moving points removed (97.4%) and
  153,890 fixed points removed (at most 0.35% of the surface).
- With the default setting: 71.9% removed, at most 1.23% of the surface lost.
- From 0.06 to 0.25 degrees recall stays between 97.2 and 97.7%, while the surface
  loss falls from 1.40% to 0.16% as the bins get coarser. The default picks
  0.04 degree bins, finer than any other row, and is the weak row.
- 0.25 degrees loses less surface than 0.18. I did not move the headline to it, because
  that would be choosing the setting after seeing the answer.
- Quote both headline rows, always with the surface number next to the recall number.

Synthetic scene (README command, re-run): 89.69% of moving points removed, 0 of
437,141 fixed points lost.

Findings worth telling:

- **Recall alone is misleading.** On the synthetic scene, dropping the angular
  agreement check gains 8 to 9 points of recall (to 98.2 to 99.3%) and destroys
  21.9 to 41.9% of the real surface.
- **A bug that ate floors.** Testing visibility at voxel centres instead of the
  measured points adds a few centimetres of error. On a floor 40 m away, seen at
  2.3 degrees, 7 cm becomes 1.75 m of apparent range error. The fix is to test at
  the measured points.
- **How much of the scan is loaded decides the answer.** The first run, on 2 M
  points per station in September 2026 and before the loader change, removed 18%
  at the default. With 0.18 degree bins, 3.7 M points per station give 94.6%
  (re-run, not recorded in the repo) and the full scan gives 97.4%.
- Speed: the full 44.6 M points carve in 42 to 51 s per setting on a laptop
  CPU (Intel i5-8265U), range images included, loading not included.

## 6. How it was checked

- Synthetic scene with exact labels by construction, scored in three classes so
  that removed noise is not counted as damage.
- The fast method was compared with a separate ray-marching version on the same
  labels (`scripts/adjudicate.py`, re-run): fast 93.69% recall and 0 damage, ray
  marching 82.89% and 6.05% damage. The ray-marching version is the weaker one,
  so this compares two methods and does not prove the fast one.
- The real data's units and pose were checked against the data: median distance
  between overlapping scans 0.046 m with the full pose, 0.229 m without it. The
  re-run on the fixed loader gave 0.043 m and the 6.714 m station distance.
- The media script re-runs both settings and stops if the counts differ from the
  recorded run.
- The repo has no automated test suite.

## 7. Limits

- One real scan, one room, two stations. Two is the minimum the method can use.
- 0.18 degrees came from an earlier sweep on a subset of this scan, scored on the
  same labels. It is not a held-out result.
- Results depend on how many points fill the range image. On thinned scans recall
  drops a lot: 53% at 1.9 M points per station with the same 0.18 degree bins
  (re-run, not recorded in the repo).
- The real labels have no noise class, so every noise point correctly removed
  counts as damage. All real-data damage numbers are upper bounds under these labels.
- It cannot remove something no second station saw through, something that stayed
  in place for both scans, or vegetation. It cannot rebuild what was hidden.
- It needs each scanner's position. Many files lose it when scans are merged.
- The Semantic3D and UOSR loaders were written from specs and never run on real files.
- The method is not new. It follows Schauer and Nüchter, The Peopleremover, IEEE
  RA-L 3(3), 2018.

## Media

Both stations merged, all points, drawn from the recorded run by
`scripts/media_export.py`. The data credit must stay visible: the repository requires
naming the recorder and the institution ("Johannes Schauer, University of Würzburg").

The moving object is two people: the dataset page says "one scan contains two people
while the other one does not". Drawn upright they look like they hold a sheet between
them; the page does not mention one, so leave the sheet out of the text.

| File | Size | What it shows |
|---|---|---|
| lecturehall_before_after_1600x900.webp | 1600x900, 267 kB | Before and after on a real two-station scan of a lecture hall. Left: the scan with its 521,923 labelled moving points in orange. Right: after carving at the 0.18 degree setting, 97.4% of them removed; 13,359 left behind (orange) and 153,890 fixed-surface points removed (pink, at most 0.35%) |
| lecturehall_portrait_1080x1350.webp | 1080x1350, 227 kB | The same before and after, stacked 4:5 for small screens |
| lecturehall_moving_points_1600x900.webp | 1600x900, 116 kB | The points labelled as moving on their own, the right way up, from the front, the side and above: two people |
| lecturehall_before_after_loop.mp4 | 1280x720, 7.2 s, 887 kB | Three states of the same view, fading in a loop with no cut: as scanned, after the default setting (71.9% of moving points removed, at most 1.23% of surface lost) and after the 0.18 degree setting (97.4%, at most 0.35%). H.264, no audio |
| lecturehall_before_after_loop.gif | 800 px wide, 1827 kB | GIF fallback of the loop |

Markup: `<video autoplay muted loop playsinline preload="metadata" width="1280" height="720" poster="lecturehall_before_after_1600x900.webp">`,
GIF only as a fallback, the portrait file as a `<picture>` source for
`(max-width: 700px)`, width and height set on every image.

Also usable: `assets/fig1_sweep.png` and `assets/fig3_mechanism.png` (how the method
works, redrawn 23 Nov 2026).

Do not use: `out/fig12_person_views.png` (upside down; use the moving-points file
instead), or any text that calls the object a motion smear.
