# scan2twin

An investigation into automated ghost removal from terrestrial laser scans,
using cross-station visibility as the only signal. No ML, no shape
recognition, no temporal data.

Built September 2026. This README documents what worked, what did not, and
where the method hits its ceiling.

## The idea

A terrestrial scanner on a tripod fires a laser in every direction. Someone
walks past during the sweep. Their body stops beams, so the scan records
points where the person was standing. Those points are ghosts.

Move the tripod to a second position and scan the same room. The person has
moved on. Beams that pass through the exact space where the person was now
travel straight through and hit the wall behind.

If scanner B's beam went through the space where scanner A recorded
something, nothing was there when B fired. So whatever A recorded there
moved. That is the whole method.

The idea is not new. It follows Schauer and Nüchter, "The Peopleremover",
IEEE RA-L 3(3), 2018.

## What it found

### On synthetic data (4 stations, controlled ground truth)

The method works well when it has enough viewpoints and a standing target:

- Ghost recall: 89.69% with the command below (93.69% in `scripts/adjudicate.py`)
- Surface damage: 0 of 437,141 points
- Ghost precision: 100%

### On real data (2 stations, Riegl VZ-400 lecture hall)

All 44.6 M points, 521,923 of them labelled as moving
(`scripts/run_lecturehall_full.py`, recorded in `results/lecturehall_full.json`):

| Direction bins | Ghost recall | Surface damage, upper bound | Ghost precision |
|---|---|---|---|
| auto (0.042 / 0.043 deg), the default | 71.93% | 1.232% | 40.9% |
| 0.060 deg | 97.27% | 1.400% | 45.1% |
| 0.100 deg | 97.70% | 0.925% | 55.6% |
| 0.140 deg | 97.62% | 0.528% | 68.7% |
| 0.180 deg | 97.44% | 0.349% | 76.8% |
| 0.250 deg | 97.18% | 0.155% | 88.1% |

Surface damage is an upper bound: the labels have two classes and no noise
class. 0.180 deg was already the highest-recall row of an earlier sweep on a
subset. It was chosen on these same labels, so it is not a held-out result.

**How much of the scan is loaded decides the result.** The first subset run
(2,000,000 points per station, September 2026) removed 18% of the moving points
with 0.033% damage at the default setting. The full scan removes 72%, and 97%
with 0.180 deg bins. An empty bin in the range image is never evidence, so a thin
subset can lose it. Subsets now use a stride rounded up, so they span the whole
sweep; the same 2 M cap gives 1.9 M points and 13.5%.

**What moved.** The dataset page says one scan contains two people and the other
does not. The stored z axis points down, so turn the cloud over to see them: they
look like they hold a sheet between them. The carver has no up direction, so this
changes no number.

### Two engineering findings

**1. The angular agreement check is not optional.**

Turn it off, on the scene `make_figures.py` builds: recall rises from 88.8-91.2%
to 98.2-99.3%, and surface damage rises from at most 0.002% to 21.9-41.9%.
Eight to nine points of recall for a fifth to two fifths of the scene. A demo
reporting only recall would look better and be catastrophically wrong.

**2. Sub-voxel offset destroys floors at grazing incidence.**

Querying visibility at voxel centres instead of measured points introduces
a few centimetres of positional error. On a wall that costs you nothing.
On a floor seen at 2.3 degrees from 40 m away:

    0.07 m / sin(2.3 deg) = 1.75 m of apparent range error

The ground plane was deleted at one voxel size and fine at others,
making it look like a tuning problem for hours. Fix: query at the measured
points, which sit on the surface by definition.

## Structure

scan2twin/ Core package
types.py Station and Scene interchange types
grid.py Sparse voxel grid
visibility.py Spherical range image (fast path)
traversal.py Explicit ray marching (reference)
carve.py Cross-station conflict resolution
sor.py Statistical outlier removal (commodity)
metrics.py 3-class scoring with asymmetric reporting
io/
synthetic.py Ray-cast scene generator with exact ground truth
readers.py Adapters for Semantic3D and Robotic 3D Scan Repository

run_demo.py Synthetic scene end-to-end with metrics
make_figures.py Generate case study figures
scripts/ Investigation and validation scripts
assets/ Pre-generated figures


## Running it

```bash
pip install numpy scipy pandas matplotlib laspy scikit-learn
python run_demo.py --az 900 --el 180 --walkers 12
python make_figures.py
```

Expected: ~89.7% ghost recall, 0 damaged surface points on synthetic data.

## Limitations, stated plainly

- Two-station capture is close to the worst case for this method. More
  stations means more evidence and higher recall.
- The method removes floating ghost points in open space. It does not
  recover geometry that was never measured (occluded by the person), fix
  colour projected onto walls, or remove vegetation.
- The real-data ground truth is two-class (static/dynamic). Every noise
  point correctly removed is scored as surface damage. All real-data
  damage numbers are upper bounds.
- The lecturehall adapter was validated (0.046 m median overlap NN) but
  the Semantic3D and UOSR adapters are written from specs and untested.

## Licence

Code: MIT (or your choice, not yet declared).
Lecturehall data: recorded by Johannes Schauer, University of Würzburg, Riegl
VZ-400, from the Robotic 3D Scan Repository
(https://robotik.informatik.uni-wuerzburg.de/telematics/3dscans/), which requires
naming the recorder and the institution.
Semantic3D data (if used): CC BY-NC-SA 3.0, non-commercial, ShareAlike.
