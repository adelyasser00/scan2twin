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

## What it found

### On synthetic data (4 stations, controlled ground truth)

The method works well when it has enough viewpoints and a standing target:

- Ghost recall: 89-93%
- Surface damage: 0.0018% (8 points of 437,141)
- Ghost precision: 99.8%

### On real data (2 stations, Riegl VZ-400 lecture hall)

- Ghost recall: 18%
- Surface damage: 0.033% upper bound (two-class labels, no noise class)
- The "person" in the dataset is not a person. It is a 2.1 m motion smear
  from someone walking past during the sweep. A scanner mirror takes minutes
  to complete a rotation. Nobody stands still for that long.

18% is low. But 93.5% of the missed ghosts had a sightline from the second
station, meaning the geometry could have caught them with a less conservative
config. The ceiling is real but higher than the default setting reaches.

### Two engineering findings

**1. The angular agreement check is not optional.**

Turn it off: recall jumps to 98%, and surface damage jumps to 9-50%.
Eight points of recall for up to half the scene destroyed. A demo
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

Expected: ~89% ghost recall, ~0.0018% surface damage on synthetic data.

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
Lecturehall data: attribution to recorders and institution, no NC clause.
Semantic3D data (if used): CC BY-NC-SA 3.0, non-commercial, ShareAlike.
