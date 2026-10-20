# Design overview

A high-level map of how `cryodisc` is organized and why. The math behind the core relations is in
[`derivations.md`](derivations.md); this document is about the code.

## The question

Given two conformations of a molecule, how many particles does telling them apart take? The answer is
a squared distance `d^2` per pose, depressed because part of any conformational difference looks like a
small pose change and is unrecoverable at any particle count. The package turns a pair of structures
(atomic models or maps) plus a microscope and collection into that number, and into the particle count
an experiment needs to reach a target error on the population fraction.

## Three tiers, imports run one way

The repository is split into three tiers, and imports only ever run *downward*:

- **`src/cryodisc/`** — the published library. The only tier packaged for PyPI. It must never import
  from the tiers above it.
- **`reproduction/`** — the paper toolkit: reusable machinery for reproducing this paper's results
  (predictors, analysis, validation gates, record/survey/target bookkeeping) that is not general enough
  to ship.
- **`reproduction/experiments/`** — the per-figure drivers. Each reads committed records from
  `results/` and rebuilds one figure or number. No driver imports another driver; shared code moves
  down a tier.

`reproduction/experiments/` → `reproduction/` → `src/cryodisc/`, never back up. The architectural tests
in `tests/` enforce this direction so the shipped package cannot grow a dependency on the paper code.

## The library, layered

`src/cryodisc/` is itself layered, with imports running only up this list:

- **`grids`** — the Fourier grid every other module shares, and the conventions that ride on it
  (unshifted `fft2` layout, DC-zeroing whitening weight, the sign of a translation ramp, where Nyquist
  lands). These live together because they are one contract; mixing a shifted array into an unshifted
  index swaps low and high spatial frequency silently rather than failing.
- **`statistics`** — discriminability, information per particle, composition and convergence: the
  estimators that turn spectra into the paper's quantities.
- **`microscope`** — the CTF, transfer and collection plan that define how a specimen is imaged.
- **`specimen`** — densities and atomic structures, including the conformer alignment that defines what
  "the same molecule, two states" means.
- **`forward`** — the forward image-formation model (built on cryoJAX) and its optics.
- **`resolution`** — the noise and transfer model behind resolution prediction.
- **`particles`** — reading, simulating and fitting particle stacks (from `.cs` or `.star`).
- **`rungs`** — the refinement rungs (achieved, local, marginal, refine) that bracket what pose
  recovery can and cannot reach.
- **`pools`** and **`pipeline`** — a particle set and the end-to-end entry points (`predict`,
  `estimate_fraction`, `rung_ratio`) re-exported at the top of the package.

Only the entry points are re-exported from `cryodisc`; everything else is reached by module path.

## Reproducing the figures

Every paper figure rebuilds from committed records in `results/` without the cluster or the raw data:
the drivers in `reproduction/experiments/` read those records and the house-style plotting in
`reproduction/reporting/` renders them. `reproduction/experiments/reproduce.py` is the registry of
those experiments. The cluster-only machinery that *produced* the records from raw EMPIAR/cryoSPARC data
(preprocessing, cross-session whitening, dataset ingest) is not part of this public repository.
