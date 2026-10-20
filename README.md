# cryodisc

**Given two known conformations of a protein, can cryo-EM tell them apart — and if so, how many
particles would it take? And in a real mixed dataset, what fraction is each state?**

Single-particle cryo-EM normally assumes one rigid structure. A flexible protein occupies several
conformations, and averaging them blurs exactly the features that distinguish the states. Tools for
that heterogeneity exist (3D classification, cryoDRGN, 3DVA), but there is no accepted way to know *in
advance* whether a target carries enough signal for the separation to succeed. This library predicts
that from the structures alone, validates the prediction by attempting the separation in silico, and
estimates the population fraction of each state from real particles.

## The approach

**1 — Discriminability.** From two atomic models and a stated collection plan, the library predicts a
per-particle discriminability `d²` and inverts it to the number of particles needed to separate the
states at a target error. Two facts shape it:

- The **pose is unknown**, so a conformational change that resembles a small rigid rotation is
  unrecoverable at any particle count. The signal that survives is the part of the difference map
  orthogonal to every pose change — so the honest quantity is the *pose-marginalized* `d²`, not the
  known-pose one.
- **Aligning pure noise to a template reproduces the template** ("Einstein from noise"), so a
  separation result cannot validate itself. The theory supplies the null a real result must exceed,
  and the prediction is checked by simulating a known-answer stack and recovering it.

The count is set by the **Fisher information `I(p)`** the score carries about the population fraction —
the quantity the efficient estimator actually attains — not by the raw state difference, which
overstates it on a pose-marginalized score.

**2 — Population fraction, pool-free.** Given the two structures and a *mixed* dataset, the library
estimates how much of each state is present — without pure reference pools to calibrate against. The
obstacle is that the pose-marginalized log-likelihood ratio is *overdispersed*, which compresses the
naive estimate toward an even mixture. The fix measures that overdispersion with a **decoy**: score the
particles a second time against a difference map carrying the true difference's power spectrum but none
of its structure (a rigid rotation of it, or its phase-randomization). That score is single-state, and
its variance is the within-state spread the mixture's own moments cannot isolate — giving the fraction
in closed form with no labeled set anywhere.

## Install & setup

```sh
python -m venv .venv
.venv/bin/pip install -e .                   # installs src/ editable (the only packaged tier)
.venv/bin/python deploy/link_repo_root.py    # dev only: see "Repository layout" below
```

## Quickstart

Predict how separable two conformations are, and how many particles it would take:

```python
import jax
from cryodisc import align_conformers, CollectionPlan, predict

pair = align_conformers("state_a.pdb", "state_b.pdb")
plan = CollectionPlan(dose=40.0, pixel_size=1.0)          # e/Å², Å/pixel
result = predict(pair, plan, key=jax.random.key(0))

result.marginal.d2           # pose-marginalized discriminability d²
result.marginal.information  # Fisher information I(p) about the fraction
result.marginal.n_for_error  # particles needed to separate the states
result.known_pose.d2         # the known-pose ceiling, for reference
print(result.summary())      # the two rungs as a table
```

`predict` is the prospective half: the two structures and the plan alone, no pipeline run. Attempting
the separation *in silico* to check what an algorithm actually recovers against the prediction — the
full validation ladder with the pose search, the EM fit, and the efficiency/calibration controls — is
method-validation machinery and lives in the research toolkit (`reproduction/chain.py`, `run_chain`),
not the installed package.

Given a *real* mixed dataset instead, estimate how much of each state is present:

```python
from cryodisc import estimate_fraction, Pool

pool = Pool(images=images, transfer=transfer, box_size=128, pixel_size=1.0)
result = estimate_fraction(pair, plan, pool, key=jax.random.key(0))  # pool-free by default

result.fraction       # estimated fraction of state A
result.fraction_err   # bootstrap standard error
result.method         # which estimator: "pool_free" (decoy), "mle", or "pooled"
```

`estimate_fraction` calibrates the template amplitude against the pool, scores every particle, and
reads the fraction off that score. The estimator is swappable via `method=` — the pool-free decoy
route is the default, needing no reference pools.

Worked drivers live in `reproduction/experiments/` — `reproduction/experiments/panel/score_survey.py` ranks a panel of targets,
`reproduction/experiments/realdata/score_pool.py` scores a real particle set.

## Repository layout

Three tiers, split by what ships:

- **`src/cryodisc/`** — the published library; the only tier packaged to PyPI.
- **`reproduction/`** — the paper toolkit (survey/dataset assembly, benchmarks, predictors, validation
  gates). Reusable but specific to reproducing this paper; never shipped. `src/` never imports it.
- **`reproduction/experiments/`** — thin per-figure / per-analysis drivers that call the two tiers above.

`reproduction/` and `reproduction/experiments/` are deliberately **not** installed. `pip install -e .` covers `src/`;
`deploy/link_repo_root.py` writes a `repo_root.pth` into the venv so the other two import when a driver
is run as a script (tests get the same repo root from pytest's `pythonpath = ["."]`). The `.pth` is a
venv artifact — it changes nothing about what ships, and deleting it reverses the step.

## Citation

The method and results are described in the accompanying paper *(in preparation)* — citation to follow.
