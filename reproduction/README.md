# `reproduction/` — the paper toolkit

Reusable machinery to reproduce this paper's results, one tier below the drivers in `reproduction/experiments/`
and one tier above the published library in `src/cryodisc/`.

## What belongs where

| Tier | Rule | Examples |
|------|------|----------|
| `src/cryodisc/` | Ships to PyPI. The Figure-1 method + physical knobs a general user sets. The test: *would a PyPI end user need this?* | scoring, the marginal rung, CTF/plan/band, pools, `noise_spectrum`, `spectrum` shells, structure alignment |
| `reproduction/` | Real, reusable, but **specific to reproducing this paper**. Never shipped. | survey assembly, dataset pipelines, the CR validation chain, the parakeet gate, the SSNR & memorization predictors |
| `reproduction/experiments/` | Thin per-figure / per-analysis drivers that call the two tiers below. | `reproduction/experiments/panel/score_survey.py`, `reproduction/experiments/figures/*` |

## The one hard rule

`src/` **must never import `reproduction/`.** Dependencies flow `reproduction/experiments/` → `reproduction/` → `src/`,
never back up. A `reproduction/` module importing `cryodisc` is expected; a `src/` module
importing `reproduction` is a tier violation and breaks the shippable package.

## Reproducing the paper's exact numbers

The published results were produced with the exact environment pinned in `requirements.lock` at the
repository root -- `cryojax==0.5.6`, `jax==0.11.0`, and the rest. Reproduce from that lock
(`pip install -r requirements.lock`), **not** from the floating ranges in `pyproject.toml`: the forward
model's internals can shift between dependency releases, so a newer `cryojax` may change a figure's
numbers in the last digits even when the method is unchanged. The dependency ranges in `pyproject.toml`
exist so a general user can install the `cryodisc` tool alongside their own stack; the lock is what makes
this paper's measurements re-derivable.

Not packaged (`pyproject` finds packages under `src/` only). Importable from the repo root exactly as
`reproduction/experiments/` is (`pythonpath = ["."]`).

## Current contents

Top level: `chain` (the CR validation chain, `run_chain`), `survey`, `targets`, `record`.

- `reproduction/analysis/` — `recovery`, `purity`, `pure_pools`, `shells`, `fits`, and `si_estimators`
  (the retired composition estimators kept for the SI bake-off).
- `reproduction/datasets/` — the dataset pipelines: `empiar`, `cryosparc`, `picking`, `triage`, `ladder`.
- `reproduction/predictors/` — `memorization` (Fig 6), `scattering` (SSNR / Rosenthal-Henderson), `spectrum`.
- `reproduction/reporting/` — `figures` (the figure-assembly helpers the drivers share).
- `reproduction/validation/` — `bank` (renderer-vs-reference gate) and the independent `parakeet` engine
  it measures against, `template_bank`, `coherence`, `floor`.

The refactor that created this tier is recorded in the git history.
