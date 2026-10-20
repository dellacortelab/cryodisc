# reproduction/experiments/

A driver that needs data, a GPU, or hours, and records a number somebody quotes. Nothing else.

    .venv/bin/python reproduction/experiments/<thread>/<script>.py --help

## Where a piece of work goes

Four destinations, and the boundary between them is not a matter of taste:

    src/           library code a PyPI user would install and call
    reproduction/      reusable machinery specific to reproducing this paper, never shipped
    tests/         anything whose answer is pass/fail and runs in seconds with no data
    reproduction/experiments/   anything needing data, a GPU, or hours, that records a number

`reproduction/experiments/` is the expensive category, so it is the one to argue yourself out of. A check that
certifies the machinery -- a quadrature identity, a convergence rate, a recovery of an injected
truth -- reads like an experiment and is a test: it has a right answer, it needs no data, and
running it on every commit is the whole point. Filing it here means it runs when somebody
remembers to, which is never.

## No experiment may import another experiment

This is the rule the layout exists to enforce, and it is checked in
`tests/test_experiments_layout.py` rather than trusted to this file.

A driver that imports a sibling turns that sibling into a library nobody declared one. Once a
diagnostic is importable, a driver depends on it, and the diagnostic can no longer be changed to
answer its own question better -- validation and production become indistinguishable. So a function
two drivers both want leaves `reproduction/experiments/` entirely:

- to **`src/`** if a user computing discriminability would want it, where it earns a docstring, a
  test, and a place in the layer map;
- to **`reproduction/`** if it is machinery for reproducing this paper -- survey assembly, a validation
  gate, a retired estimator kept for comparison -- reusable but never shipped.

`reproduction/` is where the machinery that VALIDATES the package lives without being part of it: the
second, independent multislice engine `forward/gate.py` measures our single-slice templates against
is `reproduction/validation/parakeet`, not a sibling of the driver. Dependencies flow one way only --
`reproduction/experiments/` -> `reproduction/` -> `src/` -- and `src/` may import neither tier above it, so an
optional heavyweight dependency can never leak into an install. All three directions are checked.

## What a driver owes

- **Every number it records is reproducible from the file itself.** Paths come from arguments, not
  from a constant somebody edited before the last run. A hand-edited path in a benchmark whose
  state label IS the dataset is how a pool gets scored against the wrong state's template, which
  is the one error nothing downstream catches -- it reads as a smaller signal.
- **A plotter lives with the driver that produces its data**, not in a `plots/` bucket. A figure
  and the run behind it fail together and are read together.
- **A finished experiment is deleted.** Its conclusion belongs in a docstring or a constant in
  `src/` or `reproduction/`, where it is load-bearing; kept here it is dead code that looks
  re-runnable. Git history is the archive.

## Cluster

Scripts that only run on a cluster say so in their docstring, in the first line, and take every
path as an argument. Nothing here may require a cluster to be TESTED -- a driver that cannot be
exercised at a toy size on a laptop has no way to fail fast, and the failure then costs a queue
slot instead of a second.
