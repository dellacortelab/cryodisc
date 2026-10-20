"""Paper toolkit: reusable machinery for reproducing this paper's results, not shipped to PyPI.

The middle tier between the published library (`src/cryodisc/`) and the per-figure drivers
(`reproduction/experiments/`). Dependencies flow `experiments/` -> the rest of `reproduction/` ->
`src/cryodisc/`; `src/cryodisc/` must never import `reproduction/`. Not packaged for PyPI, but
importable from the repo root.
"""
