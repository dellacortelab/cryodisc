"""Estimate the population fraction of each state in a real, mixed particle pool.

The real-data counterpart to `chain.predict`: render the two templates, calibrate the template
contrast against the pool (d^2 goes as a^2, so a wrong contrast is a wrong fraction), score every
particle's pose-marginalized log-ratio, and read the fraction of state A off that score with a
bootstrap error bar. Which estimator reads the fraction off the score is chosen by `method`, one of
`_METHODS`, in `_read_fraction`, indexed by how many pure reference pools are supplied.

Rationale: docs/rationale/statistics/composition.md
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, PRNGKeyArray

from cryodisc.grids.poses import PoseGrid, pose_bank
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.em import fit_noise_model
from cryodisc.pools import Pool
from cryodisc.rungs import pose_log_ratio
from cryodisc.specimen.structures import ConformerPair
from cryodisc.statistics.composition import (
    fraction_no_pool,
    fraction_one_pool,
    fraction_two_pool,
)

_DEFAULT_NODES = 96
"""Nodes in the bank the score marginalizes the pose over when a caller names none. Coarse: the
marginal rung is a lower bound converging from below."""

_EM_ITERATIONS = 200
"""Iteration cap on the calibration fit, matching `chain.run_chain`."""

_DEFAULT_BOOTSTRAP = 200
"""Resamples behind the fraction's error bar. Cheap: each is a fast re-evaluation of the estimator on
a resampled score, not a re-score of the images."""


@dataclass(frozen=True)
class FractionEstimate:
    """The fraction of a pool in state A, with the error bar that is the point of the method.

    Args:
        fraction: `p_hat` in (0, 1), the estimated fraction of the pool in state A.
        fraction_err: Bootstrap standard error of `fraction`, from resampling the pool's particles.
            The sampling error at this pool size, NOT the error on the underlying discriminability.
        method: Which route produced it (`"no_pool"`, `"one_pool"`, `"two_pool"`).
        n_particles: Particles in the pool the estimate was read from.
        contrast: The calibrated template scale `a` the pool was scored at. A diagnostic, because
            `d^2` goes as `a^2`: an implausible `a` is the first sign a fraction should not be quoted.
    """

    fraction: float
    fraction_err: float
    method: str
    n_particles: int
    contrast: float


_METHODS = ("no_pool", "one_pool", "two_pool")
"""The fraction routes `method` selects, indexed by how many pure reference pools they need: the bare
fixed point (none), the reference mixture MLE against one pool mirrored about the log-ratio boundary
(one), and the same MLE against two measured pure pools (two)."""


def _read_fraction(
    method: str,
    pool_lambda: np.ndarray,
    pure_a: np.ndarray | None,
    pure_b: np.ndarray | None,
) -> float:
    """Read `p_hat` in (0, 1) off the scored log-ratios by the chosen route.

    `no_pool` reads the pool's log-ratios alone; `one_pool` runs the reference mixture MLE against one
    pure pool and its reflection; `two_pool` runs it against both measured pools. `estimate_fraction`
    scores exactly the extras the route reads, so the unused arguments are `None`.
    """
    if method == "no_pool":
        return fraction_no_pool(jnp.asarray(pool_lambda))
    if method == "one_pool":
        assert pure_a is not None
        return fraction_one_pool(pool_lambda, pure_a)
    assert pure_a is not None and pure_b is not None
    return fraction_two_pool(pool_lambda, pure_a, pure_b)


def _marginal_log_ratio(
    images: Complex[Array, "n y x"],
    bank_a: Complex[Array, "p y x"],
    bank_b: Complex[Array, "p y x"],
    noise_spectrum: Array,
    *,
    contrast: float,
    grid: PoseGrid,
    transfer: Transfer,
) -> np.ndarray:
    """The pose-marginalized log-ratio per particle, scored against one template pair."""
    return np.asarray(
        pose_log_ratio(
            images,
            bank_a,
            bank_b,
            noise_spectrum,
            amplitude=contrast,
            pose_log_weights=grid.log_weights,
            transfer=transfer,
        ).marginal
    )


def _bootstrap_error(
    method: str,
    pool_lambda: np.ndarray,
    pure_a: np.ndarray | None,
    pure_b: np.ndarray | None,
    key: PRNGKeyArray,
    n_bootstrap: int,
) -> float:
    """Standard error of the estimate, from resampling the pool's particles with replacement.

    Two pure reference pools are held fixed, being references rather than the sampled quantity. Zero
    when fewer than two resamples are asked, a standard deviation over one draw being undefined.
    """
    m = pool_lambda.shape[0]
    if n_bootstrap < 2 or m < 2:
        return 0.0
    indices = np.asarray(jax.random.randint(key, (n_bootstrap, m), 0, m))
    draws = [float(_read_fraction(method, pool_lambda[row], pure_a, pure_b)) for row in indices]
    return float(np.std(draws, ddof=1))


def estimate_fraction(
    pair: ConformerPair,
    plan: CollectionPlan,
    pool: Pool,
    *,
    key: PRNGKeyArray,
    method: str = "no_pool",
    pure_pools: tuple[Pool, Pool] | None = None,
    contrast: float | None = None,
    pose_grid: PoseGrid | None = None,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
) -> FractionEstimate:
    """The fraction of a mixed pool in state A, from two structures and the particles.

    Renders the two templates from `pair` and `plan`, calibrates the template contrast against the
    pool, scores each particle's pose-marginalized log-ratio, and reads the fraction off it with the
    chosen `method`, plus a bootstrap error bar. The pose is a nuisance a real pool does not record,
    so it is marginalized over a bank; the known-pose oracle is a simulation benchmark, not here.

    Args:
        pair: Two conformers in one frame, from `structures.align_conformers`.
        plan: The collection the templates are built under.
        pool: The particles and the CTF each was taken under (`pools.Pool`).
        key: PRNG key for the bootstrap resampling.
        method: Which route reads the fraction -- `"no_pool"` (the bare fixed point, the default,
            needing no reference), `"one_pool"` (the reference mixture MLE against one pure pool
            mirrored about the log-ratio boundary), or `"two_pool"` (that MLE against two pure pools).
        pure_pools: Two pure reference pools `(state_a, state_b)`. Required by `"two_pool"`, which uses
            both; `"one_pool"` uses only the first; ignored by `"no_pool"`.
        contrast: The template scale to score at. `None` calibrates it against the pool with an EM
            fit; a value skips the fit and scores at exactly that scale.
        pose_grid: Bank the pose is marginalized over. `None` gives a coarse default, which makes the
            score a lower bound converging from below.
        n_bootstrap: Resamples behind the error bar. Under two returns a zero error.

    Returns:
        The fraction, its bootstrap error, the route, the pool size, and the calibrated contrast.

    Raises:
        ValueError: If `method` is unknown, or if it needs pure pools and `pure_pools` is not given.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    if method in ("one_pool", "two_pool") and pure_pools is None:
        raise ValueError(f"method {method!r} needs pure reference pools; pass pure_pools=(state_a, state_b)")

    grid = pose_bank(_DEFAULT_NODES) if pose_grid is None else pose_grid
    model_a, model_b = pair.models(plan, box_size=pool.box_size)
    bank_a, bank_b = model_a.bare_contrast(grid.angles), model_b.bare_contrast(grid.angles)

    # calibrate the contrast and the noise spectrum against the pool, unless the contrast is given:
    # d^2 goes as a^2, so the fraction is only as good as the scale the score is read at
    fit = fit_noise_model(
        pool.images,
        jnp.stack([bank_a, bank_b]),
        transfer=pool.transfer,
        pose_log_weights=grid.log_weights,
        n_iterations=_EM_ITERATIONS,
    )
    noise_spectrum = fit.noise_spectrum
    scale = float(fit.contrast) if contrast is None else contrast

    pool_lambda = _marginal_log_ratio(
        pool.images, bank_a, bank_b, noise_spectrum, contrast=scale, grid=grid, transfer=pool.transfer
    )
    pure_a = pure_b = None
    if method in ("one_pool", "two_pool"):
        assert pure_pools is not None
        pure_a = _marginal_log_ratio(
            pure_pools[0].images,
            bank_a,
            bank_b,
            noise_spectrum,
            contrast=scale,
            grid=grid,
            transfer=pure_pools[0].transfer,
        )
        if method == "two_pool":
            pure_b = _marginal_log_ratio(
                pure_pools[1].images,
                bank_a,
                bank_b,
                noise_spectrum,
                contrast=scale,
                grid=grid,
                transfer=pure_pools[1].transfer,
            )

    return FractionEstimate(
        fraction=float(_read_fraction(method, pool_lambda, pure_a, pure_b)),
        fraction_err=_bootstrap_error(method, pool_lambda, pure_a, pure_b, key, n_bootstrap),
        method=method,
        n_particles=pool.n_particles,
        contrast=scale,
    )
