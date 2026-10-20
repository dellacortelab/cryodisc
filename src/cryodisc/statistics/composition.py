"""What fraction of a pool is state A, from its log-ratios and how many pure pools are in hand.

Three estimators, indexed by the reference data available. `fraction_no_pool` reads the fraction off
the mixed pool's log-ratios alone (the bare soft-EM fixed point), biased when that score is miscalibrated.
`fraction_one_pool` measures ONE pure pool's score density and mirrors it about the log-ratio boundary
to stand in for the unseen state, then runs the reference mixture MLE. `fraction_two_pool` is that same
MLE with BOTH densities measured, the most accurate, running EM for the mixing weight against them.

Rationale: docs/rationale/statistics/composition.md
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

_CLIP = 1e-6
"""How close to an endpoint `p_hat` may come.

A pool that separates perfectly drives the logit to infinity and the next iterate to NaN.
"""

_LOGIT_LIMIT = 60.0
"""Bracket for the fixed point in `u = logit(p)`. Wide enough that an anchor reading 1 or 0 is a
saturation the derivation predicts rather than a bracket that was too narrow."""

_BISECTION_STEPS = 60
"""Halvings of the `+/- logit(_CLIP)` bracket, about 27.6 wide. Sixty take it below float32
resolution, so the count is a statement about the arithmetic and not about the pool."""

_WEIGHT_ITERATIONS = 200
"""EM steps for the two-pool mixing weight. The two component densities are FIXED, so the weight
update is a contraction and this many steps take it below float32 resolution."""


@eqx.filter_jit
def _em_fraction(log_ratio: Float[Array, " n"]) -> Float[Array, ""]:
    """The soft-EM fixed point, solved by bisection in `u = logit(p)` rather than iterated from 0.5.

    Particle `i`'s responsibility for state A is `sigmoid(Lambda_i + logit(p))` and `p` is their mean;
    this is the root of `logit(mean(sigmoid(Lambda + u))) - u`. Iterating the map instead is not safe at
    small `d^2`: its contraction factor goes to one, so a fixed count returns an answer compressed toward
    0.5. The residual is non-increasing in `u`, so bisection is valid; `jnp.where` rather than
    `lax.cond` keeps it vmappable.
    """
    limit = float(np.log((1.0 - _CLIP) / _CLIP))

    def halve(_: int, bounds: tuple[Float[Array, ""], Float[Array, ""]]):
        low, high = bounds
        middle = 0.5 * (low + high)
        mean = jnp.clip(jax.nn.sigmoid(log_ratio + middle).mean(), _CLIP, 1.0 - _CLIP)
        above = jnp.log(mean / (1.0 - mean)) > middle
        return jnp.where(above, middle, low), jnp.where(above, high, middle)

    low, high = jax.lax.fori_loop(0, _BISECTION_STEPS, halve, (-limit, limit))
    return jnp.clip(jax.nn.sigmoid(log_ratio + 0.5 * (low + high)).mean(), _CLIP, 1.0 - _CLIP)


def fraction_no_pool(log_ratio: Float[Array, " n"]) -> float:
    """The fraction of a pool in state A from its log-ratios alone -- NO reference pool (0 pools).

    The bare soft-EM fixed point. Solved rather than iterated, and there is no iteration count to pass:
    see `_em_fraction` for why a fixed one compressed the estimate toward 0.5 exactly where the
    separation is weakest. On a real, miscalibrated score (mean responsibility misses the prior,
    `E[sigma(Lambda + logit p)] != p`) this is BIASED -- toward 0.5 under overdispersion, otherwise as the
    miscalibration runs -- an error the fixed point cannot undo without a reference. `fraction_one_pool` and
    `fraction_two_pool` remove it with one and two pure pools' measured densities respectively.

    Args:
        log_ratio: Per-particle Lambda for the mixed pool. Real or simulated alike -- the estimator
            does not know and must not care.

    Returns:
        `p_hat` in (0, 1), never exactly at an endpoint. A pure pool therefore reads near 1 or near
        0 rather than at them, and the residual is the estimator's floor rather than an impurity.

    Raises:
        ValueError: If the pool is empty.
    """
    pool = jnp.asarray(log_ratio).ravel()
    if pool.size == 0:
        raise ValueError("the pool is empty")
    return float(_em_fraction(pool))


def fraction_one_pool(log_ratio: np.ndarray, reference: np.ndarray) -> float:
    """The fraction of a pool in state A from ONE pure pool, mirrored about the LLR boundary (1 pool).

    `fraction_two_pool` needs both component densities; with one pure pool only one is measured. For a
    swap-symmetric log-ratio `Lambda = log[P(Y|A)/P(Y|B)]` maps to `-Lambda` when A and B are exchanged,
    so the unseen state's score density is the measured one reflected about the decision boundary
    `Lambda = 0`: `f_B(x) = f_A(-x)`. The fraction then follows from the same reference-mixture EM as
    `fraction_two_pool`, run against the measured pool and its reflection. The reference is oriented by
    its own mean so the estimate is the fraction of the positive (state-A) component whichever pure
    pool is supplied.

    This is the two-pool MLE with the missing reference supplied by reflection: it keeps the two-pool's
    calibration-freedom for the shape it measures and adds only swap symmetry for the shape it does not.
    Exact when the two states are exchangeable in their score distribution; it degrades where they are
    not, and `fraction_two_pool` is the recourse.

    Args:
        log_ratio: Per-particle Lambda on the mixed pool.
        reference: Per-particle Lambda on ONE pure pool, of either state. Its full density is used.

    Returns:
        `p_hat` in `[0, 1]`, the fraction in the positive (state-A) component.

    Raises:
        ValueError: If either pool is empty.
    """
    mix = np.asarray(log_ratio, float).ravel()
    ref = np.asarray(reference, float).ravel()
    if mix.size == 0 or ref.size == 0:
        raise ValueError("the pool is empty")
    state_a = ref if ref.mean() >= 0.0 else -ref  # orient so the first density is the positive state
    return float(fraction_from_reference(mix, two_pool_reference(state_a, -state_a, span=mix)))


def _gaussian_kde(sample: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A Gaussian kernel density of `sample`, Scott's-rule bandwidth, evaluated at `grid`."""
    bandwidth = float(sample.std(ddof=1)) * sample.size ** (-0.2)  # Scott's rule
    if not bandwidth > 0.0:
        bandwidth = 1.0  # a constant pool has no spread; a unit width keeps the density finite
    z = (grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(1) / (sample.size * bandwidth * np.sqrt(2.0 * np.pi))


def fraction_two_pool(log_ratio: np.ndarray, pool_a: np.ndarray, pool_b: np.ndarray) -> float:
    """The fraction of a pool in state A by the reference mixture MLE (2 pools). The most accurate.

    EM for the mixing weight `p` against the two pure pools' EMPIRICAL densities, held fixed as the
    kernel-density estimates of the two pure pools. Because it uses the full densities rather than two
    moments it is nonparametric -- it assumes no shape for the score, reads correctly at any scale, and
    is unmoved by the miscalibration that biases `fraction_no_pool`. Handles heavy tails and states
    whose score distributions are not exchangeable, which the reflection in `fraction_one_pool` cannot.

    Args:
        log_ratio: Per-particle Lambda for the mixed pool.
        pool_a: Per-particle Lambda on the pure state-A pool.
        pool_b: Per-particle Lambda on the pure state-B pool.

    Returns:
        `p_hat` in `[0, 1]`, clipped because the weight is confined to the simplex only up to the
        density floor.

    Raises:
        ValueError: If any pool is empty, or the two pools' densities are indistinguishable on the
            mixture, when no fraction is identifiable.
    """
    mix = np.asarray(log_ratio, float).ravel()
    a = np.asarray(pool_a, float).ravel()
    b = np.asarray(pool_b, float).ravel()
    if mix.size == 0 or a.size == 0 or b.size == 0:
        raise ValueError("the pool is empty")
    # each pool's density on a shared grid, then interpolated onto the mixture: O(pool x grid), not
    # the O(pool x mix) a direct per-particle kernel sum would cost -- the latter is a several-GB
    # matrix at pool sizes in the tens of thousands
    reference = two_pool_reference(a, b, span=mix)
    return float(fraction_from_reference(mix, reference))


def fraction_classifier(log_ratio: np.ndarray, pool_a: np.ndarray, pool_b: np.ndarray) -> float:
    """The fraction in state A by hard-assigning each particle to its more-likely pure pool, then counting.

    The classify-and-count baseline the body contrasts against: each particle is labeled by whichever pure
    pool's measured density is higher at its score, and the fraction is the count of state-A labels. On
    overlapping states this count is compressed toward an even mixture (slope below one), the bias the
    soft per-image estimators avoid. Uses the same grid references as `fraction_two_pool`.

    Args:
        log_ratio: Per-particle Lambda on the mixed pool.
        pool_a: Per-particle Lambda on the pure state-A pool.
        pool_b: The same for state B.

    Returns:
        The fraction of particles labeled state A, in `[0, 1]`.

    Raises:
        ValueError: If any pool is empty.
    """
    mix = np.asarray(log_ratio, float).ravel()
    a = np.asarray(pool_a, float).ravel()
    b = np.asarray(pool_b, float).ravel()
    if mix.size == 0 or a.size == 0 or b.size == 0:
        raise ValueError("the pool is empty")
    grid, density_a, density_b = two_pool_reference(a, b, span=mix)
    return float(np.mean(np.interp(mix, grid, density_a) >= np.interp(mix, grid, density_b)))


def fraction_classifier_adjusted(log_ratio: np.ndarray, pool_a: np.ndarray, pool_b: np.ndarray) -> float:
    """The Forman (2008) adjusted classify-and-count fraction: the naive count inverted through the pools' rates.

    Dividing `fraction_classifier`'s compressed count by the pure pools' own true- and false-positive rates
    at the same decision rule removes the bias, `p = (p_obs - FPR) / (TPR - FPR)` clipped to `[0, 1]`.
    Consistent but, unlike the soft estimators, not minimum-variance, since it still hard-labels each particle.

    Args:
        log_ratio: Per-particle Lambda on the mixed pool.
        pool_a: Per-particle Lambda on the pure state-A pool.
        pool_b: The same for state B.

    Returns:
        The adjusted fraction in state A, in `[0, 1]`.
    """
    a = np.asarray(pool_a, float).ravel()
    b = np.asarray(pool_b, float).ravel()
    tpr = fraction_classifier(a, a, b)
    fpr = fraction_classifier(b, a, b)
    p_obs = fraction_classifier(np.asarray(log_ratio), a, b)
    if tpr == fpr:
        return float(np.clip(p_obs, 0.0, 1.0))
    return float(np.clip((p_obs - fpr) / (tpr - fpr), 0.0, 1.0))


def two_pool_reference(pool_a: np.ndarray, pool_b: np.ndarray, *, span: np.ndarray | None = None) -> tuple:
    """The two pure pools' kernel densities on a shared grid, built once for many mixtures.

    Args:
        pool_a: Per-particle Lambda on the pure state-A pool.
        pool_b: The same for state B.
        span: Further values the grid must cover, such as a mixture beyond the pools' support.

    Returns:
        `(grid, density_a, density_b)`, what `fraction_from_reference` reads.
    """
    arrays = (pool_a, pool_b) if span is None else (pool_a, pool_b, np.asarray(span, float))
    grid = _reference_grid(*arrays, nodes=512)
    return grid, _gaussian_kde(pool_a, grid), _gaussian_kde(pool_b, grid)


def fraction_from_reference(log_ratio: np.ndarray, reference: tuple) -> np.ndarray:
    """The two-pool MLE fraction of one mixture, or of each row of a stack of them, by EM.

    Args:
        log_ratio: Per-particle Lambda, `(n,)` for one pool or `(pools, n)` for many at once.
        reference: `two_pool_reference` of the two pure pools.

    Returns:
        `p_hat` in `[0, 1]`, a scalar or one per row.

    Raises:
        ValueError: If the two densities coincide on the mixture, when no fraction is identifiable.
    """
    grid, density_a, density_b = reference
    mix = np.asarray(log_ratio, float)
    f_a = np.interp(mix, grid, density_a) + 1e-300
    f_b = np.interp(mix, grid, density_b) + 1e-300
    if np.allclose(f_a, f_b):
        raise ValueError("the two pools' densities coincide; no fraction is identifiable")
    p = np.full(mix.shape[:-1] + (1,), 0.5)
    for _ in range(_WEIGHT_ITERATIONS):
        p = (p * f_a / (p * f_a + (1.0 - p) * f_b)).mean(axis=-1, keepdims=True)
    return np.clip(p[..., 0], 0.0, 1.0)


def _reference_grid(*arrays: np.ndarray, nodes: int = 400) -> np.ndarray:
    """The common grid a distribution comparison runs on -- the range spanned by ALL of them, so a
    mixture particle beyond the pure pools' support is covered rather than clamped to an endpoint."""
    lo = min(float(np.min(x)) for x in arrays)
    hi = max(float(np.max(x)) for x in arrays)
    return np.linspace(lo, hi, nodes)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, evaluated without overflow on either tail."""
    out = np.empty(x.shape, dtype=float)
    hi = x >= 0.0
    out[hi] = 1.0 / (1.0 + np.exp(-x[hi]))
    lo = np.exp(x[~hi])
    out[~hi] = lo / (1.0 + lo)
    return out


def information_from_pools(pool_a: np.ndarray, pool_b: np.ndarray, *, fraction: float) -> tuple[float, float]:
    """The per-particle Fisher information about the fraction, and the calibration gate, from two pools.

    `I_p = Var_mix(r)/[p(1-p)]^2` (derivations §9b(e), the responsibility-variance form of the Eq. 6
    integral) with `r` the density-based two-pool posterior, so it is calibration-free and assumes no
    shape for the score -- exact for a calibrated bank, and correct on a real pool where the Gaussian
    `fisher_from_deflection` map is not. The gate `E_mix[sigma(Lambda + logit p)]` equals `fraction`
    iff the raw `Lambda` is a calibrated LLR (§9b(f)); its departure is how far the 0-pool estimate is
    biased.

    Args:
        pool_a: Per-particle `Lambda` on the pure state-A pool.
        pool_b: The same for state B.
        fraction: The operating population fraction of state A.

    Returns:
        `(information, calibration_gate)`.

    Raises:
        ValueError: If either pool is empty or the two carry no information about the fraction.
    """
    a = np.asarray(pool_a, float).ravel()
    b = np.asarray(pool_b, float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("the pool is empty")
    p = float(fraction)
    grid = _reference_grid(a, b, nodes=512)
    dens_a = _gaussian_kde(a, grid)
    dens_b = _gaussian_kde(b, grid)

    def responsibility(sample: np.ndarray) -> np.ndarray:  # density-based two-pool posterior at p
        f_a = np.interp(sample, grid, dens_a) + 1e-300
        f_b = np.interp(sample, grid, dens_b) + 1e-300
        return p * f_a / (p * f_a + (1.0 - p) * f_b)

    r_a, r_b = responsibility(a), responsibility(b)
    # E_mix[(r-p)^2] as p*E_A + (1-p)*E_B; the score is (r-p)/[p(1-p)], so I_p is its variance
    second = p * float(np.mean((r_a - p) ** 2)) + (1.0 - p) * float(np.mean((r_b - p) ** 2))
    information = second / (p * (1.0 - p)) ** 2
    if not information > 0.0:
        raise ValueError("the two pools carry no information about the fraction")
    logit_p = float(np.log(p / (1.0 - p)))
    gate = p * float(_sigmoid(a + logit_p).mean()) + (1.0 - p) * float(_sigmoid(b + logit_p).mean())
    return float(information), float(gate)
