"""What a pipeline actually delivered, against what the theory permitted.

A per-particle posterior `r_i = P(A | Y_i)` reads as information about the population fraction, on
the scale `statistics.information` and `statistics.power` use:

    I_ach = mean_i (r_i - p)^2 / [p (1 - p)]^2

the observed Fisher information (exact; the mixture log-likelihood is linear in p). `pipeline_efficiency`
is the ratio to a rung's `I_p`; `fraction_bias` and `calibration_curve` check the responsibilities.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from cryodisc.statistics.information import fisher_from_deflection


class Calibration(eqx.Module):
    """Reliability of responsibilities read as probabilities.

    Args:
        bin_center: Center of each equal-width bin across [0, 1], (b,).
        predicted: Mean responsibility in the bin, NaN where empty, (b,).
        empirical: Fraction of the bin whose true state is A, NaN where empty, (b,).
        count: Particles in the bin, (b,).
        expected_error: Count-weighted mean `|predicted - empirical|`, zero for a calibrated posterior.
    """

    bin_center: Float[Array, " b"]
    predicted: Float[Array, " b"]
    empirical: Float[Array, " b"]
    count: Int[Array, " b"]
    expected_error: Float[Array, ""]


@eqx.filter_jit
def achieved_information(
    responsibilities: Float[Array, " n"],
    *,
    fraction: Float[Array, ""] | float | None = None,
) -> Float[Array, ""]:
    """The per-particle information a set of responsibilities carries about the fraction.

    Args:
        responsibilities: `r_i = P(A | Y_i)` in [0, 1], one per particle.
        fraction: The fraction to score at. Defaults to `mean(r)`, the estimator's own fixed point,
            which is self-limiting. Name a known fraction ONLY where the estimator is near-unbiased
            for it: elsewhere `mean((r - p)^2)` absorbs the squared bias and stops being an
            information.

    Returns:
        I_ach, comparable to `statistics.information.fisher_from_deflection`. Zero at a degenerate
        fraction, and not clipped to the ceiling, a breach being the thing worth seeing.
    """
    r = jnp.asarray(responsibilities)
    p = r.mean() if fraction is None else jnp.asarray(fraction)
    denominator = (p * (1.0 - p)) ** 2
    return jnp.where(denominator > 0.0, jnp.mean((r - p) ** 2) / jnp.maximum(denominator, 1e-300), 0.0)


@eqx.filter_jit
def pipeline_efficiency(
    responsibilities: Float[Array, " n"],
    *,
    deflection: Float[Array, ""] | float,
    fraction: Float[Array, ""] | float | None = None,
) -> Float[Array, ""]:
    """Achieved over predicted information, which is the algorithmic loss.

    Args:
        responsibilities: `r_i = P(A | Y_i)`.
        deflection: The d^2 the corresponding rung predicts.
        fraction: The fraction to score both at. Defaults to `mean(r)`.

    Returns:
        `I_ach / I_p`, at most one for a pipeline that merely loses information. Above one means the
        responsibilities are not estimating this fraction, not that the bound was beaten. `statistics.power`
        takes it as a static argument, so `float()` it on the way there.
    """
    r = jnp.asarray(responsibilities)
    p = r.mean() if fraction is None else jnp.asarray(fraction)
    predicted = fisher_from_deflection(deflection, p)
    return jnp.where(
        predicted > 0.0,
        achieved_information(r, fraction=p) / jnp.maximum(predicted, 1e-300),
        jnp.inf,
    )


@eqx.filter_jit
def fraction_bias(
    responsibilities: Float[Array, " n"],
    *,
    fraction: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """How far the responsibilities' own fraction sits from a known one, in binomial sd.

    The cheapest check that they estimate the fraction they are about to be read as estimating, and
    the only one that catches a `mean(r)` far from the truth, since scoring there is self-limiting.
    A few sd is sampling noise; tens of sd is a miscalibrated posterior or a swapped label
    convention, either of which makes the information at that fraction meaningless.

    Args:
        responsibilities: `r_i = P(A | Y_i)`.
        fraction: The known population fraction.

    Returns:
        `(mean(r) - p) / sqrt(p (1 - p) / n)`, signed.
    """
    r = jnp.asarray(responsibilities)
    p = jnp.asarray(fraction)
    return (r.mean() - p) / jnp.sqrt(p * (1.0 - p) / r.shape[0])


@eqx.filter_jit
def calibration_curve(
    responsibilities: Float[Array, " n"],
    labels: Int[Array, " n"] | Bool[Array, " n"],
    n_bins: int = 10,
) -> Calibration:
    """Bin the responsibilities and compare each bin's claim against the labels.

    Args:
        responsibilities: `r_i = P(A | Y_i)`.
        labels: An is-A mask, 1/True where the true state is A. This is the OPPOSITE of
            `pools.Pool.labels` (0 for A, 1 for B); pass `pool.labels == 0` or the chain's `is_a`,
            never the raw labels, or every bin's claim is compared against the wrong class.
        n_bins: Equal-width bins across [0, 1].

    Returns:
        The curve, and the expected calibration error summarizing it.
    """
    r = jnp.asarray(responsibilities)
    is_a = (jnp.asarray(labels) == 1).astype(r.dtype)
    index = jnp.clip((r * n_bins).astype(int), 0, n_bins - 1)
    count = jnp.bincount(index, length=n_bins)
    filled = count > 0
    safe = jnp.maximum(count, 1)
    predicted = jnp.where(filled, jnp.bincount(index, weights=r, length=n_bins) / safe, jnp.nan)
    empirical = jnp.where(filled, jnp.bincount(index, weights=is_a, length=n_bins) / safe, jnp.nan)
    gap = jnp.where(filled, jnp.abs(predicted - empirical), 0.0)
    return Calibration(
        bin_center=(jnp.arange(n_bins) + 0.5) / n_bins,
        predicted=predicted,
        empirical=empirical,
        count=count,
        expected_error=jnp.sum(count * gap) / jnp.maximum(count.sum(), 1),
    )
