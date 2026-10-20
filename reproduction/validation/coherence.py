"""Is the score CALIBRATED? One number per pool, no labels, no second pool, no estimator.

For a pool whose particles all carry the SAME `d^2`, `Lambda | A ~ N(+d^2/2, d^2)` forces
`2 E[Lambda] / Var[Lambda] = 1` exactly. Real pools vary in `d_i^2`, so the target is
`predicted_coherence`'s baseline rather than one. Reading ABOVE one is a scale error and nothing
else -- the one check valid before the baseline is known, and the only readout here that sees a
scale error at all. Read it BEFORE the `deflection`, which is blind to an affine map of Lambda.
Rationale: docs/rationale/validation/coherence.md
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class Coherence:
    """How well one pool's scores match the model that produced them.

    Args:
        value: `2 |E[Lambda]| / Var[Lambda]`, `baseline` for a pure pool scored by a correct model.
        mean: `E[Lambda]`. Its SIGN is a separate finding that `value` would hide.
        variance: `Var[Lambda]`. A numerically tiny one makes the ratio meaningless.
        n: Particles the pool held.
        baseline: What a CORRECTLY scored pool must read, one only when every particle carries the
            same `d_i^2`. Defaults to one; pass `predicted_coherence(...)` where poses or defocus vary.

    Rationale: docs/rationale/validation/coherence.md
    """

    value: float
    mean: float
    variance: float
    n: int
    baseline: float = 1.0

    @property
    def points_at_its_own_bank(self) -> bool:
        """Whether `mean` has the sign the pool's own label implies.

        Returns:
            True when the mean is positive. False is a mislabeled pool or a swapped bank, not a
            weak one, and no extra particles fix it.
        """
        return self.mean > 0.0

    @property
    def decoherence(self) -> float:
        """`baseline - |value|`, which tracks a degraded template's pull.

        Returns:
            Zero for a correctly scored pure pool, positive under decoherence. NEGATIVE is a Lambda
            scaled DOWN, not less-than-perfect decoherence -- fix the calibration before reading it.
        """
        return self.baseline - abs(self.value)

    @property
    def scale_error(self) -> float:
        """The `s` in `Lambda -> s Lambda` that would explain the reading, ASSUMING EQUAL TEMPLATE POWER.

        Decoherence and a scale error both move `value`, so this is the whole story only where
        something independent has ruled decoherence out. A LOWER bound on the amplitude deficit when
        the two templates differ in power, and it can be wrong in SIGN -- prefer `use_power_asymmetry`.

        Returns:
            `baseline / |value|`, so a reading above the baseline gives `s < 1` and one below `s > 1`.
            NaN where `value` is zero.
        """
        return self.baseline / abs(self.value) if self.value != 0.0 else float("nan")

    def use_power_asymmetry(self, power_a: float, power_b: float, d2: float) -> float:
        """The amplitude ratio `r`, corrected for the two templates differing in whitened power.

        Args:
            power_a: Whitened power of state A's template, what `score_pool` prints as SNR^2.
            power_b: The same for state B.
            d2: The pair's `d^2` at the same calibration.

        Returns:
            `r = 1 + (coherence - 1) / (1 + L)` with `L = (power_a - power_b) / d2`. NaN at `L = -1`,
            where the reading carries no information about the amplitude.

        Raises:
            ValueError: If `d2` is not positive, since `L` is then undefined.
        """
        if not d2 > 0:
            raise ValueError(f"d2 must be positive to form the power asymmetry, got {d2}")
        lever = 1.0 + (power_a - power_b) / d2
        return float("nan") if abs(lever) < 1e-12 else 1.0 + (abs(self.value) - 1.0) / lever

    def indicates_scale_error(self, tolerance: float = 0.25, min_particles: int = 128) -> bool:
        """Whether the reading is above one by more than `tolerance`, which only a scale error does.

        Against ONE and not `baseline`: heterogeneity can only push a correct pool below one, so the
        check needs no baseline to be valid.

        Args:
            tolerance: Fractional headroom over one, absorbing sampling error in a ratio of moments.
            min_particles: Below this the answer is False whatever the reading says, a variance over
                `n` points being itself uncertain by `sqrt(2/(n-1))`.

        Returns:
            True when the score is too SMALL by more than the tolerance allows, on a big enough pool.
        """
        if self.n < min_particles or self.value != self.value:
            return False
        return abs(self.value) > 1.0 + tolerance


def predicted_coherence(d2: Float[Array, " n"]) -> float:
    """What a correctly scored pool of these `d_i^2` must read, one only if they are equal.

    From `Var[Lambda] = E[d^2] + Var[d^2] / 4`, the variance a mixture of Gaussians carries when each
    particle's `d_i^2` sets its mean and width. NOT scale-free, so it inherits the amplitude it was
    computed at.

    Args:
        d2: `d_i^2` per particle, at each particle's own pose and defocus.

    Returns:
        `1 / (1 + Var[d^2] / (4 E[d^2]))`, in `(0, 1]`.

    Raises:
        ValueError: If the mean is not positive, a pool of zeros having no baseline.
    """
    mean, spread = float(jnp.mean(d2)), float(jnp.var(d2))
    if not mean > 0.0:
        raise ValueError(f"d^2 must have a positive mean to predict a coherence, got {mean:.6g}")
    return mean / (mean + spread / 4.0)


def fractional_error(log_ratio: Float[Array, " n"], d2: Float[Array, " n"]) -> tuple[float, float]:
    """The per-particle relative error on `d_i^2`, with the score's own noise subtracted off.

    `Lambda_i` has mean `d_i^2 / 2`, so `delta_i = 2 Lambda_i / d_i^2 - 1` is that particle's
    fractional error -- MEAN a scale error, SPREAD what `decoherence` reads. The spread is
    noise-corrected because `Var[Lambda_i] = d_i^2` makes `delta_i` itself noisy by `4 / d_i^2`.

    Args:
        log_ratio: Per-particle Lambda, ORIENTED so a state-B particle's entry is negated -- the same
            array `coherence` takes.
        d2: `d_i^2` for the same particles in the same order, from `rungs.marginal.known_pose_d2`.
            Available only at the oracle rung.

    Returns:
        `(mean, sd)` of `delta_i`. The mean is a scale error, positive where the fitted amplitude is
        too small; the sd is the noise-corrected spread, ZERO where sampling accounts for all of it.

    Raises:
        ValueError: If the arrays disagree in length, or any `d_i^2` is not positive.
    """
    if log_ratio.shape != d2.shape:
        raise ValueError(f"need one d^2 per particle, got {d2.shape} against {log_ratio.shape} of Lambda")
    if not bool(jnp.all(d2 > 0.0)):
        raise ValueError("every d_i^2 must be positive to take a fractional error against it")

    delta = 2.0 * log_ratio / d2 - 1.0
    # `Var[Lambda_i] = d_i^2` propagated through the division and averaged over the pool, that being
    # the term the pool's sampled variance contains
    sampling = float(jnp.mean(4.0 / d2))
    excess = float(jnp.var(delta, ddof=1)) - sampling
    return float(jnp.mean(delta)), float(jnp.sqrt(excess)) if excess > 0.0 else 0.0


def coherence(log_ratio: Float[Array, " n"], *, state_a: bool = True, baseline: float = 1.0) -> Coherence:
    """The label-free calibration readout of one pure pool.

    Args:
        log_ratio: Per-particle Lambda for a pool believed PURE in one state. A mixture returns a
            number that is not this statistic.
        state_a: Whether this pool's true state is A, whose Lambda is positive in the mean. False
            negates the mean so both pools report on the same scale.
        baseline: What a correctly scored pool must read, from `predicted_coherence`. The default of
            one is right only where every particle carries the same `d_i^2`.

    Returns:
        The readout.

    Raises:
        ValueError: If the pool holds fewer than two particles, a variance over one being undefined
            rather than zero, or if `baseline` is outside `(0, 1]`.
    """
    n = int(log_ratio.shape[0])
    if n < 2:
        raise ValueError(f"a variance needs at least two particles, got {n}")
    if not 0.0 < baseline <= 1.0:
        raise ValueError(f"a coherence baseline is 1/(1 + Var/4E) and so lies in (0, 1], got {baseline:.6g}")
    mean = float(jnp.mean(log_ratio)) * (1.0 if state_a else -1.0)
    variance = float(jnp.var(log_ratio, ddof=1))
    return Coherence(
        value=2.0 * mean / variance if variance > 0.0 else float("nan"),
        mean=mean,
        variance=variance,
        n=n,
        baseline=baseline,
    )


def coherence_pair(
    log_ratio_a: Float[Array, " na"],
    log_ratio_b: Float[Array, " nb"],
    *,
    baseline_a: float = 1.0,
    baseline_b: float = 1.0,
) -> tuple[Coherence, Coherence]:
    """Both pools' readouts, which are not redundant.

    A defect shared by the two banks moves them together; a defect in ONE template moves only its own
    pool, separating "our forward model is uniformly off" from "one of these structures is wrong".

    Args:
        log_ratio_a: Lambda on the pool whose true state is A.
        log_ratio_b: Lambda on the pool whose true state is B.
        baseline_a: What pool A must read when correctly scored, from `predicted_coherence`.
        baseline_b: The same for pool B, a SEPARATE number: the pools share a `d^2` only when they
            share a pose and defocus draw.

    Returns:
        `(coherence_a, coherence_b)`, both normalized so a positive mean is the ideal.
    """
    return (
        coherence(log_ratio_a, state_a=True, baseline=baseline_a),
        coherence(log_ratio_b, state_a=False, baseline=baseline_b),
    )
