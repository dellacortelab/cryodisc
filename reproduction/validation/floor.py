"""The null a rung must clear before it means anything.

Aligning pure noise to a template reproduces the template, so a separation result does not validate
itself. Three floors, failing differently: `shuffle_floor` permutes the true labels and re-reads the
same Lambda (cheap); `noise_floor` scores images with no molecule, isolating an asymmetry BETWEEN the
banks (the `a -> 0` limit, not a bound); `twin_floor` re-runs the whole chain with the two states
IDENTICAL (the only null that sees the forward model, search and fit). A floor does not transfer
between targets. Rationale: docs/rationale/validation/floor.md
"""

import warnings
from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray

from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.sim import colored_noise
from cryodisc.rungs.marginal import pose_log_ratio
from cryodisc.specimen.structures import ConformerPair
from reproduction.chain import ChainResult, run_chain

_STARVED_POOL = 200
"""Particles per pool below which a floor is reported but not to be trusted.

Every statistic here divides by a within-pool variance, which a thin pool corrupts first. A warning
threshold, not a hard error -- a thin floor is still worth seeing, as long as it is not quoted.
"""


@dataclass(frozen=True)
class Floor:
    """A null distribution for `d^2`, and where a measurement sits inside it.

    The statistic is whatever the producer measured -- a `d^2` from `shuffle_floor` and `twin_floor`,
    a mean Lambda from `noise_floor` -- so `z` is read in the units that producer names.

    Args:
        values: The statistic under each null draw. Keep them: the summaries below assume the null is
            roughly symmetric, and a null that is not is itself the finding.
        mean: Center of the null, not zero in general -- `d2 - mean` is what a count should be derived
            from when a floor is non-negligible.
        sd: Spread of the null, and the unit `z` is quoted in.
        upper_95: 95th percentile. `clears` tests against this rather than `mean + 2 sd`, since the
            percentile survives a skewed null.
        observed: The SAME statistic on the true labels, where the producer can form it. `None` when
            the null has no unpermuted counterpart. The producer must state it, not rebuild it by hand.

    Rationale: docs/rationale/validation/floor.md
    """

    values: Float[Array, " r"]
    mean: float
    sd: float
    upper_95: float
    observed: float | None = None

    def z(self, d2: float) -> float:
        """The excess in units of the null's own spread.

        Args:
            d2: The measured rung.

        Returns:
            `(d2 - mean) / sd`, or `inf` for a positive excess on a null of zero width.
        """
        if self.sd == 0.0:
            return float(np.inf) if d2 > self.mean else 0.0
        return float((d2 - self.mean) / self.sd)

    def clears(self, d2: float) -> bool:
        """Whether a measurement is outside the null at all.

        Args:
            d2: The measured rung.

        Returns:
            True when `d2` exceeds the null's 95th percentile. A rung that does not clear its floor
            has not been shown to differ from no separation.
        """
        return bool(d2 > self.upper_95)


def _summarize(values: Float[Array, " r"]) -> Floor:
    """Package a set of null draws (`d^2` under each) with the statistics a floor is read through."""
    return Floor(
        values=values,
        mean=float(jnp.mean(values)),
        sd=float(jnp.std(values, ddof=1)) if values.size > 1 else 0.0,
        upper_95=float(jnp.quantile(values, 0.95)),
    )


def shuffle_floor(
    log_ratio: Float[Array, " n"],
    is_a: Bool[Array, " n"],
    *,
    key: PRNGKeyArray,
    n_repeats: int = 200,
) -> Floor:
    """The null `d^2` obtained by permuting which particle belongs to which state.

    Every property of the score except its association with the truth is preserved -- pose and
    defocus spread, noise realization, numerical scale -- so it is the right null for a score that
    may be dust.

    Args:
        log_ratio: Per-particle Lambda from any pool rung.
        is_a: True where the particle's true state is A. Its COUNT is preserved by the permutation,
            so the null has the same pool sizes and imbalance as the measurement.
        key: PRNG key.
        n_repeats: Permutations to draw.

    Returns:
        The floor, whose `mean` is zero up to sampling noise and whose `sd` is what matters.
        `observed` carries the unpermuted statistic, so `z(observed)` is the significance.

    Raises:
        ValueError: If the shapes disagree, if either pool is empty, or if `n_repeats` is below two.

    Rationale: docs/rationale/validation/floor.md
    """
    if log_ratio.shape != is_a.shape:
        raise ValueError(f"log_ratio and is_a must agree, got {log_ratio.shape} and {is_a.shape}")
    n_a = int(jnp.sum(is_a))
    if n_a == 0 or n_a == is_a.size:
        raise ValueError(f"both pools must be occupied, got {n_a} of {is_a.size} in state A")
    # two, because `sd` is a ddof=1 variance and one draw makes it undefined rather than zero
    if n_repeats < 2:
        raise ValueError(f"need at least two permutations, got {n_repeats}")
    if min(n_a, is_a.size - n_a) < _STARVED_POOL:
        warnings.warn(
            f"a pool of {min(n_a, is_a.size - n_a)} particles is below {_STARVED_POOL}: the null's "
            "width is itself poorly estimated here, and a floor read off it should not be quoted.",
            stacklevel=2,
        )
    floor = _summarize(_shuffled_divergences(log_ratio, is_a, key, n_repeats))
    # the identity permutation: the same expression `one` computes, with `weight` the true labels
    membership = is_a.astype(log_ratio.dtype)
    observed = float((log_ratio * membership).sum() / n_a - (log_ratio * (1.0 - membership)).sum() / (is_a.size - n_a))
    return replace(floor, observed=observed)


@eqx.filter_jit
def _shuffled_divergences(
    log_ratio: Float[Array, " n"],
    is_a: Bool[Array, " n"],
    key: PRNGKeyArray,
    n_repeats: int,
) -> Float[Array, " r"]:
    """The mean gap `mean(Lambda|A) - mean(Lambda|B)` under `n_repeats` label permutations.

    A permutation significance test on the gap, deliberately its own statistic and not the deflection:
    the question is whether the pool separated at all. Through weights rather than boolean masks -- the
    pool sizes are fixed by the permutation, but a boolean index has a traced length and cannot be
    jitted, while the weighted means are exact.
    """
    membership = is_a.astype(log_ratio.dtype)

    def one(subkey: PRNGKeyArray) -> Float[Array, ""]:
        weight = jax.random.permutation(subkey, membership)
        n_a = weight.sum()
        return (log_ratio * weight).sum() / n_a - (log_ratio * (1.0 - weight)).sum() / (weight.size - n_a)

    return jax.vmap(one)(jax.random.split(key, n_repeats))


def noise_floor(
    templates_a: Complex[Array, "p y x"],
    templates_b: Complex[Array, "p y x"],
    noise_spectrum: Float[Array, "y x"],
    *,
    key: PRNGKeyArray,
    n_particles: int = 2000,
    n_repeats: int = 20,
    amplitude: float = 1.0,
    pose_log_weights: Float[Array, " p"] | None = None,
    transfer: Transfer | None = None,
) -> Floor:
    """The mean Lambda the two banks return on images containing no molecule at all.

    Banks of unequal power differ by `-0.5 (P_A - P_B)` before any data is seen, so this is the null
    for an OFFSET specifically. Neither the offset nor its sign survives a molecule, so it is not a
    bound. Read the mean as the `a -> 0` limit and nothing else. Give it the SAME banks, spectrum,
    amplitude, weights and transfer the measurement used.

    Rationale: docs/rationale/validation/floor.md

    Args:
        templates_a: fft2 of the state-A bank the measurement scored with, CTF-free.
        templates_b: The state-B bank, on the same poses.
        noise_spectrum: sigma^2(k) the measurement was calibrated at, on the unshifted fft grid.
        key: PRNG key. Use a different one from the measurement, whose noise this must not share.
        n_particles: Molecule-free boxes per repeat.
        n_repeats: Independent draws, resolving the null's own width.
        amplitude: Template scale `a`, as scored.
        pose_log_weights: Log pose prior, as scored.
        transfer: Per-group transfer. Its `index` is CYCLED to the drawn particle count, so the null
            carries the measurement's own mix of defocus groups.

    Returns:
        The floor over `mean_i Lambda_i`, whose `mean` is the constant the banks manufacture on an
        EMPTY box. Whether it is resolved is `mean / sd`, NOT `z(0.0)` (which inverts the sign).

    Raises:
        ValueError: If the two banks disagree in shape, or if the counts are not positive.
    """
    if templates_a.shape != templates_b.shape:
        raise ValueError(f"template banks differ: {templates_a.shape} vs {templates_b.shape}")
    if n_particles < 1 or n_repeats < 2:
        raise ValueError(f"need n_particles >= 1 and n_repeats >= 2, got {n_particles} and {n_repeats}")

    grouped = None if transfer is None else transfer.for_particle_count(n_particles)
    means = []
    for subkey in jax.random.split(key, n_repeats):
        images = colored_noise(subkey, noise_spectrum, n_particles)
        scored = pose_log_ratio(
            images,
            templates_a,
            templates_b,
            noise_spectrum,
            amplitude=amplitude,
            pose_log_weights=pose_log_weights,
            transfer=grouped,
        )
        means.append(jnp.mean(scored.marginal))
    return _summarize(jnp.stack(means))


def twin_pair(pair: ConformerPair) -> ConformerPair:
    """The same pair with its second state replaced by its first.

    Args:
        pair: Any aligned pair. Only `state_a` is read.

    Returns:
        A pair whose two states are the SAME atom table object, so no rendering difference is possible
        and `a_A == a_B` holds by construction. The alignment is carried through unchanged.
    """
    return replace(pair, state_b=pair.state_a)


def twin_floor(
    measured: ChainResult,
    pair: ConformerPair,
    plan: CollectionPlan,
    *,
    key: PRNGKeyArray,
    **run_kwargs: object,
) -> ChainResult:
    """Re-run the whole chain on identical states, at the operating point of a measurement.

    Every rung of the result is the floor for the corresponding rung of `measured`: what the forward
    model, pose search and fit produce when there is provably nothing to find.

    Args:
        measured: The run this is a null for. Its settings are reused and its operating point checked
            against the twin's.
        pair: The conformer pair. Only state A is used; see `twin_pair`.
        plan: The collection, which must be the one `measured` ran under.
        key: PRNG key. Use a DIFFERENT key from the measurement: a twin sharing its noise is
            correlated with the thing it is supposed to bound.
        **run_kwargs: Passed to `run_chain`. `box_size`, `rmsf`, `n_particles`, `fraction`, `poses`,
            `shift_sigma` and `form_factor` default to the measurement's.

    Returns:
        The twin's ladder.

    Raises:
        ValueError: If the twin's operating point does not match the measurement's.
    """
    settings: dict[str, object] = {
        "box_size": measured.operating_point.box_size,
        "rmsf": measured.operating_point.rmsf,
        "n_particles": measured.n_particles,
        "fraction": measured.fraction,
        "poses": measured.poses,
        "shift_sigma": measured.shift_sigma,
        # the FORM FACTOR too: `matches()` compares template power, so a default twin would refuse an
        # `--ablate form-factor` run instead of bounding it
        "form_factor": measured.form_factor,
    }
    settings.update(run_kwargs)
    twin = run_chain(twin_pair(pair), plan, key=key, **settings)  # type: ignore[arg-type]
    if not twin.operating_point.matches(measured.operating_point):
        raise ValueError(
            f"the twin ran at template power {twin.operating_point.template_power:.4g} against the "
            f"measurement's {measured.operating_point.template_power:.4g}. A floor is specific to "
            "the information it was measured at, so this one does not bound that run. Pass the same "
            "plan, box, rmsf and amplitude."
        )
    return twin
