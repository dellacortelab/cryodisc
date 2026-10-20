"""`statistics.information` against a reference derived from the mathematics.

The estimators are cheap to write and easy to get subtly wrong -- a `ddof`, a pooling convention, a
missing bias term -- and every such error produces a plausible number. So the reference is derived in
`ref_estimators` from the Gaussian law rather than transcribed, and the expectations it asserts
against are closed forms, not Monte Carlo estimates of production's own output.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.statistics.information import (
    deflection,
    fisher_from_deflection,
    information_ceiling,
)

from . import ref_estimators as ref


def test_the_reference_agrees_with_itself_before_anything_is_compared_to_it():
    """d^2 is defined two ways -- as the mean gap and as the symmetrized KL -- and for a
    common-covariance Gaussian pair they must be the same number. If they are not, the reference is
    wrong and every comparison below is meaningless."""
    rng = np.random.default_rng(0)
    mean_a = rng.normal(size=64)
    mean_b = mean_a + rng.normal(0.0, 0.3, 64)
    variance = rng.uniform(0.5, 2.0, 64)

    d2 = ref.true_d2_gaussian(mean_a, mean_b, variance)
    symmetrized = ref.kl_equal_covariance(mean_a, mean_b, variance) + ref.kl_equal_covariance(mean_b, mean_a, variance)
    assert symmetrized == pytest.approx(d2, rel=1e-12)


def test_the_sampled_law_reproduces_the_index_it_was_built_from():
    """The other half of the reference's self-check: log-ratios drawn from N(+-d^2/2, d^2) must have
    a mean gap of d^2 and a within-class variance of d^2, which is what makes the pool simulations
    below a test of production rather than of the sampler."""
    rng = np.random.default_rng(1)
    values, is_a = ref.sample_llr(3.0, 200_000, 200_000, rng)
    assert ref.ref_mean_gap(values, is_a) == pytest.approx(3.0, rel=0.02)
    assert values[is_a].var() == pytest.approx(3.0, rel=0.02)


def test_the_deflection_matches_the_reference_exactly():
    """The pooling, the ddof and the bias subtraction all have to agree with the derivation, so this
    is asserted on the SAME sample where there is nothing to approximate and nothing to excuse."""
    rng = np.random.default_rng(2)
    values, is_a = ref.sample_llr(2.0, 500, 700, rng)
    assert float(deflection(jnp.asarray(values[is_a]), jnp.asarray(values[~is_a]))) == pytest.approx(
        ref.ref_deflection(values, is_a), rel=1e-5
    )


def test_the_code_deflection_matches_the_reference_and_carries_its_floor():
    """The code `deflection` reproduces the numpy `ref_deflection` sample-for-sample, and both read the
    squared gap high by its `~2/n` floor at small `n`, settling onto `d^2` as the pool grows. No
    correction is applied, so the estimate is never negative -- the floor is convergence, not bias.
    """
    rng = np.random.default_rng(6)
    d2, n = 4.0, 60
    small = []
    for _ in range(800):
        pool_a, pool_b = (np.asarray(p) for p in _split(ref.sample_llr(d2, n, n, rng)))
        llr = np.concatenate([pool_a, pool_b])
        is_a = np.concatenate([np.ones(pool_a.size, bool), np.zeros(pool_b.size, bool)])
        code = float(deflection(jnp.asarray(pool_a), jnp.asarray(pool_b)))
        assert code == pytest.approx(ref.ref_deflection(llr, is_a), rel=1e-4)  # code == reference
        small.append(code)
    large = [
        float(deflection(*(jnp.asarray(np.asarray(p)) for p in _split(ref.sample_llr(d2, 4_000, 4_000, rng)))))
        for _ in range(200)
    ]
    assert float(np.mean(small)) > d2  # the squared gap reads high by its floor at small n
    assert float(np.min(small)) >= 0.0  # never negative
    assert float(np.mean(large)) == pytest.approx(d2, rel=0.01)  # floor gone once the pool is large


@pytest.mark.parametrize("cv", [0.3, 0.6])
def test_a_mixed_pose_pool_reads_its_pooled_deflection(cv):
    """On a pool whose true index varies, the pooled variance gains the spread of the conditional means
    and the deflection falls to `E[D] / (1 + CV^2 E[D] / 4)` -- the honest reading of a marginalized
    pool, checked against the derived law at two spreads."""
    rng = np.random.default_rng(7)
    mean_index = 3.0
    strata = ref.make_strata(mean_index, cv, 24)
    values, is_a, truth = ref.sample_llr_strata(strata, 900, rng)
    assert truth == pytest.approx(mean_index, rel=1e-9)

    measured = float(deflection(jnp.asarray(values[is_a]), jnp.asarray(values[~is_a])))
    predicted = ref.expected_pooled_deflection(mean_index, cv)
    assert measured == pytest.approx(predicted, rel=0.06)
    assert measured < 0.98 * mean_index


@pytest.mark.parametrize("d2", [0.05, 0.5, 2.0, 20.0])
def test_fisher_from_deflection_matches_the_gaussian_reference(d2):
    """The production Gaussian map must reproduce the exact Fisher integral of the two Gaussians."""
    assert float(fisher_from_deflection(d2)) == pytest.approx(ref.ref_gaussian_information(d2), rel=2e-3)


def test_fisher_from_deflection_is_the_deflection_in_the_weak_signal_limit():
    """`I_p -> d^2` as `d^2 -> 0` at p = 1/2, so in the experimental regime a count reads off d^2."""
    for small in (0.01, 0.05):
        assert float(fisher_from_deflection(small, 0.5)) == pytest.approx(small, rel=0.02)


def test_the_information_ceiling_is_the_perfect_label_value():
    """`1/[p(1-p)]`, which the overlap identity makes a theorem. Checked at an asymmetric fraction too,
    where a hard-coded 4 would be wrong."""
    assert float(information_ceiling(0.5)) == pytest.approx(4.0)
    assert float(information_ceiling(0.2)) == pytest.approx(1.0 / 0.16)


def test_the_information_never_exceeds_its_ceiling():
    """A theorem, so it is enforced rather than hoped for: quadrature error must not out-vote it."""
    for d2 in (0.5, 20.0, 400.0, 1e6):
        assert float(fisher_from_deflection(d2)) <= float(information_ceiling(0.5)) + 1e-9


def _split(sampled: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """`(values, is_a)` as the two labeled pools `deflection` takes."""
    values, is_a = sampled
    return values[is_a], values[~is_a]
