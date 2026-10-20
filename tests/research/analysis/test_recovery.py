"""`statistics.recovery`: the prospective count and the draws that test it.

Two drivers share this module so that a survey row and a benchmark row mean the same thing, and the
thing that has to hold is that the draw is the one the prediction is about -- binomial in the state
counts, resampled with replacement from scored pools, and estimated by an efficient estimator.
"""

import numpy as np
import pytest

from reproduction.analysis.recovery import (
    count_from_pools,
    fraction_estimate,
    gap_from_pools,
    measure_scatter,
    predicted_count,
)


def test_gap_from_pools_is_the_signed_mean_gap():
    """`gap_from_pools` is the difference of the two pure pools' means (the Jeffreys divergence when
    Lambda is the exact LLR), kept only for the direction check: a pair whose state A reads LOWER has
    no count to test, and the drivers branch on exactly this being negative."""
    lam_a = np.array([2.0, 3.0, 4.0])
    lam_b = np.array([-1.0, 0.0, 1.0])

    assert gap_from_pools(lam_a, lam_b) == pytest.approx(3.0)
    assert gap_from_pools(lam_b, lam_a) == pytest.approx(-3.0), "swapping the pools flips the sign"


def test_the_count_is_the_package_prediction_and_not_a_reimplementation():
    """The closed form `[p(1-p) + 1/d^2]/sigma^2` is an approximation valid where the states overlap
    heavily. At large `d^2` the package instead puts `I_p` at the counting limit, so the count bottoms
    out at `p(1-p)/sigma^2` however large `d^2` gets. A module that re-derived the closed form here
    would test itself and miss that."""
    floor = 0.05 * 0.95 / 0.02**2

    huge = predicted_count(500.0, fraction=0.05, sigma=0.02)
    assert huge == pytest.approx(floor, rel=0.02), "a trivially separable pair still pays the counting floor"

    modest = predicted_count(5.0, fraction=0.05, sigma=0.02)
    assert modest > huge, "less discriminable costs more particles"
    # and the floor is a FLOOR: nothing buys a count below it
    assert predicted_count(1e6, fraction=0.05, sigma=0.02) >= floor * 0.99


def test_the_bare_fixed_point_finds_a_pure_pool():
    """`fraction_estimate` delegates to the 0-pool fixed point, which reads a pure state-B pool near 0."""
    rng = np.random.default_rng(0)
    d2 = 9.0
    lam = rng.normal(-d2 / 2, np.sqrt(d2), 4000)
    assert fraction_estimate(lam) == pytest.approx(0.0, abs=0.05), "the fixed point finds a pure pool"


def test_the_draw_is_binomial_so_the_term_being_tested_is_not_removed():
    """A real pool's composition is itself a draw, and that variance is the `p(1-p)/N` term which
    DOMINATES the prediction at small p. Fixing `n_a = round(p*n)` removes exactly the term under test.

    Pinned at PERFECT separation, where the estimator recovers `n_a / n` exactly and so has no
    scatter of its own. Everything left is the draw, and it must come out at the binomial `sd`; fix
    `n_a` instead and this reads zero, which is the failure the test exists to catch."""
    rng = np.random.default_rng(1)
    fraction, n, d2 = 0.3, 500, 400.0
    # so far apart that sign(Lambda) is never wrong, hence p_hat = n_a/n with no estimator noise
    lam_a = rng.normal(+d2 / 2, np.sqrt(d2), 20_000)
    lam_b = rng.normal(-d2 / 2, np.sqrt(d2), 20_000)

    sigma_p, bias = measure_scatter(lam_a, lam_b, n=n, fraction=fraction, repeats=800, rng=np.random.default_rng(2))
    binomial = np.sqrt(fraction * (1 - fraction) / n)

    assert sigma_p == pytest.approx(binomial, rel=0.12), f"binomial sd {binomial:.5f}, measured {sigma_p:.5f}"
    assert sigma_p > 0.5 * binomial, "a fixed n_a would collapse this to the estimator's own noise"
    assert abs(bias) < 0.01


def test_count_from_pools_meets_sigma_on_calibrated_pools():
    """The product claim, from the pools' OWN Eq. 6 information (`count_from_pools`, no Gaussian map):
    size `N`, draw that many, and the two-pool recovery meets `sigma`. Nothing is fitted to the
    scatter, so this can fail. The gate reads the prior, confirming the score is calibrated here."""
    rng = np.random.default_rng(3)
    d2, fraction, sigma = 6.0, 0.05, 0.02
    lam_a = rng.normal(+d2 / 2, np.sqrt(d2), 20_000)
    lam_b = rng.normal(-d2 / 2, np.sqrt(d2), 20_000)

    n_float, information, gate = count_from_pools(lam_a, lam_b, fraction=fraction, sigma=sigma)
    assert gate == pytest.approx(fraction, abs=0.01), "a calibrated score's gate reads the prior"
    assert information > 0.0
    n = round(n_float)
    measured, bias = measure_scatter(
        lam_a, lam_b, n=n, fraction=fraction, repeats=600, rng=np.random.default_rng(4), two_pool=True
    )

    assert measured / sigma == pytest.approx(1.0, abs=0.25), f"predicted {n} particles, measured {measured:.5f}"
    assert abs(bias) < 0.02, "the two-pool MLE is close to unbiased at this count"


def test_the_gate_flags_an_uncalibrated_score_where_the_bare_fixed_point_is_biased():
    """An overdispersed score (within != gap, so not a calibrated LLR, §9b(f)): the gate `E[r]` departs
    from the prior and the 0-pool fixed point is badly biased, while the two-pool MLE -- which reads the
    measured densities -- stays unbiased. `count_from_pools` uses that same calibration-free
    information, so it is correct on either score. Moments chosen to mirror the real cas9 pools."""
    rng = np.random.default_rng(5)
    fraction, gap, within = 0.05, 3.5, 10.0
    lam_a = rng.normal(+gap / 2, np.sqrt(within), 20_000)
    lam_b = rng.normal(-gap / 2, np.sqrt(within), 20_000)

    _, _, gate = count_from_pools(lam_a, lam_b, fraction=fraction, sigma=0.02)
    assert abs(gate - fraction) > 0.02, "the gate flags the miscalibration"

    _, bias_no = measure_scatter(
        lam_a, lam_b, n=400, fraction=fraction, repeats=400, rng=np.random.default_rng(6), two_pool=False
    )
    _, bias_two = measure_scatter(
        lam_a, lam_b, n=400, fraction=fraction, repeats=400, rng=np.random.default_rng(6), two_pool=True
    )
    assert abs(bias_two) < 0.02, "the two-pool MLE stays unbiased on the uncalibrated score"
    assert abs(bias_no) > 2 * abs(bias_two), "the 0-pool fixed point is biased where two-pool is not"
