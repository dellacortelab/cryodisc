"""Checks on the null a rung has to clear.

The floor's whole job is to be right when the score is meaningless, so the cases that matter are the
degenerate ones: a numerically zero log-ratio, a pool with no separation, a twin whose two states
are the same object. A floor that reads confidently on any of those is worse than no floor.
"""

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reproduction.validation.floor import noise_floor, shuffle_floor, twin_pair

BOX = 12


def flat_banks(rng, n_poses=4, box=BOX):
    """Two independent template banks, as fft2 of real images."""
    return tuple(jnp.asarray(np.fft.fft2(rng.normal(size=(n_poses, box, box)))) for _ in range(2))


def test_identical_banks_score_molecule_free_boxes_at_exactly_zero():
    """Not a measurement but an identity, and the one that says the scorer itself adds no constant:
    with the two banks the same object every particle's Lambda is zero term by term."""
    rng = np.random.default_rng(0)
    bank, _ = flat_banks(rng)
    null = noise_floor(bank, bank, jnp.ones((BOX, BOX)), key=jax.random.key(0), n_particles=64, n_repeats=3)

    assert null.mean == pytest.approx(0.0, abs=1e-6)
    assert null.sd == pytest.approx(0.0, abs=1e-6)


def test_two_different_banks_manufacture_a_constant_on_boxes_with_no_molecule():
    """The finding this exists for. A noise box has no state, so a scorer that treated the banks
    even-handedly would return zero; a bank with more freedom to fit a given draw does not."""
    rng = np.random.default_rng(1)
    bank_a, bank_b = flat_banks(rng)
    # state A gets four times the power, so it fits any noise realization better
    null = noise_floor(2.0 * bank_a, bank_b, jnp.ones((BOX, BOX)), key=jax.random.key(1), n_particles=256)

    assert null.sd > 0.0
    assert abs(null.z(0.0)) > 3.0  # the offset is resolved, not a fluctuation of the null


def test_the_null_carries_the_measurements_own_mix_of_defocus_groups():
    """The transfer's index is cycled rather than collapsed, so a two-group measurement is not
    bounded by a null scored entirely in whichever group happened to come first."""
    from cryodisc.microscope import Transfer

    rng = np.random.default_rng(2)
    bank_a, bank_b = flat_banks(rng)
    values = jnp.stack([jnp.full((BOX, BOX), 1.0), jnp.full((BOX, BOX), -1.0)])
    transfer = Transfer(values=values, index=jnp.array([0, 1, 0, 1]))

    null = noise_floor(
        bank_a, bank_b, jnp.ones((BOX, BOX)), key=jax.random.key(2), n_particles=64, n_repeats=3, transfer=transfer
    )
    assert np.isfinite(null.mean)


def test_the_noise_null_checks_its_inputs():
    rng = np.random.default_rng(3)
    bank_a, bank_b = flat_banks(rng)
    spectrum = jnp.ones((BOX, BOX))

    with pytest.raises(ValueError, match="template banks differ"):
        noise_floor(bank_a, bank_b[:2], spectrum, key=jax.random.key(0))
    with pytest.raises(ValueError, match="n_repeats >= 2"):
        noise_floor(bank_a, bank_b, spectrum, key=jax.random.key(0), n_repeats=1)


def test_the_permutation_null_is_centered_on_no_separation():
    """Permuting the labels destroys the association and nothing else, so the null must sit at zero
    however far from zero the score itself is."""
    rng = np.random.default_rng(0)
    log_ratio = jnp.asarray(rng.normal(3.0, 2.0, 800))  # a large offset shared by both pools
    is_a = jnp.asarray(np.arange(800) < 400)

    floor = shuffle_floor(log_ratio, is_a, key=jax.random.key(0), n_repeats=300)
    # the null's own standard error is sd/sqrt(repeats); three of those is a generous band
    assert abs(floor.mean) < 3.0 * floor.sd / np.sqrt(300)
    assert floor.sd > 0.0
    assert floor.upper_95 > floor.mean


def test_a_real_separation_clears_its_floor_and_a_shared_offset_does_not():
    """The pair of cases the floor exists to distinguish. Both pools carry the same +3 offset, so a
    statistic reading the mean alone would call the second one separated."""
    rng = np.random.default_rng(1)
    is_a = jnp.asarray(np.arange(1000) < 500)
    key = jax.random.key(1)

    separated = jnp.asarray(np.where(np.arange(1000) < 500, 5.0, -5.0) + rng.normal(0.0, 1.0, 1000))
    floor = shuffle_floor(separated, is_a, key=key)
    measured = float(jnp.mean(separated[:500]) - jnp.mean(separated[500:]))
    assert floor.clears(measured)
    assert floor.z(measured) > 10.0

    unseparated = jnp.asarray(3.0 + rng.normal(0.0, 1.0, 1000))
    floor = shuffle_floor(unseparated, is_a, key=key)
    measured = float(jnp.mean(unseparated[:500]) - jnp.mean(unseparated[500:]))
    assert not floor.clears(measured)


def test_a_numerically_zero_score_lands_inside_its_own_null():
    """Defect 14: a scale-free rank statistic reads a confident z on an LLR that is numerical dust. A
    shuffle floor cannot, because the null it builds is scaled by the same dust the measurement is."""
    dust = jnp.asarray(np.random.default_rng(2).normal(0.0, 5e-8, 600))
    is_a = jnp.asarray(np.arange(600) < 300)
    floor = shuffle_floor(dust, is_a, key=jax.random.key(2), n_repeats=300)

    measured = float(jnp.mean(dust[:300]) - jnp.mean(dust[300:]))
    assert not floor.clears(measured)
    assert abs(floor.z(measured)) < 3.0
    # and the null's width tracks the dust rather than sitting at some absolute scale
    assert floor.sd < 1e-7


def test_the_permutation_preserves_the_pool_sizes():
    """An unbalanced pool has a wider null than a balanced one at the same total, and a permutation
    that resampled the labels rather than shuffling them would lose that."""
    rng = np.random.default_rng(3)
    log_ratio = jnp.asarray(rng.normal(0.0, 1.0, 1000))
    balanced = shuffle_floor(log_ratio, jnp.asarray(np.arange(1000) < 500), key=jax.random.key(3))
    lopsided = shuffle_floor(log_ratio, jnp.asarray(np.arange(1000) < 250), key=jax.random.key(3))
    assert lopsided.sd > balanced.sd


def test_a_degenerate_or_starved_pool_is_refused_or_flagged():
    """An empty pool has no divergence at all, and a thin one has a null whose own width is barely
    resolved -- the second is a warning rather than an error, a thin floor still being worth seeing."""
    log_ratio = jnp.asarray(np.random.default_rng(4).normal(size=400))
    with pytest.raises(ValueError, match="both pools"):
        shuffle_floor(log_ratio, jnp.zeros(400, bool), key=jax.random.key(4))
    with pytest.raises(ValueError, match="at least two permutations"):
        shuffle_floor(log_ratio, jnp.asarray(np.arange(400) < 200), key=jax.random.key(4), n_repeats=1)
    with pytest.raises(ValueError, match="must agree"):
        shuffle_floor(log_ratio, jnp.asarray(np.arange(300) < 150), key=jax.random.key(4))
    with pytest.warns(UserWarning, match="below 200"):
        shuffle_floor(log_ratio, jnp.asarray(np.arange(400) < 20), key=jax.random.key(4), n_repeats=50)


def test_a_zero_width_null_reports_an_infinite_z_rather_than_dividing_by_zero():
    """Every permutation of a constant score gives the same divergence, so the null has no width."""
    constant = jnp.zeros(400)
    floor = shuffle_floor(constant, jnp.asarray(np.arange(400) < 200), key=jax.random.key(5), n_repeats=20)
    assert floor.sd == 0.0
    assert floor.z(1.0) == np.inf
    assert floor.z(-1.0) == 0.0


def test_a_twin_shares_one_atom_table_rather_than_two_equal_ones(bent_pair_factory):
    """Defect 12: the twin's two pools must differ by conformation ALONE, and `a_A == a_B` is the
    detection. Sharing the object makes it hold by construction rather than by a tolerance."""
    pair = bent_pair_factory(160, 6.0, 2.0)  # 40 residues, above the 20 C-alpha floor
    twin = twin_pair(pair)
    assert twin.state_b is pair.state_a
    assert twin.state_a is pair.state_a
    assert twin.alignment is pair.alignment


def test_a_twin_inherits_the_measurements_renderer_along_with_its_exposure(bent_pair_factory, monkeypatch):
    """A twin bounds a measurement only if it is the SAME forward model with the conformational
    difference removed, so every render setting -- `form_factor` included -- has to carry over: a twin
    rendered with a different renderer has a different template power and cannot bound the measurement.
    Captured at the call rather than run, which needs no microscope."""
    from reproduction.validation import floor as floor_module

    captured = {}

    twin_result = SimpleNamespace(operating_point=SimpleNamespace(matches=lambda _: True))

    def fake_run_chain(pair, plan, **kwargs):
        captured.update(kwargs)
        return twin_result

    monkeypatch.setattr(floor_module, "run_chain", fake_run_chain)
    measured = SimpleNamespace(
        operating_point=SimpleNamespace(box_size=64, rmsf=0.5),
        n_particles=32,
        fraction=0.5,
        poses="haar",
        shift_sigma=0.0,
        form_factor="atomic_number",
    )

    floor_module.twin_floor(measured, bent_pair_factory(160, 6.0, 2.0), None, key=jax.random.key(0))
    assert captured["form_factor"] == "atomic_number", "the twin renders with what the measurement did"
    # and an explicit kwarg still wins, which is what makes it a default and not a lock
    floor_module.twin_floor(measured, bent_pair_factory(160, 6.0, 2.0), None, key=jax.random.key(0), form_factor="peng")
    assert captured["form_factor"] == "peng"


def test_the_floor_states_the_statistic_it_permuted():
    """`Floor.observed` exists so a caller cannot pair a measurement with a null of another shape.

    `shuffle_floor` permutes `mean(Lambda|A) - mean(Lambda|B)`, so the producer states its own
    statistic rather than letting a caller rebuild it as a half-sized `d2_pool_labeled / 2` and
    compare it against a full-sized null. Asserted on an UNBALANCED pool too, because the two formulas
    agree at `n_A = n_B`, which is exactly why the mismatch stayed invisible.
    """
    rng = np.random.default_rng(0)
    for n_a, n_b in ((300, 300), (200, 400)):
        labels = np.arange(n_a + n_b) < n_a
        values = np.where(labels, 1.0, -1.0) + rng.normal(0.0, 1.0, n_a + n_b)
        log_ratio, is_a = jnp.asarray(values), jnp.asarray(labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            floor = shuffle_floor(log_ratio, is_a, key=jax.random.key(0), n_repeats=200)

        difference = float(jnp.mean(log_ratio[is_a]) - jnp.mean(log_ratio[~is_a]))
        assert floor.observed == pytest.approx(difference, rel=1e-5)
        assert floor.z(floor.observed) > 3.0, "a pool built to separate must clear its own null"

        # the half-sized quantity the driver used to form. It coincides with the statistic ONLY when
        # the pools are balanced, so on 200/400 it is a different number and pairing it with this
        # null is the bug.
        halved = float(jnp.mean(log_ratio * jnp.where(is_a, 1.0, -1.0)))
        if n_a == n_b:
            assert floor.observed == pytest.approx(2.0 * halved, rel=1e-5)
        else:
            # the point is only that they are DIFFERENT quantities once the pools are uneven
            assert abs(floor.observed - 2.0 * halved) > 1e-3 * abs(floor.observed)
