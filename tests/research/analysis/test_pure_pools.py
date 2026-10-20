"""The mis-specification arm must be off by default, and off must mean bit-identical.

`defocus_error` perturbs what the scorer is told about each particle's defocus. With it unset the
default has to reproduce the un-perturbed pools exactly rather than approximately.
"""

import numpy as np
import pytest

from reproduction.analysis.pure_pools import simulate_pure_pools

SETTINGS = {
    "n_particles": 16,
    "box_size": 32,
    "pixel_size": 2.5,
    "dose": 40.0,
    "nodes": 54,
    "poses": "haar",
    "seed": 3,
}


@pytest.fixture(scope="module")
def pair(bent_pair_factory):
    """One small bent pair, read once for the whole module."""
    return bent_pair_factory(n_atoms=120, spread=9.0, bend=4.0)


def _pools(pair, **overrides):
    return simulate_pure_pools(pair, **{**SETTINGS, **overrides})


def test_the_default_is_bit_identical_to_no_defocus_error(pair):
    """Zero and unset must agree exactly, not merely closely."""
    unset = _pools(pair)
    zero = _pools(pair, defocus_error=0.0)
    for name in ("log_ratio_a", "log_ratio_b", "oracle_log_ratio_a", "oracle_log_ratio_b"):
        assert np.array_equal(np.asarray(getattr(unset, name)), np.asarray(getattr(zero, name))), name


def test_a_defocus_error_moves_the_score_but_not_the_target(pair):
    """The scorer is perturbed; `oracle_d2` is the target it is read against and must not move."""
    clean = _pools(pair)
    dirty = _pools(pair, defocus_error=0.05)

    assert not np.array_equal(np.asarray(clean.log_ratio_a), np.asarray(dirty.log_ratio_a))
    # the target keeps the TRUE transfer, so it is unchanged and the coherence drop stays visible
    assert np.array_equal(np.asarray(clean.oracle_d2_a), np.asarray(dirty.oracle_d2_a))
    # and `snr` describes the pool rather than the scorer's belief about it
    assert float(clean.snr) == float(dirty.snr)


def test_the_perturbation_is_seeded(pair):
    """Same seed, same perturbation -- otherwise the arm is not reproducible."""
    once = _pools(pair, defocus_error=0.05)
    twice = _pools(pair, defocus_error=0.05)
    assert np.array_equal(np.asarray(once.log_ratio_a), np.asarray(twice.log_ratio_a))

    other = _pools(pair, defocus_error=0.05, seed=4)
    assert not np.array_equal(np.asarray(once.log_ratio_a), np.asarray(other.log_ratio_a))


def test_a_subsample_draws_at_random_stays_in_table_order_and_repeats_on_a_seed():
    """`--subsample` exists because `--limit` takes the FRONT of an acquisition-ordered table.

    A cap for cost must not also be a cap on which session the particles came from: the pool it caps
    has to stay representative of the half-map it is scoring.
    """
    from cryodisc.pools import subsample_rows

    n_rows = 10_000
    keep = subsample_rows(n_rows, 500, seed=0)

    assert keep is not None and len(keep) == 500
    assert len(set(keep.tolist())) == 500, "drawn without replacement"

    # NOT the front, which is what --limit gives. A prefix of 500 would have max 499.
    assert keep.max() > n_rows * 0.9, f"the draw looks like a prefix: max index {keep.max()} of {n_rows}"
    assert keep.min() < n_rows * 0.1
    # spread over the table rather than clumped: the mean index sits near the middle
    assert abs(keep.mean() / n_rows - 0.5) < 0.05

    # ascending, so meta stays aligned with any per-particle array written beside it
    assert (np.diff(keep) > 0).all(), "the kept rows must stay in table order"

    # reproducible on the seed, and a different seed is a different pool
    assert (subsample_rows(n_rows, 500, seed=0) == keep).all()
    assert not (subsample_rows(n_rows, 500, seed=1) == keep).all()

    # nothing to do is signaled as None, so a full pool is not rebuilt through a no-op index
    assert subsample_rows(n_rows, 0, seed=0) is None
    assert subsample_rows(n_rows, n_rows, seed=0) is None
    assert subsample_rows(n_rows, n_rows + 1, seed=0) is None


def test_the_streamed_path_reproduces_the_whole_bank_path(pair):
    """Streaming is a memory knob, and it has to stay one.

    `simulate_pure_pools` fuses the two pools into a single sweep so the bank is rendered twice
    rather than four times. That fusion is only exact while each pool keeps its own defocus grouping;
    rebuilding one `Transfer` over the concatenated pools would move these log ratios.
    """
    whole = _pools(pair, stream=False)
    streamed = _pools(pair, stream=True)
    for name in ("log_ratio_a", "log_ratio_b"):
        expected = np.asarray(getattr(whole, name))
        actual = np.asarray(getattr(streamed, name))
        assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5 * np.sqrt(np.mean(expected**2)))
    assert np.isclose(whole.snr, streamed.snr, rtol=1e-5)


def test_the_fused_sweep_survives_a_defocus_error(pair):
    """The fused transfer must carry each pool's SCORING defocus, not its true one."""
    whole = _pools(pair, stream=False, defocus_error=0.02)
    streamed = _pools(pair, stream=True, defocus_error=0.02)
    for name in ("log_ratio_a", "log_ratio_b"):
        expected = np.asarray(getattr(whole, name))
        actual = np.asarray(getattr(streamed, name))
        assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5 * np.sqrt(np.mean(expected**2)))
