"""`predict` is the prospective half of `run_chain`, and must agree with it on the rungs they share.

The point of the shared `_score_pair` is that a user who only wants the prediction pays for the two
rungs and nothing else, while getting exactly the numbers the full chain would report -- so the
strong check here is bit-identity against `run_chain` at the same pose measure on one key.
"""

import jax
import pytest

from cryodisc import CollectionPlan, predict
from cryodisc.grids.poses import pose_bank
from reproduction.chain import run_chain

PLAN = CollectionPlan(dose=40.0, pixel_size=2.0, defocus_range=(8000.0, 20000.0))
BOX = 48
GRID = pose_bank(96)


@pytest.fixture(scope="module")
def pair(bent_pair_factory):
    """A bend large enough that the marginal rung carries real signal at this box."""
    return bent_pair_factory(n_atoms=5000, spread=10.0, bend=2.5)


def test_predict_returns_the_two_prospective_rungs(pair):
    """The oracle bounds the marginal, and the marginal carries a positive information and count."""
    result = predict(pair, PLAN, key=jax.random.PRNGKey(0), box_size=BOX, n_particles=400, pose_grid=GRID)

    assert result.known_pose.d2 >= result.marginal.d2
    assert result.marginal.information > 0.0
    assert result.marginal.n_for_error > 0.0
    assert result.known_pose.d2_err == 0.0  # closed form from the model, no sampling error
    assert result.n_particles == 400


def test_predict_agrees_with_the_haar_chain_on_the_shared_rungs(pair):
    """`predict` is `run_chain`'s known_pose and marginal at poses='haar', shift_sigma=0, so on one
    key the two return bit-identical rungs -- neither re-simulates, both read one `_score_pair`."""
    shared = {"box_size": BOX, "n_particles": 400, "pose_grid": GRID}
    prediction = predict(pair, PLAN, key=jax.random.PRNGKey(0), **shared)
    chain = run_chain(pair, PLAN, key=jax.random.PRNGKey(0), poses="haar", shift_sigma=0.0, **shared)

    assert prediction.known_pose == chain.known_pose
    assert prediction.marginal == chain.marginal
    assert prediction.operating_point == chain.operating_point


def test_predict_refuses_a_box_that_would_clip_the_templates(pair):
    """The floor guard lives in the shared setup, so `predict` inherits it."""
    with pytest.raises(ValueError, match="below the"):
        predict(pair, PLAN, key=jax.random.PRNGKey(0), box_size=8, n_particles=50, pose_grid=GRID)
