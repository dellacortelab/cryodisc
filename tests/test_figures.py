"""The figure drivers read what the fit recorded, and refuse rather than draw something wrong.

A figure is the one artifact nobody diffs, so these check the two ways a panel can be quietly false:
applying the stored exponents wrong, and ordering a legend by something other than the data.
"""

import json

import numpy as np
import pytest

from reproduction.analysis.si_estimators import stretch
from reproduction.experiments.figures.calculator import read_check
from reproduction.experiments.figures.d2_lawfulness import law_prediction, measured_at_reference, rmsd_alone_prediction
from reproduction.experiments.figures.real_recovery import best_accuracy

pytest.importorskip("matplotlib")


def test_measured_at_reference_returns_each_plateau_unscaled():
    """The survey is collected at the single reference dose, so a plateau comes back as measured."""
    result = {"rows": [{"d2_plateau": 100.0, "dose": 40.0}, {"d2_plateau": 50.0, "dose": 40.0}]}
    assert list(measured_at_reference(result)) == pytest.approx([100.0, 50.0])


def test_law_prediction_applies_the_frozen_mechanistic_law():
    """C * mass * RMSD^2 / (1 + (RMSD/r_star)^q), read straight off the fit record."""
    result = {"canonical": {"C": 2.0, "r_star": 2.0, "q": 2.0}, "rows": [{"mass_kda": 10.0, "ca_rmsd": 2.0}]}
    assert law_prediction(result)[0] == pytest.approx(2.0 * 10.0 * 2.0**2 / (1 + (2.0 / 2.0) ** 2.0))


def test_rmsd_alone_prediction_is_the_best_power_law_in_rmsd():
    """exp(intercept) * RMSD^exp, the panel (b) abscissa the fit record carries."""
    result = {"singles": {"ca_rmsd": {"intercept": np.log(3.0), "exp": 2.0}}, "rows": [{"ca_rmsd": 4.0}]}
    assert rmsd_alone_prediction(result)[0] == pytest.approx(3.0 * 4.0**2)


def test_best_accuracy_is_a_half_when_the_pools_coincide():
    """Two identical pools cannot be told apart, so no threshold beats a coin."""
    pool = np.linspace(-3.0, 3.0, 501)
    accuracy, _ = best_accuracy(pool, pool.copy())
    assert accuracy == pytest.approx(0.5, abs=0.01)


def test_best_accuracy_is_one_when_the_pools_are_disjoint():
    """A separable pair must read one, or the panel understates what a classifier could do."""
    accuracy, threshold = best_accuracy(np.linspace(5.0, 6.0, 200), np.linspace(-6.0, -5.0, 200))
    assert accuracy == pytest.approx(1.0)
    assert -5.0 < threshold < 5.0


def test_stretch_inverts_the_purity_line_exactly():
    """The calibration is the inverse of the compression, so planted anchors must undo exactly."""
    f_a, f_b = 0.8, 0.15
    truth = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    compressed = f_b + (f_a - f_b) * truth
    assert stretch(compressed, f_a, f_b) == pytest.approx(truth)


def test_stretch_is_identity_on_an_uncompressed_pair():
    """Anchors at 0 and 1 mean nothing to undo, and a calibration that still moved the points
    would be silently rescaling a measurement that was already right."""
    values = np.array([0.1, 0.4, 0.9])
    assert stretch(values, 1.0, 0.0) == pytest.approx(values)


def _check_record(path, trials=2001):
    """A minimal `calculator_check.py` record: one SpCas9 point per fraction, one system."""
    row = {"n": 100, "predicted": 0.1, "measured": 0.11, "floor": 0.05, "measured_rel_err": 0.02, "trials": trials}
    path.write_text(
        json.dumps(
            {
                "result": {
                    "cas9_sweep": {"0.5": [row], "0.05": [dict(row, predicted=0.05)]},
                    "systems": [{"system": "hb", "d2_obs": 5.9, "by_fraction": {"0.5": dict(row, ratio=1.1)}}],
                }
            }
        )
    )
    return path


def test_read_check_keys_the_sweep_by_fraction(tmp_path):
    """Panel b draws each fraction from its own rows, so the sweep must come back keyed by it."""
    check = read_check(_check_record(tmp_path / "check.json"))
    assert set(check["sweep"]) == {0.5, 0.05}
    assert check["sweep"][0.05]["predicted"][0] == pytest.approx(0.05)
    assert check["trials"] == 2001


def test_read_check_scales_the_error_bar_by_the_measurement(tmp_path):
    """A sample SD's error is relative, so the bar is the measured value times its relative error --
    a bar taken off the prediction would shrink whenever the prediction is low."""
    run = read_check(_check_record(tmp_path / "check.json"))["sweep"][0.5]
    assert run["err"][0] == pytest.approx(0.11 * 0.02)
    assert run["floor"][0] == pytest.approx(0.05)


def _arms_of(coords):
    """Lever arms from the recorded shape scalars, as figure 5 reads them."""
    from reproduction.analysis.kc_survey import lever_arms
    from reproduction.experiments.panel.pair_features import gyration

    shape = gyration(coords, np.ones(len(coords)))
    return lever_arms(shape["rg"], shape["asphericity"], shape["acylindricity"])


def test_lever_arms_recover_the_axis_distances_of_a_point_cloud():
    """The scalars are inverted exactly, so the arms must equal the RMS distances measured directly."""
    coords = np.random.default_rng(0).normal(size=(4000, 3)) * [9.0, 4.0, 2.0]
    centered = coords - coords.mean(0)
    axes = np.linalg.eigh(centered.T @ centered)[1]
    along = centered @ axes
    direct = [np.sqrt(np.mean(np.sum(np.delete(along, i, axis=1) ** 2, axis=1))) for i in range(3)]
    assert sorted(_arms_of(coords)) == pytest.approx(sorted(direct), rel=1e-6)


def test_a_rod_has_one_short_lever_arm():
    """Turning a rod about its own axis moves nothing, the case three arms exist to capture."""
    rod = np.column_stack([np.linspace(-50, 50, 501), np.zeros(501), np.zeros(501)])
    arms = np.sort(_arms_of(rod))
    assert arms[0] == pytest.approx(0.0, abs=1e-6)
    assert arms[1] == pytest.approx(arms[2])


def test_correlation_cells_scale_as_the_cube_of_frequency_and_each_arm():
    """N_cells is a product of three decorrelation rates, so doubling k multiplies it by eight."""
    from reproduction.analysis.kc_survey import correlation_cells

    arms = np.array([10.0, 20.0, 30.0])
    assert correlation_cells(0.2, arms) / correlation_cells(0.1, arms) == pytest.approx(8.0)
    assert correlation_cells(0.1, arms * [2, 1, 1]) / correlation_cells(0.1, arms) == pytest.approx(2.0)
