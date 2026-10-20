"""`reproduction/experiments/benchmark/known_fraction.py`: recover a mixing fraction nothing was told.

p_true is exact by construction from the pool counts and no estimator, reference or calibration
touches it, so p_hat tracking it is a measurement rather than a fit. Both arms run here: the
synthetic one that renders its own pools, and the real-data one that is handed measured log-ratios.
"""

import json

import numpy as np
import pytest
from conftest import load_driver


def test_the_known_fraction_benchmark_recovers_a_fraction_it_was_told_nothing_about(separable_pair, tmp_path):
    """p_true is exact by construction from the pool counts, and no estimator, reference or
    calibration touches it, so p_hat tracking it is a measurement and not a fit.

    The coherence assertions catch a driver that forgets the transfer: the banks are CTF-free and
    the stack is rendered THROUGH the plan's defocus, so a pool then scores against the wrong bank
    outright, which a sign check sees and a magnitude check does not.
    """
    driver = load_driver("benchmark/known_fraction.py")
    out = tmp_path / "kf.json"
    argv = ["--pair", str(separable_pair[0]), str(separable_pair[1]), "--out", str(out), "--assembly", "none"]

    # 1200 because --holdout mixes only half of each pool, so the sweep draws 600 per pool
    assert driver.main([*argv, "--box-size", "48", "--n-particles", "1200", "--cap-bank", "504"]) == 0

    result = json.loads(out.read_text())["result"]
    assert result["holdout"]
    assert result["sweep"]["p_true"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert result["coherence"]["a"]["points_at_its_own_bank"]
    assert result["coherence"]["b"]["points_at_its_own_bank"]
    # on the inverse-crime arm the axis spans its full range and sits centered; a driver that mixed
    # the pools backwards reports it NEGATIVE
    assert result["purity"]["f_a"] == pytest.approx(1.0, abs=0.15)
    assert result["purity"]["f_b"] == pytest.approx(0.0, abs=0.15)
    assert result["purity"]["separation"] > 0.8
    assert result["purity"]["midpoint"] == pytest.approx(0.5, abs=0.1)
    assert result["snr"] > 0.0
    # the --pair arm converged the pose bank; the realized bank it settled on is recorded
    assert result["convergence"]["status"] in ("converged", "censored")
    assert result["nodes_realized"] == max(int(k) for k in result["convergence"]["curve"])


def test_the_oracle_arm_reads_the_coherence_against_a_target_instead_of_against_one(separable_pair, tmp_path):
    """The check that convicts a calibration error, which reading against one cannot do.

    Coherence is the only currency this driver reports that sees a scale error at all; every other --
    the sweep, the purity line, both nulls -- is invariant under `Lambda -> s Lambda`. The identity
    it rests on is exact only for the KNOWN-POSE log-ratio, so the driver scores the same images at
    the poses it drew them at, where `predicted_coherence` supplies the number to beat. The
    assertions are on the GAP and on the baseline being below one, not on the reading itself.
    """
    driver = load_driver("benchmark/known_fraction.py")
    out = tmp_path / "oracle.json"
    argv = ["--pair", str(separable_pair[0]), str(separable_pair[1]), "--out", str(out), "--assembly", "none"]

    assert driver.main([*argv, "--box-size", "48", "--n-particles", "600", "--cap-bank", "504"]) == 0

    result = json.loads(out.read_text())["result"]
    # the marginal block says it is marginal, so a reader cannot mistake its ceiling for a target
    assert result["coherence"]["marginalized"] and result["coherence"]["baseline_is_a_ceiling"]
    for state in ("a", "b"):
        read = result["oracle"][state]
        assert read["points_at_its_own_bank"]
        # the baseline is the formula `1 / (1 + Var[d^2] / (4 E[d^2]))`, not a constant: it fails the
        # moment anything reverts to a bare 1.0, the defect being a driver passing no baseline at all
        variance = read["d2_closed_form"] + read["d2_spread"] / 4.0
        assert read["baseline"] == pytest.approx(read["d2_closed_form"] / variance)
        # below one BECAUSE the pool is heterogeneous: `d_i^2` varies with viewing direction and
        # defocus, so no real pool is at the degenerate target a bare one assumes
        assert read["d2_spread"] > 0.0
        assert 0.0 < read["baseline"] < 1.0
        # the tolerance is derived, not chosen: `E[Lambda] = E[d^2]/2` at this rung, so `d2_pool` is a
        # sample mean whose variance `Var[Lambda] = E[d^2] + Var[d^2]/4` gives, and twice its standard
        # error is what a correct pool scatters by
        assert abs(read["d2_pool"] - read["d2_closed_form"]) < 3.0 * 2.0 * (variance / result["n_particles"]) ** 0.5
        assert not read["indicates_scale_error"]
        # and the reading sits at its target: `value` is a ratio of two sampled moments, so it inherits
        # the mean's `sqrt(Var[Lambda]/n)` and a variance's own `sqrt(2/(n-1))`
        relative = (variance / result["n_particles"] / (read["d2_closed_form"] / 2.0) ** 2 + 2.0 / 599.0) ** 0.5
        assert abs(read["decoherence"]) < 3.0 * abs(read["value"]) * relative


def test_the_benchmark_reads_measured_log_ratios_without_banks_or_an_oracle(tmp_path):
    """The real-data arm. Measured Lambda comes from disk without banks or recorded poses, so the
    pool carries no SNR and no oracle rung -- only the purity line the two pools support."""
    driver = load_driver("benchmark/known_fraction.py")
    rng = np.random.default_rng(2)
    for name, center in (("a.npy", +2.0), ("b.npy", -2.0)):
        np.save(tmp_path / name, rng.normal(center, 2.0, 500))
    out = tmp_path / "measured.json"

    argv = ["--log-ratio-a", str(tmp_path / "a.npy"), "--log-ratio-b", str(tmp_path / "b.npy")]
    assert driver.main([*argv, "--out", str(out)]) == 0

    result = json.loads(out.read_text())["result"]
    assert result["snr"] is None
    assert result["purity"]["separation"] > 0.5
    # and no oracle either: a measured pool's pose is what a search had to find, so there is nothing
    # to score at the true pose. ABSENT rather than defaulted to a misleading baseline of one
    assert "oracle" not in result


def test_the_benchmark_refuses_one_pool_because_a_purity_line_takes_two(tmp_path):
    driver = load_driver("benchmark/known_fraction.py")
    np.save(tmp_path / "a.npy", np.zeros(10))
    with pytest.raises(SystemExit):
        driver.main(["--log-ratio-a", str(tmp_path / "a.npy"), "--out", str(tmp_path / "o.json")])


def test_the_benchmark_defaults_to_the_same_assembly_the_aligner_does():
    """`known_fraction --assembly` must default to the same value `align_conformers` does.

    At "1" the aligner refuses entries whose deposited copies do not correspond; at None it
    superposes both copies at once and reports crystal packing as conformational change, which
    nothing downstream can see. Pinned against the aligner's own default rather than a literal, since
    the bug is the DISAGREEMENT: a hardcoded "1" would go green the moment the aligner changed.
    """
    from cryodisc.specimen.structures import DEFAULT_ASSEMBLY as aligner

    for name in ("benchmark/known_fraction.py",):
        driver = load_driver(name)
        assert driver.parse_args(["--pair", "a.pdb", "b.pdb"]).assembly == aligner, name
        # and 'none' still reaches the deposited coordinates, or the default is a cage
        assert driver.parse_args(["--pair", "a.pdb", "b.pdb", "--assembly", "none"]).assembly is None, name


def test_every_driver_that_aligns_a_pair_agrees_on_the_assembly_default():
    """Every driver that aligns a pair must take `--assembly` with the same default, so a later
    driver cannot quietly reintroduce the disagreement.

    Parsed from the source rather than imported, because these drivers pull in heavy optional
    dependencies and the property under test is textual: what the option's default IS.
    """
    import ast
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path("reproduction/experiments").rglob("*.py")):
        source = path.read_text()
        if "align_conformers" not in source:
            continue
        for call in ast.walk(ast.parse(source)):
            if not (isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "add_argument"):
                continue
            if not any(isinstance(a, ast.Constant) and a.value == "--assembly" for a in call.args):
                continue
            keywords = {k.arg: k.value for k in call.keywords}
            default = keywords.get("default")
            converter = keywords.get("type")
            ok = (
                isinstance(default, ast.Constant)
                and default.value == "1"
                and isinstance(converter, ast.Name)
                and converter.id == "assembly_option"
            )
            if not ok:
                offenders.append(path.as_posix())

    assert not offenders, (
        "these drivers align a pair but do not take --assembly as `default='1', "
        f"type=assembly_option`: {offenders}. At None the aligner superposes every deposited copy "
        "at once instead of refusing a pair whose copies do not correspond."
    )


def test_the_fraction_sweep_is_not_invariant_under_a_scale_on_lambda(tmp_path):
    """The 0-pool sweep is NOT invariant under a scale on Lambda; the default two-pool sweep is.

    `no_pool` reads Lambda as a calibrated log-likelihood ratio, so its responsibilities come from
    Lambda's ABSOLUTE size. Shrinking it makes them diffuse and the fit over-separates its components
    to explain the spread, driving `separation` above one -- which looks like the estimator breaking
    down under weak votes and is instead a calibration error. Asserted as a MONOTONE improvement
    rather than at one scale, so the test cannot pass by landing on a lucky constant. `two_pool` reads
    the mixture against the pure pools' own densities, so the same scale moves nothing.
    """
    driver = load_driver("benchmark/known_fraction.py")
    rng = np.random.default_rng(7)
    # weak votes: pools a couple of sigma apart, then crushed toward zero as a low amplitude does
    truth_a, truth_b = rng.normal(+2.5, 1.0, 800), rng.normal(-2.5, 1.0, 800)

    def sweep(estimator: str) -> tuple[list[float], list[float]]:
        residuals, separations = [], []
        for scale in (0.05, 0.2, 1.0):
            np.save(tmp_path / "a.npy", truth_a * scale)
            np.save(tmp_path / "b.npy", truth_b * scale)
            out = tmp_path / f"{estimator}_s{scale}.json"
            argv = [
                "--log-ratio-a", str(tmp_path / "a.npy"), "--log-ratio-b", str(tmp_path / "b.npy"),
                "--fractions", "0.0", "0.25", "0.5", "0.75", "1.0", "--out", str(out),
                "--estimator", estimator,
            ]  # fmt: skip
            assert driver.main(argv) == 0
            purity = json.loads(out.read_text())["result"]["purity"]
            residuals.append(purity["residual_rms"])
            separations.append(purity["separation"])
        return residuals, separations

    pooled_residuals, pooled_separations = sweep("two_pool")
    assert pooled_residuals == pytest.approx([pooled_residuals[-1]] * 3, rel=1e-6)
    assert pooled_separations == pytest.approx([pooled_separations[-1]] * 3, rel=1e-6)

    residuals, separations = sweep("no_pool")

    # a scale changes the answer at all, which is the refuted claim
    assert residuals[0] != pytest.approx(residuals[-1], rel=0.1)
    # and it changes it in ONE direction: crushing Lambda makes the fit worse, monotonically
    assert residuals[0] > residuals[1] > residuals[2]
    # and the crushed fit OVER-separates, reading above one, the signature that sends a reader
    # looking for a weak-vote breakdown; it saturates, so this is a margin, not an ordering
    assert separations[0] > 1.05
    assert separations[1] > 1.05
    assert separations[2] < separations[0]


def test_the_counting_floor_decomposition_matches_a_resampled_estimate():
    """The algebra is worth nothing if the estimator does not attain it, so the driver's own analytic
    line is checked against pools resampled from synthetic pure-state draws.

    Deliberately NOT a grid-scale measurement -- this is pure numpy over 1-D arrays.
    """
    driver = load_driver("benchmark/counting_floor.py")
    rng = np.random.default_rng(0)
    pure_a = rng.normal(+16.0, 6.8, 4000)
    pure_b = rng.normal(-15.0, 6.8, 4000)

    geometry = driver.axis(pure_a, pure_b)
    term = geometry["discriminability_term"]
    for fraction in (0.5, 0.1):
        want = driver.predicted(fraction, 400, term)
        got = driver.resample(pure_a, pure_b, fraction=fraction, n=400, trials=3000, seed=1)
        assert got["sigma_p"] == pytest.approx(want["sigma_p"], rel=0.06)
        assert abs(got["bias"]) < 0.01


def test_the_floor_share_falls_away_from_a_balanced_population():
    """The whole point: the floor's dominance is a property of p = 0.5, so a minority population is
    where the count becomes informative about d^2 rather than about the error target."""
    driver = load_driver("benchmark/counting_floor.py")
    term = 0.04626  # a representative discriminability term
    half = driver.predicted(0.5, 200, term)
    rare = driver.predicted(0.05, 200, term)

    assert half["floor_share"] > 0.8
    assert rare["floor_share"] < 0.55
    # the information is correspondingly further from its cap, which `floor_share` IS -- `I_p / cap`
    # reduces to `binomial / total`, so asserting it separately would restate the two lines above
    assert rare["information"] / rare["cap"] < half["information"] / half["cap"]
    # N at a fixed error target FALLS with p, because the floor term p(1-p) falls
    assert rare["information"] > half["information"]


def test_a_twin_pool_has_no_axis_rather_than_a_narrow_one():
    """Two pools of the same state separate by nothing, and dividing by that separation would report a
    finite `sigma_p` from noise. `twin_floor` builds exactly this case."""
    driver = load_driver("benchmark/counting_floor.py")
    same = np.zeros(64)
    with pytest.raises(SystemExit, match="no axis"):
        driver.axis(same, same)
