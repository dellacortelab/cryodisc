"""`reproduction/experiments/panel/score_survey.py`: score a panel of candidates and record the ranking.

The thread that answers "which target next", so what it has to survive is a panel that is wrong: a
candidate whose structures are not on disk, and one whose two states are the same molecule.
"""

import json
import math
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ALPHA_CARBON, atom_records, load_driver, write_pdb
from scipy.spatial.transform import Rotation

from cryodisc import CollectionPlan
from cryodisc.grids.poses import _choose_bank, median_miss, pose_bank
from cryodisc.statistics.convergence import CROSSOVER_EXPONENT, RefusedExtrapolation, extrapolate_in_k
from reproduction import targets as targets_module
from reproduction.targets import score_target


@pytest.fixture(scope="module")
def toy_panel(tmp_path_factory):
    """Candidates built from synthetic structures, so the panel needs no network.

    One pair is a TWIN -- both states from one atom table -- which a driver has to survive: it
    separates by nothing, so any row reading a separation off it found it in the machinery.
    """
    directory = tmp_path_factory.mktemp("panel")
    rng = np.random.default_rng(0)
    coords = rng.uniform(-9.0, 9.0, (50, 3))
    bent = coords + np.where(coords[:, :1] > 0.0, np.array([[2.5, 0.0, 0.0]]), 0.0)

    for name, xyz in (("AAAA", coords), ("BBBB", bent), ("CCCC", coords)):
        write_pdb(directory / f"{name}.pdb", atom_records(xyz, atoms=ALPHA_CARBON))
    # the PKA shape: the same molecule deposited under a DIFFERENT chain name, plus a second chain
    # that only one entry carries -- 1J3H deposits two copies as A and B where 1ATP deposits one as E
    write_pdb(
        directory / "DDDD.pdb",
        atom_records(coords, atoms=ALPHA_CARBON, chain="A")
        + atom_records(coords + 40.0, atoms=ALPHA_CARBON, chain="B", start_serial=len(coords) + 1),
    )
    write_pdb(directory / "EEEE.pdb", atom_records(bent, atoms=ALPHA_CARBON, chain="E"))

    panel = directory / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "ligands": [],
                "candidates": [
                    {"name": "bent", "pdb_a": "AAAA", "pdb_b": "BBBB", "motion": "hinge", "assembly": None},
                    {"name": "twin", "pdb_a": "AAAA", "pdb_b": "CCCC", "motion": "none", "assembly": None},
                    {
                        "name": "renamed",
                        "pdb_a": "DDDD",
                        "pdb_b": "EEEE",
                        "motion": "hinge",
                        "assembly": None,
                        "chains_a": ["A"],
                        "chains_b": ["E"],
                        "labels_a": ["A"],
                        "labels_b": ["A"],
                    },
                ],
            }
        )
    )
    return panel, directory


def test_the_panel_driver_scores_a_toy_panel_and_records_it(toy_panel, tmp_path):
    """The whole path: read a panel, resolve its structures, score every row, write a result. What
    it really catches is an argument that does not exist -- the driver calls into `targets` and
    `chain` by keyword, and a rename there fails here rather than three hours into a real panel."""
    panel, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    out = tmp_path / "ranking.json"

    assert driver.main(["--panel", str(panel), "--data-dir", str(data_dir), "--out", str(out), "--box-size", "48"]) == 0

    envelope = json.loads(out.read_text())
    assert envelope["driver"] == "reproduction/experiments/panel/score_survey.py"
    assert envelope["commit"]
    assert envelope["result"]["table"]
    # both rows SCORED, not merely accounted for: a panel where every candidate was rejected would
    # satisfy a count that only had to add up, which is the way this kind of test goes green empty
    assert envelope["result"]["n_scored"] == 3
    assert envelope["result"]["n_rejected"] == 0

    rows = {row["name"]: row for row in envelope["result"]["rows"]}
    assert rows["bent"]["d2_floor"] > 0.0
    # the per-entry row pins `load_panel`'s plumbing: `chains_a`/`chains_b` and `labels_a`/`labels_b`
    # together align two entries that name the same subunit differently; drop any and it aligns nothing
    assert rows["renamed"]["d2_floor"] > 0.0
    # the SAME motion as `bent`, since it is the same two coordinate sets under other chain names
    assert rows["renamed"]["ca_rmsd"] == pytest.approx(rows["bent"]["ca_rmsd"], rel=1e-6)
    # the twin separates by nothing, so a row reading anything above zero here found its separation
    # in the machinery rather than in the structures
    assert rows["twin"]["d2_floor"] == 0.0
    # `null` and not `Infinity`, which RFC 8259 cannot hold: the twin needs unboundedly many
    # particles and the bent pair does not, spelled as two facts so a consumer must say which it means
    assert rows["twin"]["n_floor"] is None
    assert math.isfinite(rows["bent"]["n_floor"]) and rows["bent"]["n_floor"] > 0.0


def test_the_panel_driver_refuses_a_missing_structure_rather_than_reaching_the_network(toy_panel, tmp_path):
    """`--fetch` is off by default so a cluster node with no outbound route fails naming the file it
    wanted, instead of timing out somewhere inside a scoring loop."""
    _, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    panel = tmp_path / "missing.json"
    panel.write_text(json.dumps({"candidates": [{"name": "x", "pdb_a": "ZZZZ", "pdb_b": "YYYY"}]}))

    with pytest.raises(FileNotFoundError, match="--fetch"):
        driver.main(["--panel", str(panel), "--data-dir", str(data_dir), "--out", str(tmp_path / "o.json")])


def test_a_refused_marginal_rung_costs_one_row_and_not_the_panel(toy_panel, monkeypatch):
    """A refused marginal rung must cost one row, not the panel: `attach_marginal`'s oracle check
    fires on the row whose pose is BEST determined, so it lands late, after cheaper rows are paid for,
    and has to sit inside the same guard that catches a chain which will not run.

    The refusal is injected rather than provoked, since what this fixes is where the call sits
    relative to the `except`. The successful call comes first so a chain failing for some other reason
    cannot pass the second assertion for free.
    """
    panel, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    args = driver.parse_args(["--panel", str(panel), "--data-dir", str(data_dir), "--n-particles", "64"])
    plan = CollectionPlan(dose=args.dose, pixel_size=args.pixel_size)
    candidate = next(c for c in driver.load_panel(panel, data_dir, fetch=False)[0] if c.name == "bent")
    scored = score_target(candidate, plan, box_size=48)

    measured, _ = driver.measure_marginal(scored, candidate, plan, args)
    assert measured.n_marginal is not None

    def refuse(*_args, **_kwargs):
        raise ValueError("the marginal rung 653.5 exceeds GltPh's known-pose oracle 646.4")

    monkeypatch.setattr(driver, "attach_marginal", refuse)
    survived, null = driver.measure_marginal(scored, candidate, plan, args)
    assert survived is scored
    assert survived.n_marginal is None
    # the refusal travels with the row it cost, so a panel that loses a row still says why
    assert null["measured"] is False
    assert "oracle" in null["error"]


def test_a_recorded_result_carries_the_arguments_that_produced_it(toy_panel, tmp_path):
    """A recorded result carries the arguments that produced it; a payload alone is untraceable."""
    panel, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    out = tmp_path / "args.json"

    driver.main(
        ["--panel", str(panel), "--data-dir", str(data_dir), "--out", str(out), "--box-size", "48", "--sigma", "0.05"]
    )

    arguments = json.loads(out.read_text())["arguments"]
    assert arguments["sigma"] == 0.05
    assert arguments["box_size"] == 48
    assert str(panel) in arguments["panel"]


def test_a_bank_drawn_pool_nests_into_the_ladder_and_a_haar_one_does_not():
    """The hazard `--poses haar` exists for, measured from both sides.

    A `K^(-p)` fit reads how a rung approaches a fine bank. If the pool's own poses ARE bank nodes,
    some rungs contain it exactly and score with no quadrature error, so the fit measures which node
    counts divide the pool's grid. Both halves are asserted because only the contrast identifies the
    cause: the SAME ladder is non-monotone against a bank-drawn pool and monotone against a Haar one.
    """
    ladder = [pose_bank(n) for n in (1024, 2048, 3456, 8192, 16384, 27648, 65536)]

    on_bank = [median_miss(pose_bank(54).angles, grid) for grid in ladder]
    assert on_bank != sorted(on_bank, reverse=True), f"expected nesting, got {on_bank}"
    assert min(on_bank) == pytest.approx(0.0, abs=1e-9), "a nesting rung should contain the pool exactly"

    haar = Rotation.random(400, random_state=3).as_euler("zyz", degrees=True)
    off_bank = [median_miss(haar, grid) for grid in ladder]
    assert off_bank == sorted(off_bank, reverse=True), f"not monotone against a Haar pool: {off_bank}"


def test_a_refused_extrapolation_falls_back_to_the_raw_marginal(toy_panel, monkeypatch):
    """When the K^(-2/3) fit is refused there is no converged number, so the driver quotes the RAW
    marginal at the base bank as the lower bound it is -- bounded above by `known_pose` -- rather than
    a finite-bank correction. The honest number is the measurement, and it carries no correction block.
    """
    panel, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    shared = ["--panel", str(panel), "--data-dir", str(data_dir), "--n-particles", "64", "--extrapolate"]
    args = driver.parse_args(shared)
    plan = CollectionPlan(dose=args.dose, pixel_size=args.pixel_size)
    candidate = next(c for c in driver.load_panel(panel, data_dir, fetch=False)[0] if c.name == "bent")
    scored = score_target(candidate, plan, box_size=48)

    # a ladder the K^(-2/3) fit refuses: rising, but not on the assumed power law
    floor = scored.d2_known_pose
    rungs = sorted(args.extrapolate_nodes)
    rising = [floor * (0.60 + 0.12 * i / max(len(rungs) - 1, 1)) for i in range(len(rungs))]
    ladder = extrapolate_in_k(rungs, rising, [0.001 * floor] * len(rungs))
    with pytest.raises(RefusedExtrapolation):
        ladder.check(ceiling=floor)

    # `run_chain` RETURNS the unchecked ladder rather than raising it. The stub's `marginal` rung is
    # what the driver now falls back to when the ladder is refused.
    def refusable(*_args, **_kwargs):
        return _chain_result(marginal_limit=ladder, ceiling=floor)

    monkeypatch.setattr(driver, "run_chain", refusable)
    measured, record = driver.measure_marginal(scored, candidate, plan, args)
    # no correction block: the deficit corrector is gone
    assert "correction" not in record
    # the refusal does not drop the row on a name error or leave a correction behind; whether the raw
    # marginal clears the floor is the ordinary marginal test, applied to a not-extrapolated rung
    if measured.n_marginal is not None:
        assert record.get("extrapolation", {}).get("method") != "deficit correction"


def test_a_refused_row_still_reads_the_rest_of_its_ladder(toy_panel, monkeypatch):
    """The failure the refusal-inside-the-producer shape actually caused, which nothing caught.

    `--twin-null` reads `result` AFTER the extrapolation is judged. While `run_chain` raised the
    refusal, `result` was never bound on that path, so the row died of an `UnboundLocalError` inside
    the blanket per-row handler and was recorded as `{"error": ...}` -- silently, on exactly the
    refused rows the raw-marginal fallback now keeps. Nothing failed loudly enough to notice.
    """
    panel, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    args = driver.parse_args(
        [
            *["--panel", str(panel), "--data-dir", str(data_dir), "--n-particles", "64"],
            *["--extrapolate", "--twin-null"],
        ]
    )
    plan = CollectionPlan(dose=args.dose, pixel_size=args.pixel_size)
    candidate = next(c for c in driver.load_panel(panel, data_dir, fetch=False)[0] if c.name == "bent")
    scored = score_target(candidate, plan, box_size=48)

    floor = scored.d2_known_pose
    # built from the driver's own node counts rather than a literal, so a ladder of the wrong length
    # dies on a shape mismatch when that default changes
    rungs = sorted(args.extrapolate_nodes)
    rising = [floor * (0.60 + 0.12 * i / max(len(rungs) - 1, 1)) for i in range(len(rungs))]
    ladder = extrapolate_in_k(rungs, rising, [0.001 * floor] * len(rungs))
    with pytest.raises(RefusedExtrapolation):
        ladder.check(ceiling=scored.d2_known_pose)

    result = _chain_result(marginal_limit=ladder, ceiling=scored.d2_known_pose)
    monkeypatch.setattr(driver, "run_chain", lambda *_a, **_k: result)
    # a twin whose rungs are zero by construction, which is what `twin_floor` returns on a real pair
    monkeypatch.setattr(driver, "twin_floor", lambda *_a, **_k: _chain_result(marginal_limit=None, ceiling=0.0))

    measured, record = driver.measure_marginal(scored, candidate, plan, args)

    assert "error" not in record, f"the row died instead of falling back to the raw marginal: {record.get('error')}"
    assert measured.n_marginal is not None, "the raw-marginal fallback should have kept this row"
    assert record["measured"] is True
    # the whole point: the twin null is read off the SAME result whose extrapolation was refused
    assert record["measurement_marginal_d2"] == pytest.approx(result.marginal.d2)
    assert record["d2_rungs_are_zero_by_construction"] is True


def _chain_result(*, marginal_limit, ceiling: float):
    """A stand-in for `ChainResult` carrying only what `measure_marginal` reads off one.

    A real one costs a chain run per call, and the branch under test is reached by what the
    extrapolation says rather than by anything the pool did.

    Args:
        marginal_limit: The `Extrapolation` to hand back, or `None`.
        ceiling: The `known_pose` rung's `d^2`, the bound the extrapolation is checked against.

    Returns:
        An object with the attributes the driver touches.
    """
    # `information` is read by the driver beside `d2`; a stub without it fails the row for a missing
    # attribute rather than for the thing under test
    rung = SimpleNamespace(d2=0.5 * ceiling if ceiling else 0.0, d2_err=0.01, information=0.35)
    return SimpleNamespace(
        marginal_limit=marginal_limit,
        marginal=rung,
        known_pose=SimpleNamespace(d2=ceiling, d2_err=0.0),
        fraction_bias=0.0,
        calibration=SimpleNamespace(expected_error=0.0),
    )


def test_only_selects_named_rows_and_refuses_a_name_the_panel_does_not_have(toy_panel, tmp_path):
    """`--only` is what makes the marginal arm parallel and restartable, one row per job.

    The refusal matters more than the selection: a mistyped name that silently matched nothing would
    score zero rows, write a valid-looking result, and exit 0. The error names the rows that do exist,
    since knowing a name is wrong is not the same as knowing the right one.
    """
    panel, data_dir = toy_panel
    driver = load_driver("panel/score_survey.py")
    out = tmp_path / "only.json"

    argv = ["--panel", str(panel), "--data-dir", str(data_dir), "--box-size", "48"]
    assert driver.main([*argv, "--only", "bent", "--out", str(out)]) == 0

    rows = json.loads(out.read_text())["result"]["rows"]
    assert [row["name"] for row in rows] == ["bent"]

    with pytest.raises(ValueError, match="which this panel does not contain") as refusal:
        driver.main([*argv, "--only", "bnet", "--out", str(tmp_path / "x.json")])
    # and it lists what IS there, so the caller can fix the typo from the message alone
    assert "bent" in str(refusal.value)


def test_a_deposition_rcsb_will_not_serve_as_pdb_is_fetched_as_cif(tmp_path, monkeypatch):
    """RCSB serves no legacy PDB file for a deposition past the format's limits, which is most large
    recent cryo-EM work. Without the `.pdb`-to-`.cif` fallback the survey cannot reach its newer
    sources at all."""
    asked = []

    class Response:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(url):
        asked.append(url)
        if url.endswith(".pdb"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return Response(b"data_9ABC\n")

    monkeypatch.setattr(targets_module.urllib.request, "urlopen", fake_urlopen)
    path = targets_module.fetch_structure("9ABC", tmp_path, fetch=True)

    assert path == tmp_path / "9ABC.cif"
    assert path.read_bytes() == b"data_9ABC\n"
    assert [url.rsplit(".", 1)[1] for url in asked] == ["pdb", "cif"]
    # a second call takes the cached .cif and asks for nothing
    assert targets_module.fetch_structure("9ABC", tmp_path, fetch=True) == path
    assert len(asked) == 2


def test_one_entry_rcsb_will_not_serve_drops_its_row_and_not_the_panel(tmp_path, monkeypatch):
    """One entry RCSB will not serve drops its row, not the run: a row that cannot be scored is a
    row, not a run.

    A 404 is skipped; a file merely absent from disk with --fetch off still raises, because that is an
    operator error and skipping it would turn a wrong --data-dir into a silent "0 scored". The test
    below this one pins that half."""
    data_dir = tmp_path / "pdb"
    data_dir.mkdir()
    xyz = np.random.default_rng(0).uniform(-12.0, 12.0, (24, 3))
    write_pdb(data_dir / "GOOD.pdb", atom_records(xyz, atoms=ALPHA_CARBON))
    write_pdb(data_dir / "ALSO.pdb", atom_records(xyz, atoms=ALPHA_CARBON))

    def always_404(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(targets_module.urllib.request, "urlopen", always_404)

    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "candidates": [
                    {"name": "unserved", "pdb_a": "GOOD", "pdb_b": "MISS", "assembly": None},
                    {"name": "present", "pdb_a": "GOOD", "pdb_b": "ALSO", "assembly": None},
                ]
            }
        )
    )
    candidates, unresolved = targets_module.load_panel(panel, data_dir, fetch=True)

    assert [c.name for c in candidates] == ["present"]
    assert len(unresolved) == 1
    assert "unserved" in unresolved[0] and "GOOD/MISS" in unresolved[0]
    assert "404" in unresolved[0]


def test_the_default_extrapolation_ladder_clears_the_span_gate_and_is_all_class_i():
    """`--extrapolate-nodes`'s default must survive `Extrapolation.check`, which no other test pinned.

    `check` is called by `score_survey` on every row, so a default that fails its span gate is not a
    crash but a silent demotion of every panel row from a fitted limit to a raw-marginal lower bound.
    Two properties, regression-prone in opposite directions: widening the span pushes the top rung's
    cost up, and narrowing it re-breaks the gate.
    """
    driver = load_driver("panel/score_survey.py")
    # through `parse_args` with the flag ABSENT, which is the only path that proves the DEFAULT is what
    # a real run gets -- reading the literal out of the source would pass even if the flag were rewired
    rungs = sorted(driver.parse_args(["--panel", "unused.json"]).extrapolate_nodes)

    span = rungs[-1] / rungs[0]
    assert span >= 64.0, f"the default ladder {rungs} spans {span:.1f}x, under check()'s min_span of 64"

    for requested in rungs:
        triangulation, n_psi = _choose_bank(requested)
        assert (10 * triangulation + 2) * n_psi == requested, (
            f"--nodes {requested} does not land on itself; the default is meant to be hit exactly"
        )
        assert math.isqrt(triangulation) ** 2 == triangulation, f"{requested} selects a non-Class-I bank"

    # and the gate refuses on nothing to do with span: asserted on the refusal MESSAGE rather than by
    # constructing a curve that passes outright, so the test stays about the span check
    grids = np.array(rungs, dtype=float)
    values = 1.0 - 2.0 * grids ** (-CROSSOVER_EXPONENT)
    try:
        extrapolate_in_k(grids, values, np.full_like(values, 0.001)).check(ceiling=1.0)
    except RefusedExtrapolation as refusal:
        assert "span" not in str(refusal), f"the default ladder is refused on span: {refusal}"
