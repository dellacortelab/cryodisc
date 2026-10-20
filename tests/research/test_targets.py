"""Checks on the ranked panel.

The scoring itself is `rungs`' and is tested there. What is tested here is the bookkeeping that
decides what a row is allowed to CLAIM: which count is quotable, what an unmeasured row prints, and
that a rejected candidate is reported rather than dropped.
"""

import dataclasses
import json
import math
from pathlib import Path

import pandas
import pytest

from reproduction.targets import (
    Candidate,
    CandidateRejected,
    TargetRow,
    attach_marginal,
    format_table,
    load_panel,
    quotable_count,
    rank_targets,
    verdict,
)


def row(name="pair", n_floor=5000.0, d2_known_pose=2.0, **overrides):
    """A `TargetRow` with plausible fields, for the bookkeeping tests."""
    fields = {
        "name": name,
        "motion": "hinge",
        "mass_kda": 64.0,
        "ca_rmsd": 2.5,
        "radius": 40.0,
        "box_size": 128,
        "d2_known_pose": d2_known_pose,
        "d2_crlb": d2_known_pose * 0.98,
        "information": 0.4,
        "saturation": 0.1,
        "n_floor": n_floor,
        "n_marginal": None,
        "marginal_extrapolated": False,
        "pose_efficiency": None,
        "note": "",
    }
    return TargetRow(**{**fields, **overrides})


def test_a_row_with_no_rung_and_nothing_borrowed_is_unmeasured_rather_than_unreachable():
    """A row with no rung and nothing borrowed is unmeasured, not unreachable: calling it 'out of
    reach' would be a claim about the target, when it is just a row nobody measured."""
    count, source = quotable_count(row())
    assert count == math.inf
    assert source == "unmeasured"
    assert verdict(row()) == "unmeasured"
    assert "--" in format_table([row()])


def test_there_is_no_way_to_spend_a_borrowed_pose_cost():
    """A row with no measured rung has exactly one reading, so `pose_efficiency` must not be a
    spendable argument on `quotable_count`, `verdict` or `format_table`.

    Asserted by CALLING rather than by inspecting the signature: the way this regresses is a
    `**kwargs` that forwards the argument silently, and `inspect.signature` on `**kwargs` reports
    `{"kwargs"}`, so a name-absence check passes under exactly the regression it is written for."""
    for function in (quotable_count, verdict, format_table):
        with pytest.raises(TypeError, match="pose_efficiency"):
            function([row()] if function is format_table else row(), pose_efficiency=0.1)


def test_a_single_grid_rung_is_marked_as_an_upper_bound():
    """One grid reads the marginal rung low, so the count that follows is an UPPER bound, and that
    is a different defect from a borrowed constant with a different size."""
    measured = attach_marginal(row(d2_known_pose=2.0), 0.5, extrapolated=False)
    assert measured.pose_efficiency == pytest.approx(0.25)
    assert quotable_count(measured)[1] == "single_grid"
    assert "^" in format_table([measured])

    extrapolated = attach_marginal(row(d2_known_pose=2.0), 0.5, extrapolated=True)
    assert quotable_count(extrapolated)[1] == "measured"
    assert "^" not in format_table([extrapolated])


def test_a_marginal_rung_above_its_own_oracle_is_refused():
    """Marginalizing a pose cannot beat knowing it, so the two sides were built against different
    models. With no error bar the inequality is exact, which is right only for a model-only rung."""
    with pytest.raises(ValueError, match="exceeds"):
        attach_marginal(row(d2_known_pose=2.0), 3.0, extrapolated=True)
    with pytest.raises(ValueError, match="finite"):
        attach_marginal(row(), float("nan"), extrapolated=True)
    with pytest.raises(ValueError, match="non-negative"):
        attach_marginal(row(d2_known_pose=2.0), 1.0, d2_marginal_err=-0.1, extrapolated=True)


def test_a_zero_marginal_rung_is_admitted_with_an_infinite_count():
    """An indistinguishable pool reads a deflection of zero -- no measurable separation. That is a
    reading, not an error: the row is admitted with an infinite count rather than refused, so a survey
    does not lose its weakest pairs. A negative deflection cannot arise from the estimator and is
    refused as a bug rather than silently admitted."""
    admitted = attach_marginal(row(d2_known_pose=2.0), 0.0, extrapolated=True)
    assert not math.isfinite(admitted.n_marginal)
    with pytest.raises(ValueError, match="non-negative"):
        attach_marginal(row(d2_known_pose=2.0), -0.05, extrapolated=True)


def test_a_rung_at_the_oracle_is_admitted_within_its_own_scatter():
    """ON BANK the marginal rung IS the oracle, so which side it lands on is the pool's sampling
    noise and not a defect. A small overshoot within the pool's own scatter must be admitted, reading
    a pose efficiency just above one rather than a target that beat its own oracle."""
    admitted = attach_marginal(row(d2_known_pose=646.4), 653.5, d2_marginal_err=12.0, extrapolated=False)
    assert admitted.pose_efficiency == pytest.approx(653.5 / 646.4)
    assert admitted.n_marginal is not None

    # the same overshoot against a pool tight enough to resolve it is still refused
    with pytest.raises(ValueError, match="sigma"):
        attach_marginal(row(d2_known_pose=646.4), 653.5, d2_marginal_err=0.5, extrapolated=False)


def test_the_verdict_brackets_the_quotable_count_against_the_budget():
    """The three-way call, with each budget derived from the count the row actually carries rather
    than from a `d2` guessed to land in a band -- the mapping from `d2` to a count runs through
    `I_p` and is not something a test should be re-deriving by hand."""
    scored = attach_marginal(row(d2_known_pose=2.0), 0.5, extrapolated=True)
    count, source = quotable_count(scored)
    assert source == "measured"

    assert verdict(scored, budget=int(20 * count)) == "clear"  # count <= 0.1 * budget
    assert verdict(scored, budget=int(2 * count)) == "marginal"  # 0.1 * budget < count <= budget
    assert verdict(scored, budget=int(0.5 * count)) == "out of reach"


def test_a_smaller_marginal_rung_never_improves_the_verdict():
    """The monotonicity the table rests on: less signal cannot make a target easier. Checked across
    the whole range rather than at one point, since the count runs through a nonlinear `I_p`."""
    counts = [
        quotable_count(attach_marginal(row(d2_known_pose=2.0), d2, extrapolated=True))[0]
        for d2 in (2.0, 1.0, 0.2, 0.02)
    ]
    assert counts == sorted(counts)


def test_the_table_sorts_by_the_floor_and_reports_its_rejections(tmp_path):
    """A panel that silently omitted a failed pair would look like a table of everything tried."""
    from cryodisc.microscope import CollectionPlan

    missing = [
        Candidate("nonexistent a", tmp_path / "no_a.pdb", tmp_path / "no_b.pdb"),
        Candidate("nonexistent b", tmp_path / "no_c.pdb", tmp_path / "no_d.pdb"),
    ]
    rows, rejections = rank_targets(missing, CollectionPlan(dose=40.0, pixel_size=1.5))
    assert rows == []
    assert len(rejections) == 2
    # the order the candidates were given in, not whatever order the failures happened in
    assert rejections[0].startswith("nonexistent a")
    assert rejections[1].startswith("nonexistent b")


def scoring_stubs(monkeypatch, seen, *, minimum_box=8, rmsd=1.0, residues_b=("ALA", "GLY")):
    """Replace everything `score_target` renders with recorders, keeping only its plumbing.

    What is under test below is which PLAN reaches each consumer, which a real render cannot show: a
    moved grid comes back as a different `d2`, indistinguishable from ordinary numerical drift.
    """
    from types import SimpleNamespace

    import jax.numpy as jnp

    from reproduction import targets as module

    # aliased, because `rmsd = rmsd` in a CLASS body does not read the enclosing function's
    # parameter -- class bodies are not closures, so the name resolves to itself and raises
    alignment_rmsd = rmsd
    state_b_residues = list(residues_b)

    class FakeAlignment:
        rmsd = alignment_rmsd
        radius = 40.0
        # matches the two C-alpha the frames below carry, since `score_target` now RECORDS the
        # correspondence the guards read rather than only acting on it
        n_matched = 2
        shared_fraction = 1.0

        def minimum_box_size(self, pixel_size):
            seen["floor_pixel"] = pixel_size
            return minimum_box

    class FakePair:
        alignment = FakeAlignment()
        # a real correspondence, since `evaluate_candidate` checks it unconditionally: two matching
        # C-alpha frames satisfy the guard without stubbing it out
        state_a = pandas.DataFrame({"atom": ["CA", "CA"], "residue": ["ALA", "GLY"]})
        state_b = pandas.DataFrame({"atom": ["CA", "CA"], "residue": state_b_residues})

        # `**_` for the same reason `align` takes it: these tests are about which PLAN reaches each
        # consumer, and an ablation knob added to the real signature must not break them
        def models(self, plan, *, box_size, **_):
            seen["models_plan"] = plan
            return "model_a", "model_b"

    class FakeOptics:
        @staticmethod
        def from_plan(plan, defocus):
            seen["optics_plan"] = plan
            seen["optics_defocus"] = defocus
            return "optics"

    def fake_noise_spectrum(plan, box, *, rmsf):
        seen["spectrum_plan"] = plan
        return jnp.ones((box, box))

    # `**_` so an ablation knob added to `align` does not break the plumbing tests, which are about
    # which PLAN reaches each consumer and have no opinion about what the pair is made of
    monkeypatch.setattr(Candidate, "align", lambda self, **_: FakePair())
    monkeypatch.setattr(module, "noise_spectrum", fake_noise_spectrum)
    monkeypatch.setattr(module, "CTFParams", FakeOptics)
    monkeypatch.setattr(module, "_molecular_mass", lambda pair: 64.0)
    monkeypatch.setattr(module, "pose_bank", lambda *a: SimpleNamespace(angles=None, log_weights=None))
    monkeypatch.setattr(
        module,
        "local_information",
        # both rungs, since `score_target` reads `LocalRung.crlb` for `d2_crlb` as well as `known_pose`
        lambda *a, **k: SimpleNamespace(known_pose=jnp.array([2.0, 2.0]), crlb=jnp.array([1.95, 1.95])),
    )


def test_every_consumer_is_handed_the_panel_plan_untouched(monkeypatch):
    """One plan reaches the box floor, the render, the noise spectrum and the optics, and it is the
    caller's own object rather than a copy.

    `Candidate` carries no per-candidate `pixel_size` or `box_size` override: a row at its own
    sampling is not comparable with the rest of the panel. Asserted on the dataclass, because the way
    this regresses is a field added to `Candidate` that only one arm reads."""
    from cryodisc.microscope import CollectionPlan
    from reproduction.targets import score_target

    fields = {field.name for field in dataclasses.fields(Candidate)}
    assert not fields & {"pixel_size", "box_size"}

    seen = {}
    scoring_stubs(monkeypatch, seen)
    plan = CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(8000.0, 20000.0))

    score_target(Candidate("plain", "a.pdb", "b.pdb"), plan)
    assert seen["floor_pixel"] == pytest.approx(1.5)
    for key in ("models_plan", "spectrum_plan", "optics_plan"):
        assert seen[key] is plan
    # the optics come from the shared midpoint accessor, which `chain.run_chain` also scores through
    assert seen["optics_defocus"] == pytest.approx(14000.0)


def test_a_box_below_the_pairs_own_floor_is_rejected_before_anything_is_rendered(monkeypatch):
    """A clipped template is clipped identically in both states, so it shrinks `d^2` without moving
    any ratio -- nothing downstream can see it, and the box is the only place it is visible."""
    from cryodisc.microscope import CollectionPlan
    from reproduction.targets import score_target

    seen = {}
    scoring_stubs(monkeypatch, seen, minimum_box=256)
    with pytest.raises(CandidateRejected, match="below the 256"):
        score_target(Candidate("too small", "a.pdb", "b.pdb"), CollectionPlan(dose=40.0))
    assert "models_plan" not in seen


def test_a_slipped_correspondence_is_rejected_at_a_MODEST_rmsd(monkeypatch):
    """The guard must not ask the RMSD's permission before checking the correspondence.

    `_MAX_PLAUSIBLE_RMSD` is the wrong signal -- real motions reach large RMSDs -- so a correspondence
    that has slipped without inflating the RMSD must still be examined. Mis-numbered residues spanning
    similar-length stretches do exactly that, and Kabsch absorbs part of the offset. So: a modest RMSD
    no threshold would question, and zero matching residue types.
    """
    from cryodisc.microscope import CollectionPlan
    from reproduction.targets import score_target

    seen = {}
    scoring_stubs(monkeypatch, seen, rmsd=1.0, residues_b=("TRP", "PHE"))
    with pytest.raises(CandidateRejected, match="correspondence is wrong"):
        score_target(Candidate("slipped", "a.pdb", "b.pdb"), CollectionPlan(dose=40.0))
    # refused BEFORE rendering, the whole point of a guard
    assert "models_plan" not in seen


def test_a_rejection_carries_its_reason():
    """`CandidateRejected` is caught by name, so it has to be raised by name."""
    assert issubclass(CandidateRejected, Exception)
    error = CandidateRejected("pair: C-alpha RMSD 40.0 A exceeds 15 A")
    assert "RMSD" in str(error)


def test_a_count_dominated_by_the_binomial_floor_says_so_in_its_own_cell():
    """A row whose information has saturated is largely quoting the error target back, not measuring.

    `N` bottoms out at `p(1-p)/sigma^2` however discriminable the pair is, so a wide spread in the
    physics can print the same count; the cell must say so. Both directions, since a marker on every
    row would carry no information -- and the two rows below differ ONLY in the count, at identical
    saturation. The marker must read the printed COUNT and not `saturation`, which describes the floor
    rung: a saturated `I_p/cap` at a marginal count above the floor must not be marked as on it. The
    floor needs no sigma to recover: `n_floor * saturation` is `1/(ceiling sigma^2)`, exactly.
    """
    on_floor = row(n_floor=625.0, saturation=1.0, n_marginal=625.0, marginal_extrapolated=True)
    above_it = row(n_floor=625.0, saturation=1.0, n_marginal=761.0, marginal_extrapolated=True)

    assert "625=" in format_table([on_floor])
    assert "761=" not in format_table([above_it])
    assert "761" in format_table([above_it])
    # and the marker is about the COUNT, not about the pair: both rows saturate `I_p/cap`
    assert format_table([on_floor]).count("1.00*") == format_table([above_it]).count("1.00*") == 1


def test_a_written_row_still_carries_the_legacy_key_beside_the_new_one():
    """`d2_floor` is the legacy name in tracked result files and every reader indexes it BY NAME.

    Renaming the field to `d2_known_pose` was right -- it is the largest rung, not a floor -- but the
    written record cannot lose the old key: `reproduction/experiments/figures/marginalization_floor.py` filters
    rows on its presence and exits when none survives. Spelled as literal key names rather than looped
    over the implementation's own alias table, which would pass against a backwards mapping.
    """
    record = row(d2_known_pose=7.5).as_dict()

    assert record["d2_known_pose"] == 7.5
    assert record["d2_floor"] == 7.5
    # the new rung rides along in the same record, since a survey row is where it becomes quotable
    assert record["d2_crlb"] == pytest.approx(7.5 * 0.98)


def test_a_label_asym_id_row_demands_the_mmcif_and_loses_only_itself_without_one(tmp_path):
    """`FORMATS` prefers `.pdb`, and no legacy PDB file records a `label_asym_id`.

    So a label-chain row resolving to `.pdb` would die in `label_subchain_atoms`, and must reach for
    the mmCIF instead. A pdb-only label row is then REFUSED and not quietly dropped, matching
    `--fetch`'s contract: an absent file with fetching off means the tree is not built, and the
    operator has to hear it naming the path rather than lose the row silently.
    """
    for name in ("1AAA.pdb", "1BBB.pdb", "2AAA.cif", "2BBB.cif"):
        (tmp_path / name).write_text("")

    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "ligands": [],
                "candidates": [
                    {"name": "author row (1AAA/1BBB)", "pdb_a": "1AAA", "pdb_b": "1BBB"},
                    {
                        "name": "label row with a cif (2AAA/2BBB)",
                        "pdb_a": "2AAA",
                        "pdb_b": "2BBB",
                        "chain_id_kind": "label_asym_id",
                    },
                    {
                        "name": "label row with only a pdb (1AAA/1BBB)",
                        "pdb_a": "1AAA",
                        "pdb_b": "1BBB",
                        "chain_id_kind": "label_asym_id",
                    },
                ],
            }
        )
    )

    with pytest.raises(FileNotFoundError, match=r"1AAA\.cif"):
        load_panel(panel, tmp_path, fetch=False)

    # without that row the other two resolve, each to the format its naming requires
    rows = json.loads(panel.read_text())
    rows["candidates"] = rows["candidates"][:2]
    panel.write_text(json.dumps(rows))
    candidates, unresolved = load_panel(panel, tmp_path, fetch=False)

    assert not unresolved
    assert {candidate.name: Path(candidate.path_a).suffix for candidate in candidates} == {
        "author row (1AAA/1BBB)": ".pdb",
        "label row with a cif (2AAA/2BBB)": ".cif",
    }


def test_a_chain_id_kind_that_is_neither_naming_is_refused(tmp_path):
    """`_read_assembly` branches on `== "label_asym_id"` with an unguarded `else`, so a typo reads as
    author chains and builds a pair from the wrong subunit in silence. Panels are hand-edited."""
    (tmp_path / "1AAA.pdb").write_text("")
    (tmp_path / "1BBB.pdb").write_text("")
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "ligands": [],
                "candidates": [
                    {"name": "typo (1AAA/1BBB)", "pdb_a": "1AAA", "pdb_b": "1BBB", "chain_id_kind": "label_asym"}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="neither 'auth_asym_id' nor 'label_asym_id'"):
        load_panel(panel, tmp_path, fetch=False)
