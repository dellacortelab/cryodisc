"""`reproduction/experiments/forward/gate.py`: certify a template bank against a second engine, shell by shell.

Only the cheap half runs on a laptop -- a bank rendered, saved, reloaded and compared against
itself. That is the gate's own null: a verdict that cannot pass a bank against itself cannot pass
anything, and the save-and-reload is where the provenance would quietly be lost.
"""

import json

import numpy as np
import pytest
from conftest import load_driver


def test_the_forward_gate_certifies_a_bank_against_itself_and_records_the_provenance(separable_pair, tmp_path):
    """The cheap half of the gate, which is the half a laptop can run: a bank read off disk, a
    cryoJAX bank rendered to match it, and the per-shell verdict. Saving and reloading is where the
    provenance would be lost -- `None` does not survive an npz, so CTF_FREE round-trips as a NaN."""
    from reproduction.validation.template_bank import CTF_FREE, TemplateBank

    driver = load_driver("forward/gate.py")
    config = driver.config_from(driver.parse_args(["compare", "--structure", "x", "--out", "y", "--reference", "z"]))
    angles = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    reference = driver.cryojax_bank(separable_pair[0], angles, config)

    stored = driver.save_bank(reference, tmp_path / "ref.npz")
    reloaded = driver.load_bank(stored)
    assert reloaded.provenance() == reference.provenance()
    assert reloaded.engine == "cryojax"

    out = tmp_path / "gate.json"
    # a bank against itself must certify: this is the gate's own null, and one that failed here
    # would mean the verdict cannot pass anything
    assert (
        driver.main(["compare", "--structure", str(separable_pair[0]), "--reference", str(stored), "--out", str(out)])
        == 0
    )

    result = json.loads(out.read_text())["result"]
    assert result["passes"]
    assert result["median_ncc"] > 0.99
    assert result["engines"] == ["cryojax", "cryojax"]
    assert result["provenance"][3] is not None  # the defocus traveled with the bank
    del CTF_FREE, TemplateBank


def test_the_gate_refuses_a_reference_rendered_at_another_pixel_size(separable_pair, tmp_path):
    """The precondition the whole type exists for, reached through the driver rather than the API."""
    driver = load_driver("forward/gate.py")
    config = driver.config_from(driver.parse_args(["compare", "--structure", "x", "--out", "y", "--reference", "z"]))
    reference = driver.cryojax_bank(separable_pair[0], np.zeros((2, 3)), config)
    stored = driver.save_bank(reference, tmp_path / "ref.npz")

    argv = ["compare", "--structure", str(separable_pair[0]), "--reference", str(stored)]
    with pytest.raises(ValueError, match="rendered at"):
        driver.main([*argv, "--out", str(tmp_path / "g.json"), "--pixel-size", "3.0"])


def test_the_gate_needs_something_to_measure_against(tmp_path):
    driver = load_driver("forward/gate.py")
    with pytest.raises(SystemExit):
        driver.main(["compare", "--structure", "a.pdb", "--out", str(tmp_path / "g.json")])


def test_the_laptop_preset_is_overridden_field_by_field_and_not_wholesale():
    """`--laptop` is a preset for the cheap arm; `--defocus` states a condition being tested. A value
    stated on the command line must survive the preset rather than being discarded with it, so the
    defaults live in `_OPTICS` and `add_argument` declares `None`."""
    driver = load_driver("forward/gate.py")
    base = ["render", "--structure", "s.pdb", "--out", "o.npz"]

    preset = driver.config_from(driver.parse_args([*base, "--laptop"]))
    stated = driver.config_from(driver.parse_args([*base, "--laptop", "--defocus", "5000"]))
    plain = driver.config_from(driver.parse_args(base))

    # 5000 against the preset's own 12000, which is the pair the defocus series was run over
    assert stated.defocus_in_angstroms == 5000.0, "the stated value survives the preset"
    assert preset.defocus_in_angstroms == 12000.0, "and the preset is what an unstated one falls to"
    # everything not stated still comes from the preset, so this is an override and not a bypass
    assert stated.ice_thickness_in_angstroms == preset.ice_thickness_in_angstroms
    assert stated.box_size == preset.box_size
    # and without --laptop the documented defaults are what an unstated argument means
    assert (plain.defocus_in_angstroms, plain.voltage_in_kv) == (5000.0, 300.0)
