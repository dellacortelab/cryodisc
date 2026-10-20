"""Checks on the rendered-bank type and the provenance it makes a precondition.

Most of these are about REFUSAL. The gate `compare_engines` feeds reports a level and a shape per
shell, and both statistics are perfectly happy to compare two banks that are of different things --
so everything that would make the comparison meaningless has to fail before it runs, and that is
what is checked here. The multislice engine itself is not exercised: it needs a GPU with MULTEM, so
`_shared.parakeet` is arranged to keep everything BUT the render testable on a laptop.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from reproduction.validation.bank import bank_verdict, compare_engines
from reproduction.validation.parakeet import (
    MultisliceConfig,
    atoms_only_copy,
    config_overrides,
    crop_templates,
    euler_to_rotvec,
    laptop_config,
    micrograph_layout,
    render_bank,
)
from reproduction.validation.template_bank import CTF_FREE, TemplateBank

BOX = 16


def make_bank(rng, *, n_poses=4, engine="cryojax", pixel_size=1.5, defocus=CTF_FREE, angles=None, scale=1.0):
    """A bank of random real templates with stated provenance."""
    return TemplateBank(
        images=jnp.asarray(scale * rng.normal(size=(n_poses, BOX, BOX))),
        angles=jnp.asarray(np.zeros((n_poses, 3)) if angles is None else angles),
        pixel_size=pixel_size,
        voltage_in_kv=300.0,
        defocus_in_angstroms=defocus,
        engine=engine,
    )


def test_a_malformed_bank_is_refused_at_construction_not_inside_a_rung():
    with pytest.raises(ValueError, match="square box"):
        TemplateBank(jnp.zeros((3, BOX, BOX + 1)), jnp.zeros((3, 3)), 1.0, 300.0, CTF_FREE, "cryojax")
    with pytest.raises(ValueError, match="empty"):
        TemplateBank(jnp.zeros((0, BOX, BOX)), jnp.zeros((0, 3)), 1.0, 300.0, CTF_FREE, "cryojax")
    with pytest.raises(ValueError, match=r"angles must be \(3, 3\)"):
        TemplateBank(jnp.zeros((3, BOX, BOX)), jnp.zeros((2, 3)), 1.0, 300.0, CTF_FREE, "cryojax")


def test_the_box_size_is_derived_so_it_cannot_disagree_with_the_images():
    bank = make_bank(np.random.default_rng(1))
    assert bank.box_size == BOX
    assert len(bank) == 4
    assert bank.fourier.shape == bank.images.shape


def test_ctf_free_is_not_a_defocus_of_zero():
    """Zero defocus is a real optical setting whose transfer was applied; CTF_FREE means it was
    never applied and one still has to be. Conflating them counts the CTF twice or never."""
    rng = np.random.default_rng(2)
    assert make_bank(rng, defocus=CTF_FREE).is_ctf_free
    assert not make_bank(rng, defocus=0.0).is_ctf_free
    with pytest.raises(ValueError, match="rendered at"):
        make_bank(rng, defocus=CTF_FREE).assert_comparable(make_bank(rng, defocus=0.0))


@pytest.mark.parametrize(
    ("field", "value"),
    [("pixel_size", 2.0), ("defocus", 8000.0)],
)
def test_banks_rendered_at_different_settings_refuse_to_be_compared(field, value):
    """A gate measures the engines only when everything else about the render is shared."""
    rng = np.random.default_rng(3)
    other = {"pixel_size": 1.5, "defocus": CTF_FREE} | {field: value}
    with pytest.raises(ValueError, match="rendered at"):
        make_bank(rng).assert_comparable(make_bank(rng, **other))


def test_banks_of_different_lengths_are_refused_before_anything_is_correlated():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="templates"):
        make_bank(rng, n_poses=4).assert_comparable(make_bank(rng, n_poses=6))


def test_a_pose_mismatch_is_caught_because_no_spectral_statistic_can_see_one():
    """The failure worth naming twice: two banks at different poses are of different things, and
    both the amplitude ratio and the FRC compare them shell by shell without complaint."""
    rng = np.random.default_rng(5)
    poses = np.tile(np.array([10.0, 20.0, 30.0]), (4, 1))
    drifted = poses + np.array([0.0, 0.0, 0.5])

    make_bank(rng, angles=poses).assert_comparable(make_bank(rng, angles=poses))
    with pytest.raises(ValueError, match="degrees in pose"):
        make_bank(rng, angles=poses).assert_comparable(make_bank(rng, angles=drifted))


def test_the_error_message_names_which_engine_is_which():
    """A gate that fails at three in the morning has to say which side was rendered wrong."""
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="parakeet"):
        make_bank(rng, engine="cryojax").assert_comparable(make_bank(rng, engine="parakeet", pixel_size=3.0))


def test_a_comparable_pair_reaches_the_gate_and_certifies_itself():
    """Provenance checked, then the existing statistics run unchanged."""
    rng = np.random.default_rng(7)
    images = rng.normal(size=(5, BOX, BOX))
    one = TemplateBank(jnp.asarray(images), jnp.zeros((5, 3)), 1.5, 300.0, CTF_FREE, "cryojax")
    two = TemplateBank(jnp.asarray(images), jnp.zeros((5, 3)), 1.5, 300.0, CTF_FREE, "parakeet")

    comparison = compare_engines(one, two)
    assert comparison.pixel_size == 1.5
    assert bank_verdict(comparison).passes


def test_the_gate_refuses_an_incomparable_pair_rather_than_reporting_on_it():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="rendered at"):
        compare_engines(make_bank(rng), make_bank(rng, pixel_size=3.0))


def test_the_multislice_configuration_refuses_a_slab_it_would_cross_in_one_step():
    """One step through the whole specimen IS a projection, so a gate at those settings would be
    comparing the projection model against itself and would pass for the wrong reason."""
    with pytest.raises(ValueError, match="which IS a projection"):
        MultisliceConfig(box_size=64, pixel_size=1.0, slice_thickness_in_angstroms=600.0)
    with pytest.raises(ValueError, match="must be positive"):
        MultisliceConfig(box_size=0, pixel_size=1.0)
    with pytest.raises(ValueError, match="dose must be positive"):
        MultisliceConfig(box_size=64, pixel_size=1.0, dose=0.0)


def test_the_laptop_configuration_is_noise_free_and_still_takes_more_than_one_slice():
    """It exists to exercise wiring in seconds, but a config that quietly became a projection would
    make the wiring test pass while testing nothing."""
    config = laptop_config()
    assert config.dose is None
    assert config.slice_thickness_in_angstroms < config.ice_thickness_in_angstroms


def test_the_multislice_poses_are_checked_before_the_gpu_dependency_is_touched():
    """A shape error must fail on a laptop, and it must fail with the SHAPE error -- checking the
    arguments after the import would hand a developer a message about MULTEM instead."""
    with pytest.raises(ValueError, match=r"angles must be \(p, 3\)"):
        render_bank("nowhere.pdb", np.zeros((4, 2)), laptop_config())


def test_a_well_formed_call_reaches_the_dependency_and_says_what_is_missing(tmp_path):
    """The other side of the same ordering: once the arguments are right, the failure is the honest
    one, and it names the extra rather than raising a bare ModuleNotFoundError.

    The structure is REAL here, because `render_bank` reads the file before it reaches the engine (it
    strips the assembly records so both engines see one molecule), so a missing file would raise
    FileNotFoundError first and this test would pass on the wrong exception."""
    structure = tmp_path / "tiny.pdb"
    structure.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    try:
        import parakeet  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[parakeet\]"):
            render_bank(structure, np.zeros((2, 3)), laptop_config())


@pytest.fixture(scope="module")
def rendered_model(tmp_path_factory):
    """A small cryoJAX model, so `from_model` is exercised against the real render path."""
    from conftest import ALPHA_CARBON, atom_records, write_pdb

    from cryodisc.forward.cryojax_model import CryoJAXModel

    coords = np.random.default_rng(0).uniform(-10.0, 10.0, (40, 3))
    path = write_pdb(tmp_path_factory.mktemp("bank") / "s.pdb", atom_records(coords, atoms=ALPHA_CARBON))
    return CryoJAXModel.from_structure(path, box_size=BOX, pixel_size=2.0)


def test_from_model_takes_its_provenance_from_the_render_rather_than_restating_it(rendered_model):
    """The box, pixel and voltage come off the model's own `image_config`, so a bank cannot claim a
    pixel size the templates were not rendered at -- which is the one error the gate cannot see."""
    angles = jnp.asarray([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    bank = TemplateBank.from_model(rendered_model, angles)

    assert bank.engine == "cryojax"
    assert bank.box_size == BOX
    assert bank.pixel_size == pytest.approx(2.0)
    assert bank.voltage_in_kv == pytest.approx(300.0)
    assert bank.is_ctf_free


def test_a_ctf_free_bank_reproduces_bare_contrast_exactly(rendered_model):
    """`fourier` is what the pool rungs consume, so it has to BE the bank they were built on -- the
    real-space round trip through `ifft2` must not move it or change its sign."""
    angles = jnp.asarray([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    bank = TemplateBank.from_model(rendered_model, angles)

    assert np.allclose(np.asarray(bank.fourier), np.asarray(rendered_model.bare_contrast(angles)), atol=1e-4)


def test_optics_without_a_recorded_defocus_is_refused(rendered_model):
    """A bank recording CTF_FREE while carrying a CTF has its transfer applied twice, and one
    recording a defocus while carrying none has it applied never. Neither raises downstream."""
    from cryodisc.forward.optics import CTFParams

    optics = CTFParams(defocus_in_angstroms=12000.0)
    angles = jnp.zeros((2, 3))

    with pytest.raises(ValueError, match="must be given together"):
        TemplateBank.from_model(rendered_model, angles, ctf_params=optics)
    with pytest.raises(ValueError, match="must be given together"):
        TemplateBank.from_model(rendered_model, angles, defocus_in_angstroms=12000.0)

    paired = TemplateBank.from_model(rendered_model, angles, ctf_params=optics, defocus_in_angstroms=12000.0)
    assert not paired.is_ctf_free


def test_the_two_render_paths_agree_about_the_sign(rendered_model):
    """The claim the module docstring makes, measured rather than asserted: a CTF-free bank times
    the transfer IS the bank rendered through that transfer. If the two paths disagreed in sign a
    Parakeet gate would read as a total forward-model failure, and a CTF-free bank fed to a rung
    would anti-correlate every template with its own data and still return a number.

    The envelope is turned off on purpose. `CTFParams.b_factor` defaults to 80 A^2 and
    `microscope.ctf` carries none, so a comparison at the default is close enough to look like
    agreement and far enough to hide a real defect.
    """
    from cryodisc.forward.optics import CTFParams
    from cryodisc.grids.fourier import frequency_vectors
    from cryodisc.microscope import CollectionPlan, ctf

    angles = jnp.asarray([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    defocus, pixel_size = 12000.0, 2.0
    free = TemplateBank.from_model(rendered_model, angles)
    shaped = TemplateBank.from_model(
        rendered_model,
        angles,
        ctf_params=CTFParams(defocus_in_angstroms=defocus, b_factor=0.0),
        defocus_in_angstroms=defocus,
    )

    ky, kx = frequency_vectors(BOX, pixel_size)
    plan = CollectionPlan(dose=40.0, pixel_size=pixel_size)
    transfer = ctf(jnp.sqrt(ky**2 + kx**2), defocus_in_angstroms=defocus, plan=plan)
    predicted = jnp.real(jnp.fft.ifft2(free.fourier * transfer))

    scale = float(jnp.max(jnp.abs(predicted)))
    assert float(jnp.max(jnp.abs(predicted - shaped.images))) < 1e-3 * scale


def test_the_micrograph_holds_every_instance_with_room_around_it():
    """A whole bank renders as one micrograph, so the layout is this module's problem rather than
    Parakeet's. What it must not do is let two instances overlap: a defocused particle's rings reach
    past its own box, and a neighbor's rings inside a cropped template read as a forward-model
    deficit in the gate this feeds."""
    config = MultisliceConfig(box_size=32, pixel_size=2.0)
    positions, micrograph_px = micrograph_layout(17, config)

    assert positions.shape == (17, 2)
    extent = micrograph_px * config.pixel_size
    half_box = 0.5 * config.box_size * config.pixel_size
    assert positions.min() - half_box > 0.0
    assert positions.max() + half_box < extent

    gaps = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min() >= config.box_size * config.pixel_size


def test_the_layout_refuses_an_empty_bank():
    with pytest.raises(ValueError, match="at least one pose"):
        micrograph_layout(0, MultisliceConfig(box_size=32, pixel_size=2.0))


def test_poses_convert_to_rotvecs_in_radians_not_degrees():
    """Degrees would rotate every template by a factor of 57 and still return a plausible bank."""
    rotvecs = euler_to_rotvec(np.array([[0.0, 0.0, 0.0], [0.0, 180.0, 0.0]]))

    assert np.allclose(rotvecs[0], 0.0)
    assert np.linalg.norm(rotvecs[1]) == pytest.approx(np.pi, abs=1e-6)


def test_the_config_negates_the_defocus_because_parakeets_c10_is_negative_for_underfocus():
    """Read off Parakeet 0.6.8, whose own default is c_10 = -20000. A repository this was ported
    from passes it POSITIVE, which is not a contradiction -- it goes through roodmus, which flips
    the sign. Writing a config directly means using Parakeet's convention, and backwards inverts
    every template while returning a bank that looks entirely reasonable."""
    config = MultisliceConfig(box_size=16, pixel_size=2.0, defocus_in_angstroms=9000.0)
    positions, micrograph_px = micrograph_layout(2, config)
    overrides = config_overrides("m.pdb", positions, np.zeros((2, 3)), micrograph_px, config)

    assert overrides["microscope"]["lens"]["c_10"] == pytest.approx(-9000.0)
    assert overrides["microscope"]["lens"]["c_30"] == pytest.approx(2.7)
    # keV, so 300 rather than 0.3 -- a factor of a thousand that changes every wavelength
    assert overrides["microscope"]["beam"]["energy"] == pytest.approx(300.0)
    assert overrides["microscope"]["detector"]["pixel_size"] == pytest.approx(2.0)


def test_the_solvent_goes_in_the_simulation_block_and_not_the_sample_one():
    """`sample.ice` is the SLOW atomic model, whose fields are `generate` and `density`;
    `simulation.ice` is the fast amorphous one a gate wants. A made-up key in either block leaves
    the default standing rather than raising."""
    positions, micrograph_px = micrograph_layout(1, MultisliceConfig(box_size=16, pixel_size=2.0))
    for solvent in (True, False):
        config = MultisliceConfig(box_size=16, pixel_size=2.0, solvent=solvent)
        overrides = config_overrides("m.pdb", positions, np.zeros((1, 3)), micrograph_px, config)
        assert overrides["simulation"]["ice"] is solvent
        assert overrides["sample"]["ice"] is None


def test_every_pose_becomes_one_instance_with_its_own_orientation():
    config = MultisliceConfig(box_size=16, pixel_size=2.0)
    positions, micrograph_px = micrograph_layout(3, config)
    rotvecs = euler_to_rotvec(np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))
    overrides = config_overrides("m.pdb", positions, rotvecs, micrograph_px, config)

    instances = overrides["sample"]["molecules"]["local"][0]["instances"]
    assert len(instances) == 3
    assert overrides["sample"]["molecules"]["local"][0]["filename"] == "m.pdb"
    assert np.allclose([i["orientation"] for i in instances], rotvecs)


def test_the_crop_recovers_each_instance_from_the_shared_micrograph():
    """The other half of the layout: what goes in at a position has to come back out of it."""
    config = MultisliceConfig(box_size=8, pixel_size=2.0)
    positions, micrograph_px = micrograph_layout(4, config)
    micrograph = np.zeros((micrograph_px, micrograph_px), np.float32)
    for i, (x, y) in enumerate(positions):
        micrograph[round(y / config.pixel_size), round(x / config.pixel_size)] = i + 1.0

    templates = crop_templates(micrograph, positions, config)
    assert templates.shape == (4, 8, 8)
    assert [float(t[4, 4]) for t in templates] == [1.0, 2.0, 3.0, 4.0]


def test_a_crop_that_runs_off_the_micrograph_is_refused():
    """The layout and the render disagreeing about the size makes every template past that point
    garbage, and a silently truncated patch would be scored as a real one."""
    config = MultisliceConfig(box_size=8, pixel_size=2.0)
    positions, _ = micrograph_layout(4, config)
    with pytest.raises(ValueError, match="does not fit"):
        crop_templates(np.zeros((16, 16), np.float32), positions, config)


def test_an_odd_box_crops_to_the_size_it_asked_for():
    """`center +- half` is one pixel short per axis on an odd box, and the shape check downstream
    reports that as the layout and the render disagreeing -- after the GPU time has been spent."""
    config = MultisliceConfig(box_size=9, pixel_size=2.0)
    positions, micrograph_px = micrograph_layout(3, config)
    micrograph = np.zeros((micrograph_px, micrograph_px), np.float32)
    for i, (x, y) in enumerate(positions):
        micrograph[round(y / config.pixel_size), round(x / config.pixel_size)] = i + 1.0

    templates = crop_templates(micrograph, positions, config)
    assert templates.shape == (3, 9, 9)
    assert [float(t[4, 4]) for t in templates] == [1.0, 2.0, 3.0]


def test_a_float32_round_trip_of_the_pixel_size_does_not_refuse_a_matching_pair():
    """`from_model` reads its settings back out of a float32 config while `render_bank` keeps the
    Python float, so exact equality refused a correctly-configured gate at any pixel size that is
    not float32-exact -- which the defaults in use happen to be, hiding it."""
    rng = np.random.default_rng(11)
    exact = 1.31
    rounded = float(np.float32(exact))
    assert rounded != exact
    make_bank(rng, pixel_size=exact).assert_comparable(make_bank(rng, pixel_size=rounded, engine="parakeet"))


def test_a_real_difference_in_the_pixel_size_is_still_refused():
    """The tolerance is for a float32 round trip and nothing wider: 1.5 against 1.31 is two
    different renders and the gate would report the difference as a forward-model deficit."""
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="rendered at"):
        make_bank(rng, pixel_size=1.5).assert_comparable(make_bank(rng, pixel_size=1.31))


def test_the_rotvec_is_cryojaxs_intrinsic_convention_and_not_the_extrinsic_one():
    """The conversion that was wrong, pinned against the convention on the other side of the gate.

    `euler_to_rotvec` was reading its input as EXTRINSIC ZYZ while `cryojax_model` documents and
    passes intrinsic zyz, which swaps phi and psi. The bank came back well formed and at the wrong
    poses, and the gate reported that as a forward-model deficit. The old test pinned only radians
    against degrees, which both readings satisfy, so it could not see this at all.
    """
    from scipy.spatial.transform import Rotation

    # asymmetric on purpose: the two conventions AGREE whenever phi equals psi
    angles = np.array([[30.0, 50.0, 70.0], [12.0, 100.0, -45.0]])
    produced = Rotation.from_rotvec(euler_to_rotvec(angles))
    intrinsic = Rotation.from_euler("zyz", angles, degrees=True)
    extrinsic = Rotation.from_euler("ZYZ", angles, degrees=True)

    assert float(np.max((produced * intrinsic.inv()).magnitude())) < 1e-9
    # and the two really are different here, so the assertion above has something to fail on
    assert float(np.min((intrinsic * extrinsic.inv()).magnitude())) > 0.1


def test_only_the_atoms_are_handed_to_the_render_so_both_engines_see_one_molecule(tmp_path):
    """Parakeet expands REMARK 350 and cryoJAX does not, which puts a different molecule through each
    engine -- a deposited dimer becomes the assembly's tetramer under one engine only. The gate then
    reads that as a forward-model deficit, and the assembly's symmetry can render distinct poses
    identically, which looks like the in-plane angle being lost and is not. The atoms must be stripped
    so both engines see one molecule.
    """
    deposited = tmp_path / "deposited.pdb"
    deposited.write_text(
        "HEADER    TEST\n"
        "REMARK 350 BIOMOLECULE: 1\n"
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
        "REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000        0.00000\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C\n"
        "TER\nEND\n"
    )
    stripped = atoms_only_copy(deposited, tmp_path / "atoms.pdb")
    text = stripped.read_text()

    assert "REMARK 350" not in text, "the assembly records are what Parakeet expands on"
    assert "HEADER" not in text
    # and the atoms themselves survive untouched, coordinates included
    assert text.count("\nATOM  ") + text.startswith("ATOM  ") == 2
    assert "1.000   2.000   3.000" in text
    assert "4.000   5.000   6.000" in text
