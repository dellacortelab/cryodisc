"""What `conformers` has to get right for a difference map to mean anything.

The structures are synthetic, so the tests need no network, and each is built to make one failure
mode visible: a frame the superposition did not remove, an assembly left unexpanded, a composition
difference surviving the restriction, or a center applied per state instead of per pair. Those all
produce a plausible number rather than an error, which is why they are pinned here.
"""

import gemmi
import numpy as np
import pytest
from conftest import BACKBONE, atom_records, write_pdb
from scipy.spatial.transform import Rotation

from cryodisc.forward.cryojax_model import CryoJAXModel, pad_to_common_atom_count
from cryodisc.microscope import CollectionPlan
from cryodisc.specimen.structures import _read_assembly, align_conformers, label_subchain_atoms

N_RESIDUES = 40


def _biomt(operators):
    """A REMARK 350 record applying `operators` (each a `(rotation, translation)`) to chain A."""
    lines = ["REMARK 350 BIOMOLECULE: 1", "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A"]
    for index, (rotation, translation) in enumerate(operators, start=1):
        for row in range(3):
            values = "".join(f"{value:10.6f}" for value in rotation[row])
            lines.append(f"REMARK 350   BIOMT{row + 1}{index:4d}{values}{translation[row]:15.5f}")
    return lines


@pytest.fixture(scope="module")
def coords():
    """A compact blob of backbone atoms, well inside any box these tests render into."""
    return np.random.default_rng(0).uniform(-15.0, 15.0, (N_RESIDUES * len(BACKBONE), 3))


@pytest.fixture(scope="module")
def displaced(tmp_path_factory, coords):
    """One structure, and the same structure rigidly moved: the difference is entirely frame."""
    directory = tmp_path_factory.mktemp("displaced")
    rotation = Rotation.from_euler("zyz", [37.0, 61.0, -22.0], degrees=True).as_matrix()
    moved = coords @ rotation.T + np.array([120.0, -45.0, 9.0])
    return (
        write_pdb(directory / "a.pdb", atom_records(coords)),
        write_pdb(directory / "b.pdb", atom_records(moved)),
    )


def test_a_rigid_displacement_does_not_survive(displaced):
    """The whole reason the module exists: two frames differ, the conformation does not.

    The floor is the PDB's own 0.001 A coordinate field, which the displacement then rotates.
    """
    pair = align_conformers(*displaced, assembly=None)
    assert pair.alignment.rmsd == pytest.approx(0.0, abs=2e-3)
    assert pair.alignment.n_matched == N_RESIDUES
    np.testing.assert_allclose(pair.state_a[["x", "y", "z"]], pair.state_b[["x", "y", "z"]], atol=5e-3)


def _center_of_mass(state):
    weight = state["atomic_weight"].to_numpy()
    return state[["x", "y", "z"]].to_numpy().T @ weight / weight.sum()


def test_the_pair_is_centered_once_not_twice(tmp_path, coords):
    """The center is the pair's, so a shift between the states' centers of mass reaches the templates.

    Only the carbonyl oxygens move here, so the C-alpha superposition is the identity and the whole
    difference is carried by where the mass sits. Centering each state on its own center of mass --
    which is what cryoJAX does unless told not to -- would delete exactly that.
    """
    shifted = coords.copy()
    shifted[3 :: len(BACKBONE)] += np.array([0.0, 0.0, 6.0])
    pair = align_conformers(
        write_pdb(tmp_path / "a.pdb", atom_records(coords)),
        write_pdb(tmp_path / "b.pdb", atom_records(shifted)),
        assembly=None,
    )
    assert pair.alignment.rmsd == pytest.approx(0.0, abs=2e-3)
    np.testing.assert_allclose(_center_of_mass(pair.state_a) + _center_of_mass(pair.state_b), 0.0, atol=1e-6)
    assert np.linalg.norm(_center_of_mass(pair.state_a)) > 0.5


def test_the_assembly_is_built_not_the_deposited_chain(tmp_path, coords):
    """A deposited half-molecule is expanded by its own operators before anything is compared."""
    flip = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
    operators = [(np.eye(3), np.zeros(3)), (flip, np.array([0.0, 0.0, 40.0]))]
    paths = [
        write_pdb(tmp_path / f"{name}.pdb", _biomt(operators) + atom_records(atoms))
        for name, atoms in (("a", coords), ("b", coords + np.array([0.0, 0.0, 0.3])))
    ]
    deposited = align_conformers(*paths, assembly=None)
    assembled = align_conformers(*paths, assembly="1")
    assert assembled.alignment.n_atoms == 2 * deposited.alignment.n_atoms
    assert assembled.alignment.n_matched == 2 * deposited.alignment.n_matched


def _generated_against_deposited(tmp_path, coords, shift=0.3):
    """The 2DN1/2DN2 shape: one entry GENERATES the dimer by a two-fold, the other deposits both chains.

    Returns `(generated_path, deposited_path)`. The generator is an exact 180 degree rotation with no
    translation, as 2DN1's assembly 1 is, so the two labelings inside its orbit are equivalent.
    """
    flip = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
    generated = write_pdb(
        tmp_path / "generated.pdb",
        _biomt([(np.eye(3), np.zeros(3)), (flip, np.zeros(3))]) + atom_records(coords),
    )
    moved = coords + np.array([0.0, 0.0, shift])
    deposited = write_pdb(
        tmp_path / "deposited.pdb",
        atom_records(moved, chain="A") + atom_records(moved @ flip.T, chain="B", start_serial=len(moved) + 1),
    )
    return generated, deposited


def test_a_generated_assembly_cannot_be_matched_to_a_deposited_one_without_a_label_map(tmp_path, coords):
    """The defect that withdrew the only independent check on the primary haemoglobin row.

    A generator does not rename what it produces, so expanding one chain gives `(0, A)` and `(1, A)`
    while the entry that deposits the same dimer has `(0, A)` and `(0, B)`. Half the keys match, which
    is a correspondence failure and is refused -- rightly, since nothing in either file says which
    generated copy is which deposited chain.
    """
    generated, deposited = _generated_against_deposited(tmp_path, coords)

    with pytest.raises(ValueError, match="share only"):
        align_conformers(generated, deposited, assembly_a="1", assembly_b=None)


def test_a_label_map_matches_the_generated_copies_onto_the_deposited_chains(tmp_path, coords):
    """And with the map, the whole dimer is compared instead of half of it.

    The point is the ATOM COUNT: falling back to one chain each also aligns, and reads a smaller motion
    off a molecule that is missing the subunit the motion is relative to.
    """
    generated, deposited = _generated_against_deposited(tmp_path, coords)

    whole = align_conformers(generated, deposited, assembly_a="1", assembly_b=None, labels_a=("A", "B"))
    # the fallback: both entries cut to the one chain they share unexpanded, which is the comparison
    # panel.json had to settle for and which keeps only a fraction of a quaternary motion
    half = align_conformers(generated, deposited, assembly=None, chains=("A",))

    assert whole.alignment.n_atoms == 2 * half.alignment.n_atoms
    assert whole.alignment.n_matched == 2 * N_RESIDUES
    assert whole.alignment.n_dropped == (0, 0)


def test_relabeling_within_the_generators_orbit_cannot_change_the_answer(tmp_path, coords):
    """A proof, not a tolerance: copy 1 is exactly `R` copy 0, so swapping which copy is called `A`
    composes the entry with `R`, and the superposition fitted afterwards absorbs exactly that.

    It matters because it says the remaining freedom in `labels_a` carries no risk. A labeling outside
    the orbit is not free, but the RMSD catches that; there is no such labeling on a two-chain dimer,
    so this pins only the invariance.
    """
    generated, deposited = _generated_against_deposited(tmp_path, coords)

    forward = align_conformers(generated, deposited, assembly_a="1", assembly_b=None, labels_a=("A", "B"))
    swapped = align_conformers(generated, deposited, assembly_a="1", assembly_b=None, labels_a=("B", "A"))

    assert swapped.alignment.rmsd == pytest.approx(forward.alignment.rmsd, rel=1e-9)
    assert swapped.alignment.n_matched == forward.alignment.n_matched


def test_a_shared_setting_and_a_per_entry_one_are_refused_together(tmp_path, coords):
    """Which was meant to win is a guess, and the two answers differ in what molecule gets built."""
    generated, deposited = _generated_against_deposited(tmp_path, coords)

    with pytest.raises(ValueError, match="not both"):
        align_conformers(generated, deposited, assembly="1", assembly_b=None)


def test_a_label_map_that_does_not_cover_the_copies_is_refused(tmp_path, coords):
    """A short list would silently drop a subunit, which reads as a smaller molecule and not as an
    error."""
    generated, deposited = _generated_against_deposited(tmp_path, coords)

    with pytest.raises(ValueError, match="One label per pair"):
        align_conformers(generated, deposited, assembly_a="1", assembly_b=None, labels_a=("A",))


def test_hydrogens_one_entry_models_and_the_other_does_not_are_dropped_from_both(tmp_path, coords):
    """Whether a deposition models its hydrogens is a property of the method, and when two entries
    disagree it wrecks the correspondence guard unless the hydrogens are dropped from both.

    A pair whose heavy atoms correspond can be pushed below the guard's threshold by one entry's
    hydrogens alone; dropping them lets the pair align on its heavy atoms.
    """
    hydrogenated = atom_records(coords, atoms=BACKBONE)
    # one hydrogen per heavy atom, so they are half the entry -- a lighter sprinkling leaves the
    # guard's denominator inside its threshold and proves nothing
    with_protons = hydrogenated + atom_records(coords + 0.4, atoms=(("H", "H"),), start_serial=len(coords) + 1)
    paths = (
        write_pdb(tmp_path / "nmr.pdb", with_protons),
        write_pdb(tmp_path / "xray.pdb", hydrogenated),
    )

    dropped = align_conformers(*paths, assembly=None)
    assert dropped.alignment.n_atoms == len(coords)
    assert dropped.alignment.n_dropped == (0, 0)
    # and asking for them back puts the entries back out of correspondence, which is the real state
    # of affairs rather than a limitation: one entry has atoms the other never modeled
    with pytest.raises(ValueError, match="share only"):
        align_conformers(*paths, assembly=None, hydrogens=True)


def test_an_mmcif_assembly_is_built_the_same_as_a_pdb_one(tmp_path, coords):
    """The two formats name a generator's parts differently, and only one of them is author chains.

    mmCIF lists label subchains -- a chain's polymer and its waters are separate ones -- so reading
    only `generator.chains` selects nothing and the assembly comes back empty. gemmi's own conversion
    reproduces that exactly, which is what this goes through rather than a downloaded entry.
    """
    flip = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
    lines = _biomt([(np.eye(3), np.zeros(3)), (flip, np.array([0.0, 0.0, 40.0]))])
    pairs = []
    for suffix in (".pdb", ".cif"):
        paths = []
        for name, atoms in (("a", coords), ("b", coords + np.array([0.0, 0.0, 0.3]))):
            path = write_pdb(tmp_path / f"{name}.pdb", lines + atom_records(atoms))
            if suffix == ".cif":
                structure = gemmi.read_structure(str(path))
                structure.setup_entities()
                path = tmp_path / f"{name}.cif"
                structure.make_mmcif_document().write_file(str(path))
            paths.append(path)
        pairs.append(align_conformers(*paths, assembly="1"))

    from_pdb, from_cif = pairs
    assert from_cif.alignment.n_atoms == from_pdb.alignment.n_atoms == 2 * len(coords)
    assert from_cif.alignment.rmsd == pytest.approx(from_pdb.alignment.rmsd, abs=1e-6)


def test_a_missing_assembly_raises_rather_than_falling_back(tmp_path, coords):
    paths = [write_pdb(tmp_path / f"{name}.pdb", atom_records(coords)) for name in ("a", "b")]
    with pytest.raises(ValueError, match="no assembly"):
        align_conformers(*paths, assembly="7")


def test_unresolved_atoms_leave_the_two_states_with_equal_composition(tmp_path, coords):
    """Atoms one entry resolved and the other did not are not motion, and are dropped from both."""
    pair = align_conformers(
        write_pdb(tmp_path / "a.pdb", atom_records(coords)),
        write_pdb(tmp_path / "b.pdb", atom_records(coords)[:-12]),
        assembly=None,
    )
    assert pair.alignment.n_dropped == (12, 0)
    assert pair.alignment.n_atoms == len(coords) - 12
    np.testing.assert_array_equal(pair.state_a["atomic_number"], pair.state_b["atomic_number"])
    assert pair.state_a["atomic_weight"].sum() == pytest.approx(pair.state_b["atomic_weight"].sum())


def test_too_little_overlap_raises_instead_of_superposing(tmp_path, coords):
    """Half an atom list matching is the assembly-against-asymmetric-unit trap, not a comparison."""
    with pytest.raises(ValueError, match="do not express the same molecule"):
        align_conformers(
            write_pdb(tmp_path / "a.pdb", atom_records(coords)),
            write_pdb(tmp_path / "b.pdb", atom_records(coords)[: len(coords) // 2]),
            assembly=None,
        )


def test_a_ligand_enters_only_when_it_is_named(tmp_path, coords):
    """A ligand bound in one state and not the other is an occupancy difference, not conformation."""
    het = [
        f"HETATM{999:5d}  FE  HEM A 900    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          FE",
    ]
    paths = [write_pdb(tmp_path / f"{name}.pdb", atom_records(coords) + het) for name in ("a", "b")]
    assert align_conformers(*paths, assembly=None).alignment.n_atoms == len(coords)
    assert align_conformers(*paths, assembly=None, ligands=("HEM",)).alignment.n_atoms == len(coords) + 1


def _hetatm(serial, atom, residue, position, element, chain="A", residue_id=202):
    """One HETATM line, in the column layout `test_a_ligand_enters_only_when_it_is_named` uses."""
    x, y, z = position
    return (
        f"HETATM{serial:5d}  {atom:<4s}{residue:>3s} {chain}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _ligated(tmp_path, coords, bound_a, bound_b):
    """Two entries of one molecule carrying different HETATM at the SAME numbered site.

    The haemoglobin shape, and the numbering is theirs: 8WIX deposits `CMO 202` in chain B where 8WIZ
    deposits `BCT 202`, and 8WJ2 deposits no fourth ligand at all.

    Args:
        tmp_path: Where the two files go.
        coords: The backbone both entries share.
        bound_a: `(residue, [(atom, element), ...])` for state A, or `None` for nothing bound.
        bound_b: The same for state B.

    Returns:
        `(path_a, path_b)`.
    """
    paths = []
    for name, bound in (("a", bound_a), ("b", bound_b)):
        records = atom_records(coords)
        if bound is not None:
            residue, atoms = bound
            records = records + [
                _hetatm(len(coords) + index, atom, residue, (0.0, 0.0, float(index)), element)
                for index, (atom, element) in enumerate(atoms, start=1)
            ]
        paths.append(write_pdb(tmp_path / f"{name}.pdb", records))
    return tuple(paths)


def test_a_ligand_one_state_binds_and_the_other_does_not_is_kept_by_the_state_that_has_it(tmp_path, coords):
    """A ligand present in one state and absent in the other IS the difference between two ligation
    states, and naming it in `ligands` is how a caller says so.

    A one-sided allowlisted ligand must reach one state's template and not the other's. The trap is an
    intersection that discards it from both because the partner state has no matching residue, leaving
    the templates with identical density; the last assertion catches that.
    """
    paths = _ligated(tmp_path, coords, bound_a=("CMO", (("C", "C"), ("O", "O"))), bound_b=None)

    unnamed = align_conformers(*paths, assembly=None)
    assert unnamed.alignment.n_state_specific == (0, 0)
    assert len(unnamed.state_a) == len(unnamed.state_b) == len(coords)

    named = align_conformers(*paths, assembly=None, ligands=("CMO",))
    assert named.alignment.n_atoms == len(coords)
    assert named.alignment.n_state_specific == (2, 0)
    # not DROPPED: the whole point is that these atoms are neither corresponded nor discarded
    assert named.alignment.n_dropped == (0, 0)
    assert (len(named.state_a), len(named.state_b)) == (len(coords) + 2, len(coords))

    model_a, model_b = named.models(CollectionPlan(dose=40.0), box_size=96)
    assert float(np.asarray(model_a.forward(0.0, 0.0, 0.0)).sum()) != pytest.approx(
        float(np.asarray(model_b.forward(0.0, 0.0, 0.0)).sum()), rel=1e-4
    )


def test_the_exemption_cannot_move_the_frame_it_is_placed_by(tmp_path, coords):
    """The superposition, the pair's center and the RMSD are taken over the CORRESPONDING atoms only.

    Otherwise allowlisting a one-sided ligand would pull the shared center toward the state that has
    it, so turning the ligation state on would displace both templates against the particles -- which
    reads as a weaker measurement rather than as a placement error.
    """
    paths = _ligated(tmp_path, coords, bound_a=("CMO", (("C", "C"), ("O", "O"))), bound_b=None)

    bare = align_conformers(*paths, assembly=None)
    named = align_conformers(*paths, assembly=None, ligands=("CMO",))

    assert named.alignment.rmsd == pytest.approx(bare.alignment.rmsd, rel=1e-9)
    corresponding = slice(0, named.alignment.n_atoms)
    for state in ("state_a", "state_b"):
        np.testing.assert_allclose(
            getattr(named, state)[["x", "y", "z"]].to_numpy()[corresponding],
            getattr(bare, state)[["x", "y", "z"]].to_numpy(),
            atol=1e-9,
        )


def test_a_one_sided_ligand_still_leaves_two_models_that_can_be_stacked(tmp_path, coords):
    """`particles.sim` stacks the two volumes and gathers per particle, which keeps a pool's cost flat
    in the number of states, and it cannot when the two atom tables differ in length.

    `models` pads with zero-amplitude atoms so a one-sided pair still stacks, and the second assertion
    is why that is a fix rather than a compromise: the padded render is exactly the unpadded one.
    """
    paths = _ligated(tmp_path, coords, bound_a=("CMO", (("C", "C"), ("O", "O"))), bound_b=None)
    pair = align_conformers(*paths, assembly=None, ligands=("CMO",))
    assert len(pair.state_a) != len(pair.state_b)

    model_a, model_b = pair.models(CollectionPlan(dose=40.0), box_size=96)
    assert model_a.volume.positions.shape == model_b.volume.positions.shape

    # on the plane `models` renders on, so this isolates the padding rather than also comparing a
    # cropped projection against a full-box one
    unpadded = CryoJAXModel.from_structure(
        pair.state_b,
        box_size=96,
        pixel_size=1.0,
        center=False,
        projection_shape=pair.alignment.projection_box_size(1.0, 96),
    )
    np.testing.assert_array_equal(
        np.asarray(model_b.forward(20.0, 80.0, -10.0)), np.asarray(unpadded.forward(20.0, 80.0, -10.0))
    )

    # a voxel map has no atom axis, so returning it unchanged would leave the caller's stack to fail
    # later and somewhere else
    voxels = CryoJAXModel.from_voxel_grid(np.zeros((16, 16, 16), np.float32), pixel_size=1.0)
    with pytest.raises(TypeError, match="no atom axis"):
        pad_to_common_atom_count((voxels, model_a))


def test_rendering_on_a_cropped_plane_reproduces_the_full_box(displaced):
    """`models` evaluates the erf rasterization only where the molecule is, and that has to be a
    speedup and not an approximation.

    A production box holds the CTF's delocalization as well as the density, so most of what the
    renderer evaluates is empty -- haemoglobin at box 128 and 2.0 A/px needs a plane of 44. What
    makes the crop safe is the margin: the Gaussian tails run past the furthest atom, and at zero
    margin haemoglobin's mass moves 0.13%. Pinned against the uncropped render rather than against a
    stored number, because the claim is equality and not a tolerance anyone chose.
    """
    pair = align_conformers(*displaced, assembly=None)
    plane = pair.alignment.projection_box_size(1.0, 96)
    assert plane < 96, "a blob well inside its box has to crop, or the test proves nothing"

    cropped, _ = pair.models_on_grid(box_size=96, pixel_size=1.0)
    full = CryoJAXModel.from_structure(pair.state_a, box_size=96, pixel_size=1.0, center=False)
    got, exact = (np.asarray(model.forward(20.0, 80.0, -10.0)) for model in (cropped, full))
    assert np.abs(got - exact).max() / np.abs(exact).max() < 1e-5
    assert abs(float(got.sum() / exact.sum()) - 1.0) < 1e-4


def test_a_molecule_that_fills_its_box_is_not_cropped(displaced):
    """The plane follows the image box's parity and never exceeds it.

    Unlike parity would center the molecule half a pixel out, which is the failure that reads as a
    slightly worse rung rather than as an error. A box too small for the margin returns the box.
    """
    alignment = align_conformers(*displaced, assembly=None).alignment
    assert alignment.projection_box_size(1.0, 8) == 8
    for box in (48, 49):
        assert alignment.projection_box_size(4.0, box) % 2 == box % 2


def test_two_different_ligands_at_one_residue_number_are_not_a_correspondence(tmp_path, coords):
    """The key carries the residue NUMBER, so two different ligands bound at the same site match on it.

    Two different ligands at one site put same-element atoms under one key and would be corresponded
    silently -- the atom count merely rises. Requiring the residue name to agree separates them, and
    only for an allowlisted residue: a polymer position that mutated keeps its atom names, and cutting
    those would change every row already measured.
    """
    paths = _ligated(
        tmp_path,
        coords,
        bound_a=("CMO", (("C", "C"), ("O", "O"))),
        bound_b=("BCT", (("C", "C"), ("O1", "O"), ("O2", "O"), ("O3", "O"))),
    )

    pair = align_conformers(*paths, assembly=None, ligands=("CMO", "BCT"))

    assert pair.alignment.n_atoms == len(coords)
    assert pair.alignment.n_state_specific == (2, 4)
    assert pair.alignment.n_dropped == (0, 0)


def test_an_alternate_conformation_is_counted_once(tmp_path, coords):
    """cryoJAX weighs every row fully, so a second altloc would place the atom twice."""
    altloc = atom_records(coords[:4], altloc="B", start_serial=len(coords) + 1, occupancy=0.5)
    paths = [
        write_pdb(tmp_path / "a.pdb", atom_records(coords, altloc="A", occupancy=0.5) + altloc),
        write_pdb(tmp_path / "b.pdb", atom_records(coords)),
    ]
    assert align_conformers(*paths, assembly=None).alignment.n_atoms == len(coords)


def test_templates_carry_equal_mass_and_an_unequal_image(tmp_path, coords):
    """The end of the chain: identical composition, and a difference that is only conformation."""
    bent = coords.copy()
    bent[len(bent) // 2 :] += np.array([0.0, 4.0, 0.0])
    pair = align_conformers(
        write_pdb(tmp_path / "a.pdb", atom_records(coords)),
        write_pdb(tmp_path / "b.pdb", atom_records(bent)),
        assembly=None,
    )
    model_a, model_b = pair.models(CollectionPlan(dose=40.0), box_size=96)
    image_a = np.asarray(model_a.forward(0.0, 0.0, 0.0))
    image_b = np.asarray(model_b.forward(0.0, 0.0, 0.0))
    assert image_a.sum() == pytest.approx(image_b.sum(), rel=1e-5)
    assert np.linalg.norm(image_a - image_b) / np.linalg.norm(image_a) > 0.05


def test_minimum_box_size_holds_the_molecule(displaced):
    alignment = align_conformers(*displaced, assembly=None).alignment
    assert alignment.minimum_box_size(pixel_size=1.0) >= 2 * alignment.radius
    assert alignment.minimum_box_size(pixel_size=2.0) * 2 >= 2 * alignment.radius


def test_an_offset_places_the_pairs_center_where_it_is_asked_and_leaves_the_motion_alone(displaced):
    """A pair whose images came from off-center volumes needs its templates off-center too.

    Two claims. The pair's mass-weighted center lands ON the requested offset rather than near it,
    since a template displaced from its particles suppresses mean Lambda while leaving the d^2 spread
    alone -- which reads as a weak measurement, not as a bug. And the offset is a RIGID translation:
    the inter-state motion is what every rung measures, so shifting the pair must not touch it.
    """
    plain = align_conformers(*displaced, assembly=None)
    offset = (7.0, -3.0, 11.0)
    moved = align_conformers(*displaced, assembly=None, offset=offset)

    mass = plain.state_a["atomic_weight"].to_numpy()[:, None]

    def center(pair):
        both = 0.5 * (pair.state_a[["x", "y", "z"]].to_numpy() + pair.state_b[["x", "y", "z"]].to_numpy())
        return (both * mass).sum(0) / mass.sum()

    assert center(plain) == pytest.approx(np.zeros(3), abs=1e-4)
    assert center(moved) == pytest.approx(np.asarray(offset), abs=1e-4)

    # the motion is untouched: same RMSD, and the state-to-state displacement of every atom identical
    assert moved.alignment.rmsd == pytest.approx(plain.alignment.rmsd)

    def motion(pair):
        return pair.state_b[["x", "y", "z"]].to_numpy() - pair.state_a[["x", "y", "z"]].to_numpy()

    assert motion(moved) == pytest.approx(motion(plain), abs=1e-4)

    # `frame_a` has to agree with where the atoms actually went, or a caller placing a deposited MAP
    # against these templates is handed the origin-centered placement while the templates sit elsewhere
    deposited = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert moved.alignment.frame_a.apply(deposited) == pytest.approx(
        plain.alignment.frame_a.apply(deposited) + np.asarray(offset), abs=1e-4
    )


def _renumbered(records, offset):
    """The same ATOM lines with every residue id shifted by `offset`, columns 23-26."""
    return [line[:22] + f"{int(line[22:26]) + offset:4d}" + line[26:] for line in records]


def test_a_numbering_mismatch_is_diagnosed_differently_from_a_different_construct(tmp_path, coords):
    """Two failures wear the same refusal and want opposite responses.

    Equal atom counts with nothing corresponding is the same molecule keyed differently -- a residue
    numbering offset, which a sequence alignment would fix. Unequal counts with partial overlap really
    is one entry's assembly against part of the other's, where naming a common subunit is the fix. The
    two must be diagnosed with different messages.
    """
    baseline = write_pdb(tmp_path / "base.pdb", atom_records(coords))
    offset = write_pdb(tmp_path / "off.pdb", _renumbered(atom_records(coords), 500))
    with pytest.raises(ValueError, match="residue-NUMBERING mismatch"):
        align_conformers(baseline, offset, assembly=None)

    half = write_pdb(tmp_path / "half.pdb", atom_records(coords[: len(coords) // 2]))
    with pytest.raises(ValueError, match="assembly against part of the other"):
        align_conformers(baseline, half, assembly=None)


def test_a_label_asym_id_selects_its_own_subchain_and_not_the_author_chain_holding_it():
    """A `label_asym_id` is not a chain name, and resolving it to one returns the wrong molecule.

    mmCIF splits one author chain into polymer, ligand and water subchains, so mapping a label to its
    parent chain's name hands back the whole subunit while looking like a successful selection (8S4D's
    label `D`), or substitutes a different chain when the label finds no author chain (9BA0). Both
    directions are asserted, because a map that merely refuses would satisfy only the first; 6Z4Q is
    the case where the label does name an author chain.
    """
    ligand = label_subchain_atoms("data/pdb/8S4D.cif", ("D",))
    assert len(ligand) == 1, "label D of 8S4D is one ligand residue, not a subunit"
    assert {chain for chain, _, _ in ligand} == {"A"}

    whole = _read_assembly("data/pdb/8S4D.cif", None, ("A",), (), 1)
    assert len(whole) > 100 * len(ligand), "the ligand must not resolve to the chain that holds it"

    # the discriminating assertion: through the label, `D` is its own 16 atoms, not the whole subunit
    # of the chain that holds it. `OLA` is allowlisted or the ligand is dropped and the call refuses
    through_label = _read_assembly("data/pdb/8S4D.cif", None, ("D",), ("OLA",), 1, chain_id_kind="label_asym_id")
    assert len(through_label) == 16, "a label selection is its subchain's atoms, not its chain's"

    polymer = label_subchain_atoms("data/pdb/6Z4Q.cif", ("A",))
    assert len(polymer) == 438
    assert {chain for chain, _, _ in polymer} == {"AAA"}
    assert len(_read_assembly("data/pdb/6Z4Q.cif", None, ("A",), (), 1, chain_id_kind="label_asym_id")) == 3026


def test_a_chain_name_that_is_not_an_author_chain_refuses_rather_than_falling_back_to_a_label():
    """Wrong here is silent, so the two namings are told apart by argument and never tried in turn.

    A fallback -- author first, labels if that selects nothing -- reads as a convenience and is not
    one: on 9BA0 the author chains are `R` and `N`, so `("A",)` selects nothing, and label `A` is a
    subchain of `R`. The fallback then returns a different subunit at full confidence. It also fires
    only when the selection is EMPTY, so `("A", "B")` where one name is an author chain and the other
    a label drops the label with no fallback at all, handling a two-chain request differently from a
    one-chain one.
    """
    with pytest.raises(ValueError, match="no atoms left"):
        _read_assembly("data/pdb/9BA0.cif", None, ("A",), (), 1)

    with pytest.raises(ValueError, match="no label_asym_id"):
        label_subchain_atoms("data/pdb/9BA0.cif", ("ZZ",))

    with pytest.raises(ValueError, match="carries no label_asym_id"):
        label_subchain_atoms("data/pdb/2OO8.pdb", ("A",))


def test_a_label_selection_is_not_widened_by_a_shared_residue_number(tmp_path):
    """`(chain, residue_id)` is not unique across the subchains of one author chain.

    A ligand is often numbered inside the polymer's range, so keying on chain and residue number alone
    pulls the polymer residue sharing its number into a one-atom ligand selection -- the same silent
    widening the label route prevents, one level down. Built synthetically because no entry under
    `data/` has the clash.
    """
    structure = gemmi.Structure()
    structure.name, structure.spacegroup_hm = "TEST", "P 1"
    structure.cell = gemmi.UnitCell(50, 50, 50, 90, 90, 90)
    model, chain = gemmi.Model("1"), gemmi.Chain("A")
    for seq, name, atoms in ((1, "GLY", ("N", "CA", "C")), (2, "ALA", ("N", "CA"))):
        residue = gemmi.Residue()
        residue.name, residue.seqid, residue.het_flag = name, gemmi.SeqId(seq, " "), "A"
        for index, atom_name in enumerate(atoms):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element("N" if atom_name == "N" else "C")
            atom.pos, atom.occ, atom.b_iso = gemmi.Position(index, seq, 0), 1.0, 10.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    # a Zn HETATM carrying the SAME residue number as the ALA above, in its own subchain
    zinc = gemmi.Residue()
    zinc.name, zinc.seqid, zinc.het_flag = "ZN", gemmi.SeqId(2, " "), "H"
    atom = gemmi.Atom()
    atom.name, atom.element = "ZN", gemmi.Element("Zn")
    atom.pos, atom.occ, atom.b_iso = gemmi.Position(9, 9, 9), 1.0, 10.0
    zinc.add_atom(atom)
    chain.add_residue(zinc)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()

    path = tmp_path / "clash.cif"
    structure.make_mmcif_document().write_file(str(path))

    ligand_label = next(
        sub.subchain_id()
        for entry in gemmi.read_structure(str(path))[0]
        for sub in entry.subchains()
        if any(residue.name == "ZN" for residue in sub)
    )
    keys = label_subchain_atoms(path, (ligand_label,))
    assert keys == (("A", 2, "ZN"),), "the residue NAME is part of the key"

    selected = _read_assembly(path, None, (ligand_label,), ("ZN",), 1, chain_id_kind="label_asym_id")
    assert len(selected) == 1 and set(selected["residue"]) == {"ZN"}, (
        "keyed without the residue name this also takes the ALA sharing residue number 2"
    )
