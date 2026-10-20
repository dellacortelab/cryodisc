"""Checks on a molecule held as a grid.

Every one of these is a convention that produces a plausible map when it is wrong: a low-pass that
rings, a resampler that rescales the molecule, a rotation applied in the wrong index order. So each
is pinned against a map built analytically from the physics the operation claims to implement.
"""

import numpy as np
import pytest
from conftest import gaussian_cloud, physical_grid

from cryodisc.specimen.density import (
    center_of_mass_offset,
    correlation,
    from_model,
    lowpass,
    read_map,
    write_map,
)

BOX = 48
PIXEL = 1.0
IDENTITY = np.eye(3)


def gaussian(center, widths, rotation=IDENTITY, box=BOX, pixel=PIXEL):
    """An anisotropic Gaussian, so a rotation is visible in the map and not only in its centroid."""
    x, y, z = physical_grid(box, pixel)
    offset = np.stack([x - center[0], y - center[1], z - center[2]], -1)
    return np.exp(-0.5 * ((offset @ rotation / widths) ** 2).sum(-1))


def test_a_lowpass_removes_the_band_it_says_it_does():
    """And leaves the rest: a filter that attenuated everything would still look like a low-pass in
    a picture and would rescale every coefficient the regression downstream fits."""
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(BOX, BOX, BOX))
    filtered = lowpass(volume, PIXEL, 8.0)

    frequency = np.fft.fftfreq(BOX, PIXEL)
    radius = np.sqrt(frequency[:, None, None] ** 2 + frequency[None, :, None] ** 2 + frequency[None, None, :] ** 2)
    spectrum = np.abs(np.fft.fftn(filtered))
    assert spectrum[radius > 1.0 / 8.0].max() < 1e-9
    # inside 0.8 of the cutoff the taper is exactly 1, so those modes are untouched
    passband = radius < 0.8 / 8.0
    np.testing.assert_allclose(spectrum[passband], np.abs(np.fft.fftn(volume))[passband], rtol=1e-9)


def test_a_lowpass_past_nyquist_is_refused_rather_than_doing_nothing():
    """It would filter nothing while reading as a stated band, and the band is what a map fraction
    is quoted at."""
    with pytest.raises(ValueError, match="past the"):
        lowpass(np.zeros((8, 8, 8)), 2.0, 3.0)


def test_a_map_survives_a_round_trip_through_a_file(tmp_path):
    """The voxel size especially: a map imported at the wrong one is the wrong SIZE, which a picker
    and a fit both use without complaint."""
    pytest.importorskip("mrcfile")
    volume = gaussian((1.0, 2.0, 3.0), np.array([4.0, 4.0, 4.0]))

    path = write_map(tmp_path / "v.mrc", volume, 1.31)
    reloaded, pixel_size, origin = read_map(path)

    np.testing.assert_allclose(reloaded, volume, atol=1e-6)
    assert pixel_size == pytest.approx(1.31)
    assert origin is not None


def test_an_npy_map_reports_no_pixel_size_rather_than_guessing_one():
    """A wrong voxel size rescales every spatial frequency invisibly, which is why `load_star`
    refuses to default one either."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/v.npy"
        np.save(path, np.zeros((8, 8, 8)))
        volume, pixel_size, origin = read_map(path)
    assert volume.shape == (8, 8, 8)
    assert pixel_size is None and origin is None


def test_a_map_that_is_not_a_cube_is_refused():
    """Every operation here is cubic, and a rectangular map would be cropped asymmetrically."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/v.npy"
        np.save(path, np.zeros((8, 8, 12)))
        with pytest.raises(ValueError, match="cubic map"):
            read_map(path)


def test_correlation_is_centered_and_says_nothing_about_a_constant_map():
    volume = gaussian((0.0, 0.0, 0.0), np.array([4.0, 4.0, 4.0]))
    assert correlation(volume, volume) == pytest.approx(1.0)
    assert correlation(volume, 3.0 * volume + 7.0) == pytest.approx(1.0)
    assert np.isnan(correlation(volume, np.ones_like(volume)))


def atomic_cloud(seed, box, pixel, width, center, n=40):
    """A cluster of narrow Gaussians -- an atomic-looking density, which carries power past Nyquist.

    Which is the whole point: a real reconstruction does, and a model map rendered directly at the
    grid it will be fitted on does not.
    """
    positions = np.random.default_rng(seed).normal(0.0, 6.0, (n, 3)) + np.asarray(center)
    return gaussian_cloud(positions, box, pixel, width)


def test_a_map_origin_is_read_from_either_of_the_two_header_conventions(tmp_path):
    """An MRC records its first voxel's position in one of two places: `ORIGIN` (words 50-52) in
    angstroms, or `NXSTART`/`NYSTART`/`NZSTART` (words 5-7) as unit-cell indices times the sampling.
    Both must be read, or a map written the second way reports origin zero and slips past the map
    frame guard. `ORIGIN` when non-zero and the start words otherwise; at most one is ever set."""
    mrcfile = pytest.importorskip("mrcfile")

    def origin_of(name, **header):
        path = tmp_path / name
        with mrcfile.new(path, overwrite=True) as handle:
            handle.set_data(np.zeros((8, 8, 8), np.float32))
            handle.voxel_size = (1.5, 1.5, 1.5)
            for field, value in header.items():
                setattr(handle.header, field, value)
        return read_map(path)[2]

    np.testing.assert_allclose(origin_of("centered.mrc"), 0.0)
    np.testing.assert_allclose(origin_of("origin.mrc", origin=(3.0, -6.0, 9.0)), [3.0, -6.0, 9.0])
    # 4 voxels at 1.5 A/voxel is 6 A
    np.testing.assert_allclose(origin_of("start.mrc", nxstart=4, nystart=0, nzstart=-2), [6.0, 0.0, -3.0])
    # the start words are indexed by column/row/section, so mapc/mapr/maps put them back on x, y, z
    permuted = origin_of("permuted.mrc", nxstart=4, nystart=0, nzstart=-2, mapc=3, mapr=1, maps=2)
    np.testing.assert_allclose(permuted, [0.0, -3.0, 6.0])


def test_a_written_map_carries_the_origin_it_was_read_with(tmp_path):
    """A `read_map` -> `write_map` round trip is how every resampled map reaches a driver, so
    `write_map` must record the origin: a corner-origin map that came back claiming to be box-centered
    would slip past the map frame guard, which fails open rather than raising."""
    mrcfile = pytest.importorskip("mrcfile")
    source = tmp_path / "corner.mrc"
    with mrcfile.new(source, overwrite=True) as handle:
        handle.set_data(np.zeros((8, 8, 8), np.float32))
        handle.voxel_size = (1.5, 1.5, 1.5)
        handle.header.origin = (3.0, -6.0, 9.0)

    volume, pixel_size, origin = read_map(source)
    reloaded_origin = read_map(write_map(tmp_path / "round.mrc", volume, pixel_size, origin=origin))[2]

    np.testing.assert_allclose(reloaded_origin, [3.0, -6.0, 9.0])
    # and the default is still the box-centered convention everything here renders in, not a leak
    np.testing.assert_allclose(read_map(write_map(tmp_path / "plain.mrc", volume, pixel_size))[2], 0.0)


def test_a_permuted_axis_order_is_put_back_on_zyx_rather_than_returned_as_it_lies(tmp_path):
    """`mrcfile` validates `mapc`/`mapr`/`maps` and never reorders the array, so a map naming an
    unusual permutation comes back indexed something other than (z, y, x). The cubic check cannot see
    it, and `center_of_mass_offset` would then report two axes swapped. The reader must put the array
    back on (z, y, x)."""
    mrcfile = pytest.importorskip("mrcfile")

    # one voxel at STORED index (2, 5, 9). With mapc, mapr, maps = 2, 1, 3 the array is indexed
    # (section, row, column) = (z, x, y), so that voxel is physically at (x, y, z) = (5, 9, 2)
    data = np.zeros((16, 16, 16), np.float32)
    data[2, 5, 9] = 1.0
    path = tmp_path / "permuted.mrc"
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(data)
        handle.voxel_size = (1.0, 1.0, 1.0)
        handle.header.mapc, handle.header.mapr, handle.header.maps = 2, 1, 3

    volume, _, _ = read_map(path)
    assert volume.shape == (16, 16, 16)
    # read back as (z, y, x): z = 2, y = 9, x = 5
    assert [int(i) for i in np.argwhere(volume > 0.5)[0]] == [2, 9, 5]
    # and the offset from the box center agrees, in (x, y, z) -- not two axes of it swapped.
    # `center_of_mass_offset` centers on n/2 = 8, so (5, 9, 2) is (-3, +1, -6) A at 1 A/voxel
    np.testing.assert_allclose(center_of_mass_offset(volume, 1.0), [-3.0, 1.0, -6.0])


def test_a_voxel_size_that_differs_by_axis_is_refused_rather_than_read_off_x(tmp_path):
    """The cubic check constrains the shape and says nothing about `cella`, so a map with an
    anisotropic cell passes it. Read as a single uniform voxel size, the molecule is silently
    compressed along the odd axis, which reads downstream as a different conformation."""
    mrcfile = pytest.importorskip("mrcfile")

    path = tmp_path / "aniso.mrc"
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.zeros((16, 16, 16), np.float32))
        handle.header.cella.x, handle.header.cella.y, handle.header.cella.z = 32.0, 32.0, 64.0

    with pytest.raises(ValueError, match="expected one voxel size"):
        read_map(path)


def test_a_rendered_map_is_the_molecule_the_templates_are(tmp_path):
    """The certification `from_model` exists for. A map read off disk is asserted to be the pair's
    molecule; one rendered from the pair's own model IS it, and this is the measurement that says so:
    project the map and project the atoms at the same pose, and they have to agree."""
    from conftest import atom_records, write_pdb

    from cryodisc.forward.cryojax_model import CryoJAXModel
    from cryodisc.forward.optics import CTFParams

    box, pixel = 64, 1.5
    coords = np.random.default_rng(0).normal(0.0, 6.0, (300, 3))
    structure = write_pdb(tmp_path / "a.pdb", atom_records(coords))
    model = CryoJAXModel.from_structure(structure, box_size=box, pixel_size=pixel, center=False)

    rendered = from_model(model)
    assert rendered.shape == (box, box, box)
    assert rendered.min() >= 0.0, "an electrostatic potential is non-negative"

    optics = CTFParams(defocus_in_angstroms=8000.0, spherical_aberration_in_mm=2.7, b_factor=0.0)
    identity = np.zeros((1, 3))
    from_atoms = np.asarray(model.forward_batch(identity, optics))[0]
    from_map = np.asarray(
        CryoJAXModel.from_voxel_grid(rendered, pixel_size=pixel, box_size=box).forward_batch(identity, optics)
    )[0]
    # at the identity pose the voxel path does no slice interpolation, so this is the renderer alone
    assert correlation(from_atoms, from_map) > 0.999


def test_a_rendered_map_sits_where_the_atoms_do(tmp_path):
    """The convention every other function here assumes: indexed (z, y, x) and box-centered at n / 2.
    An axis transposition or a half-box offset still produces a map, and `to_frame` would then move
    it somewhere else again."""
    from conftest import atom_records, write_pdb

    from cryodisc.forward.cryojax_model import CryoJAXModel

    box, pixel = 64, 1.0
    # one lobe far off-center along x alone, so a transposition is unmissable
    coords = np.random.default_rng(1).normal(0.0, 2.0, (200, 3)) + np.array([12.0, 0.0, 0.0])
    structure = write_pdb(tmp_path / "a.pdb", atom_records(coords))
    model = CryoJAXModel.from_structure(structure, box_size=box, pixel_size=pixel, center=False)

    rendered = from_model(model)
    index = np.indices(rendered.shape)
    mass = rendered / rendered.sum()
    center = np.array([float((mass * (index[axis] - box / 2.0)).sum() * pixel) for axis in range(3)])

    # center is (z, y, x) against the atoms' (x, y, z)
    np.testing.assert_allclose(center[::-1], coords.mean(0), atol=0.2)


def test_two_states_render_onto_one_grid_in_one_frame(tmp_path):
    """What rendering buys over reading two files: `align_conformers` superposed the pair and
    `ConformerPair.models` builds both on one image config, so the two maps cannot disagree about a
    grid, a frame, or which assembly was built. A driver taking map FILES can check none of that."""
    from conftest import atom_records, write_pdb

    from cryodisc import CollectionPlan
    from cryodisc.specimen.structures import align_conformers
    from reproduction.datasets.picking import center_of_mass_separation

    rng = np.random.default_rng(2)
    coords = rng.uniform(-12.0, 12.0, (150, 3))
    bent = coords + np.where(coords[:, :1] > 0.0, np.array([[4.0, 0.0, 0.0]]), 0.0)
    pair = align_conformers(
        write_pdb(tmp_path / "a.pdb", atom_records(coords)),
        write_pdb(tmp_path / "b.pdb", atom_records(bent)),
        assembly=None,
    )
    model_a, model_b = pair.models(CollectionPlan(dose=40.0, pixel_size=2.0), box_size=48)

    map_a, map_b = from_model(model_a), from_model(model_b)
    assert map_a.shape == map_b.shape == (48, 48, 48)
    # rendered from ONE superposition, so superposed -- which is the property the map-fraction fit
    # previously had to take on trust from two files on disk
    assert center_of_mass_separation(map_a, map_b) < 1.0


def test_a_rendered_map_carries_its_molecule_only_in_band(tmp_path):
    """The caveat every consumer has to honor. The Peng potential is nearly point-like -- under 1%
    of voxels carry anything -- so a RAW voxel correlation reads the sampling and gets WORSE as the
    grid gets finer, which is the signature of comparing aliases. Band-limited, the same comparison
    is grid-independent, and that independence is the tell that what is left is the molecule."""
    from conftest import atom_records, write_pdb

    from cryodisc import CollectionPlan
    from cryodisc.specimen.structures import align_conformers

    rng = np.random.default_rng(2)
    coords = rng.uniform(-12.0, 12.0, (150, 3))
    bent = coords + np.where(coords[:, :1] > 0.0, np.array([[4.0, 0.0, 0.0]]), 0.0)
    pair = align_conformers(
        write_pdb(tmp_path / "a.pdb", atom_records(coords)),
        write_pdb(tmp_path / "b.pdb", atom_records(bent)),
        assembly=None,
    )

    raw, banded = [], []
    for box, pixel in ((48, 2.0), (96, 1.0)):
        model_a, model_b = pair.models(CollectionPlan(dose=40.0, pixel_size=pixel), box_size=box)
        map_a, map_b = from_model(model_a), from_model(model_b)
        assert (map_a > 0.01 * map_a.max()).mean() < 0.01, "the potential is nearly point-like"
        raw.append(correlation(map_a, map_b))
        banded.append(correlation(lowpass(map_a, pixel, 8.0), lowpass(map_b, pixel, 8.0)))

    # raw: grid-dependent and low, and the FINER grid looks worse
    assert raw[1] < raw[0] < 0.4
    # banded: the same number on both grids, and it is the molecule
    assert banded[0] == pytest.approx(banded[1], abs=0.02)
    assert banded[0] > 0.8


def test_a_voxel_model_is_refused_rather_than_re_rendered(tmp_path):
    """It already has its map; round-tripping one through a renderer would resample it silently."""
    from cryodisc.forward.cryojax_model import CryoJAXModel

    voxel = CryoJAXModel.from_voxel_grid(np.zeros((16, 16, 16), np.float32), pixel_size=1.0)
    with pytest.raises(TypeError, match="renders an ATOMIC model"):
        from_model(voxel)


def test_a_maps_center_of_mass_offset_is_reported_in_the_atom_frames_axis_order():
    """The axis reversal is the whole point, and getting it wrong half-works rather than failing.

    A map is indexed (z, y, x) and a structure's coordinates are (x, y, z), so an offset measured in
    index order and applied to atoms has two axes swapped -- which displaces templates by roughly the
    right MAGNITUDE in the wrong direction and reads as a partial fix. Density is placed on ONE axis
    here so a swap cannot hide: the answer must be nonzero in x alone.
    """
    box, pixel_size = 16, 2.0
    volume = np.zeros((box, box, box))
    # `box // 2`, this module's convention and cryoJAX's rotation center, NOT `(box-1)/2`
    center = box // 2  # so y and z straddle it symmetrically and contribute nothing
    # index order is (z, y, x), so density written at x = 12 must come back as an offset along x
    volume[7:10, 7:10, 12] = 1.0

    offset = center_of_mass_offset(volume, pixel_size)

    assert offset[0] == pytest.approx((12 - center) * pixel_size)
    # the x component is the ONLY nonzero one, which is what a z/x swap would break
    assert offset[1] == pytest.approx(0.0)
    assert offset[2] == pytest.approx(0.0)


def test_a_rendered_centered_pair_reads_no_offset_at_either_parity(tmp_path):
    """The convention tied to what actually renders, rather than to a hand-built blob.

    `align_conformers` centers the pair, so a map rendered from either state must report an offset of
    zero -- and must do so at EVEN and ODD box sizes alike, since `(box-1)/2` and `box // 2` coincide
    only at odd parity. A half-voxel convention error reads a nonzero offset on the even box, small
    enough to look like discretization and large enough to displace every template placed with it.
    """
    from conftest import atom_records, write_pdb

    from cryodisc import CollectionPlan
    from cryodisc.specimen.structures import align_conformers

    rng = np.random.default_rng(0)
    coords = rng.uniform(-8.0, 8.0, (200, 3))
    path = write_pdb(tmp_path / "a.pdb", atom_records(coords))
    pair = align_conformers(path, path, assembly=None)

    pixel_size = 1.5
    for box in (48, 49):
        model, _ = pair.models(CollectionPlan(dose=40.0, pixel_size=pixel_size), box_size=box)
        offset = center_of_mass_offset(from_model(model), pixel_size)
        # a third of a voxel, which is the discretization of a finite atom render and not a shift
        assert np.max(np.abs(offset)) < 0.5 * pixel_size, f"box {box} reads an offset of {offset}"


def test_a_centered_map_reports_no_offset_and_an_empty_one_is_refused():
    """The two ends. A symmetric map must read zero rather than half a voxel, since a systematic
    half-pixel would displace every template it placed; and a map summing to zero has no center of
    mass at all, which is a refusal rather than a nan quietly entering a coordinate."""
    box = 16
    volume = np.zeros((box, box, box))
    volume[7:10, 7:10, 7:10] = 1.0  # straddles the box // 2 = 8 center symmetrically

    assert center_of_mass_offset(volume, 2.0) == pytest.approx(np.zeros(3))

    # and at ODD parity, where `(box-1)/2` and `box // 2` coincide -- so a convention error shows up
    # on the even box alone, which is every box this package actually renders at
    odd = np.zeros((15, 15, 15))
    odd[6:9, 6:9, 6:9] = 1.0  # centered on 15 // 2 = 7
    assert center_of_mass_offset(odd, 2.0) == pytest.approx(np.zeros(3))

    with pytest.raises(ValueError, match="no center of mass"):
        center_of_mass_offset(np.zeros((box, box, box)), 2.0)

    with pytest.raises(ValueError, match="cubic"):
        center_of_mass_offset(np.zeros((box, box, box + 2)), 2.0)
