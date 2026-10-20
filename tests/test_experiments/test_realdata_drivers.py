"""`reproduction/experiments/realdata/`: the arm that reads deposited data -- score_pool.

Every input here is synthetic and every reader is the real one: a RELION 3.1 star file with an
optics block, particle stacks under a project root, a cryoSPARC particle table carrying the picker's
own `location/*` fields. What is exercised is the path from a file on disk to a number, which is the
half of the pipeline no library test reaches.
"""

import json

import numpy as np
import pytest
from conftest import OPTICS, cryosparc_particles, load_driver, located_particles, save_cs, write_relion_star


def ctf_filtered_noise(box: int, pixel_size: float, defocus: float) -> np.ndarray:
    """White noise through one CTF, which is a micrograph as far as a defocus fit is concerned.

    Args:
        box: Micrograph edge in pixels.
        pixel_size: A/px of that micrograph, NOT of any extracted box.
        defocus: The truth the fit has to come back with, in A.

    Returns:
        `(box, box)` real, its power spectrum carrying the Thon rings of `defocus`.
    """
    from cryodisc.microscope import ctf_astigmatic

    frequency = np.fft.fftfreq(box, d=pixel_size)
    grid_x, grid_y = np.meshgrid(frequency, frequency)
    transfer = np.asarray(ctf_astigmatic(grid_y, grid_x, defocus_u=defocus, defocus_v=defocus, **OPTICS))
    return np.fft.ifft2(np.fft.fft2(np.random.default_rng(4).normal(size=(box, box))) * transfer).real


@pytest.fixture(scope="module")
def real_data_pool(tmp_path_factory, separable_pair):
    """A metadata file and the stacks it points at, built to look like a deposition.

    Synthetic, but through the REAL readers: a RELION 3.1 star file with an optics block, particle
    images in `.mrcs` under a project root, and defocus and origin columns the driver has to follow.
    What it exercises is the path from a file on disk to a scored pool, which had no caller at all
    outside its own unit tests before this driver existed.
    """
    mrcfile = pytest.importorskip("mrcfile")
    directory = tmp_path_factory.mktemp("realdata")
    box, n_particles = 48, 60

    # PARTICLES, not noise: a pool with no molecule in it has no well-posed template amplitude, since
    # `score_pool` fits `a` from the data and on pure noise its sign is a coin flip. The path under
    # test includes a calibration, so the images have to contain the molecule.
    import jax

    from cryodisc import align_conformers
    from cryodisc.microscope import CollectionPlan
    from cryodisc.particles.sim import simulate_stack
    from cryodisc.resolution import noise_spectrum

    plan = CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(11000.0, 13400.0))
    pair = align_conformers(separable_pair[0], separable_pair[1], assembly=None)
    stack = simulate_stack(
        pair.models(plan, box_size=box),
        key=jax.random.key(3),
        n_particles=n_particles,
        noise_spectrum=noise_spectrum(plan, box, rmsf=0.5),
        plan=plan,
    )
    with mrcfile.new(directory / "stack.mrcs", overwrite=True) as handle:
        handle.set_data(np.asarray(jax.numpy.fft.ifftn(stack.images, axes=(-2, -1)).real, dtype=np.float32))

    # a defocus ramp spanning 11000 to 13400 A, so the driver's grouping has more than one group to
    # find, and the two axes 100 A apart so an astigmatic pair cannot be read as one number
    index = np.arange(n_particles)
    star = write_relion_star(
        directory / "particles.star",
        [f"{i + 1:06d}@stack.mrcs" for i in index],
        np.stack([11000.0 + 40.0 * index, 11100.0 + 40.0 * index], axis=1),
        pixel_size=1.5,
        box_size=box,
        defocus_angle=30.0,
    )
    return star, directory, separable_pair


def test_the_real_data_driver_goes_from_a_metadata_file_to_a_scored_pool(real_data_pool, tmp_path):
    """Every piece of the real-data arm in one line, which is what none of them had. The images are
    noise, so the log-ratios carry no signal and are not supposed to -- what is being checked is
    that the file is read, the stacks are found, the per-particle CTF is grouped, the calibration
    converges and a Lambda comes out the far end."""
    star, root, pair = real_data_pool
    driver = load_driver("realdata/score_pool.py")
    out, lam = tmp_path / "pool.json", tmp_path / "lambda.npy"

    argv = [
        "--meta",
        str(star),
        "--root",
        str(root),
        "--structure-a",
        str(pair[0]),
        "--structure-b",
        str(pair[1]),
        "--assembly",
        "none",
    ]
    assert driver.main([*argv, "--out", str(out), "--log-ratio", str(lam), "--cap-bank", "504"]) == 0

    result = json.loads(out.read_text())["result"]
    assert result["n_particles"] == 60
    assert result["box_size"] == 48
    assert result["pixel_size"] == pytest.approx(1.5)
    # the defocus ramp must produce more than one group; a single group would score every particle
    # through one CTF, the failure `microscope.defocus_groups_for` exists to prevent
    assert result["n_defocus_groups"] > 1
    assert np.load(lam).shape == (60,)


def test_a_deposited_pixel_size_can_be_overridden_because_it_can_be_wrong(real_data_pool, tmp_path):
    """EMPIAR-10076 says 1.0 A/px against a true 1.31, and the error rescales every spatial
    frequency invisibly. The override has to reach the recorded result, not just the reader."""
    star, root, pair = real_data_pool
    driver = load_driver("realdata/score_pool.py")
    out = tmp_path / "override.json"

    argv = [
        "--meta",
        str(star),
        "--root",
        str(root),
        "--structure-a",
        str(pair[0]),
        "--structure-b",
        str(pair[1]),
        "--assembly",
        "none",
    ]
    driver.main([*argv, "--out", str(out), "--pixel-size", "1.31", "--cap-bank", "504"])

    assert json.loads(out.read_text())["result"]["pixel_size"] == pytest.approx(1.31)


def test_a_ctf_refit_without_a_micrograph_is_refused(real_data_pool, tmp_path):
    """There is nothing to refit a CTF from without one, and failing at the argument is better than
    failing after the stacks have been read. The micrograph's PIXEL SIZE is refused for the same
    reason and separately: this `--meta` is a `.star`, which states none, and defaulting it to the
    particle boxes' is the silent wrong answer the flag exists to avoid."""
    star, root, pair = real_data_pool
    driver = load_driver("realdata/score_pool.py")
    argv = [
        "--meta",
        str(star),
        "--root",
        str(root),
        "--structure-a",
        str(pair[0]),
        "--structure-b",
        str(pair[1]),
        "--assembly",
        "none",
    ]
    with pytest.raises(SystemExit):
        driver.main([*argv, "--out", str(tmp_path / "x.json"), "--refit-ctf"])
    with pytest.raises(SystemExit):
        driver.main([*argv, "--out", str(tmp_path / "x.json"), "--refit-ctf", "--micrograph", str(tmp_path / "m.npy")])


def test_the_micrograph_pixel_size_is_taken_from_the_pool_and_only_overridden_by_the_flag(
    real_data_pool, tmp_path, capsys
):
    """A flag catches "never stated"; it cannot catch "stated wrong". A `.cs` pool from
    `build_pools.py` is a class2D output joined with its passthrough, and that join brings
    `location/micrograph_psize_A` -- the micrograph's own sampling, which is what a refit needs and
    what the extracted boxes are a downsample of. So the file settles it, the flag overrides it, and
    only a file with neither is refused."""
    star, root, pair = real_data_pool
    driver = load_driver("realdata/score_pool.py")
    pool = tmp_path / "pool.cs"
    located = located_particles(4, pixel_size=1.5)
    located["location/micrograph_psize_A"] = 0.75
    save_cs(pool, located)

    def parsed(meta, *extra):
        return driver.parse_args([
            "--meta", str(meta), "--root", str(root), "--structure-a", str(pair[0]),
            "--structure-b", str(pair[1]), "--assembly", "none", "--out", str(tmp_path / "o.json"),
            "--refit-ctf", "--micrograph", str(tmp_path / "m.npy"), *extra,
        ])  # fmt: skip

    assert parsed(pool).micrograph_pixel_size == pytest.approx(0.75)
    # the flag still wins, because a deposition can be wrong, but the disagreement is said out loud,
    # since a silent one is what a wrong pixel size read back as a wildly wrong defocus looks like
    assert parsed(pool, "--micrograph-pixel-size", "1.31").micrograph_pixel_size == pytest.approx(1.31)
    assert "overrides the file's 0.75" in capsys.readouterr().out
    # a .star carries no such field, so there the flag is the only source and stays required
    with pytest.raises(SystemExit):
        parsed(star)
    assert parsed(star, "--micrograph-pixel-size", "0.9").micrograph_pixel_size == pytest.approx(0.9)


def test_a_pool_spanning_two_micrograph_samplings_is_refused_rather_than_picking_one(real_data_pool, tmp_path):
    """One fit at either sampling is wrong for the rest of the particles, and picking one silently is
    the same class of defect as reading the boxes' pixel size."""
    _, root, pair = real_data_pool
    driver = load_driver("realdata/score_pool.py")
    located = located_particles(4)
    located["location/micrograph_psize_A"] = [0.75, 0.75, 1.31, 1.31]
    save_cs(tmp_path / "mixed.cs", located)

    argv = [
        "--meta", str(tmp_path / "mixed.cs"), "--root", str(root), "--structure-a", str(pair[0]),
        "--structure-b", str(pair[1]), "--assembly", "none", "--out", str(tmp_path / "o.json"),
        "--refit-ctf", "--micrograph", str(tmp_path / "m.npy"),
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        driver.parse_args(argv)


def test_a_ctf_refit_fits_the_micrograph_at_the_micrographs_own_pixel_size(real_data_pool, tmp_path):
    """A CTF refit must fit the micrograph at the micrograph's own pixel size, not the extracted
    boxes'. `chi = pi lambda dz k^2` absorbs an error in k into a larger error in dz, so fitting at
    the boxes' coarser pixel size returns a defocus several times the truth. What is checked is that
    the number reaching `with_ctf` is the truth and not that inflated value."""
    import argparse

    from cryodisc.particles.metadata import load

    star, _, _ = real_data_pool
    driver = load_driver("realdata/score_pool.py")
    micrograph_pixel, truth = 0.75, 4000.0
    np.save(tmp_path / "micrograph.npy", ctf_filtered_noise(512, micrograph_pixel, truth))

    meta = load(star)
    assert float(np.median(meta.pixel_size)) == pytest.approx(1.5), "the boxes are the 2x downsample"

    def refit(pixel_size):
        arguments = argparse.Namespace(
            refit_ctf=True, micrograph=tmp_path / "micrograph.npy", micrograph_pixel_size=pixel_size
        )
        return driver.refit_ctf(meta, arguments)[1]

    stated = refit(micrograph_pixel)
    assert stated["micrograph_pixel_size"] == micrograph_pixel
    assert stated["median_defocus_after"] == pytest.approx(truth, rel=0.05)
    # and the boxes' own pixel size -- what the driver used to pass -- is off by nearly threefold
    assert refit(1.5)["median_defocus_after"] > 2.5 * truth


MAP_FRACTION_BOX = 40
"""Above the 200-atom pair's own floor at 2 A/px, so nothing is clipped."""

MAP_FRACTION_PIXEL = 2.0
"""Coarse enough that an 8 A band is well inside Nyquist and the toy renders stay fast."""


@pytest.fixture(scope="module")
def rendered_models(separable_pair):
    """The two model maps the map-fraction fit will render for itself, at the grid it will be given.

    The test plants its density as a blend of THESE, so the recovery is a statement about the
    driver's wiring: rendering a different assembly or alignment would not recover the planted
    fraction at all.
    """
    from cryodisc import CollectionPlan, align_conformers
    from cryodisc.specimen.density import from_model

    pair = align_conformers(separable_pair[0], separable_pair[1], assembly=None)
    models = pair.models(CollectionPlan(dose=40.0, pixel_size=MAP_FRACTION_PIXEL), box_size=MAP_FRACTION_BOX)
    return tuple(from_model(model) for model in models)


def test_the_map_fraction_models_are_superposed_because_they_are_rendered(rendered_models):
    """What rendering buys the driver. Two model MAP files could be in any two frames and nothing
    downstream would fail -- an unsuperposed pair reads as a smaller fraction, not as an error. Both
    states come off one superposition through one image config, so this cannot be false."""
    from reproduction.datasets.picking import center_of_mass_separation

    model_a, model_b = rendered_models
    assert model_a.shape == model_b.shape == (MAP_FRACTION_BOX,) * 3
    assert center_of_mass_separation(model_a, model_b) < 1.0


def test_the_recentering_driver_reads_a_pool_that_does_carry_shifts(tmp_path, separable_pair):
    """The other side of the refusal: a `.cs` with `alignments3D/shift` and `/pose` gets past both
    guards, which is what says the guards are reading the table rather than refusing everything.

    Stops at the guards deliberately -- the measurement itself is thousands of correlations and is
    what the driver's own run is for. What is asserted here is that the shared pool layer hands this
    driver the same table `score_pool` would get.
    """
    pytest.importorskip("mrcfile")
    import mrcfile

    from cryodisc.pools import read_pool

    box, n = 32, 8
    rng = np.random.default_rng(3)
    with mrcfile.new(tmp_path / "stack.mrcs", overwrite=True) as handle:
        handle.set_data(rng.normal(size=(n, box, box)).astype(np.float32))
    data = cryosparc_particles(
        n,
        box=box,
        pixel_size=1.5,
        stack="stack.mrcs",
        extra={
            "alignments3D/pose": rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32),
            "alignments3D/shift": rng.uniform(-2.0, 2.0, (n, 2)).astype(np.float32),
            # the refinement's own pixel size, which is what its shift is in units of -- a
            # downsampled refinement records a shift the stack's psize would read wrong
            "alignments3D/psize_A": np.full(n, 1.5, np.float32),
        },
    )
    meta_path = save_cs(tmp_path / "particles.cs", data)

    meta, images, pixel_size, report, labels = read_pool(meta_path, root=tmp_path, recenter_images=False)

    assert meta.has_shift and meta.has_euler, "both guards must see what the table carries"
    assert images.shape == (n, box, box)
    assert pixel_size == pytest.approx(1.5)
    assert report["applied"] is False, "--no-recenter must leave the images as stored"
    assert labels is None, "no --state was given, so there is nothing to label"


@pytest.fixture
def map_pair(tmp_path):
    """A writer for box-32 maps at 2.0 A/voxel, the blob to write, and the plan to score them with.

    Three tests build the same pair of Gaussian blobs and the same `mrcfile` writer; the header
    keywords are here because the origin test needs them and the other two are a subset. Shared so
    that "what a well-formed map looks like" is stated once.
    """
    mrcfile = pytest.importorskip("mrcfile")
    from cryodisc.microscope import CollectionPlan

    box, pixel = 32, 2.0
    grid = np.indices((box,) * 3) - (box - 1) / 2.0
    blob = np.exp(-(grid**2).sum(axis=0) / 32.0).astype(np.float32)

    def write(name, volume, nxstart=None, origin=None):
        path = tmp_path / name
        with mrcfile.new(path, overwrite=True) as handle:
            handle.set_data(np.ascontiguousarray(volume, np.float32))
            handle.voxel_size = (pixel,) * 3
            if nxstart is not None:
                handle.header.nxstart, handle.header.nystart, handle.header.nzstart = (nxstart,) * 3
            if origin is not None:
                handle.header.origin.x = handle.header.origin.y = handle.header.origin.z = origin
        return path

    return write, blob, CollectionPlan(dose=40.0, pixel_size=pixel), box, pixel


def test_a_map_pair_that_is_not_on_one_amplitude_scale_is_refused(map_pair):
    """One amplitude is fitted for the PAIR, so an imbalance between the two maps is unfittable: it
    adds the same constant to every particle's log-ratio whatever that particle's state. A pair whose
    maps differ by a flat amplitude factor puts both pools at a positive mean Lambda, a degenerate
    readout wearing the shape of a perfect one, so it must be refused."""
    from cryodisc.pools import build_states

    write, blob, plan, box, pixel = map_pair
    balanced = (write("a.mrc", blob), write("b.mrc", np.roll(blob, 1, axis=0)))
    _, report = build_states(plan, box_size=box, pixel_size=pixel, map_a=balanced[0], map_b=balanced[1])
    assert report["amplitude_ratio"] == pytest.approx(1.0, abs=0.05), "a real pair passes, and on the record"

    # a flat 1.7x amplitude imbalance between the two maps, which the guard must refuse
    with pytest.raises(ValueError, match="amplitude"):
        build_states(
            plan,
            box_size=box,
            pixel_size=pixel,
            map_a=write("loud.mrc", 1.7 * blob),
            map_b=balanced[1],
        )


def test_a_box_centered_map_is_accepted_under_either_mrc_convention_and_a_corner_origin_is_not(map_pair):
    """A box-centered map must be accepted under either MRC convention -- `ORIGIN` set, or
    `NXSTART = -N/2` -- since both name the box center as the world origin, and a `|origin| > 0` guard
    wrongly refuses the second.

    A genuine corner origin must still be refused: `from_voxel_grid` centers the density whatever the
    header says, so the offset displaces every template from the particles and reads as a smaller
    signal rather than failing.
    """
    from cryodisc.pools import build_states

    write, blob, plan, box, pixel = map_pair
    shifted = np.roll(blob, 1, axis=0)

    build_states(plan, box_size=box, pixel_size=pixel, map_a=write("a.mrc", blob), map_b=write("b.mrc", shifted))
    build_states(
        plan,
        box_size=box,
        pixel_size=pixel,
        map_a=write("ca.mrc", blob, nxstart=-box // 2),
        map_b=write("cb.mrc", shifted, nxstart=-box // 2),
    )

    corner = write("corner.mrc", blob, origin=40.0)
    with pytest.raises(ValueError, match="box center"):
        build_states(plan, box_size=box, pixel_size=pixel, map_a=corner, map_b=write("b2.mrc", shifted))
    build_states(
        plan,
        box_size=box,
        pixel_size=pixel,
        map_a=corner,
        map_b=write("b3.mrc", shifted),
        ignore_origin=True,
    )


def test_an_offset_read_off_maps_is_refused_on_the_map_route_rather_than_ignored(map_pair):
    """`offset_from_maps` moves a rendered ATOMIC pair, and the map route has no atoms to move, so
    passing it beside a map pair must be refused rather than accepted and discarded in silence.
    `check_pool_arguments` refuses it at a CLI; a library caller got nothing.
    """
    from cryodisc.pools import build_states

    write, blob, plan, box, pixel = map_pair
    maps = (write("a.mrc", blob), write("b.mrc", np.roll(blob, 1, axis=0)))
    with pytest.raises(ValueError, match="no atoms to move"):
        build_states(
            plan,
            box_size=box,
            pixel_size=pixel,
            map_a=maps[0],
            map_b=maps[1],
            offset_from_maps=list(maps),
        )
