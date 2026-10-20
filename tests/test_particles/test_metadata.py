"""Checks on the real-data metadata readers.

Everything here is a unit conversion or a sign, and every one of them is invisible downstream: a
defocus read in the wrong units still produces a transfer, a shift applied the wrong way still
produces a recentered stack. So each is pinned against a value written into the file by the test, or
against an image whose true offset the test planted.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import cryosparc_particles, save_cs

from cryodisc.grids.fourier import phase_ramp
from cryodisc.grids.poses import euler_from_relion, rotation_of
from cryodisc.microscope import wavelength
from cryodisc.particles.metadata import (
    CS_POSE_TRANSPOSE,
    check_shift_sign,
    defocus_bin_width,
    load,
    load_star,
    recenter,
    relion_euler_from_cs_pose,
    select_states,
    transfer_from_meta,
)

RELION_31 = """
data_optics

loop_
_rlnOpticsGroup #1
_rlnVoltage #2
_rlnSphericalAberration #3
_rlnAmplitudeContrast #4
_rlnImagePixelSize #5
_rlnImageSize #6
1 300.0 2.7 0.1 1.5 64

data_particles

loop_
_rlnImageName #1
_rlnDefocusU #2
_rlnDefocusV #3
_rlnDefocusAngle #4
_rlnOpticsGroup #5
_rlnOriginXAngst #6
_rlnOriginYAngst #7
000001@stack.mrcs 10000.0 10200.0 30.0 1 3.0 -1.5
000002@stack.mrcs 12000.0 12100.0 45.0 1 0.0 0.0
"""

RELION_1X = """
data_

loop_
_rlnImageName #1
_rlnDefocusU #2
_rlnDefocusV #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
_rlnDetectorPixelSize #7
_rlnMagnification #8
_rlnOriginX #9
_rlnOriginY #10
000001@old.mrcs 15000.0 15000.0 200.0 2.0 0.07 14.0 100000.0 2.0 -3.0
"""


def write(tmp_path, name, text):
    """Write a star file and return its path."""
    path = tmp_path / name
    path.write_text(text.strip() + "\n")
    return path


def test_the_31_form_is_read_in_the_units_the_module_promises(tmp_path):
    """Angstroms and RADIANS throughout, and the shift transposed to (y, x) once at the reader."""
    meta = load(write(tmp_path, "p.star", RELION_31))

    assert len(meta) == 2
    assert meta.box_size == 64
    np.testing.assert_allclose(meta.defocus_u, [10000.0, 12000.0])
    np.testing.assert_allclose(meta.defocus, [10100.0, 12050.0])
    # the file records degrees; the reader owes radians
    np.testing.assert_allclose(meta.astigmatism_angle, np.radians([30.0, 45.0]))
    np.testing.assert_allclose(meta.voltage_in_kv, [300.0, 300.0])
    np.testing.assert_allclose(meta.pixel_size, [1.5, 1.5])
    assert meta.has_shift
    # written as (x=3.0, y=-1.5) angstroms, stored as (y, x)
    np.testing.assert_allclose(meta.shift[0], [-1.5, 3.0])
    np.testing.assert_allclose(meta.shift_in_pixels()[0], [-1.0, 2.0])
    # the frame index is 1-based on disk and 0-based here
    np.testing.assert_array_equal(meta.stack_index, [0, 1])
    assert meta.stack_path[0] == "stack.mrcs"


def test_the_older_form_converts_both_of_its_own_conventions(tmp_path):
    """1.x records the pixel size as detector size over magnification, and the origin in PIXELS."""
    meta = load_star(write(tmp_path, "old.star", RELION_1X))

    # 14 um / 100000 x = 1.4e-4 um = 1.4 A
    np.testing.assert_allclose(meta.pixel_size, [1.4])
    np.testing.assert_allclose(meta.voltage_in_kv, [200.0])
    assert meta.has_shift
    # (x=2.0, y=-3.0) pixels at 1.4 A/px, stored as (y, x) angstroms
    np.testing.assert_allclose(meta.shift[0], [-3.0 * 1.4, 2.0 * 1.4])
    np.testing.assert_allclose(meta.shift_in_pixels()[0], [-3.0, 2.0])
    # a missing DefocusV falls back to DefocusU rather than to zero
    np.testing.assert_allclose(meta.defocus_v, meta.defocus_u)


def test_a_file_that_cannot_state_its_pixel_size_is_refused(tmp_path):
    """A file that does not state its pixel size is refused rather than silently defaulted to 1.0."""
    text = """
data_

loop_
_rlnImageName #1
_rlnDefocusU #2
_rlnDefocusV #3
000001@old.mrcs 15000.0 15000.0
"""
    with pytest.raises(ValueError, match="cannot determine the pixel size"):
        load_star(write(tmp_path, "nopixel.star", text))


def test_an_unknown_extension_is_refused(tmp_path):
    """Dispatch is on the extension, and a silently-wrong reader would be worse than none."""
    path = tmp_path / "p.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="expected a .cs or .star"):
        load(path)


def test_a_refit_defocus_replaces_the_stored_one_and_nothing_else(tmp_path):
    """The seam `ctffit` exists for. A refit is a statement about the defocus alone, so the optics,
    the pixel size, the shifts and the stack pointers have to survive it untouched."""
    meta = load(write(tmp_path, "p.star", RELION_31))
    refit = meta.with_ctf(
        defocus_u=np.array([21000.0, 22000.0]),
        defocus_v=np.array([20800.0, 21900.0]),
        astigmatism_angle=np.array([0.5, 0.75]),
    )

    assert np.allclose(refit.defocus, [20900.0, 21950.0])
    assert np.allclose(refit.astigmatism_angle, [0.5, 0.75])
    for carried in ("voltage_in_kv", "spherical_aberration_in_mm", "pixel_size", "shift", "stack_index"):
        assert np.array_equal(getattr(refit, carried), getattr(meta, carried))
    assert np.allclose(meta.defocus, [10100.0, 12050.0])  # and the original is untouched


def test_a_refit_may_leave_the_astigmatism_azimuth_alone(tmp_path):
    """A radial fit has no azimuth to offer, and overwriting it with zero would rotate every
    particle's astigmatism to the x axis rather than declining to update it."""
    meta = load(write(tmp_path, "p.star", RELION_31))
    refit = meta.with_ctf(defocus_u=np.array([1.0, 2.0]), defocus_v=np.array([1.0, 2.0]))

    assert np.array_equal(refit.astigmatism_angle, meta.astigmatism_angle)


def test_a_refit_of_the_wrong_length_is_refused_rather_than_broadcast(tmp_path):
    """A one-value refit would otherwise broadcast into a table that still looks well formed."""
    meta = load(write(tmp_path, "p.star", RELION_31))
    with pytest.raises(ValueError, match="one value per particle"):
        meta.with_ctf(defocus_u=np.array([1.0]), defocus_v=np.array([1.0]))


def test_the_bin_width_is_the_phase_tolerance_over_the_phase_slope():
    """`chi = pi lambda dz k^2`, so `d chi / d dz = pi lambda k^2`. Checked against the closed form
    rather than against the function's own output, and against its `k^-2` scaling."""
    k_max, tolerance = 1.0 / 10.0, 0.1
    expected = tolerance / (np.pi * float(wavelength(300.0)) * k_max**2)
    assert defocus_bin_width(k_max, 300.0, tolerance) == pytest.approx(expected, rel=1e-12)
    # halving the band limit quadruples the width, which is why k_max cannot be defaulted
    assert defocus_bin_width(k_max / 2.0) == pytest.approx(4.0 * defocus_bin_width(k_max), rel=1e-12)
    with pytest.raises(ValueError, match="k_max must be positive"):
        defocus_bin_width(0.0)


def test_grouping_collapses_particles_the_band_cannot_tell_apart(tmp_path):
    """Two particles 200 A apart in defocus share a row at a 10 A band and not at a 4 A one, and
    `exact` never groups at all."""
    meta = load(write(tmp_path, "p.star", RELION_31))
    coarse = transfer_from_meta(meta, k_max=1.0 / 40.0, phase_tolerance=0.5)
    assert coarse.values.shape[0] == 1
    assert coarse.values.shape[1:] == (64, 64)

    fine = transfer_from_meta(meta, k_max=1.0 / 4.0)
    assert fine.values.shape[0] == 2
    exact = transfer_from_meta(meta, k_max=1.0 / 40.0, phase_tolerance=0.5, exact=True)
    assert exact.values.shape[0] == 2


def test_an_envelope_multiplies_every_row_once(tmp_path):
    """Every consumer expects the transfer to carry the whole frequency response, so an envelope
    applied separately downstream would be squared."""
    meta = load(write(tmp_path, "p.star", RELION_31))
    plain = transfer_from_meta(meta, k_max=1.0 / 40.0, phase_tolerance=0.5)
    damped = transfer_from_meta(meta, k_max=1.0 / 40.0, phase_tolerance=0.5, envelope=jnp.full((64, 64), 0.5))
    np.testing.assert_allclose(np.asarray(damped.values), 0.5 * np.asarray(plain.values), rtol=1e-6)


def test_a_transfer_cannot_be_built_on_a_guessed_box(tmp_path):
    """A star with no rlnImageSize records no box, and building one on a default would put every
    zero in the wrong place."""
    text = RELION_31.replace("_rlnImageSize #6\n", "").replace("1 300.0 2.7 0.1 1.5 64", "1 300.0 2.7 0.1 1.5")
    meta = load_star(write(tmp_path, "nobox.star", text))
    assert meta.box_size == 0
    with pytest.raises(ValueError, match="box size is unknown"):
        transfer_from_meta(meta, k_max=1.0 / 10.0)


def test_recentering_goes_through_the_one_ramp_convention():
    """A second copy of the sign would let a stack be recentered one way and scored the other, and
    the two agree only on already-centered particles."""
    rng = np.random.default_rng(0)
    images = jnp.asarray(np.fft.fft2(rng.normal(size=(3, 16, 16))))
    offsets = jnp.asarray([[2.0, -1.0], [0.0, 0.0], [-3.0, 4.0]])
    np.testing.assert_allclose(
        np.asarray(recenter(images, offsets, 1.0)),
        np.asarray(images * phase_ramp(offsets, 16)),
        rtol=1e-6,
    )
    # and it undoes itself, which a sign that flipped somewhere inside would not
    np.testing.assert_allclose(
        np.asarray(recenter(recenter(images, offsets, 1.0), offsets, -1.0)), np.asarray(images), rtol=1e-5
    )


def test_the_shift_sign_is_recovered_from_planted_offsets():
    """Defect: the natural-looking reading of a stored offset is the wrong one, and it lands 4x
    below doing nothing. Here the truth is planted, so the check has a right answer to find."""
    rng = np.random.default_rng(1)
    box = 32
    y, x = np.meshgrid(np.arange(box) - box / 2, np.arange(box) - box / 2, indexing="ij")
    blob = np.exp(-(y**2 + x**2) / (2.0 * 3.0**2))

    offsets = rng.integers(-4, 5, size=(24, 2)).astype(float)
    moved = np.stack([np.roll(blob, (int(dy), int(dx)), axis=(0, 1)) for dy, dx in offsets]) + 0.02 * rng.normal(
        size=(24, box, box)
    )
    images = jnp.asarray(np.fft.fft2(moved))

    # stored as the CORRECTION, so the recentering translation is PLUS it
    stored = jnp.asarray(-offsets)
    sign, swapped, margin, scores = check_shift_sign(images, stored, pixel_size=1.0, band_in_angstroms=8.0)
    assert sign == 1.0
    assert not swapped
    assert scores["plus"] > scores["none"]
    assert scores["plus"] > scores["minus"]
    # the margin is the winner over the RUNNER-UP, not over the do-nothing baseline
    assert margin == pytest.approx(scores["plus"] / max(sorted(scores.values())[-2], 1e-30))
    # this case is right by construction, so its margin is pinned as evidence that the margin must
    # not be thresholded: an easy case must not read as a warning
    assert 1.9 < margin < 2.1

    # handing it the offsets with the axes transposed has to be caught as a swap, not as a sign
    sign, swapped, _, _ = check_shift_sign(images, stored[:, ::-1], pixel_size=1.0, band_in_angstroms=8.0)
    assert swapped


def test_a_tied_field_reports_a_margin_near_one():
    """Pure noise makes the five readings interchangeable, and the statistic still names a winner --
    which is the whole reason the margin is returned. The sign comes back regardless: refusing would
    leave the caller choosing a convention with less information than this had."""
    rng = np.random.default_rng(0)
    images = jnp.asarray(np.fft.fft2(rng.normal(size=(16, 32, 32))))
    offsets = jnp.asarray(rng.integers(-3, 4, size=(16, 2)).astype(float))

    sign, _, margin, scores = check_shift_sign(images, offsets, pixel_size=1.0, band_in_angstroms=8.0)
    assert len(scores) == 5
    assert 1.0 <= margin < 1.5
    assert sign in (-1.0, 0.0, 1.0)


def test_a_cryosparc_file_is_read_in_the_units_the_module_promises(tmp_path):
    """The `.cs` half of the reader. A `.cs` states its angles in RADIANS where a star states
    DEGREES, so the two readers disagree about what they are given and must agree about what they return."""
    save_cs(tmp_path / "particles.cs", cryosparc_particles(3, box=64, pixel_size=1.31))

    meta = load(tmp_path / "particles.cs")

    assert len(meta) == 3
    assert meta.box_size == 64
    assert meta.pixel_size == pytest.approx(1.31)
    assert meta.defocus[0] == pytest.approx(11050.0)
    assert not meta.has_shift
    assert list(meta.stack_path) == ["J5/extract/stack.mrcs"] * 3
    assert list(meta.stack_index) == [0, 1, 2]


def test_the_passthrough_join_survives_the_unsigned_fields_every_particle_table_has(tmp_path):
    """The passthrough join must survive unsigned fields. `recfunctions.append_fields` routes through
    a masked array whose fill value raises `OverflowError` on any unsigned field, and `blob/idx`,
    `blob/shape` and `alignments2D/class` are all unsigned in every cryoSPARC table.
    """
    from numpy.lib import recfunctions

    full = cryosparc_particles(6, classes=(0, 1))
    inherited = [name for name in full.dtype.names if name.startswith(("blob/", "ctf/"))]
    save_cs(tmp_path / "p.cs", recfunctions.drop_fields(full, inherited, usemask=False))
    # reversed, because the join is on `uid`: a reordered output must not silently mismatch, and
    # position would pair every particle with someone else's optics while still reading as a table
    save_cs(tmp_path / "pt.cs", recfunctions.drop_fields(full, ["alignments2D/class"], usemask=False)[::-1])

    meta = load(tmp_path / "p.cs", passthrough=tmp_path / "pt.cs")

    assert len(meta) == 6
    assert list(meta.defocus_u) == list(full["ctf/df1_A"])
    assert list(meta.stack_index) == list(full["blob/idx"])
    assert meta.pixel_size[0] == pytest.approx(2.04)


def test_a_passthrough_covering_other_particles_is_refused(tmp_path):
    save_cs(tmp_path / "p.cs", cryosparc_particles(6))
    other = cryosparc_particles(6)
    other["uid"] += 9000
    save_cs(tmp_path / "pt.cs", other)

    with pytest.raises(ValueError, match="does not cover the same particles"):
        load(tmp_path / "p.cs", passthrough=tmp_path / "pt.cs")


def test_a_refined_shift_converts_through_the_job_it_was_measured_in(tmp_path):
    """cryoSPARC stores an alignment shift in PIXELS of the box that job ran in, which is often a
    downsampled one. Converting it through the stack's pixel size instead is a scale error that
    still produces a recentered stack."""
    base = cryosparc_particles(2, box=64, pixel_size=1.0)
    shifted = np.zeros(
        2, dtype=base.dtype.descr + [("alignments2D/shift", "<f4", (2,)), ("alignments2D/psize_A", "<f4")]
    )
    for name in base.dtype.names:
        shifted[name] = base[name]
    shifted["alignments2D/shift"] = [[3.0, -1.5], [0.0, 2.0]]  # (x, y) on disk, in the JOB's pixels
    shifted["alignments2D/psize_A"] = 4.0  # a 4x downsampled refinement box
    save_cs(tmp_path / "aligned.cs", shifted)

    meta = load(tmp_path / "aligned.cs")

    assert meta.has_shift
    # 3.0 job-pixels at 4.0 A/job-pixel is 12 A, and the reader transposes to (y, x)
    assert meta.shift[0] == pytest.approx([-6.0, 12.0])
    # asked for in the STACK's 1.0 A/px, that is 12 pixels and not 3
    assert meta.shift_in_pixels()[0] == pytest.approx([-6.0, 12.0])


def test_the_passthrough_join_follows_uid_through_an_arbitrary_permutation(tmp_path):
    """A REVERSAL is its own inverse, so it cannot tell the join's permutation from its inverse. The
    join builds the inverse by a scatter rather than by a third argsort -- two sorts over tables of
    1e5 to 1e6 rows -- and getting that backwards pairs every particle with someone else's optics
    while still producing a well-formed table."""
    from numpy.lib import recfunctions

    full = cryosparc_particles(9, classes=(0, 1))
    full["uid"] = np.random.default_rng(3).permutation(full["uid"])
    inherited = [name for name in full.dtype.names if name.startswith(("blob/", "ctf/"))]
    save_cs(tmp_path / "p.cs", recfunctions.drop_fields(full, inherited, usemask=False))
    shuffled = recfunctions.drop_fields(full, ["alignments2D/class"], usemask=False)[
        np.random.default_rng(7).permutation(9)
    ]
    save_cs(tmp_path / "pt.cs", shuffled)

    meta = load(tmp_path / "p.cs", passthrough=tmp_path / "pt.cs")

    # every row keeps the optics its OWN uid carried, in the left table's order
    assert list(meta.defocus_u) == list(full["ctf/df1_A"])
    assert list(meta.stack_index) == list(full["blob/idx"])


def test_a_pools_optics_travel_as_four_scalars_rather_than_four_call_site_medians(tmp_path):
    """`ctffit.fit_micrograph` DEFAULTS the three optics to 300 kV / 2.7 mm / 0.1, so a caller who
    assembles them by hand and forgets one gets a plausible-looking fit at the wrong electron
    wavelength and a `score` that says nothing about it."""
    from cryodisc.particles.metadata import PoolOptics

    data = cryosparc_particles(5, box=64, pixel_size=1.5)
    data["ctf/accel_kv"] = [200.0, 200.0, 300.0, 300.0, 300.0]
    data["ctf/cs_mm"] = 2.7
    data["ctf/amp_contrast"] = 0.07
    save_cs(tmp_path / "p.cs", data)
    meta = load(tmp_path / "p.cs")

    optics = meta.optics()
    # medians, so one deviant row cannot move it; the table stores float32, hence approx
    assert optics.pixel_size == pytest.approx(1.5)
    assert optics.voltage_in_kv == pytest.approx(300.0)
    assert optics.spherical_aberration_in_mm == pytest.approx(2.7)
    assert optics.amplitude_contrast_ratio == pytest.approx(0.07)
    # the micrographs are not the box: an extracted box is often downsampled, so the fitted pixel
    # size is an override rather than the stored one
    assert meta.optics(pixel_size=0.6).pixel_size == pytest.approx(0.6)
    # the raw-column route, for a driver holding a `.cs` with no `blob/*` to build a table from
    assert PoolOptics.from_columns(1.5, data["ctf/accel_kv"], data["ctf/cs_mm"], data["ctf/amp_contrast"]) == optics
    assert set(optics.as_kwargs()) == {
        "pixel_size",
        "voltage_in_kv",
        "spherical_aberration_in_mm",
        "amplitude_contrast_ratio",
    }


def test_row_selection_works_on_a_table_that_carries_no_poses(tmp_path):
    """A cryoSPARC `.cs` records no Euler angles, and `--state`/`--limit` slice regardless.

    `euler` is the one optional per-row field, so selection must guard it. Both index forms are
    checked, since a mask and a slice take different routes in.
    """
    save_cs(tmp_path / "p.cs", cryosparc_particles(6))
    meta = load(tmp_path / "p.cs")
    assert meta.euler is None and not meta.has_euler

    masked = meta[np.array([True, False, True, False, True, False])]
    assert len(masked.pixel_size) == 3
    assert masked.euler is None and not masked.has_euler

    limited = meta[:2]
    assert len(limited.pixel_size) == 2
    assert limited.euler is None


def test_row_selection_carries_poses_when_the_table_has_them(tmp_path):
    """The other half: a star WITH rlnAngle* must still slice its angles alongside everything else,
    or the oracle rung would score particle i at particle j's pose."""
    from conftest import write_relion_star

    eulers = np.arange(18.0).reshape(6, 3)
    star = write_relion_star(
        tmp_path / "p.star",
        [f"{i + 1:06d}@s.mrcs" for i in range(6)],
        np.full(6, 12000.0),
        pixel_size=1.5,
        box_size=32,
        euler=eulers,
    )
    meta = load(star)
    assert meta.has_euler

    kept = meta[np.array([False, True, False, True, False, False])]
    assert kept.euler.shape == (2, 3)
    np.testing.assert_allclose(kept.euler, eulers[[1, 3]])


def test_cs_pose_round_trips_through_the_render_convention():
    """A cryoSPARC rotation vector must come back as the same rotation matrix it went in as.

    The two conventions this crosses both look like the identity and neither is. `rotation_of` builds
    `Rz(c) Ry(b) Rz(a)` -- the angles apply in REVERSE -- so extracting with the textbook
    `Rz(a) Ry(b) Rz(c)` swaps two angles while still producing a valid-looking pose. The theta = 0 and
    theta = pi poles need different algebra. Neither failure is visible downstream: a wrong pose
    renders a plausible template and the amplitude fit merely reads low.
    """
    rng = np.random.default_rng(0)
    vector = rng.normal(size=(500, 3))
    vector *= (rng.uniform(0.0, np.pi, 500) / np.linalg.norm(vector, axis=1))[:, None]
    poles = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-9], [np.pi, 0.0, 0.0], [0.0, np.pi, 0.0], [0.0, 0.0, np.pi]])
    vector = np.vstack([vector, poles])

    angle = np.linalg.norm(vector, axis=1)
    safe = np.where(angle > 1e-12, angle, 1.0)
    axis = vector / safe[:, None]
    skew = np.zeros((len(vector), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
    identity = np.broadcast_to(np.eye(3), (len(vector), 3, 3))
    expected = identity + np.sin(angle)[:, None, None] * skew + (1.0 - np.cos(angle))[:, None, None] * (skew @ skew)
    expected = np.where((angle > 1e-12)[:, None, None], expected, identity)
    if CS_POSE_TRANSPOSE:
        expected = np.swapaxes(expected, 1, 2)

    rendered = euler_from_relion(relion_euler_from_cs_pose(vector))
    recovered = np.stack([np.asarray(rotation_of(jnp.asarray(row)).as_matrix()) for row in rendered])
    assert np.abs(recovered - expected).max() < 1e-5


def test_load_cs_reads_a_pose_and_a_2d_pool_carries_none(tmp_path):
    """`alignments3D/pose` becomes `euler`, and `alignments2D` alone must NOT.

    A 2D-classified pool carries an in-plane angle and no viewing direction, so reading its alignment
    as a full pose would silently pin two of the three angles at zero and hand `--known-pose` an
    orientation it never measured.
    """
    rng = np.random.default_rng(1)
    pose = rng.normal(size=(7, 3)) * 0.4

    with_pose = cryosparc_particles(7, extra={"alignments3D/pose": pose.astype(np.float32)})
    path = tmp_path / "with_pose.cs"
    save_cs(path, with_pose)
    meta = load(path)
    assert meta.has_euler
    assert meta.euler is not None and meta.euler.shape == (7, 3)
    # tolerance in DEGREES: the file stores float32, and any real pose error is degrees, not
    # millionths of one, so a tighter bound would measure numpy rather than the convention
    np.testing.assert_allclose(meta.euler, relion_euler_from_cs_pose(pose), rtol=0, atol=1e-4)

    flat = cryosparc_particles(7, extra={"alignments2D/pose": rng.normal(size=7).astype(np.float32)})
    flat_path = tmp_path / "flat.cs"
    save_cs(flat_path, flat)
    assert not load(flat_path).has_euler


def test_a_limit_keeps_that_many_particles_on_every_route_not_only_the_two_state_one(tmp_path):
    """A `--limit` must keep that many particles with no `--state`, or exactly one, not only with two.

    When there are no labels to size the mask from, it must be sized by the pool rather than by
    `limit`, or boolean-indexing an `n`-row table with a `limit`-length mask raises.
    """
    data = cryosparc_particles(5, stack="000_particles.mrcs")
    meta = load(save_cs(tmp_path / "one.cs", data))

    for states in (["000"], None):
        kept, labels = select_states(meta, states, limit=2)
        assert len(kept) == 2, f"states={states}"
        assert labels is None, "one state, or none, is a pool of known composition and not a labeled one"

    # a limit above the pool keeps the pool, and zero means no limit at all
    assert len(select_states(meta, ["000"], limit=99)[0]) == 5
    assert len(select_states(meta, None, limit=0)[0]) == 5


def test_a_limit_counts_per_state_and_not_over_the_pool(tmp_path):
    """The reason the mask exists. A benchmark's star orders its stacks one after another, so a flat
    truncation of a two-state selection keeps only the first state and labels it as two."""
    first = cryosparc_particles(6, stack="000_particles.mrcs")
    second = cryosparc_particles(6, stack="050_particles.mrcs")
    both = np.concatenate([first, second])
    both["uid"] = np.arange(len(both)) + 100
    meta = load(save_cs(tmp_path / "two.cs", both))

    kept, labels = select_states(meta, ["000", "050"], limit=2)

    assert len(kept) == 4, "two of each, not the first two of the pool"
    assert labels.tolist() == [0, 0, 1, 1]


def test_a_band_limit_stops_the_score_where_k_max_never_did():
    """`--k-max` sets the DEFOCUS GROUPING, not a band limit: changing it barely moves the amplitude
    or the SNR. `band_limit_envelope` is what actually removes modes, and it multiplies the transfer,
    so a zero there is a mode the whitened inner product never sees.

    Hard, not tapered: this is a projection onto the modes the data supports, and the whitening
    already weights what survives."""
    from cryodisc.particles.metadata import band_limit_envelope

    box, pixel = 64, 2.0
    mask = np.asarray(band_limit_envelope(box, pixel, 0.1))

    assert mask.shape == (box, box)
    assert set(np.unique(mask).tolist()) == {0.0, 1.0}, "a hard cutoff, so nothing is partly scored"
    assert mask[0, 0] == 1.0, "DC is always inside"

    # the kept set is exactly the disc of that radius, which is what makes it a band and not a box
    fy, fx = np.fft.fftfreq(box, d=pixel)[:, None], np.fft.fftfreq(box, d=pixel)[None, :]
    assert np.array_equal(mask, (np.sqrt(fy**2 + fx**2) <= 0.1).astype(float))

    # a wider band keeps strictly more, and even at Nyquist the fft CORNERS are dropped -- they are
    # above Nyquist in every direction but the two axes
    nyquist = 0.5 / pixel
    at_nyquist = np.asarray(band_limit_envelope(box, pixel, nyquist))
    assert at_nyquist.mean() > mask.mean()
    assert at_nyquist.mean() < 1.0, "the corners lie outside the Nyquist circle"

    with pytest.raises(ValueError, match="must be positive"):
        band_limit_envelope(box, pixel, 0.0)
