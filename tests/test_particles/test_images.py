"""Checks on the loader that follows a metadata table to its pixels.

Stacks are written to a tmp_path and read back, so the round trip covers the file format as well as
the indexing. What the tests are really about is the two ways a load goes wrong silently: rows
delivered in the wrong ORDER, which no shape check catches, and a `root` that resolves to somebody
else's stack.
"""

from dataclasses import replace

import numpy as np
import pytest

from cryodisc.particles.images import load_stack

mrcfile = pytest.importorskip("mrcfile")

BOX = 8


def write_stack(path, images):
    """A .mrcs holding `images`, as an extraction job would write it."""
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(images, np.float32))


def meta_for(tmp_path, stack_path, stack_index, *, sign=None, box_size=BOX, source="test.cs"):
    """A ParticleMeta carrying only what `load_stack` reads, the rest at plausible constants."""
    from cryodisc.particles.metadata import ParticleMeta

    n = len(stack_index)
    ones = np.ones(n)
    return ParticleMeta(
        defocus_u=ones * 12000.0,
        defocus_v=ones * 12000.0,
        astigmatism_angle=np.zeros(n),
        phase_shift=np.zeros(n),
        voltage_in_kv=ones * 300.0,
        spherical_aberration_in_mm=ones * 2.7,
        amplitude_contrast_ratio=ones * 0.07,
        b_factor=np.zeros(n),
        pixel_size=ones,
        box_size=box_size,
        shift=np.zeros((n, 2)),
        has_shift=False,
        sign=ones if sign is None else np.asarray(sign, float),
        stack_path=np.asarray(stack_path, object),
        stack_index=np.asarray(stack_index, int),
        source=source,
    )


def test_rows_come_back_in_the_metadatas_order_not_the_files(tmp_path):
    """The load groups by file to avoid a seek per particle, and the regrouping is where an order
    bug would live -- it would still return the right images, just attached to the wrong metadata."""
    for name, base in (("a.mrcs", 0.0), ("b.mrcs", 100.0)):
        write_stack(tmp_path / name, [np.full((BOX, BOX), base + i) for i in range(4)])
    # deliberately interleaved, so grouping cannot preserve the order by accident
    meta = meta_for(tmp_path, ["b.mrcs", "a.mrcs", "b.mrcs", "a.mrcs"], [3, 1, 0, 2])

    loaded = load_stack(meta, tmp_path)

    assert [float(image[0, 0]) for image in loaded] == [103.0, 1.0, 100.0, 2.0]


def test_the_sign_flag_is_applied_so_one_stack_has_one_contrast_convention(tmp_path):
    write_stack(tmp_path / "a.mrcs", [np.full((BOX, BOX), 5.0), np.full((BOX, BOX), 7.0)])
    meta = meta_for(tmp_path, ["a.mrcs", "a.mrcs"], [0, 1], sign=[1.0, -1.0])

    assert [float(image[0, 0]) for image in load_stack(meta, tmp_path)] == [5.0, -7.0]
    assert [float(image[0, 0]) for image in load_stack(meta, tmp_path, apply_sign=False)] == [5.0, 7.0]


def test_an_absolute_stored_path_ignores_the_root(tmp_path):
    """A table mixing absolute and relative paths resolves both, which is what a re-exported set has."""
    write_stack(tmp_path / "a.mrcs", [np.full((BOX, BOX), 3.0)])
    meta = meta_for(tmp_path, [str(tmp_path / "a.mrcs")], [0])

    assert float(load_stack(meta, tmp_path / "nowhere")[0][0, 0]) == 3.0


def test_a_single_image_mrc_is_not_transposed_into_a_stack_of_columns(tmp_path):
    """`mrcfile` hands a one-image file back 2D, and appending the axis rather than prepending it
    would turn one 8x8 image into eight images of 8x1 -- a shape error nothing downstream sees."""
    write_stack(tmp_path / "one.mrc", np.full((BOX, BOX), 2.0))
    meta = meta_for(tmp_path, ["one.mrc"], [0])

    assert load_stack(meta, tmp_path).shape == (1, BOX, BOX)


def test_a_missing_stack_names_the_root_because_that_is_usually_the_cause(tmp_path):
    meta = meta_for(tmp_path, ["gone.mrcs"], [0])
    with pytest.raises(FileNotFoundError, match="root="):
        load_stack(meta, tmp_path)


def test_an_index_past_the_end_of_its_file_is_refused(tmp_path):
    write_stack(tmp_path / "a.mrcs", [np.zeros((BOX, BOX)), np.zeros((BOX, BOX))])
    meta = meta_for(tmp_path, ["a.mrcs"], [7])
    with pytest.raises(ValueError, match="out of step"):
        load_stack(meta, tmp_path)


def test_a_box_that_disagrees_with_the_metadata_is_refused(tmp_path):
    """A stack extracted at a different box than the table records is a mispaired file, and it would
    otherwise reach the scorer as a silently rescaled particle."""
    write_stack(tmp_path / "a.mrcs", [np.zeros((BOX * 2, BOX * 2))])
    meta = meta_for(tmp_path, ["a.mrcs"], [0])
    with pytest.raises(ValueError, match="against the expected"):
        load_stack(meta, tmp_path)


def test_a_table_that_records_no_box_checks_nothing_unless_told(tmp_path):
    write_stack(tmp_path / "a.mrcs", [np.zeros((BOX, BOX))])
    meta = replace(meta_for(tmp_path, ["a.mrcs"], [0]), box_size=0)

    assert load_stack(meta, tmp_path).shape == (1, BOX, BOX)
    with pytest.raises(ValueError, match="against the expected"):
        load_stack(meta, tmp_path, expected_box_size=BOX * 2)


def test_a_micrograph_reads_from_either_format_and_a_movie_collapses_to_one_frame(tmp_path):
    """`score_pool --refit-ctf` refits a micrograph, which goes through this.
    A movie or a multi-frame stack collapses to its FIRST frame: a CTF fit wants the
    aligned sum, and a stack of frames is not one."""
    from cryodisc.particles.images import read_micrograph

    field = np.random.default_rng(0).normal(size=(16, 24)).astype(np.float32)

    np.save(tmp_path / "m.npy", field)
    np.testing.assert_allclose(read_micrograph(tmp_path / "m.npy"), field)

    with mrcfile.new(tmp_path / "m.mrc", overwrite=True) as handle:
        handle.set_data(field)
    np.testing.assert_allclose(read_micrograph(tmp_path / "m.mrc"), field)
    assert read_micrograph(str(tmp_path / "m.mrc")).shape == field.shape

    movie = np.stack([field, field + 1.0, field + 2.0])
    with mrcfile.new(tmp_path / "movie.mrc", overwrite=True) as handle:
        handle.set_data(movie.astype(np.float32))
    np.testing.assert_allclose(read_micrograph(tmp_path / "movie.mrc"), field)
    np.save(tmp_path / "movie.npy", movie)
    np.testing.assert_allclose(read_micrograph(tmp_path / "movie.npy"), field)
