"""Deriving optics off a star, and refusing when they cannot be described by one set of numbers."""

import pytest

from cryodisc.particles.star import star_columns, star_optics

HEADER = """
data_

loop_
_rlnImageName #1
_rlnVoltage #2
_rlnSphericalAberration #3
_rlnAmplitudeContrast #4
_rlnMagnification #5
_rlnDetectorPixelSize #6
"""


def _write(tmp_path, rows, header=HEADER):
    path = tmp_path / "particles.star"
    path.write_text(header + "\n".join(rows) + "\n")
    return path


def test_the_pixel_size_is_microns_over_magnification():
    """EMPIAR-10317's own numbers: 14.0 / 1.299907e5 x 1e4 is the 1.077 A that was hardcoded."""
    assert 14.0 / 1.299907e5 * 1e4 == pytest.approx(1.077, abs=5e-4)


def test_derives_the_four_numbers(tmp_path):
    path = _write(tmp_path, ["1@s.mrcs 300.0 2.7 0.1 1.299907e5 14.0"] * 3)
    optics = star_optics(path)
    assert optics["psize_A"] == pytest.approx(1.077, abs=5e-4)
    assert optics["accel_kv"] == 300.0
    assert optics["cs_mm"] == 2.7
    assert optics["amp_contrast"] == 0.1


def test_columns_are_read_by_their_declared_index(tmp_path):
    path = _write(tmp_path, ["1@s.mrcs 300.0 2.7 0.1 1.299907e5 14.0"])
    assert star_columns(path)["rlnDetectorPixelSize"] == 5
    assert star_columns(path)["rlnImageName"] == 0


def test_refuses_two_acquisitions_in_one_file(tmp_path):
    """One set of numbers cannot describe two pixel sizes, and averaging them would be silent."""
    path = _write(
        tmp_path,
        ["1@s.mrcs 300.0 2.7 0.1 1.299907e5 14.0", "2@s.mrcs 300.0 2.7 0.1 9.79021e4 16.633566"],
    )
    with pytest.raises(SystemExit, match="not constant"):
        star_optics(path)


def test_refuses_a_star_with_no_optics_columns(tmp_path):
    """EMPIAR-10076 is exactly this: ten columns and not one of them an angle or a magnification."""
    header = "\ndata_\n\nloop_\n_rlnImageName #1\n_rlnDefocusU #2\n"
    path = _write(tmp_path, ["1@s.mrcs 15301.1"], header=header)
    with pytest.raises(SystemExit, match="cannot be derived"):
        star_optics(path)


def test_refuses_a_header_with_no_rows(tmp_path):
    path = _write(tmp_path, [])
    with pytest.raises(SystemExit, match="no data rows"):
        star_optics(path)
